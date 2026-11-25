"""Server entry (./start.sh runs it in the image on the head = rank 0 and the worker = rank 1).
Settings: serve.toml + serve.local.toml (+ --set); the link addresses come from the launcher's probe (E55_HEAD_IP =
the head's CX7 address for NCCL / RDMA bootstrap, E55_PEER_IP = the worker's, for the command link).

Start: rank 0's HTTP front answers at once (/health 503 "loading"); both ranks open the command link and compare
settings and sources, load the engine with the same settings (TP2 collectives pair up), and rank 0 turns ready
(/health 200). Rank 1 replays rank 0's engine calls (link.py); rank 0 runs the scheduler — every engine call — in the
main thread, the thread that initialised the engine.
Stop (SIGTERM / SIGINT to rank 0, e.g. ./stop.sh → docker stop): new requests get 503, running ones get
server.drain_seconds to finish and are then cut with a normal ending; rank 0 closes the link, both ranks destroy the
engine (RDMA proxy and QPs, NCCL) and exit. A second signal leaves at once. A signal while loading leaves at once.
If the other rank goes away, a watcher thread fails the running requests (503 / an error event) and leaves at once."""
import os
import argparse
import glob
import hashlib
import json
import logging
import signal
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "engine", "tests"), os.path.join(ROOT, "tools")]
import qlog  # noqa: E402

import link  # noqa: E402
import settings  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--model", default="/models/ERNIE-5.5-Flash-Preview", help="checkpoint folder in the container")
ap.add_argument("--table", default="/work/results/tensor_table.tsv")
ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a setting")
a = ap.parse_args()
try:
    S = settings.load(ROOT, a.set)
except settings.SettingsError as e:
    qlog.fail("serve", str(e))
sv, cl, dc = S["server"], S["cluster"], S["decode"]
# DSpark (decode.dspark): every slot verifies up to block + 1 rows a round, so the engine keeps logits (and draft rows)
# for slots × (block + 1) rows; plain decoding: one row a slot
BLOCK = dc["block"] if dc["dspark"] else 0
if dc["dspark"] and not 1 <= BLOCK <= 15:
    qlog.fail("serve", f"decode.block must be 1 … 15, got {BLOCK}")
if dc["verify"] not in ("cost", "gate"):
    qlog.fail("serve", f"decode.verify must be \"cost\" or \"gate\", got {dc['verify']!r}")
# P168 copy drafts (decode.copy, with DSpark): a copy round verifies up to decode.copy_rows rows of one request
COPY = bool(BLOCK and dc["copy"])
if COPY and not 2 <= dc["copy_rows"] <= 64:
    qlog.fail("serve", f"decode.copy_rows must be 2 … 64, got {dc['copy_rows']}")
LOGIT_ROWS = max(sv["slots"] * (BLOCK + 1), dc["copy_rows"] if COPY else 0)
MAX_CONTEXT = 1048576  # the model's positions (YaRN factor 16 × 65,536)
if not 1 <= sv["context"] <= MAX_CONTEXT:
    qlog.fail("serve", f"server.context {sv['context']:,}: 1 … {MAX_CONTEXT:,} tokens (the model's limit)")
# P125 kv_pool (patches.kv_pool; the engine reads the same switch from E55_PATCHES, set from [patches] by the launcher)
KV_POOL = S["patches"].get("kv_pool", settings.patch_names(ROOT).get("kv_pool", False))
for x in os.environ.get("E55_PATCHES", "").split(","):
    if x.strip() in ("+kv_pool", "-kv_pool"):
        KV_POOL = x.strip()[0] == "+"
# Refuse what cannot fit instead of allocating until it fails. GB10: CUDA allocates from free pages only (the page cache
# does not make room — drop caches before a start), so the check reads MemFree. A rank takes ENGINE_GIB (measured
# 2026-11-24, tests/mem_report.py at 4 slots × 64K, chunk 4096, DSpark: MemFree drop over init + dspark_init + the first
# steps, 88.4 GiB on the larger rank, less those slots' 1.2 GiB of caches) + 1.5 GiB per 4096 chunk rows above 4096
# (estimate: activations / FFN scratch) + its slots' fixed state (37 × (128-row window + the compressor's 128 × 1024
# fp32 state)) + the caches: per token 6 DSA caches × (656 + 68) B + 37 HCA compressed rows × 584 / 128 — with the pool
# at least one chunk's prompt (kv_pool_init takes the rest and checks again), else slots × context up front.
ENGINE_GIB = 87.2 + max(0, sv["chunk"] - 4096) * 1.5 / 4096
with open("/proc/meminfo") as f:
    free = next(int(x.split()[1]) for x in f if x.startswith("MemFree:")) / 1048576
toks = (max(sv["chunk"], 8192) if KV_POOL else sv["slots"] * sv["context"])
need = (toks * (6 * 724 + 37 * 584 / 128) + sv["slots"] * 37 * (128 * 584 + 128 * 4096)) / 2**30
room = free - ENGINE_GIB - sv["memory_free_gib"]
if need > room:
    qlog.fail("serve", (f"no room for the KV pool: about {room:.1f} GiB left after the engine and memory_free_gib "
                        f"{sv['memory_free_gib']:g}" if KV_POOL else
                        f"{sv['slots']} slots × {sv['context']:,} tokens need {need:.1f} GiB of caches per machine, "
                        f"about {room:.1f} GiB fit — fewer slots, a smaller context or patches.kv_pool"))
head, peer = os.environ.get("E55_HEAD_IP", ""), os.environ.get("E55_PEER_IP", "")
if not head or not peer:
    qlog.fail("serve", "E55_HEAD_IP / E55_PEER_IP not set — start the server with ./start.sh")
LOG = "/work/results/serve"
os.makedirs(LOG, exist_ok=True)
LOG_HOST = os.environ.get("E55_LOG_HOST", LOG)  # the same folder as the host sees it (for the screen)
if a.rank == 0:
    commit = os.environ.get("E55_COMMIT", "")
    rails = len([x for x in os.environ.get("NCCL_IB_HCA", "").split(",") if x])
    qlog.banner([f"plus {settings.version()}{' · ' + commit if commit else ''} · ERNIE-5.5-Flash engine",
                 f"model    {a.model}",
                 f"image    {os.environ.get('E55_IMAGE', '?')}",
                 f"cluster  TP2 · head {head} ↔ worker {peer}" + (f" · {rails} rail{'s' if rails > 1 else ''}" if rails else "")])


def load_engine(rank):
    """The engine module: built here on the first start (or after its sources change), cached in /work/cache/ext."""
    import ext
    why = ext.stale("engine", ext.engine_sources())
    if why and rank == 0:
        qlog.info("build", f"engine · compiling ({why}, a few minutes) · log {LOG_HOST}/build0.log")
    t0 = time.time()
    m = ext.load("engine", ext.engine_sources(), log=f"{LOG}/build{rank}.log", ldflags=ext.ENGINE_LDFLAGS)
    if why and rank == 0:
        qlog.ok("build", f"engine · built · {time.time() - t0:.0f} s")
    return m


def digest():
    """What both ranks must agree on: settings, engine switches from the environment, and the sources."""
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(ROOT, "engine", "src", "**", "*"), recursive=True) +
                    glob.glob(os.path.join(HERE, "*.py")) + [os.path.join(ROOT, "engine", "tests", "bind", "engine.cpp")]):
        if os.path.isfile(f):
            h.update(f.encode())
            with open(f, "rb") as fh:
                h.update(fh.read())
    env = {k: os.environ.get(k, "") for k in ("E55_PATCHES", "E55_RDMA_INIT")}
    st = {k: v for k, v in S.items() if k != "paths"}
    st["server"] = {k: v for k, v in st["server"].items() if k != "api_key"}  # rank 0 only (HTTP)
    return {"settings": json.dumps(st, sort_keys=True),
            "engine switches": json.dumps(env, sort_keys=True), "sources": h.hexdigest(), "model": a.model}


def close_engine(eng, who):
    """Destroy the engine; if that hangs (a peer stuck mid-collective), leave anyway after 20 s."""
    t = threading.Timer(20, lambda: (qlog.error(who, "engine close hung — leaving"), os._exit(1)))
    t.daemon = True
    t.start()
    eng.close()
    t.cancel()


# ---- rank 1 ------------------------------------------------------------------------------------------------------
if a.rank == 1:
    sock = link.accept(peer, cl["link_port"])
    calls = link.reader(link.hello(sock, 1, digest()))
    eng = load_engine(1)
    eng.init(a.model, a.table, 1, head, sv["slots"], sv["context"], max(sv["chunk"], LOGIT_ROWS), LOGIT_ROWS,
             cl["nccl_port"], KV_POOL)
    n = link.replay(eng, calls)  # (dspark_init, then kv_pool_init, come from rank 0 as its first calls)
    close_engine(eng, "serve")
    qlog.ok("serve", f"rank 1 stopped · {n:,} engine calls")
    sys.stdout.flush()
    os._exit(0)

# ---- rank 0 ------------------------------------------------------------------------------------------------------
# signals go to one waiting thread, never into the middle of an engine call; threads made later inherit this mask.
# (Rank 1 keeps the default: a signal ends it at once.)
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
import uvicorn  # noqa: E402

from app import Front, build  # noqa: E402

front = Front(sv["api_key"])
logging.basicConfig(filename=f"{LOG}/http.log", level=logging.INFO)  # uvicorn's lines: log file only
http = uvicorn.Server(uvicorn.Config(front, host=sv["host"], port=sv["port"], log_config=None, access_log=False,
                                     lifespan="off"))
threading.Thread(target=http.run, daemon=True, name="http").start()
t0 = time.time()
while not http.started:
    if time.time() - t0 > 20 or not any(t.name == "http" and t.is_alive() for t in threading.enumerate()):
        qlog.fail("serve", f"HTTP front did not start on {sv['host']}:{sv['port']} (port in use?) — {LOG}/http.log")
    time.sleep(0.05)

sched = None
closing = threading.Event()


def peer_gone(msg):
    front.state = "failed"
    if sched is not None:
        sched.fail_all(f"the server stopped: {msg}")
        time.sleep(1.0)  # let the HTTP thread send the errors
    link.gone(msg + " — stopping")


def on_signal():
    signal.sigwait({signal.SIGTERM, signal.SIGINT})
    if sched is None:
        qlog.warn("serve", "stopped while loading")
        sys.stdout.flush()
        os._exit(0)
    front.state = "stopping"
    run, wait = sched.counts()
    qlog.info("serve", f"stopping · {run + wait} request(s) get up to {sv['drain_seconds']} s to finish")
    sched.drain(time.time() + sv["drain_seconds"])
    signal.sigwait({signal.SIGTERM, signal.SIGINT})
    qlog.warn("serve", "second signal — leaving now")
    sys.stdout.flush()
    os._exit(1)


threading.Thread(target=on_signal, daemon=True, name="signals").start()

sock = link.connect(peer, cl["link_port"])
link.hello(sock, 0, digest())
link.watch(sock, closing, peer_gone)

from chat import Chat  # noqa: E402
from scheduler import Scheduler  # noqa: E402

eng = load_engine(0)
eng.init(a.model, a.table, 0, head, sv["slots"], sv["context"], max(sv["chunk"], LOGIT_ROWS), LOGIT_ROWS,
         cl["nccl_port"], KV_POOL)
chat = Chat(a.model)
m = link.Mirror(eng, sock)
if BLOCK:
    t0 = time.time()
    m.dspark_init()  # the draft (model.mtp.*) on both ranks
    vl = (("P143 round cost" if dc["verify"] == "cost" and eng.patch_on("verify_joint") else
           "P121 gate " + str(dc["gate"]) if dc["verify"] == "gate" else "each on B = 1's cost (P143 off)")
          if eng.patch_on("conf_gate") else "every draft (P121 off)")
    qlog.ok("engine", f"DSpark draft · block {BLOCK} · verify lengths {vl} · {time.time() - t0:.0f} s")
from kvguard import KvGuard  # noqa: E402
# P125: last thing before serving (after dspark_init when the server drafts): both ranks run the serving paths once
# (with DSpark rounds: logit_rows = slots × (block + 1) ≥ 2 · slots), so their lazy allocations are taken, then the pool
# takes the room left (the same chunk count on both)
if KV_POOL:
    m.kv_pool_init(float(sv["memory_free_gib"]), float(sv["kv_pool_gib"]), True)
kv = KvGuard(eng, sv["context"], sv["output_reserve"], float(sv["memory_floor_gib"]))
sched = Scheduler(m, chat.eos, sv["slots"], sv["context"], sv["chunk"], think_end=chat.think_end,
                  loop_guard=S["thinking"]["loop_guard"], block=BLOCK, gate=dc["gate"], kv=kv, verify=dc["verify"],
                  checkpoints=sv["checkpoints"], checkpoint_every=sv["checkpoint_every"],
                  prefill_marks=bool(sv["prefill_marks"]), copy=COPY and eng.patch_on("copy_draft"),
                  copy_rows=dc["copy_rows"], copy_match=dc["copy_match"])
front.app = build(chat, sched, sv["served_name"], kv.max_tokens(), S["thinking"], int(sv["max_tokens"]))
d = chat.defaults
pool = (f"KV pool {kv.total * 2 / 1024:.1f} GiB · ≤ {kv.max_tokens():,} tokens a request" if kv.pool else
        f"{sv['context'] // 1024}K each")
qlog.ok("serve", f"http://{sv['host']}:{sv['port']}/v1 · {sv['served_name']} · {sv['slots']} slots · {pool} · "
                 f"chunk {sv['chunk']} · {'DSpark block ' + str(BLOCK) if BLOCK else 'plain decoding'}"
                 f"{' · copy ≤ ' + str(dc['copy_rows']) + ' rows' if sched.copy else ''} · default "
                 f"sampling T {d['temperature']} · top_k {d['top_k']} · top_p {d['top_p']}")
front.state = "ready"

sched.run()  # returns after a drain
front.state = "stopped"
time.sleep(0.5)  # the last chunks of cut requests
closing.set()
m.close()  # rank 1 closes its engine and exits
close_engine(eng, "serve")
st = sched.stats
qlog.ok("serve", f"stopped · {st['requests']:,} requests served")
sys.stdout.flush()
os._exit(0)
