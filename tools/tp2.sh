#!/bin/bash
# tp2.sh <name> <cmd...> — (Mac, development) the same command on both ranks: A = rank 0 (head), B = rank 1, each in
# its own ernie55-plus:dev container (host network, RDMA devices; NCCL over the CX7 rails read at start with
# server/settings.py probe, nothing stored). "--rank R --head-ip <A's CX7 address>" is appended. Rank 0's output
# streams here; both ranks' logs → <work>/results/runs/tp2-<name>-<time>/ on A. Containers are removed at the end.
set -uo pipefail
# one TP2 run at a time from this Mac (parallel agents, private copies included): two engines on A would not fit in
# its memory, and A is the site's hub — every run waits here for the previous one (lockf removes the file afterwards)
if [ -z "${E55_TP2_LOCKED:-}" ]; then
  E55_TP2_LOCKED=1 exec lockf -t 14400 "${TMPDIR:-/tmp}/ernie55-tp2.lock" "$0" "$@"
fi
cd "$(dirname "$0")/.." && . tools/qlog.sh && . tools/dev.sh
n=${1:?name}; shift
# E55_TREE=<name>: a private tree instead — the Mac repo → <work>/dev/<name>/src on A and B (as tools/agent.sh), run
# there (cwd /work/dev/<name>/src, extensions cached in /work/dev/<name>/ext); <work>/src is not touched
SRC=/work/src
if [ -n "${E55_TREE:-}" ]; then
  [[ $E55_TREE =~ ^[a-z0-9-]+$ ]] || bad tp2 "E55_TREE: lowercase letters, digits, -"
  D=$WORK/dev/$E55_TREE; SRC=/work/dev/$E55_TREE/src
  X=(--exclude .git --exclude .claude --exclude .venv --exclude '__pycache__' --exclude .DS_Store --exclude /ref
     --exclude /results --exclude build --exclude CLAUDE.md --exclude tools/site.local.sh)
  on A "mkdir -p $D/src $D/ext" && rsync -a --delete "${X[@]}" ./ "$HEAD:$D/src/" || bad tp2 "copy to A failed"
  on A "ssh -n -o BatchMode=yes $WORKER 'mkdir -p $D/src $D/ext' && rsync -a --delete -e 'ssh -o BatchMode=yes' $D/src/ $WORKER:$D/src/" \
    || bad tp2 "copy A→B failed"
fi
# E55_SYNC_FILES="a b …": copy only these repo paths (scripts) to <work>/src on A and B, leaving the engine sources (and
# the module built from them) as they are — for runs that must not pick up other work in progress on the Mac
if [ -n "${E55_TREE:-}" ]; then
  :
elif [ -n "${E55_SYNC_FILES:-}" ]; then
  for f in $E55_SYNC_FILES; do [ -f "$f" ] || bad tp2 "E55_SYNC_FILES: $f not found"; done
  rsync -aR $E55_SYNC_FILES "$HEAD:$WORK/src/" || bad tp2 "copy to A failed"
  on A "cd $WORK/src && rsync -aR -e 'ssh -o BatchMode=yes' $E55_SYNC_FILES $WORKER:$WORK/src/" || bad tp2 "copy A→B failed"
else
  tools/sync.sh >/dev/null || bad tp2 "sync failed"
fi
rails() { python3 -c 'import json,sys; r=json.loads(sys.argv[1]); assert r; print(" ".join(x["ip"] for x in r))' "$1" 2>/dev/null; }
H=$(on A "python3 - probe ${WORKER_CX[*]}" < server/settings.py) || bad tp2 "probe on A failed"
hip=$(rails "$H") || bad tp2 "probe on A found no rails"
# B through A without `ssh -n` (on B), so settings.py reaches it on stdin
W=$(ssh -o BatchMode=yes "$HEAD" "ssh -o BatchMode=yes $WORKER $(printf %q "python3 - probe $hip")" < server/settings.py) \
  || bad tp2 "probe on B failed"
rails "$W" >/dev/null || bad tp2 "probe on B found no rails"
envs() { python3 -c '
import json, sys
r = json.loads(sys.argv[1])
e = {"NCCL_IB_DISABLE": "0", "NCCL_IB_HCA": ",".join(x["hca"] for x in r), "NCCL_IB_GID_INDEX": str(r[0]["gid"]),
     "NCCL_IB_ROCE_VERSION_NUM": "2", "NCCL_IB_ADDR_FAMILY": "AF_INET", "NCCL_SOCKET_IFNAME": r[0]["dev"],
     "NCCL_CUMEM_ENABLE": "0", "NCCL_NVLS_ENABLE": "0", "NCCL_IGNORE_CPU_AFFINITY": "1"}
if len(r) > 1: e.update(NCCL_IB_MERGE_NICS="1", NCCL_CROSS_NIC="1")
print(" ".join(f"-e {k}={v}" for k, v in e.items()))' "$1"; }
head_ip=${hip%% *}
ok tp2 "rails A $(python3 -c 'import json,sys; print(",".join(x["hca"] for x in json.loads(sys.argv[1])))' "$H") · B $(python3 -c 'import json,sys; print(",".join(x["hca"] for x in json.loads(sys.argv[1])))' "$W") · head $head_ip"
ts=$(date +%Y%m%d-%H%M%S); RUN=$WORK/results/runs/tp2-$n-$ts
C="ernie55-tp2-$n"
common="--label project=ernie-5.5-flash --user 1000:1000 -e HOME=/tmp -e HF_HOME=/tmp/hf --gpus all --ipc=host --network host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 -v $WORK:/work -v $MODELS:/models:ro -w $SRC${E55_TREE:+ -e E55_EXT_CACHE=/work/dev/$E55_TREE/ext}"
pass=""; for v in E55_PATCHES E55_SYNC NCCL_DEBUG CUDA_LAUNCH_BLOCKING; do [ -n "${!v:-}" ] && pass="$pass -e $v=${!v}"; done
cmd="$(printf '%q ' "$@")"
# E55_GPU_LOCK=1: A's and B's GPU locks (tools/agent.sh --gpu) are both taken BEFORE either rank starts (A first, then
# B — one order everywhere), held by a sleep on each node for the whole run and released on exit. Starting rank 1 under
# B's lock while rank 0 did not wait for A's let the two ranks start at different times (or run beside A's other work).
lk=""
if [ "${E55_GPU_LOCK:-0}" = 1 ]; then
  lk=/tmp/ernie55-tp2lk-$n-$$.pid
  hold() { on "$1" "rm -f $lk; nohup flock $WORK/cache/gpu.lock bash -c 'echo \$\$ > $lk; exec sleep 14400' >/dev/null 2>&1 &
    until test -s $lk; do sleep 2; done"; }
  release() { for x in A B; do on $x "test -s $lk && kill \$(cat $lk) 2>/dev/null; rm -f $lk"; done; }
  trap release EXIT; trap 'exit 130' INT TERM
  hold A || bad tp2 "GPU lock on A failed"
  hold B || bad tp2 "GPU lock on B failed"
  ok tp2 "GPU locks A · B held"
fi
on B "docker rm -f $C >/dev/null 2>&1; docker run -d --name $C $common $(envs "$W") $pass ernie55-plus:dev $cmd --rank 1 --head-ip $head_ip >/dev/null" \
  || bad tp2 "rank 1 did not start"
on A "mkdir -p $RUN; docker rm -f $C >/dev/null 2>&1; docker run --name $C $common $(envs "$H") $pass ernie55-plus:dev $cmd --rank 0 --head-ip $head_ip 2>&1 | tee $RUN/rank0.log"
rc=$?
on B "docker wait $C >/dev/null 2>&1; docker logs $C" > /tmp/.e55_r1.$$ 2>&1; on A "cat > $RUN/rank1.log" < /tmp/.e55_r1.$$; rm -f /tmp/.e55_r1.$$
r1=$(on B "docker inspect -f {{.State.ExitCode}} $C 2>/dev/null")
on A "docker rm -f $C >/dev/null 2>&1"; on B "docker rm -f $C >/dev/null 2>&1"
[ "${r1:-1}" = 0 ] || warn tp2 "rank 1 exit ${r1:-?} · log $RUN/rank1.log"
t tp2 "logs $RUN"
