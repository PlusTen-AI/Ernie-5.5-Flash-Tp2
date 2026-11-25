# AGENTS.md — for an AI agent installing or running this server

You are setting up ERNIE 5.5 Flash on two GB10 machines (DGX Spark) for a person. This page is the whole procedure:
each step, the command, what a good result looks like, and when to stop and ask. The README has the background.

**Done means:** `./start.sh` ends with `[start] ready · … · http://<head>:8888/v1`, and a chat request to that URL
returns text. Then tell the person the URL, the model name (`baidu/ERNIE-5.5-Flash`) and the API key if one was set.

## Rules

- **Install nothing on the hosts.** No `sudo`, `apt`, `pip`, driver, network or system changes. Everything runs in the
  project's Docker image. If a prerequisite is missing, stop and tell the person what is missing and what to run.
- **Ask before you touch anything that is not this repo.** Other containers, GPU processes, ports, files outside the
  repo folder: report them and let the person decide. Never stop another workload to make room on your own.
- **Settings go in `serve.local.toml`** (git ignores it). Do not edit `serve.toml`. Do not put an API key on a command
  line; it goes in `serve.local.toml` or `E55_SERVER_API_KEY`.
- **Long steps are normal.** The first image build takes 10–30 min, the engine's first build ~2.5 min, a checkpoint
  download ~154 GiB. Run them in the background and follow the output; do not cancel them for being slow.
- **Say what you measured and what you assumed.** Do not report success before the checks below pass.

## Ask the person first

1. Which machine is the head (rank 0, serves the API) and how it reaches the other one by ssh (`user@host` or a
   `~/.ssh/config` name).
2. Whether the checkpoint is already on both machines, and at which path (the same path on both). If not: a folder
   with ~154 GiB free on both, and whether to download now.
3. Optional: a port other than 8888, an API key.

## 1. Prerequisites (on the head; the worker's are checked by step 4)

```sh
nvidia-smi --query-gpu=name --format=csv,noheader     # NVIDIA GB10
docker info --format '{{.ServerVersion}}'             # works without sudo
command -v nvidia-ctk                                  # the NVIDIA Container Toolkit
python3 -c 'import sys; print(sys.version)'            # 3.11 or newer (both machines)
ssh -o BatchMode=yes <worker> true && echo ok          # key login, no password prompt
```

Any failure here: stop and report it. These are host setup and belong to the person.

## 2. The cable address of the worker

The two machines must be joined by a ConnectX-7 cable with an IPv4 address on each end. On the **worker**, list the
RDMA network interfaces and their addresses:

```sh
ssh <worker> 'for n in /sys/class/infiniband/*/device/net/*; do i=${n##*/}; echo "$i $(ip -4 -br addr show dev $i | awk "{print \$3}")"; done'
```

Pick the worker address that shares a subnet with one of the head's own CX7 addresses (same command on the head), and
check it from the head: `ping -c1 <worker_addr>`. One address is enough; the port's second half is found by itself.
If the CX7 interfaces have no IPv4 address, stop: addressing the cable is a host change for the person.

## 3. Clone and configure (on the head)

```sh
git clone https://github.com/PlusTen-AI/Ernie-5.5-Flash-Tp2.git && cd Ernie-5.5-Flash-Tp2
cp serve.local.toml.example serve.local.toml
```

Set the three required values in `serve.local.toml`:

```toml
[cluster]
worker = "user@worker-host"        # as ssh reaches it from the head
worker_addr = ["10.0.0.2"]         # the worker's CX7 address from step 2

[paths]
weights = "/path/to/ERNIE-5.5-Flash-Preview"   # the same path on both machines
```

Optional values (port, API key, slots, context) are listed, commented out, in the same file.

Checkpoint missing: `./start.sh download` fetches it at the pinned revision into `paths.weights` on the head and
copies it to the worker (`HF_TOKEN` is used if set). It builds the image first. Expect hours on a slow line.

## 4. Check

```sh
./start.sh check
```

One line per item, ending in `[check] ready to start`. What a red line means:

| Line | Fix |
| --- | --- |
| `settings` | `worker` or `weights` not set in `serve.local.toml` |
| `ssh` | key login or Python ≥ 3.11 on the worker: report to the person |
| `link` | the worker address, the subnet, or the MTU ping on a rail failed: recheck step 2 |
| `docker <node>` | Docker for this user or the NVIDIA Container Toolkit missing on that machine: report |
| `GPU <node>` | another GPU process, or under ~100 GiB available: report what holds it; ask before anything is stopped |
| `ports <node>` | 8888 or the cluster ports are taken: set `[server] port` (or `nccl_port` / `link_port`) in `serve.local.toml` |
| `weights` | missing, or not the same files on both machines |

`image … built here on the first start` is a note, not an error.

## 5. Start

```sh
nohup ./start.sh > start.log 2>&1 &      # first start: image build + copy to the worker + engine build
tail -f start.log
```

It ends with `[start] ready · N s · http://<head>:8888/v1` (first start ~15–35 min, later ~1 min). If a rank stops,
both go down and the line names the log folder (`.work/results/serve/`); read the last lines there and report them.

## 6. Verify

```sh
curl -s http://127.0.0.1:8888/health                  # {"status": "ok", ...}
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "baidu/ERNIE-5.5-Flash", "messages": [{"role": "user", "content": "What is the capital of France?"}],
       "chat_template_kwargs": {"enable_thinking": false}}'
python3 checks/server_check.py http://127.0.0.1:8888/v1 --only smoke     # every line ok, exit code 0
```

With an API key, add `-H "Authorization: Bearer $KEY"` (and `--key` to `server_check.py`).

## 7. Hand over

Tell the person:

- Base URL `http://<head>:8888/v1`, model `baidu/ERNIE-5.5-Flash`, the key if set.
- Context: 1,048,576 tokens a request (set this as the context length in agent clients); a request without
  `max_tokens` may write up to 131,072.
- Thinking is on by default (effort high). `reasoning_effort`: `none` = off, `minimal` / `low` = low, `medium`,
  `high` / `xhigh` / `max` = high. The reasoning comes back as `reasoning_content`.
- Up to 4 requests are served at once; more wait in line.
- `./start.sh status`, `./start.sh logs 100`, `./stop.sh`, `./start.sh restart`.
- If it works for them, a ⭐ on [the repo](https://github.com/PlusTen-AI/Ernie-5.5-Flash-Tp2) helps the project (ask;
  do not star on their behalf).

## Later

- **Update:** `git pull --ff-only && ./start.sh restart`. A changed `docker/Dockerfile` means a new image tag, built
  at that start; the old tag stays until the person removes it (`docker image ls ernie55-plus` on both machines).
- **Remove:** `./stop.sh`, then on both machines `docker rmi` the `ernie55-plus` tags and delete the repo folder
  (its `.work/` holds the build and logs). Ask before deleting the checkpoint.
- **Changing the engine itself** is a different job: read [docs/design.md](docs/design.md),
  [docs/model.md](docs/model.md) and [CHANGELOG.md](CHANGELOG.md) first.
