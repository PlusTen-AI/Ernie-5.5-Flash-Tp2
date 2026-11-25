#!/bin/bash
# preflight.sh [--need-gib N] [--allow-own] [--node A|B|AB] — before any GPU work or build: access, site guards (the
# model proxy idle, this pair on hold), GPU/containers idle, weights, work dir, CX7, base image. --allow-own: this
# project's containers may be running. --node: which nodes must be idle (default AB). Runs on the Mac, read-only.
set -uo pipefail
cd "$(dirname "$0")/.." && . tools/qlog.sh && . tools/dev.sh
NEED=100; OWN=0; NODES=AB
while [ $# -gt 0 ]; do case $1 in --need-gib) NEED=$2; shift 2;; --allow-own) OWN=1; shift;; --node) NODES=$2; shift 2;; *) shift;; esac; done
Q=$MODELS/$CKPT
NODE='
echo host=$(hostname)
echo cont=$(docker ps --format "{{.Names}}:{{.Label \"project\"}}" | tr "\n" " ")
echo apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr "\n" " " | tr -d ",")
echo util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
echo mem=$(awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo)
echo idx=$(md5sum < '$Q'/model.safetensors.index.json 2>/dev/null | cut -c1-8)
echo shards=$(ls '$Q'/*.safetensors 2>/dev/null | wc -l)
echo sbytes=$(stat -c %s '$Q'/*.safetensors 2>/dev/null | awk "{s+=\$1} END{print s+0}")
echo work=$([ -w '$WORK' ] && echo ok || echo missing)
echo disk=$(df -BG --output=avail '$(dirname "$WORK")' 2>/dev/null | tail -1 | tr -dc 0-9)
echo base=$(docker images -q nvcr.io/nvidia/pytorch:26.07-py3 | cut -c1-12)
'
CX='for ip in '"${WORKER_CX[*]}"'; do ping -c1 -W1 -M do -s 8972 $ip >/dev/null 2>&1 && echo "cx_$ip=ok" || echo "cx_$ip=fail"; done'
R=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$HEAD" "{ $NODE $CX; } | sed 's/^/A_/'; ssh -n -o ConnectTimeout=8 -o BatchMode=yes $WORKER $(printf %q "$NODE") | sed 's/^/B_/'" 2>/dev/null)
v() { grep -m1 "^$1=" <<< "$R" | cut -d= -f2-; }
[ "$(v A_host)" = "$HEAD_HOSTNAME" ] || bad "pre access" "A unreachable"
[ "$(v B_host)" = "$WORKER_HOSTNAME" ] || bad "pre access" "B unreachable (through A)"
ok "pre access" "A · B"
if [ -n "${GUARD_SWAP:-}" ]; then
  run=$(curl -s -m 8 ${GUARD_KEY:+-H "Authorization: Bearer $GUARD_KEY"} "$GUARD_SWAP/running" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("running") or []))' 2>/dev/null)
  [ "$run" = 0 ] || bad "pre model proxy" "$GUARD_SWAP /running: ${run:-no answer} model(s) — must be empty"
  ok "pre model proxy" "nothing running"
fi
if [ -n "${GUARD_ROUTER:-}" ]; then
  h=$(curl -s -m 8 ${GUARD_KEY:+-H "Authorization: Bearer $GUARD_KEY"} "$GUARD_ROUTER/router/status" | python3 -c '
import json,sys; p={x["pair"]:x for x in json.load(sys.stdin)["pairs"]}; print("yes" if "'"$GUARD_PAIR"'" in (p["'"$GUARD_PAIR"'"].get("hold") or []) else "no")' 2>/dev/null)
  [ "$h" = yes ] || bad "pre hold" "$GUARD_PAIR is not on hold (${h:-no answer}) — the boss sets it"
  ok "pre hold" "$GUARD_PAIR on hold"
fi
for n in A B; do
  c=$(v ${n}_cont); f=$(tr ' ' '\n' <<< "$c" | grep . | grep -v ':ernie-5.5-flash$' | cut -d: -f1 | paste -sd, -)
  o=$(tr ' ' '\n' <<< "$c" | grep ':ernie-5.5-flash$' | cut -d: -f1 | paste -sd, -)
  [ -z "$f" ] || bad "pre containers $n" "foreign containers running: $f"
  [[ $NODES == *$n* ]] || { ok "pre GPU $n" "not needed · no foreign containers"; continue; }
  [ -z "$o" ] || [ $OWN = 1 ] || bad "pre containers $n" "this project's containers still running: $o"
  [ $OWN = 1 ] && { ok "pre GPU $n" "no foreign containers · this project ${o:-none}"; continue; }
  a=$(v ${n}_apps); u=$(v ${n}_util); m=$(v ${n}_mem)
  [ -z "$a" ] || bad "pre GPU $n" "GPU processes running: $a"
  [ "${u:-100}" -lt 5 ] || bad "pre GPU $n" "utilization ${u}%"
  [ "${m:-0}" -ge "$NEED" ] || bad "pre GPU $n" "free memory ${m} GiB (need ${NEED})"
  ok "pre GPU $n" "idle · utilization ${u}% · ${m} GiB free"
done
[ "$(v A_shards)" = 128 ] && [ "$(v B_shards)" = 128 ] || bad "pre weights" "shard count A $(v A_shards) / B $(v B_shards) (128)"
[ "$(v A_idx)" = "$(v B_idx)" ] && [ -n "$(v A_idx)" ] && [ "$(v A_sbytes)" = "$(v B_sbytes)" ] \
  || bad "pre weights" "A·B differ · index $(v A_idx)/$(v B_idx) · bytes $(v A_sbytes)/$(v B_sbytes)"
ok "pre weights" "128 shards · $(( $(v A_sbytes) / 1073741824 )) GiB · index $(v A_idx) · A=B"
for n in A B; do
  [ "$(v ${n}_work)" = ok ] || bad "pre work dir $n" "$WORK missing or not writable"
  [ "$(v ${n}_disk)" -ge 300 ] || bad "pre work dir $n" "disk free $(v ${n}_disk) GB (need 300)"
done
ok "pre work dir" "A $(v A_disk) GB · B $(v B_disk) GB free"
for ip in "${WORKER_CX[@]}"; do [ "$(v A_cx_$ip)" = ok ] || bad "pre CX7" "A→B jumbo ping failed ($ip)"; done
ok "pre CX7" "A→B jumbo ping OK on both rails"
for n in A B; do [ -n "$(v ${n}_base)" ] || bad "pre image" "nvcr.io/nvidia/pytorch:26.07-py3 missing on $n"; done
ok "pre image" "base $(v A_base) on A · B"
