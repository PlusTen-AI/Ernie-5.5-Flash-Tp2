#!/bin/bash
# agent.sh <name> [--gpu] <cmd...> — (Mac, development) one private sandbox per parallel worker on B: the repo (Mac) →
# <work>/dev/<name>/src on A, then → B over CX7; <cmd> runs on B in an ernie55-plus:dev container (cwd = that src,
# test extensions cached in <work>/dev/<name>/ext). --gpu: wait for B's GPU lock first, so only one timed GPU run at a
# time (compiles need no lock: build first without --gpu, then run the test with --gpu — the module is reused).
# Output streams here. Sandboxes are removed with: tools/agent.sh <name> --clean (or all: tools/agent.sh all --clean).
set -uo pipefail
cd "$(dirname "$0")/.." && . tools/qlog.sh && . tools/dev.sh
n=${1:?name}; shift
[[ $n =~ ^[a-z0-9-]+$ ]] || bad agent "name: lowercase letters, digits, -"
D=$WORK/dev/$n
if [ "${1:-}" = --clean ]; then
  [ "$n" = all ] && D=$WORK/dev
  on A "rm -rf ${D:?}" && on B "rm -rf ${D:?}" && ok agent "removed $D on A · B"
  exit
fi
lock=""; [ "${1:-}" = --gpu ] && { lock="flock $WORK/cache/gpu.lock"; shift; }
[ $# -gt 0 ] || bad agent "no command"
X=(--exclude .git --exclude .claude --exclude .venv --exclude '__pycache__' --exclude .DS_Store --exclude /ref
   --exclude /results --exclude build --exclude CLAUDE.md --exclude tools/site.local.sh)
on A "mkdir -p $D/src" && rsync -a --delete "${X[@]}" ./ "$HEAD:$D/src/" || bad agent "copy to A failed"
on A "ssh -n -o BatchMode=yes $WORKER 'mkdir -p $D/src $D/ext' && rsync -a --delete -e 'ssh -o BatchMode=yes' $D/src/ $WORKER:$D/src/" \
  || bad agent "copy A→B failed"
cmd="$(printf '%q ' "$@")"
on B "$lock docker run --rm --name ernie55-agent-$n-$$ --label project=ernie-5.5-flash --user 1000:1000 -e HOME=/tmp \
  -e HF_HOME=/tmp/hf -e E55_EXT_CACHE=/work/dev/$n/ext ${E55_PATCHES:+-e E55_PATCHES=$E55_PATCHES} --gpus all \
  -v $WORK:/work -v $MODELS:/models:ro -w /work/dev/$n/src ernie55-plus:dev $cmd"
