#!/bin/bash
# build.sh — (Mac, development) compile the engine module on the worker (B) in an ernie55-plus:dev container, then copy
# the built module to the head (A) over CX7, so a TP2 run finds it up to date on both ranks (A does no compiling).
# Errors → <work>/cache/build-engine.log on B.
set -uo pipefail
cd "$(dirname "$0")/.." && . tools/qlog.sh && . tools/dev.sh
tools/sync.sh >/dev/null || bad build "sync failed"
t0=$(date +%s)
out=$(on B "docker run --rm --label project=ernie-5.5-flash --user 1000:1000 -e HOME=/tmp --gpus all -v $WORK:/work -w /work/src ernie55-plus:dev python3 /work/src/engine/tests/build.py 2>&1 | tail -1")
echo "$out" | grep -q "engine OK" || bad build "${out:-no output} (B)"
on A "mkdir -p $WORK/cache/ext && rsync -a --delete -e 'ssh -o BatchMode=yes' $WORKER:$WORK/cache/ext/engine $WORK/cache/ext/" \
  || bad build "copy B→A failed"
ok build "engine · B $(( $(date +%s) - t0 )) s · copied to A"
