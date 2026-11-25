# sync.sh — copy the repo (Mac) to <work>/src on the head, then head → worker over CX7. rsync --delete inside src only.
set -euo pipefail
cd "$(dirname "$0")/.." && . tools/qlog.sh && . tools/dev.sh
X=(--exclude .git --exclude .claude --exclude .venv --exclude '__pycache__' --exclude .DS_Store --exclude /ref
   --exclude /results --exclude build --exclude CLAUDE.md --exclude tools/site.local.sh)
on A "mkdir -p $WORK/src" && rsync -a --delete "${X[@]}" ./ "$HEAD:$WORK/src/" || bad sync "copy to A failed"
on A "ssh -n -o BatchMode=yes $WORKER 'mkdir -p $WORK/src' && rsync -a --delete -e 'ssh -o BatchMode=yes' $WORK/src/ $WORKER:$WORK/src/" \
  || bad sync "copy A→B failed"
n=$(on B "find $WORK/src -type f | wc -l")
rev=$(git rev-parse --short HEAD 2>/dev/null || echo no-commit)$(git diff --quiet HEAD 2>/dev/null || echo ' +modified')
ok sync "A·B src · ${n} files · $rev"
