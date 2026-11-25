# dev.sh — (Mac, development, this site only — not part of the public recipe) reach the nodes. Sourced: defines
# `on A|B <cmd>` (A = the head, reached directly; B = the worker, through the head) and HEAD_REPO.
_SITE="$(dirname "${BASH_SOURCE[0]}")/site.local.sh"  # this site's addresses (git ignores it; see site.local.sh.example)
[ -f "$_SITE" ] || { echo "$_SITE missing — copy tools/site.local.sh.example to tools/site.local.sh and fill it in" >&2; exit 1; }
. "$_SITE"
HEAD_REPO=$WORK/src
on() { case $1 in
    A) ssh -o BatchMode=yes "$HEAD" "$2" ;;
    B) ssh -o BatchMode=yes "$HEAD" "ssh -n -o BatchMode=yes $WORKER $(printf %q "$2")" ;;
    *) echo "on A|B <cmd>" >&2; return 2 ;;
  esac; }
pass_env() { local v e=""; for v in E55_CONTAINER E55_COMMIT E55_PATCHES E55_SYNC E55_RDMA_INIT E55_RDMA_TIMEOUT_S CUDA_LAUNCH_BLOCKING NCCL_DEBUG; do
  [ -n "${!v:-}" ] && e="$e $v=$(printf %q "${!v}")"; done; echo "$e"; }
# Run: tools/dev.sh start|restart|stop|status|logs [args] — drives ./start.sh on the head as the development server
# (container ernie55-tp2-plus-dev, so a deployed ernie55-tp2-plus is never touched from here); start/restart run
# tools/preflight.sh (site guards) and tools/sync.sh first. Engine switches set here (E55_PATCHES, …) are passed on.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -uo pipefail
  cd "$(dirname "$0")/.." && . tools/qlog.sh
  E55_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo no-commit)$(git diff --quiet HEAD 2>/dev/null || echo ' +modified')"
  E55_CONTAINER=ernie55-tp2-plus-dev
  cmd=${1:-}; shift
  case "$cmd" in
    start) tools/preflight.sh ${E55_PRE:-} || exit 1; tools/sync.sh || exit 1 ;;
    restart) tools/preflight.sh --allow-own || exit 1; tools/sync.sh || exit 1 ;;
    stop|status|logs) ;;
    *) bad dev "usage: tools/dev.sh start|restart|stop|status|logs [args]" ;;
  esac
  on A "cd $HEAD_REPO && env $(pass_env) ./$([ "$cmd" = stop ] && echo stop.sh || echo "start.sh $cmd") ${*:+$(printf '%q ' "$@")}"
fi
