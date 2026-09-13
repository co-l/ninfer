#!/bin/bash
# Deploy the sources in this repository to the inference box and rebuild the
# serving image from them. The box tree at $REMOTE_DIR is a clean copy of this
# repository (git-tracked files only, --delete), so stale files the repo no
# longer carries are purged automatically; the box-local models/ and logs/
# directories are never touched.
#
# The build stops ninfer-serve first: the serving footprint pins ~28 of the
# box's 30 GiB RAM, so the compile (~7 GiB at -j16) only fits with the server
# down — not because the build is memory-heavy.
#
# Usage: ./deploy.sh [--dry-run]
#   --dry-run  show the sync plan only (no build)
#   env overrides: BOX (ssh host), REMOTE_DIR, IMAGE, PARALLEL (ninja jobs)
#   next steps after a real deploy:
#     ../5090/launch-gaming-pc.sh   # serve (stops vLLM/SGLang, starts ninfer)
#     # then the cache-pressure bench suite (agent-sim, needle-test, cache-pressure)

set -euo pipefail

BOX="${BOX:-gaming_pc}"
REMOTE_DIR="${REMOTE_DIR:-/home/conrad/dev/nicefox-5090-prod}"
IMAGE="${IMAGE:-localhost/ninfer:local}"
PARALLEL="${PARALLEL:-16}"
SSH=(ssh -o ConnectTimeout=8 -o BatchMode=yes)

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; fi

if ! "${SSH[@]}" "$BOX" 'true' >/dev/null 2>&1; then
  echo "error: box '$BOX' is not reachable — wake it first:" >&2
  echo "  ../5090/poweron-gaming-pc.sh" >&2
  exit 1
fi

mapfile -d '' files < <(git ls-files -z)
if (( ${#files[@]} == 0 )); then
  echo "error: no git-tracked files to deploy (git add the new files first?)" >&2
  exit 1
fi

printf '%s\0' "${files[@]}" | rsync -a --delete -0 \
  --exclude 'models/' --exclude 'logs/' \
  --files-from=- ./ "$BOX:$REMOTE_DIR/"
echo "synced ${#files[@]} tracked files -> $BOX:$REMOTE_DIR/ (models/ + logs/ preserved)"

if [ "$DRY_RUN" = 1 ]; then
  echo "dry run — image not built"
  exit 0
fi

"${SSH[@]}" "$BOX" 'sudo podman stop ninfer-serve 2>/dev/null || true'

echo "building $IMAGE on $BOX (podman --jobs 4, ninja $PARALLEL, ~4 min)..."
# Stream podman build output through a line-buffered filter: keep STEP/cache
# progress plus every compiler warning and error, drop ninja per-file progress,
# short layer echoes, and blank lines. pipefail keeps a failed build non-zero.
"${SSH[@]}" "$BOX" "cd $REMOTE_DIR && sudo podman build --jobs 4 --build-arg BUILD_PARALLEL=$PARALLEL -t $IMAGE ." 2>&1 | awk '
  /^\[[0-9]+\/[0-9]+\] (Building|Linking) / { next }
  /^--> [0-9a-f]{12}$/ { next }
  NF == 0 { next }
  { print; fflush() }'

echo
echo "deploy complete: $BOX:$REMOTE_DIR built $IMAGE"
echo "next:"
echo "  ../5090/launch-gaming-pc.sh    # serve (stops vLLM/SGLang, starts ninfer)"
echo "  # then the cache-pressure bench suite from the workstation"
