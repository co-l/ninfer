#!/bin/bash
# Stop the ninfer-serve container on the box (frees ~31 GiB VRAM).
#
# Usage: ./stop.sh
#   env override: BOX

set -euo pipefail

BOX="${BOX:-gaming_pc}"

if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$BOX" 'true' >/dev/null 2>&1; then
  echo "error: box '$BOX' is not reachable — power it on first" >&2
  exit 1
fi

ssh "$BOX" 'sudo podman stop ninfer-serve 2>/dev/null && echo "ninfer-serve stopped" || echo "ninfer-serve not running"'
