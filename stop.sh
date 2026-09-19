#!/bin/bash
# Stop the ninfer-serve container on the box (frees the serving footprint).
#
# Usage: ./stop.sh
# Configuration: shell env > ./.env > defaults
#   (BOX, CONTAINER_RUNTIME, CONTAINER_SUDO, CONTAINER_NAME).

set -euo pipefail
. "$(cd -- "$(dirname -- "$0")" && pwd)/deploy-lib.sh"
load_env

BOX="${BOX:-inference-box}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-podman}"
CONTAINER_SUDO="${CONTAINER_SUDO-sudo}"
CONTAINER_NAME="${CONTAINER_NAME:-ninfer-serve}"

if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$BOX" 'true' >/dev/null 2>&1; then
  echo "error: box '$BOX' is not reachable — power it on first" >&2
  exit 1
fi

if ssh "$BOX" "$(join_remote "${CONTAINER_SUDO:+$CONTAINER_SUDO}" "$CONTAINER_RUNTIME" stop "$CONTAINER_NAME")" 2>/dev/null; then
  echo "$CONTAINER_NAME stopped"
else
  echo "$CONTAINER_NAME not running"
fi
