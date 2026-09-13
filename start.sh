#!/bin/bash
# Launch ninfer-serve on the box (detached) and return once it is listening.
# Everything lives here: the recipe below is the single source of truth for
# the serving flags. The server survives this script exiting; it keeps running
# after the "listening on" line is seen.
#
# Usage: ./start.sh
#   env overrides: BOX, NINFER_CONCURRENCY (4), NINFER_KV_CAPACITY (460000),
#   NINFER_KV_DTYPE (nvfp4), NINFER_DEVICE_STATE_SLOTS (4), NINFER_VISION (1)

set -euo pipefail

BOX="${BOX:-gaming_pc}"
REMOTE="/home/conrad/dev/nicefox-5090-prod"
LOG="$REMOTE/serve.log"
CONCURRENCY="${NINFER_CONCURRENCY:-4}"
KV_CAPACITY="${NINFER_KV_CAPACITY:-460000}"
KV_DTYPE="${NINFER_KV_DTYPE:-nvfp4}"
DEVICE_STATE_SLOTS="${NINFER_DEVICE_STATE_SLOTS:-4}"
VISION_FLAG=""
[ "${NINFER_VISION:-1}" = "1" ] && VISION_FLAG="--vision"

if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$BOX" 'true' >/dev/null 2>&1; then
  echo "error: box '$BOX' is not reachable — power it on first" >&2
  exit 1
fi

if ssh "$BOX" 'sudo -n podman ps --format "{{.Names}}" 2>/dev/null | grep -qx ninfer-serve'; then
  echo "ninfer-serve already running — nothing to do"
  exit 0
fi

echo "launching ninfer (${KV_DTYPE} KV, ${KV_CAPACITY} pool, 12 GiB RAM tier, MTP4, vision=$([ -n "$VISION_FLAG" ] && echo on || echo off), C=${CONCURRENCY}, ds=${DEVICE_STATE_SLOTS}) on $BOX ..."

ssh -f -o ConnectTimeout=5 "$BOX" "setsid nohup sudo podman run --name ninfer-serve --rm \
  --device nvidia.com/gpu=0 \
  --security-opt label=disable \
  -p 8000:8000 \
  -v $REMOTE/models:/models:ro \
  -v $REMOTE/logs:/logs \
  localhost/ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --port 8000 \
  --model-id qwen3.8-27b \
  --max-context 262144 \
  --kv-capacity $KV_CAPACITY \
  --max-concurrency $CONCURRENCY \
  --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --device-state-slots $DEVICE_STATE_SLOTS \
  --host-state-slots 96 \
  --host-kv-mib 12288 \
  --max-private-continuations 128 \
  --max-shared-prefixes 64 \
  --request-log-jsonl /logs/requests.jsonl \
  --kv-dtype $KV_DTYPE \
  --spec mtp \
  --draft-tokens 4 \
  --lm-head-draft \
  --preserve-thinking \
  $VISION_FLAG > $LOG 2>&1 < /dev/null"

echo "waiting for the server to listen ..."
timeout 180 ssh "$BOX" "tail -F $LOG 2>/dev/null | sed -n '/listening on/{p;q}; p'"
echo "ninfer-serve is up"
