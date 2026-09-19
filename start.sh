#!/bin/bash
# Launch ninfer-serve on the box (detached) and return once it is listening.
# The serving recipe below is the single source of truth for the flags; every
# knob is configurable via .env (copy .env.example), with shell env winning.
# The server survives this script exiting.
#
# Usage: ./start.sh

set -euo pipefail
. "$(cd -- "$(dirname -- "$0")" && pwd)/deploy-lib.sh"
load_env

BOX="${BOX:-gaming_pc}"
REMOTE_DIR="${REMOTE_DIR:-/home/conrad/dev/nicefox-5090-prod}"
IMAGE="${IMAGE:-localhost/ninfer:local}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-podman}"
CONTAINER_SUDO="${CONTAINER_SUDO-sudo}"
CONTAINER_NAME="${CONTAINER_NAME:-ninfer-serve}"
GPU_DEVICE="${GPU_DEVICE:-0}"
HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"

ARTIFACT="${ARTIFACT:-/models/qwen3_8_27b_nvfp4.ninfer}"
MODEL_ID="${MODEL_ID:-qwen3.8-27b}"
MAX_CONTEXT="${MAX_CONTEXT:-262144}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_PENDING_REQUESTS="${MAX_PENDING_REQUESTS:-16}"
PENDING_TIMEOUT_MS="${PENDING_TIMEOUT_MS:-600000}"
DEVICE_STATE_SLOTS="${DEVICE_STATE_SLOTS:-4}"
HOST_STATE_SLOTS="${HOST_STATE_SLOTS:-32}"
HOST_KV_MIB="${HOST_KV_MIB:-16384}"
MAX_PRIVATE_CONTINUATIONS="${MAX_PRIVATE_CONTINUATIONS:-128}"
MAX_SHARED_PREFIXES="${MAX_SHARED_PREFIXES:-64}"
KV_CAPACITY="${KV_CAPACITY:-460000}"
KV_DTYPE="${KV_DTYPE:-nvfp4}"
SPEC="${SPEC:-mtp}"
DRAFT_TOKENS="${DRAFT_TOKENS:-4}"
LM_HEAD_DRAFT="${LM_HEAD_DRAFT:-1}"
PRESERVE_THINKING="${PRESERVE_THINKING:-1}"
VISION="${VISION:-1}"
REQUEST_LOG="${REQUEST_LOG:-/logs/requests.jsonl}"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
SERVE_LOG="${SERVE_LOG:-$REMOTE_DIR/serve.log}"

SSH=(ssh -o ConnectTimeout=5 -o BatchMode=yes)

if ! "${SSH[@]}" "$BOX" 'true' >/dev/null 2>&1; then
  echo "error: box '$BOX' is not reachable — power it on first" >&2
  exit 1
fi

if ssh "$BOX" "$(join_remote "${CONTAINER_SUDO:+$CONTAINER_SUDO}" "$CONTAINER_RUNTIME" ps --format '{{.Names}}')" 2>/dev/null | grep -qx "$CONTAINER_NAME"; then
  echo "$CONTAINER_NAME already running — nothing to do"
  exit 0
fi

run=()
[ -n "$CONTAINER_SUDO" ] && run+=("$CONTAINER_SUDO")
run+=("$CONTAINER_RUNTIME" run --name "$CONTAINER_NAME" --rm)
if [ "$CONTAINER_RUNTIME" = docker ]; then
  run+=(--gpus="device=$GPU_DEVICE")
else
  run+=(--device="nvidia.com/gpu=$GPU_DEVICE")
fi
run+=(--security-opt label=disable)
run+=(-p "$HOST_PORT:$CONTAINER_PORT")
run+=(-v "$REMOTE_DIR/models:/models:ro")
run+=(-v "$REMOTE_DIR/logs:/logs")
read -ra extra <<< "$EXTRA_MOUNTS"
run+=("${extra[@]}")
run+=("$IMAGE")
run+=("ninfer-serve" "$ARTIFACT")
run+=(--host 0.0.0.0 --port "$CONTAINER_PORT")
run+=(--model-id "$MODEL_ID")
run+=(--max-context "$MAX_CONTEXT")
run+=(--kv-capacity "$KV_CAPACITY")
run+=(--max-concurrency "$CONCURRENCY")
run+=(--max-pending-requests "$MAX_PENDING_REQUESTS")
run+=(--pending-timeout-ms "$PENDING_TIMEOUT_MS")
run+=(--device-state-slots "$DEVICE_STATE_SLOTS")
run+=(--host-state-slots "$HOST_STATE_SLOTS")
run+=(--host-kv-mib "$HOST_KV_MIB")
run+=(--max-private-continuations "$MAX_PRIVATE_CONTINUATIONS")
run+=(--max-shared-prefixes "$MAX_SHARED_PREFIXES")
run+=(--request-log-jsonl "$REQUEST_LOG")
run+=(--kv-dtype "$KV_DTYPE")
run+=(--spec "$SPEC")
run+=(--draft-tokens "$DRAFT_TOKENS")
[ "$LM_HEAD_DRAFT" = 1 ] && run+=(--lm-head-draft)
[ "$PRESERVE_THINKING" = 1 ] && run+=(--preserve-thinking)
[ "$VISION" = 1 ] && run+=(--vision)
[ -n "$CHAT_TEMPLATE" ] && run+=(--chat-template "$CHAT_TEMPLATE")

echo "launching ninfer (${KV_DTYPE} KV, ${KV_CAPACITY} pool, ${HOST_KV_MIB} MiB RAM tier, ${SPEC}${DRAFT_TOKENS}, vision=${VISION}, C=${CONCURRENCY}, ds=${DEVICE_STATE_SLOTS}) on $BOX ..."

ssh -f -o ConnectTimeout=5 "$BOX" "setsid nohup $(join_remote "${run[@]}") > $(printf '%q' "$SERVE_LOG") 2>&1 < /dev/null"

echo "waiting for the server to listen ..."
timeout 180 ssh "$BOX" "tail -F $(printf '%q' "$SERVE_LOG") 2>/dev/null | sed -n '/listening on/{p;q}; p'"
echo "$CONTAINER_NAME is up"
