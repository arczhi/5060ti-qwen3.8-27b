#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Reference hardware: RTX 5060 Ti 16 GB VRAM, 32 GB system RAM.
# Override these variables in the environment for another host or layout.
IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"
CONTAINER="${CONTAINER_NAME:-qwen38-27b-ud-iq4-xs}"
HOST_PORT="${HOST_PORT:-8024}"
MODEL_DIR="${MODEL_DIR:-/srv/models/Qwen3.8-27B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-UD-IQ4_XS.gguf}"
DRAFT_FILE="${DRAFT_FILE:-mtp-Qwen3.8-27B-Q4_0.gguf}"
MODEL="${MODEL_DIR}/${MODEL_FILE}"
DRAFT="${MODEL_DIR}/${DRAFT_FILE}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.8-27b-ud-iq4-xs-mtp1}"
CONTEXT="${CONTEXT:-98304}"

if [[ ! -f "$MODEL" || ! -f "$DRAFT" ]]; then
  echo "Missing model or MTP draft under ${MODEL_DIR}" >&2
  exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU is not available" >&2
  exit 1
fi

# Stop only an IndexTTS process if it is present; leave unrelated services alone.
mapfile -t INDEXTTS_PIDS < <(ps -eo pid=,args= | awk '/[i]ndextts/ {print $1}')
if ((${#INDEXTTS_PIDS[@]})); then
  kill "${INDEXTTS_PIDS[@]}"
  sleep 2
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# Keep the 1.68 GiB MTP draft on CPU; full GPU placement exceeds 16 GiB VRAM.
exec docker run --rm --name "$CONTAINER" --gpus all \
  --pull=never \
  -p "${HOST_PORT}:8080" \
  -v "${MODEL_DIR}:/models:ro" \
  "$IMAGE" \
  -m "/models/${MODEL_FILE}" \
  -md "/models/${DRAFT_FILE}" \
  --alias "$MODEL_ALIAS" \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size "$CONTEXT" \
  --parallel 1 \
  --fit on \
  --n-gpu-layers-draft 0 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 1 \
  --reasoning off \
  --jinja \
  --metrics
