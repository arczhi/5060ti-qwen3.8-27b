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
CONTEXT="${CONTEXT:-81920}"
N_CPU_FFN="${N_CPU_FFN:-4}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q4_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q4_0}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-1}"
SPEC_NGRAM_N_MATCH="${SPEC_NGRAM_N_MATCH:-24}"
SPEC_NGRAM_N_MIN="${SPEC_NGRAM_N_MIN:-48}"
SPEC_NGRAM_N_MAX="${SPEC_NGRAM_N_MAX:-64}"

CPU_FFN_ARGS=()
if [[ -n "$N_CPU_FFN" && "$N_CPU_FFN" != "0" ]]; then
  CPU_FFN_ARGS=(--n-cpu-ffn "$N_CPU_FFN")
fi

MODEL_ARGS=(-m "/models/${MODEL_FILE}")
if [[ "$SPEC_TYPE" == *draft-mtp* ]]; then
  if [[ ! -f "$DRAFT" ]]; then
    echo "Missing MTP draft under ${MODEL_DIR}" >&2
    exit 1
  fi
  MODEL_ARGS+=(-md "/models/${DRAFT_FILE}")
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Missing main model under ${MODEL_DIR}" >&2
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
# Offload a small number of dense FFN layers to CPU to reduce VRAM pressure.
SPEC_ARGS=(--spec-type "$SPEC_TYPE" --spec-draft-n-max "$SPEC_DRAFT_N_MAX")
if [[ "$SPEC_TYPE" == *ngram-mod* ]]; then
  SPEC_ARGS+=(
    --spec-ngram-mod-n-match "$SPEC_NGRAM_N_MATCH"
    --spec-ngram-mod-n-min "$SPEC_NGRAM_N_MIN"
    --spec-ngram-mod-n-max "$SPEC_NGRAM_N_MAX"
  )
fi

exec docker run --rm --name "$CONTAINER" --gpus all \
  --pull=never \
  -p "${HOST_PORT}:8080" \
  -v "${MODEL_DIR}:/models:ro" \
  "$IMAGE" \
  "${MODEL_ARGS[@]}" \
  --alias "$MODEL_ALIAS" \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size "$CONTEXT" \
  --batch-size "$BATCH_SIZE" \
  --ubatch-size "$UBATCH_SIZE" \
  --parallel 1 \
  --fit off \
  --n-gpu-layers all \
  "${CPU_FFN_ARGS[@]}" \
  --n-gpu-layers-draft 0 \
  --flash-attn on \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V" \
  "${SPEC_ARGS[@]}" \
  --reasoning off \
  --jinja \
  --metrics
