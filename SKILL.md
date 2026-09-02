---
name: qwen38-local-coding
description: Deploy Qwen3.8 27B on a 16GB NVIDIA GPU with llama.cpp, MTP, quantized KV cache, and compatible Claude Code, Codex, or Pi clients. Use for reproducible local coding-agent setup, troubleshooting, and task-quality benchmarking.
---

<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 Local Coding

Use this skill to deploy a quantized Qwen3.8 27B coding model on a small NVIDIA host and connect coding agents without exposing host-specific credentials.

## Reference profile

- GPU: RTX 5060 Ti, 16 GB VRAM
- System RAM: 32 GB
- Runtime: `llama.cpp` CUDA server in Docker
- Model: `Qwen3.8-27B-UD-IQ4_XS.gguf`
- Model page: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- Draft: `mtp-Qwen3.8-27B-Q4_0.gguf`
- Context: start at `81920` tokens; test lower before attempting 96K or 128K
- Parallelism: `1`
- KV cache: `q4_0` for both K and V
- MTP: one draft token, draft model kept on CPU
- FFN offload: `--n-cpu-ffn 4` by default; use `0` for no offload or `8` for comparison
- Vision: off
- Reasoning: off for coding-agent tool loops unless explicitly requested

## Deployment workflow

1. Inspect GPU memory, system RAM, disk space, Docker, and the exact model files. Do not convert or rename model files silently.
2. Set `MODEL_DIR`, `MODEL_FILE`, `DRAFT_FILE`, `LLAMA_IMAGE`, `HOST_PORT`, `CONTEXT`, and optionally `N_CPU_FFN` in the environment. The startup script has portable defaults, but deployment paths should be explicit.
3. If an IndexTTS process is consuming the GPU, stop only that process after confirming its PID. Leave unrelated services untouched.
4. Ensure the CUDA image is already present when using `--pull=never`, then run `start-qwen38-27b-5060ti.sh`.
5. Verify `GET /health` or `GET /v1/models` and send a short non-streaming request before connecting an agent.
6. Connect Pi or Codex directly to `http://<host>:<port>/v1`. Use the exact model alias returned by `/v1/models`.
7. Connect Claude Code through `anthropic_openai_proxy.py` on a separate port. Keep the base URL without `/v1`; the adapter translates Anthropic Messages requests and preserves authentication.
8. For a coding benchmark, use a clean or explicitly stashed worktree, the same design document, the same tool allowlist, and the same context limit. Never let a previous model's answer enter the next run.

## Reliability rules

- Treat a successful HTTP response as a smoke result, not proof of coding quality.
- Record first-success rate, agent turns, wall time, compaction count, final test pass rate, prefill, decode, and server generation time.
- MTP acceptance is useful evidence that speculative decoding is active; it is not a substitute for task success.
- If the server reports a prefill memory guard error, reduce context or free VRAM before raising memory limits. On a 16GB card, 80K is the tested default; 96K and 128K are aggressive experiments.
- CPU resources matter during large-model inference. A small `--n-cpu-ffn` value can release GPU pressure and improve long-context stability; higher values can slow decode because more data crosses between CPU and GPU.
- Keep proxy tokens in a root-readable private environment file. Never commit IP addresses, passwords, tokens, request dumps, or absolute home-directory paths.
- Stop the service after a benchmark unless it is intentionally being used as a shared endpoint.

## Included files

- `start-qwen38-27b-5060ti.sh`: parameterized llama.cpp + MTP startup script.
- `anthropic_openai_proxy.py`: small Anthropic Messages to OpenAI Chat Completions adapter.
- `anthropic_openai_proxy.service`: systemd template for the optional Claude Code adapter.
- `references/client-config.md`: client examples and protocol boundaries.
- `references/benchmark.md`: anonymized real-run data and interpretation.

## Upstream references

- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- Qwen3.8 GGUF family: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- MLX-LM alternative for Apple Silicon: <https://github.com/ml-explore/mlx-lm>
- Claude Code: <https://docs.anthropic.com/en/docs/claude-code/overview>
- OpenAI Codex: <https://github.com/openai/codex>
- Pi / oh-my-pi reference: <https://github.com/can1357/oh-my-pi>
