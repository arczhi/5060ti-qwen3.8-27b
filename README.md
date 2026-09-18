<!-- SPDX-License-Identifier: Apache-2.0 -->


# Qwen3.8 Local Coding on a 16GB GPU

A small, reproducible deployment package for running a quantized Qwen3.8 27B coding agent with llama.cpp, MTP, and quantized KV cache. It supports OpenAI-compatible clients such as Codex and Pi, plus an optional Anthropic adapter for Claude Code.

The useful question is not "how many tok/s?" but whether an agent can inspect a real repository, follow a design document, change code, and pass tests without wasting turns or overflowing memory. Reference hardware: **RTX 5060 Ti 16GB VRAM + 32GB RAM** (Ryzen 9 9950X, Linux, Docker, llama.cpp CUDA server).

## Experiment timeline (newest first)

<details open>
<summary><b>2026-09-18 · llama-tierkv + MTP-5 (latest)</b> — prefill 680-850 tok/s · decode 63.7 tok/s @8K (no-spec 25.7), 62.1 in the agent E2E · 256K context on 16GB · agent E2E 100/100 in 398 s</summary>

Forked upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) master (`c77ae69`): **TierKV** is an independent branch for running end-to-end agent tasks on consumer GPUs — a three-tier KV store (VRAM desk / RAM bookshelf / SSD archive) plus page-sparse attention selection. Source: **<https://github.com/arczhi/llama-tierkv>** (local: `~/coding/llama-tierkv/src`).

The model is the **same weights as the KVMem run** — `Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf` (Unsloth Qwen3.8-27B IQ4_XS with the MTP head requantized to Q4_0). The server was started with `--alias qwen38-next`, so the OpenAI API model id (and the `llama_next/qwen38-next` entry in the pi provider config) shows `qwen38-next`; that is only an API alias, not a different model.

| Item | Result |
| --- | --- |
| Spec sweep (8K, temp 0) | no-spec 25.7 → MTP-3 56.6 → **MTP-5 63.7 tok/s**; n>=6 collapses |
| Agent E2E, full TierKV stack | **100.0/100, 398 s, 72 tests full green, 0 steers** (window 49,152, Q4_0 KV + host store, hybrid selector, MTP-5) |
| Server-side E2E (full stack) | decode p50 **62.1 tok/s**, acceptance 73.7%, prefill p50 382 tok/s (no-pager baseline: 65.8 / 72.8% / 479) |
| Long context (single request, no spec) | 128K: 713 tok/s prefill / 22.74 decode · 256K: 680 / **22.70 tok/s (flat from 8K to 256K)** |
| Context ceiling / host RAM | **262,144 tokens** / ~4.6 GB (Q4_0 store, lazy) |
| vs. KVMem (100/100, 482 s, decode p50 50.3) | same score; **faster wall and decode**; 4.5x denser context ceiling; SSD session snapshots; 256K decode is below KVMem because mainline's MTP draft cannot be windowed |
| Traps worth knowing | `--fit` silently CPU-offloads layers (57→35 tok/s); the MTP draft graph is sized like the main context (needs b256/ub128 to fit 56K) |

Full report: **[references/llama-next-mtp5-experiment.md](references/llama-next-mtp5-experiment.md)** · TierKV design: **[references/kv-paging-design.md](references/kv-paging-design.md)** · TierKV README: <https://github.com/arczhi/llama-tierkv#readme>

</details>

<details>
<summary><b>2026-09-17 · KVMem 256K</b> — prefill 675-1,399 tok/s · decode 56.8 tok/s @ 261K, 38.8-65.8 over agent turns · 256K workspace on 16GB · agent E2E 100/100</summary>

Replicated the [KVMem scheme](https://github.com/ggml-org/llama.cpp/discussions/28894) ([repo](https://github.com/kvmem/kvmem-llama.cpp) v0.16.0-rc1): finished history lives in host RAM and a bounded 32K GPU window is retrieved per question — **a full 262,144-token workspace runs on the 16GB card**, with correct recall of a needle at the top of the context.

> **In plain terms:** think of KVMem as "RAG for the KV cache". Normally a 256K conversation must keep its entire KV cache resident in VRAM — which is why this model caps out around 80K on a 16GB card. KVMem instead treats finished history as a searchable store: conversation blocks that scroll out of the GPU window are parked in system RAM like documents, and for every new question the server scores them and pulls only the most relevant ~32K tokens back into VRAM (recent turns stay pinned). The model always attends to a bounded working set, so decode speed stays steady and VRAM stays capped — the cost simply moves to ~13 GB of ordinary RAM. What makes it work is that old KV blocks are usually still useful to *some* future question, but rarely all of them at once; retrieval lets the card serve the few that matter instead of holding everything.

| Item | Result |
| --- | --- |
| 256K workspace | 261,699 / 262,144 tokens processed in one request, needle recalled |
| Prefill | 675 tok/s @ 142K · 1,399 tok/s @ 261K (single-shot) |
| Decode | 56.8 tok/s @ 261K single-shot; 38.8-65.8 tok/s over agent turns (median 50.3) |
| Real agent E2E (DeliverableBench ocr-dual-channel) | **100.0/100, 8.0 min, 69 tests full green, 0 steers** |
| vs. Adaptive KV 192K | 12.7 min -> 8.0 min; agent-turn decode 21.0 -> median 50.3 tok/s (MTP on: 83.8% acceptance) |
| Resources | 15,620 MiB peak VRAM; 13.8 GB server RSS at 261K |
| Build | CUDA 13.2.86 (overlaid onto the 13.2.0-devel image), `120a-real`, `GGML_CUDA_FA_ALL_QUANTS=ON` |

Full config, complete server command line, build pitfalls (nvcc 13.2.51 must be upgraded), model prep (MTP head requantized to Q4_0) and long-context tables: **[references/kvmem-256k-experiment.md](references/kvmem-256k-experiment.md)**.

</details>

<details>
<summary><b>2026-09-11 · Adaptive KV Streaming @ 192K</b> — prefill ~350 tok/s · decode 21.0 tok/s agent (8.2-22.4 sweep) · 196,608 context on 16GB · agent E2E 99.5/100</summary>

Replicated the community "Adaptive KV Streaming" scheme ([Reddit](https://www.reddit.com/r/LocalLLM/comments/1was9n0/), [fork repo](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)): the fork streams the KV cache between VRAM and system RAM, so **196,608-token context loads and generates correctly on a 16GB card** (Q8_0 K + Q4_0 V, `--kv-stream-stage-mib 1408`, no MTP). Prefill ~350 tok/s in the corrected multi-arch build (~700 in the broken sm_120-only build); 109-209 tok/s incremental during the E2E.

| Item | Result |
| --- | --- |
| Context sweep | 22.4 tok/s decode @ 8K → 15.5 @ 90K → 13.4 @ 122K → 8.2 @ 180K (Qwen3.8-27B-UD-IQ4_XS) |
| Real agent E2E (DeliverableBench ocr-dual-channel) | **99.5/100, 12.7 min, 70 tests passed (full green), 0 steers**, decode 21.0 tok/s |
| vs. 5060 Ti historical runs | same card, same Pi agent: 14m20s / 14m04s / 25m30s (see below); decode 11-19 tok/s historically → 21 tok/s here, no regression from the 192K config |
| Practical ceiling | decode drops below 10 tok/s near ~170K; comfortable zone ≈ 100-140K |

Full config, build pitfalls (a silently-broken `sm_120`-only build, the PEG parser 500 patch, `libcuda.so.1` link fix), sweep table, and E2E numbers: **[references/adaptive-kv-192k-experiment.md](references/adaptive-kv-192k-experiment.md)**.

</details>

<details>
<summary><b>2026-09-02 · FFN offload experiment</b> — prefill 565-657 tok/s · decode 18.4-19.4 tok/s · 80K context · best variant FFN4, 14m04s</summary>

`--n-cpu-ffn N` keeps the dense FFN weights of the first `N` layers in system RAM and runs those layers on the CPU, reducing GPU memory pressure. Repeated the same Pi session / repo / design doc / implementation task at an 80K server context (A = earlier no-offload run for reference).

| Variant | Agent wall time | Requests | Weighted prefill | Weighted decode | Task result |
| --- | ---: | ---: | ---: | ---: | --- |
| A: no `--n-cpu-ffn` | 14m20s | 14 | 22.18-326.36 tok/s | 11.06-12.95 tok/s | 7 files, 5 new tests passed; 2 known pre-existing failures |
| B: `--n-cpu-ffn 4` | 14m04s | 53 | 656.86 tok/s | 19.40 tok/s | Completed; build/vet/gofmt and 5 new tests passed |
| C: `--n-cpu-ffn 8` | 25m30s (stopped) | 112 | 565.51 tok/s | 18.36 tok/s | Code mostly applied; no clean final report |

Within these real runs, FFN4 was the best balance: better prefill/decode than the no-offload baseline without FFN8's decode drop and test-loop instability. More CPU offload is not automatically better. Prompted by Reddit user **Square_Turn935** ([discussion](https://www.reddit.com/r/LocalLLM/comments/1w509o5/comment/p7c2sim/)).

</details>

<details>
<summary><b>Original baseline</b> — prefill 22-326 tok/s · decode 11.1-13.0 tok/s · 80K context · 14m20s</summary>

First complete reference run (no FFN offload): in one medium-sized Go repository task, the agent read a design document, implemented a custom prompt feature, added five tests, and passed scoped build, vet, formatting, and diff checks in **about 14m20s**. Seven files changed; two failures were confirmed pre-existing; a billable E2E was intentionally not run. The 14 server requests measured 22.18-326.36 tok/s prefill, 11.06-12.95 tok/s decode; MTP acceptance was 84.9%-100%. Details in [benchmark.md](references/benchmark.md).

</details>

## Reproducible hardware profile

Reference machine: RTX 5060 Ti 16GB + 32GB RAM, verified again on 2026-09-11 during the Adaptive KV experiment.

| Component | Detected configuration |
| --- | --- |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads, x86_64, 1 socket |
| System memory | 32GB total, 2 × 16GB DDR5, 6000 MT/s configured speed |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16311 MiB VRAM, compute capability 12.0, PCIe bus 01:00.0 |
| NVIDIA software | Driver 595.71.05, CUDA 13.2 |
| GPU power limit | 180W |
| Operating system | Ubuntu, Linux kernel 6.17.0-41-generic, x86_64 |
| System disk | 467GB NVMe SSD |
| Inference host | Linux x86_64, Docker (llama.cpp CUDA server) or native llama-server |

The GPU status snapshot taken while the model was serving showed about 51% utilization and 15767 MiB of VRAM in use. Utilization and free VRAM are workload-dependent; they are included only to make the measurement context clear.

## Quick start

1. Download the exact main and draft GGUF files from the model page; put both under one model directory on the GPU host.
2. Override the deployment variables and start the server:

```bash
export MODEL_DIR=/srv/models/Qwen3.8-27B-GGUF
export CONTEXT=81920
export HOST_PORT=8024
bash start-qwen38-27b-5060ti.sh
```

The tested default: one slot, `batch=2048`, `ubatch=512`, `q4_0` KV cache, one CPU-resident MTP draft, Flash Attention, reasoning off, `--n-cpu-ffn 4`. All settings are overridable via environment variables (`N_CPU_FFN=0/8`, `CONTEXT`, `CACHE_TYPE_K/V`, ...); `LLAMA_IMAGE` supplies an image mirror.

## Connect an agent

- Pi or Codex: `http://<model-host>:8024/v1`
- Claude Code: run the optional adapter on `http://<model-host>:8025`
- Model ID: `qwen3.8-27b-ud-iq4-xs-mtp1` ([Hugging Face model page](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF))

Claude Code must use the adapter root without `/v1` and keep its token in a private environment file. See [client-config.md](references/client-config.md).

## Limits

80K is the current tested default for this 16GB GPU profile; larger contexts compete for VRAM with weights, KV cache, CUDA workspace, and other processes. 96K/128K are experiments, not defaults. For >100K contexts there are two tested routes on 16GB: the Adaptive KV fork (196,608 context; decode falls to 8-15 tok/s beyond ~150K) and KVMem (up to 256K; 38-57 tok/s decode in our tests) — see the timeline.

## Open-source building blocks

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for CUDA inference, MTP, Flash Attention, and OpenAI-compatible serving.
- [KVMem fork](https://github.com/kvmem/kvmem-llama.cpp) for a 256K workspace on 16GB via host-RAM KV storage + retrieval.
- [Adaptive KV streaming fork](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) for >100K contexts on 16GB VRAM.
- [Qwen3.8 GGUF family](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) for the quantized model files.
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview), [OpenAI Codex](https://github.com/openai/codex), and [Pi / oh-my-pi](https://github.com/can1357/oh-my-pi) for coding-agent clients.

Licensed under Apache-2.0.
