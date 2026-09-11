<div align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</div>

<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="english"></a>

# Qwen3.8 Local Coding on a 16GB GPU

A small, reproducible deployment package for running a quantized Qwen3.8 27B coding agent with llama.cpp, MTP, and quantized KV cache. It supports OpenAI-compatible clients such as Codex and Pi, plus an optional Anthropic adapter for Claude Code.

The useful question is not "how many tok/s?" but whether an agent can inspect a real repository, follow a design document, change code, and pass tests without wasting turns or overflowing memory. Reference hardware: **RTX 5060 Ti 16GB VRAM + 32GB RAM** (Ryzen 9 9950X, Linux, Docker, llama.cpp CUDA server).

## Experiment timeline (newest first)

### 2026-09-11 · Adaptive KV Streaming @ 192K (latest)

Replicated the community "Adaptive KV Streaming" scheme ([Reddit](https://www.reddit.com/r/LocalLLM/comments/1was9n0/), [fork repo](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)): the fork streams the KV cache between VRAM and system RAM, so **196,608-token context loads and generates correctly on a 16GB card** (Q8_0 K + Q4_0 V, `--kv-stream-stage-mib 1408`, no MTP).

| Item | Result |
| --- | --- |
| Context sweep | 22.4 tok/s decode @ 8K → 15.5 @ 90K → 13.4 @ 122K → 8.2 @ 180K (Qwen3.8-27B-UD-IQ4_XS) |
| Real agent E2E (DeliverableBench ocr-dual-channel) | **99.5/100, 12.7 min, 70 tests passed (full green), 0 steers**, decode 21.0 tok/s |
| vs. 5060 Ti historical runs | same card, same Pi agent: 14m20s / 14m04s / 25m30s (see below); decode 11-19 tok/s historically → 21 tok/s here, no regression from the 192K config |
| Practical ceiling | decode drops below 10 tok/s near ~170K; comfortable zone ≈ 100-140K |

Full config, build pitfalls (a silently-broken `sm_120`-only build, the PEG parser 500 patch, `libcuda.so.1` link fix), sweep table, and E2E numbers: **[references/adaptive-kv-192k-experiment.md](references/adaptive-kv-192k-experiment.md)**.

### 2026-09-02 · FFN offload experiment

`--n-cpu-ffn N` keeps the dense FFN weights of the first `N` layers in system RAM and runs those layers on the CPU, reducing GPU memory pressure. Repeated the same Pi session / repo / design doc / implementation task at an 80K server context (A = earlier no-offload run for reference).

| Variant | Agent wall time | Requests | Weighted prefill | Weighted decode | Task result |
| --- | ---: | ---: | ---: | ---: | --- |
| A: no `--n-cpu-ffn` | 14m20s | 14 | 22.18-326.36 tok/s | 11.06-12.95 tok/s | 7 files, 5 new tests passed; 2 known pre-existing failures |
| B: `--n-cpu-ffn 4` | 14m04s | 53 | 656.86 tok/s | 19.40 tok/s | Completed; build/vet/gofmt and 5 new tests passed |
| C: `--n-cpu-ffn 8` | 25m30s (stopped) | 112 | 565.51 tok/s | 18.36 tok/s | Code mostly applied; no clean final report |

Within these real runs, FFN4 was the best balance: better prefill/decode than the no-offload baseline without FFN8's decode drop and test-loop instability. More CPU offload is not automatically better. Prompted by Reddit user **Square_Turn935** ([discussion](https://www.reddit.com/r/LocalLLM/comments/1w509o5/comment/p7c2sim/)).

### Original baseline

First complete reference run (no FFN offload): in one medium-sized Go repository task, the agent read a design document, implemented a custom prompt feature, added five tests, and passed scoped build, vet, formatting, and diff checks in **about 14m20s**. Seven files changed; two failures were confirmed pre-existing; a billable E2E was intentionally not run. The 14 server requests measured 22.18-326.36 tok/s prefill, 11.06-12.95 tok/s decode; MTP acceptance was 84.9%-100%. Details in [benchmark.md](references/benchmark.md).

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

80K is the current tested default for this 16GB GPU profile; larger contexts compete for VRAM with weights, KV cache, CUDA workspace, and other processes. 96K/128K are experiments, not defaults. For >100K contexts, the Adaptive KV fork (see timeline) is the only tested route that loads on 16GB; expect decode below ~15 tok/s beyond 100K on this card.

## Open-source building blocks

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for CUDA inference, MTP, Flash Attention, and OpenAI-compatible serving.
- [Adaptive KV streaming fork](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming) for >100K contexts on 16GB VRAM.
- [Qwen3.8 GGUF family](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) for the quantized model files.
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview), [OpenAI Codex](https://github.com/openai/codex), and [Pi / oh-my-pi](https://github.com/can1357/oh-my-pi) for coding-agent clients.

Licensed under Apache-2.0.

<a id="中文"></a>

<details>
<summary>中文</summary>

# 在 16GB 显卡上运行 Qwen3.8 本地 Coding Agent

可复用部署包：用 llama.cpp、MTP 和量化 KV cache 在 16GB 显卡、32GB 内存设备上运行量化 Qwen3.8 27B，支持 Codex、Pi 等 OpenAI 兼容客户端，并提供 Claude Code 的 Anthropic 协议适配层。真正重要的不是 tok/s，而是 agent 能否读懂真实仓库、遵循设计文档、完成修改并通过测试。参考设备：RTX 5060 Ti 16GB + 32GB RAM。

## 实验时间线（最新在前）

### 2026-09-11 · Adaptive KV Streaming 192K（最新）

复现社区方案（[Reddit](https://www.reddit.com/r/LocalLLM/comments/1was9n0/)、[fork 仓库](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)）：fork 把 KV cache 在显存与内存之间流式搬运，**16GB 单卡可加载并正常生成 196,608 context**（Q8_0 K + Q4_0 V，`--kv-stream-stage-mib 1408`，无 MTP）。

| 项目 | 结果 |
| --- | --- |
| 上下文扫描 | 8K 时 decode 22.4 tok/s → 90K 15.5 → 122K 13.4 → 180K 8.2（UD-IQ4_XS） |
| 真实 agent E2E（DeliverableBench ocr-dual-channel） | **99.5/100 分，12.7 分钟，70 tests 全绿，0 纠偏**，decode 21.0 tok/s |
| 与 5060 Ti 历史运行对比 | 同一张卡、同一 Pi agent：历史 14分20秒 / 14分04秒 / 25分30秒，decode 11-19 tok/s → 本次 21 tok/s，192K 配置在 agent 规模上下文下无速度退化 |
| 实用上限 | decode 在 ~170K 跌破 10 tok/s；舒适区约 100-140K |

完整配置、构建坑（仅 sm_120 构建静默产出乱码、PEG 解析 500 补丁、libcuda.so.1 链接修复）、扫描表和 E2E 数据：[references/adaptive-kv-192k-experiment.md](references/adaptive-kv-192k-experiment.md)。

### 2026-09-02 · FFN offload 实测

`--n-cpu-ffn N` 把前 N 层的 dense FFN 权重放到系统内存、由 CPU 计算，给显卡减负。同一 Pi 会话/仓库/设计文档/实现任务，80K 服务端上下文（A 为历史无 offload 对照）：

| 方案 | Agent 总耗时 | 请求数 | 加权 prefill | 加权 decode | 任务结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| A：不加 `--n-cpu-ffn` | 约14分20秒 | 14 | 22.18-326.36 tok/s | 11.06-12.95 tok/s | 7 文件，5 新增测试通过；2 个已知历史失败 |
| B：`--n-cpu-ffn 4` | 14分04秒 | 53 | 656.86 tok/s | 19.40 tok/s | 完成；build/vet/gofmt 和 5 个新增测试通过 |
| C：`--n-cpu-ffn 8` | 中止前 25分30秒 | 112 | 565.51 tok/s | 18.36 tok/s | 代码基本落地，无干净最终报告 |

FFN4 是最佳平衡；offload 更多并不更快。由 Reddit 网友 **Square_Turn935** 的建议启发（[原讨论](https://www.reddit.com/r/LocalLLM/comments/1w509o5/comment/p7c2sim/)）。

### 原始基准线

首次完整参考运行（无 FFN offload）：中型 Go 仓库任务中，agent 阅读设计文档、实现自定义 prompt 功能、新增 5 个测试，通过限定范围 build/vet/格式化/diff 检查，**约 14 分 20 秒**，修改 7 个文件；2 个失败为已有基线问题，计费 E2E 未运行。14 次请求 prefill 22.18-326.36 tok/s，decode 11.06-12.95 tok/s，MTP 接受率 84.9%-100%。详见 [benchmark.md](references/benchmark.md)。

## 可复现设备配置

参考设备：RTX 5060 Ti 16GB + 32GB RAM（2026-09-11 Adaptive KV 实验时再次核验）。

| 部件 | 实际检测配置 |
| --- | --- |
| CPU | AMD Ryzen 9 9950X，16 核 / 32 线程，x86_64，单路 |
| 系统内存 | 总计 32GB，2 × 16GB DDR5，配置运行频率 6000 MT/s |
| GPU | NVIDIA GeForce RTX 5060 Ti，显存 16311 MiB，计算能力 12.0，PCIe 总线 01:00.0 |
| NVIDIA 软件 | 驱动 595.71.05，CUDA 13.2 |
| GPU 功耗上限 | 180W |
| 操作系统 | Ubuntu，Linux 内核 6.17.0-41-generic，x86_64 |
| 系统磁盘 | 467GB NVMe SSD |
| 推理环境 | Linux x86_64，Docker（llama.cpp CUDA server）或原生 llama-server |

模型运行期间采集到的 GPU 快照约为 51% 利用率、已使用 15767 MiB 显存。利用率和剩余显存会随任务变化，这里只用于说明测试时的运行环境。

## 快速开始

```bash
export MODEL_DIR=/srv/models/Qwen3.8-27B-GGUF
export CONTEXT=81920
export HOST_PORT=8024
bash start-qwen38-27b-5060ti.sh
```

默认：单并发、`batch=2048`、`ubatch=512`、`q4_0` KV、CPU 上的 MTP draft、Flash Attention、关 reasoning、`--n-cpu-ffn 4`。全部可用环境变量覆盖。

## 使用边界

80K 是当前 16GB 配置实测过的默认值；更大上下文与权重/KV/工作区争抢显存，96K/128K 需单独压测。超过 100K 的唯一实测路线是 Adaptive KV fork（见时间线），该卡上 100K 之后 decode 约低于 15 tok/s。

## 开源基础

- [llama.cpp](https://github.com/ggml-org/llama.cpp)：推理与服务。
- [Adaptive KV streaming fork](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)：16GB 显存跑 >100K 上下文。
- [Qwen3.8 GGUF 模型页](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)：量化模型。
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)、[OpenAI Codex](https://github.com/openai/codex)、[Pi / oh-my-pi](https://github.com/can1357/oh-my-pi)：coding agent 客户端。

协议：Apache-2.0。

</details>