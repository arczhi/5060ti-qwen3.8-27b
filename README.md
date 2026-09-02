<div align="center">
  <a href="#english">English</a> | <a href="#中文">中文</a>
</div>

<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="english"></a>

# Qwen3.8 Local Coding on a 16GB GPU

A small, reproducible deployment package for running a quantized Qwen3.8 27B coding agent with llama.cpp, MTP, and quantized KV cache. It supports direct OpenAI-compatible clients such as Codex and Pi, plus an optional Anthropic adapter for Claude Code.

## Why this exists

The useful question is not “how many tokens per second?” It is whether an agent can inspect a real repository, follow a design document, change code, and pass tests without wasting turns or overflowing memory.

This package is based on open-source work and hands-on local testing. The reference machine was an RTX 5060 Ti with 16GB VRAM and 32GB system RAM.

## Quick start

1. Download the exact main and draft GGUF files from the model page. Do not convert them in the setup step.
2. Put both files under one model directory on the GPU host.
3. Override the deployment variables and start the server:

```bash
export MODEL_DIR=/srv/models/Qwen3.8-27B-GGUF
export CONTEXT=81920
export HOST_PORT=8024
bash start-qwen38-27b-5060ti.sh
```

The tested default uses one request slot, `q4_0` KV cache, one CPU-resident MTP draft, Flash Attention, reasoning disabled, and `--n-cpu-ffn 4`. A CUDA image mirror can be supplied with `LLAMA_IMAGE` when the default registry is unavailable. Use `N_CPU_FFN=0` for the no-offload comparison or `N_CPU_FFN=8` for the larger-offload comparison.

## Connect an agent

- Pi or Codex: `http://<model-host>:8024/v1`
- Claude Code: run the optional adapter on `http://<model-host>:8025`
- Model ID: `qwen3.8-27b-ud-iq4-xs-mtp1`

Claude Code must use the adapter root without `/v1`. Keep its token in a private environment file. See [client-config.md](references/client-config.md).

## Real coding-agent reference

In one medium-sized Go repository task, the agent read a design document, implemented a custom prompt feature, added five tests, and passed scoped build, vet, formatting, and diff checks in about 14m20s. Seven files changed. Two failures were identified as pre-existing baseline failures; a billable provider E2E test was intentionally not run.

The 14 server requests measured 22.18-326.36 tok/s prefill, 11.06-12.95 tok/s decode, and 7.89-184.60s generation time per request. MTP acceptance was 84.9%-100%. The short smoke test reached 86.54 tok/s prefill and 17.77 tok/s decode, but the long request took 184.6s. This is why task completion, turns, wall time, compactions, and final tests are the primary metrics. Full details are in [benchmark.md](references/benchmark.md).

## FFN offload experiment

`--n-cpu-ffn N` keeps the dense FFN weights of the first `N` layers in system RAM and runs those layers on the CPU. It is not a PCIe setting and it does not move the MTP draft. The point is to reduce GPU memory pressure; too much offload can make CPU-to-GPU transfers dominate.

We repeated the same Pi session, repository, design document, skill, and implementation task. A is the earlier complete no-offload run; B and C are fresh runs at an 80K server context so they could load reliably on the 16GB GPU. The numbers are from llama.cpp server logs. B completed; C was stopped after the agent entered a repeated host-only test-environment loop, so C is a stability signal rather than a completed task result.

| Variant | Agent wall time | Server requests | Weighted prefill | Weighted decode | Server time / request | Task result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A: no `--n-cpu-ffn` | 14m20s | 14 | 22.18-326.36 tok/s | 11.06-12.95 tok/s | 7.89-184.60s | 7 files, 5 new tests passed; 2 known pre-existing failures |
| B: `--n-cpu-ffn 4` | 14m04s | 53 | 656.86 tok/s | 19.40 tok/s | 15.16s average | Completed; build/vet/gofmt and 5 new tests passed |
| C: `--n-cpu-ffn 8` | 25m30s until stopped | 112 | 565.51 tok/s | 18.36 tok/s | 12.78s average | Code mostly applied; agent did not reach a clean final report |

The request counts and averages include tool calls and retries, so they are not a synthetic single-prompt benchmark. Within this real agent run, FFN4 was the best balance: it improved the observed decode and prefill behavior over the historical no-offload run without the FFN8 run's lower decode and test-loop instability. The practical lesson is that CPU resources matter too: assigning a small part of the FFN work to CPU can release GPU pressure and improve long-context stability and speed, but more CPU offload is not automatically better.

This experiment was prompted by Reddit user **Square_Turn935**. Thank you for the concrete suggestion to offload some FFN layers to CPU instead of slowing the MTP drafter: [original discussion](https://www.reddit.com/r/LocalLLM/comments/1w509o9/comment/p7c2sim/).

## Limits

80K is the current tested default for this 16GB GPU profile. A larger context can trigger prefill memory rejection because model weights, KV cache, CUDA workspace, and other processes compete for the same VRAM. Lower context or stop unrelated GPU workloads before raising limits. 96K and 128K must be treated as separate experiments, not defaults.

## Open-source building blocks

- [llama.cpp](https://github.com/ggml-org/llama.cpp) for CUDA inference, MTP, Flash Attention, and OpenAI-compatible serving.
- [Qwen3.8 GGUF family](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) for the quantized model files.
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview), [OpenAI Codex](https://github.com/openai/codex), and [Pi / oh-my-pi](https://github.com/can1357/oh-my-pi) for coding-agent clients.

Licensed under Apache-2.0.

<a id="中文"></a>

<details>
<summary>中文</summary>

# 在 16GB 显卡上运行 Qwen3.8 本地 Coding Agent

这是一个可复用的部署包：使用 llama.cpp、MTP 和量化 KV cache，在 16GB 显卡、32GB 内存的设备上运行量化 Qwen3.8 27B。它支持 Codex、Pi 等 OpenAI 兼容客户端，也提供给 Claude Code 使用的 Anthropic 协议适配层。

## 项目目标

真正重要的不是单独看 tok/s，而是 agent 能否读懂真实代码仓库、遵循设计文档、完成修改并通过测试，同时少走弯路、不频繁压缩上下文。

本项目结合开源社区成果和实际部署经验。参考设备为 RTX 5060 Ti 16GB 显存、32GB 系统内存。

## 快速开始

下载模型页中的主模型和 draft GGUF 文件，原样放在 GPU 主机的同一个目录中，不在部署步骤里自行转换：

```bash
export MODEL_DIR=/srv/models/Qwen3.8-27B-GGUF
export CONTEXT=81920
export HOST_PORT=8024
bash start-qwen38-27b-5060ti.sh
```

脚本默认单并发、`q4_0` KV cache、一个放在 CPU 的 MTP draft、Flash Attention、关闭 reasoning，并启用 `--n-cpu-ffn 4`。镜像仓库不可用时，可通过 `LLAMA_IMAGE` 指定镜像站。设置 `N_CPU_FFN=0` 可跑无 offload 对照，设置 `N_CPU_FFN=8` 可跑较大 offload 对照。

## 接入 agent

- Pi 或 Codex：`http://<模型主机>:8024/v1`
- Claude Code：额外启动适配器，使用 `http://<模型主机>:8025`
- 模型名：`qwen3.8-27b-ud-iq4-xs-mtp1`

Claude Code 要指向不带 `/v1` 的适配器根地址，令牌放在私有环境文件里。详见 [client-config.md](references/client-config.md)。

## 实战参考

一次中型 Go 代码仓库任务中，agent 阅读设计文档，落地了自定义 prompt 功能，新增 5 个测试，并通过限定范围的 build、vet、格式化和 diff 检查，总耗时约 14 分 20 秒，共修改 7 个文件。另有 2 个失败被确认是已有基线问题，没有把它们归因于本次改动；需要真实凭据和计费账号的 E2E 没有运行。

14 次服务端请求的 prefill 为 22.18-326.36 tok/s，decode 为 11.06-12.95 tok/s，单次生成耗时 7.89-184.60 秒，MTP 接受率为 84.9%-100%。短冒烟测试很快，但最长一次生成仍用了 184.6 秒，所以实际比较应优先看任务成功率、轮数、总耗时、压缩次数和最终测试通过率。详见 [benchmark.md](references/benchmark.md)。

## FFN offload 实测

`--n-cpu-ffn N` 的意思是：把前 `N` 层的 dense FFN 权重放到系统内存，由 CPU 负责这些层的计算。它不是 PCIe 参数，也不会把 MTP draft 挪走。这样做可以给显卡减负，但如果 offload 太多，CPU 和 GPU 之间搬数据的成本反而会拖慢速度。

我们使用同一个 Pi 会话、同一个代码仓库、同一份设计文档、同一个 skill 和同一个实现任务进行比较。A 是之前已经完成的无 offload 基线；B、C 为了在 16GB 显卡上稳定加载，统一使用 80K 服务端上下文。数据来自 llama.cpp 服务日志。B 已完成；C 因 agent 反复卡在宿主机测试环境问题上被停止，因此 C 只能作为稳定性信号，不能算完整成功任务。

| 方案 | Agent 总耗时 | 服务端请求数 | 加权 prefill | 加权 decode | 服务端单请求平均耗时 | 任务结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A：不加 `--n-cpu-ffn` | 约14分20秒 | 14 | 22.18-326.36 tok/s | 11.06-12.95 tok/s | 7.89-184.60秒 | 修改 7 个文件，5 个新增测试通过；2 个已知历史失败 |
| B：`--n-cpu-ffn 4` | 14分04秒 | 53 | 656.86 tok/s | 19.40 tok/s | 15.16秒 | 完成；build/vet/gofmt 和 5 个新增测试通过 |
| C：`--n-cpu-ffn 8` | 中止前 25分30秒 | 112 | 565.51 tok/s | 18.36 tok/s | 12.78秒 | 代码基本落地，但 agent 没有形成干净的最终报告 |

这里的请求数和平均值包含工具调用与重试，不是单次 prompt 的实验室跑分。就本次真实 coding-agent 任务看，FFN4 是更好的平衡：相比历史无 offload 基线，prefill 和 decode 观测值更好，也没有 FFN8 的较低 decode 和测试循环问题。实际经验是：大模型推理时 CPU 同样重要，让 CPU 承担少量 FFN 计算可以释放 GPU 压力，改善长上下文稳定性和速度，但不是 offload 越多越快。

这次实验受到 Reddit 网友 **Square_Turn935** 的建议启发。感谢他提出“把部分 FFN 层 offload 到 CPU，不要让 MTP drafter 变慢”的具体建议：[原讨论](https://www.reddit.com/r/LocalLLM/comments/1w509o9/comment/p7c2sim/)。

## 使用边界

对这套 16GB 显卡配置，80K 是目前实际验证过的默认值。上下文越大，模型权重、KV cache、CUDA 工作区和其他 GPU 进程越容易争抢显存并触发 prefill 拒绝。遇到问题应先降低上下文或停止无关 GPU 进程，96K 和 128K 都应单独压测，不建议直接作为默认值。

## 开源基础

- [llama.cpp](https://github.com/ggml-org/llama.cpp)：CUDA 推理、MTP、Flash Attention 和 OpenAI 兼容服务。
- [Qwen3.8 GGUF 模型页](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)：量化模型文件。
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)、[OpenAI Codex](https://github.com/openai/codex)、[Pi / oh-my-pi](https://github.com/can1357/oh-my-pi)：coding agent 客户端。

协议：Apache-2.0。

</details>
