# KVMem at 256K (2026-09-17)

> Reproduction of the scheme posted in [llama.cpp discussion #28894](https://github.com/ggml-org/llama.cpp/discussions/28894):
> "Qwen 3.8 27B at 256K context on a 16GB 5060 Ti — decode stays ~30–40 tok/s (KVMem + llama.cpp)".
> Repo: [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) `v0.16.0-rc1` · Paper: [arXiv:2609.04852](https://arxiv.org/abs/2609.04852)
> KVMem keeps finished KV history in host RAM and retrieves a bounded window (32K on IQ4) into VRAM for each new question, instead of streaming the full cache through VRAM (the Adaptive KV approach below).

Test machine: **RTX 5060 Ti 16GB + Ryzen 9 9950X + 32GB RAM** (remote host 10.41.3.123, Ubuntu, driver 595.71.05 / CUDA 13.2, kernel 6.17.0-41).
Model: `Qwen3.8-27B-UD-IQ4_XS.gguf` (Unsloth) with the 8 MTP matrices requantized to Q4_0 → `Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf`; `mmproj-BF16.gguf` on CPU (unused by the text task).

## Summary

| Item | Result |
| --- | --- |
| 256K workspace | **261,699 / 262,144 tokens (99.8%) processed in one request; needle at position 0 recalled correctly** |
| Decode at 256K | **56.8 tok/s** (256-token repeat-pattern generation, 100% MTP acceptance) |
| Decode, real agent turns | min 38.8 / median 50.3 / max 65.8 tok/s (151 turns, context up to 54K) |
| Prefill | 675 tok/s @ 142K single-shot; 1,399 tok/s @ 262K single-shot; 110–447 ms incremental prefill in agent turns |
| MTP acceptance | **83.8%** overall during the agent run (18,614 / 22,203 drafted tokens) |
| VRAM / host RAM | 15,620 MiB peak VRAM; server RSS 13.8 GB with the 261K workspace |
| Real coding-agent E2E | **100.0 / 100, completed in 482s (8.0 min), 69 tests full green, 0 steers** |
| Verdict | Works as advertised on this card, with higher decode than the vendor's reference laptop numbers (~33 tok/s @ 256K) |

## Model preparation

The IQ4 model used in the 2026-09-11 Adaptive KV experiment had been deleted from the host (`rm -rf Qwen3.8-27B-GGUF` in shell history; full-disk search confirmed no other copy), so the identical Unsloth file was re-downloaded from ModelScope (~16 min at ~15 MB/s, 14,252,845,984 bytes) together with `imatrix_unsloth.gguf` and `mmproj-BF16.gguf`.

The IQ4 recipe of KVMem runs a locally requantized model whose **MTP head (`blk.64`) matmul weights are Q4_0** instead of the upstream Q6_K/Q8_0 mix — only 8 matrices change; everything else keeps Unsloth's original mixed quantization. Done with the project's own helper:

```bash
python3 scripts/quantization/quantize-iq4-mtp.py \
  --model   /root/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --output  /root/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf \
  --imatrix /root/models/Qwen3.8-27B-GGUF/imatrix_unsloth.gguf
# 16.7 s, output 14,140,820,480 bytes; 8 tensors requantized iq4_xs -> q4_0
```

## Build (CUDA 13.2.86)

The project pins llama.cpp `b81c99b` and warns that **nvcc 13.2.51 produces garbage IQ kernels; 13.2.86 is the validated compiler**. The available `nvidia/cuda:13.2.0-devel-ubuntu24.04` image ships exactly 13.2.51, so CUDA 13.2 Update 2 components from NVIDIA's redist index were overlaid into the container before building:

- `cuda_nvcc`, `cuda_crt`, `libnvvm`, `cuda_tileiras` — all 13.2.86, total ~103 MB
- Gotcha: in the image `/usr/local/cuda/include` and `lib64` are **symlinks** (`→ targets/x86_64-linux/...`), so `cp -a` of an archive's `include/` fails with "cannot overwrite non-directory". Extract into the resolved `targets/x86_64-linux` tree instead.
- cmake 3.31.6 (Ubuntu 24.04's 3.28 does not know Blackwell arch names), Ninja, `-DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real` via the project's `scripts/build-cuda.sh`. Full build ~4.5 min on the 9950X (`NPROC=12`).

```bash
git clone --recurse-submodules https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp && git checkout v0.16.0-rc1
scripts/apply-patches.sh      # applies patches/llama-kvmem-current.patch to the pin
scripts/build-cuda.sh         # → build/bin/llama-kvmem-server, llama-quantize
```

## Server configuration (IQ4 recipe, as run)

```bash
MODEL=.../Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf \
MMPROJ=.../mmproj-BF16.gguf PORT=18200 \
bash scripts/start-iq4.sh --chat-template-kwargs '{"enable_thinking":false}'
```

The launcher resolves to this **complete server command line** (captured from `/proc/<pid>/cmdline` of the running process):

```bash
/root/kvmem-llama.cpp/build/bin/llama-kvmem-server \
  -m /root/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf \
  --mmproj /root/models/Qwen3.8-27B-GGUF/mmproj-BF16.gguf \
  --no-mmproj-offload --image-max-tokens 512 \
  --host 127.0.0.1 --port 18200 \
  -c 262144 -n 12288 \
  --kvmem-budget 32768 --kvmem-gen-reserve 12288 \
  --kv-dtype q5_0 --spec-type draft-mtp \
  --enable-thinking --reasoning-budget 4096 \
  --chat-template-kwargs '{"enable_thinking": false}'
```

with the environment:

```bash
LD_LIBRARY_PATH=/root/kvmem-llama.cpp/build/bin:/root/tools/cuda-libs
CUDA_VISIBLE_DEVICES=GPU-99031b4a-af49-6a9e-558b-9bf2606f151e   # the RTX 5060 Ti UUID
KVMEM_VISION_DEVICE=cpu
```

The same invocation works without the Python launcher (useful if `ss`/iproute2 or Python 3.10+ is unavailable on the host) — start it with `nohup` and those env vars; `LD_LIBRARY_PATH` must contain the build directory (bundled `libggml*`) and a CUDA runtime/libcublas directory (here `/root/tools/cuda-libs`, copied out of the CUDA devel image).

Deviations from the vendor recipe, both deliberate:

- **Thinking disabled** (`enable_thinking:false` in template defaults) for comparability with the earlier thinking-off baselines in this repo; the vendor default enables it with a 4096 budget.
- Sampling stays on the server's non-thinking defaults (temp 0.7 / top_p 0.80 / top_k 20 / presence_penalty 1.5), as the client sends no sampling parameters.

Startup log confirms the tiered layout: `KVMem slot-pool cells=45056 slots=352 block_tokens=128 budget=32768 gen_reserve=12288 type_k=q5_0 type_v=q5_0`.

The surface is an OpenAI-compatible endpoint (chat, tools, no auth) bound to `127.0.0.1`; the benchmark reached it from the Mac over an SSH tunnel:

```bash
ssh -N -L 18200:127.0.0.1:18200 root@10.41.3.123
# client sees it at http://127.0.0.1:18200/v1
```

## Real coding-agent E2E (DeliverableBench ocr-dual-channel)

Same task as the previous 5060 Ti experiments: read the repo → design a dual-channel PaddleOCR integration → implement → full test suite → summarize. Task text fed verbatim; `pi` agent; venv symlinked from `/Users/alex/coding/video-multimodal-feature/.venv` (paddleocr 3.7.0).

The exact agent invocation (launched directly instead of via `bench/run_bench.py`, see caveats):

```bash
cd results/kvmem-256k/workspace
pi --provider kvmem_256k --model 'Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf' \
   --session-id dlb-kvmem-256k -p "$(cat ../../../tasks/ocr-dual-channel/task.md)"
```

with the pi provider registered in `~/.pi/agent/models.json` as `kvmem_256k` → `http://127.0.0.1:18200/v1`.

| Metric | KVMem @ 5060 Ti 16GB |
| --- | ---: |
| Wall time | **8.0 min (482 s)** |
| Total score | **100.0 / 100** |
| Deliverables | 40 / 40 |
| Tests | 30 / 30 (**69 passed, full green**; scorer ignores `test_model_paths.py`) |
| Completion | 15 / 15 |
| Behavior | 15.0 / 15 |
| Tool calls | 91 (bash 72 / read 3 / write 3 / edit 13) |
| Steers | 0 |
| Tokens in / out | 38,989 / 15,733 (client-side usage) |

Server-side during the E2E (151 chat turns, peak context 54,255 tokens — **the retrieval path engaged well before the run ended**, since the GPU window is only 32K+12K):

- decode **min 38.8 / median 50.3 / max 65.8 tok/s**
- MTP acceptance **83.8%** overall
- KVMem retrieval active: e.g. `retrieval_ms≈55`, `checkpoint_save/restore` pairs per turn; history blocks live in host RAM (6.9 GB RSS at end of run)

The agent's own summary reported it also had to repair workspace environment gaps (trained the missing LightGBM artifact, symlinked model dirs; that test is excluded from scoring) — environment noise of the harness, not a KVMem effect.

## Long-context validation (single-shot)

Synthetic prompts with a needle at position 0 ("The secret word is OBSIDIAN"), filler tail, then the question.

| Target | prompt tokens | Prefill time | Prefill rate | Decode (256 tok) | Recall |
| --- | ---: | ---: | ---: | ---: | --- |
| ~140K | 142,359 | 210.9 s | **675 tok/s** | n/a (4-token answer) | OBSIDIAN ✓ |
| ~256K | **261,699** (99.8% of 262,144) | 187.0 s | **1,399 tok/s** | **56.8 tok/s** | OBSIDIAN ✓ |

Resources at 261K: **VRAM 15,620 MiB** (peak, 16,311 MiB card), **server RSS 13.8 GB** (fits the 32GB host), `retrieval_ms=162`, `checkpoint_save/restore=62/19 ms` on the final turn.

The vendor's 256K tool benchmark quotes 33 tok/s decode / ~463 tok/s first-pass prefill on a Core Ultra 7 255H laptop; on this desktop (9950X, faster host memory) the same scheme measures noticeably better. The desktop is also not WSL2-limited (29.4 GB visible RAM).

## Comparison with previous 5060 Ti runs (same card, same agent)

| Run | Context reachable | Agent wall | Agent-turn decode | Score |
| --- | ---: | ---: | ---: | ---: |
| Original baseline (no offload, 80K) | 80K | 14m20s | 11.1–12.9 tok/s | (different task) |
| FFN4 offload (80K) | 80K | 14m04s | 19.4 tok/s | (different task) |
| Adaptive KV streaming, 192K | 196,608 | 12.7 min | 21.0 tok/s | 99.5 / 100 |
| **KVMem, 256K (this experiment)** | **262,144** | **8.0 min** | **38.8–65.8 tok/s** | **100.0 / 100** |

Caveats for the comparison: KVMem's decode advantage partly reflects the shorter context actually used by this agent task (~54K peak) plus MTP acceptance; the Adaptive KV run kept MTP off. KVMem's 256K claim was validated separately by the single-shot tests above.

## Notes and caveats

- Thinking was off and single-run only; no repeats/variance data (same as previous in-repo experiments).
- Single generation cannot exceed `--kvmem-gen-reserve` (12,288 tokens on IQ4, including thinking) — a documented KVMem limitation; the system prompt should cap per-turn output.
- `NVMe offload is not implemented` (vendor note); the 261K workspace fits in the 32GB host RAM here.
- The benchmark runner `bench/run_bench.py` in DeliverableBench-Live has a missing `import os` (line 113) and cannot start as committed; the run was launched directly with the same `pi` command line it builds (same task text verbatim, session id `dlb-kvmem-256k`), and scored with `bench/score.py`.
- Client reachability was via SSH tunnel from the Mac; the server itself binds `127.0.0.1` and has no auth — do not expose it.
- Disk/time budget of the reproduction: model re-download ~16 min, CUDA overlay + build ~6 min, MTP requant ~17 s.

## Usage notes

- For agent workloads that stay under ~60K tokens, KVMem adds retrieval overhead but keeps decode at 40–65 tok/s on this card — it is a drop-in OpenAI endpoint.
- Beyond ~150K it is currently the fastest correct route tested on a single 16GB card (56.8 tok/s @ 261K in our repeat-pattern probe), versus 8.2–10 tok/s for Adaptive KV streaming at similar lengths.
- Keep the build CUDA at 13.2.86+ (13.2.51 IQ-kernel garbage warning is real); pin a fresh build directory after any toolkit change.
