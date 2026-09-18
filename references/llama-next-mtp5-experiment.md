# Mainline llama.cpp + MTP-5 on the 5060 Ti (2026-09-18)

> Fork of upstream `ggml-org/llama.cpp` master (c77ae69, 2026-09-17) — **not** a KVMem derivative.
> This is stage 1 of the three-tier roadmap: get the most out of the *stock* framework
> with spec-decode and VRAM-layout tuning, measure the remaining gap, then decide what a
> KV paging layer is actually worth.
> Code: `~/coding/llama-tierkv/src/` (git bundle + patch + launcher/probe).

Test machine: **RTX 5060 Ti 16GB + Ryzen 9 9950X + 32GB RAM** (10.41.3.123, driver 595.71.05).
Model: `Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf` — **the same weights as the KVMem run** (Unsloth Qwen3.8-27B UD-IQ4_XS with the 8 MTP matrices requantized to Q4_0). The server's OpenAI alias is `qwen38-next` (set with `--alias`), so API responses and the pi provider id show `qwen38-next`; it is an alias for these weights, not a different model.
Build: CUDA 13.2.86 overlay, `GGML_CUDA=ON GGML_CUDA_FA_ALL_QUANTS=ON`, `120a-real`, cmake 3.31.6.

## Summary

| Item | Result |
| --- | --- |
| No-spec decode (8K ctx, temp 0) | 25.7 tok/s |
| MTP-3 decode | 56.6 tok/s (73% draft acceptance) |
| **MTP-5 decode** | **63.7 tok/s** (best measured; n>=6 collapses) |
| MTP-5 + ngram-mod | 68.8 / 52.0 tok/s — unstable, not adopted |
| E2E (DeliverableBench ocr-dual-channel) | **100.0 / 100, R3: 7.2 min (433 s), 71 tests full green, 0 steers** |
| Server-side E2E decode (R3) | **min 46.1 / p50 65.8 / max 79.7 tok/s**, acceptance 72.8%, mean len 5.07 |
| Dense context on 16GB | **57,344 tokens** (q4_0 KV, b256/ub128) |
| VRAM at 57,344 ctx | 15,604-15,640 MiB |

## The two traps that cost the first 20 points of performance

1. **`--fit` (default `on`) silently offloads layers to the CPU.** With `-ngl` unset, the
   loader "fits" the 13.2 GiB model by keeping a margin and moving a few layers to the CPU;
   decode collapses from ~57 to ~35 tok/s with no error. Fix: always pass
   `-ngl 99 --spec-draft-ngl 99`.
2. **The MTP draft context's prefill graph is sized like the main context.** It inherits
   `-b/-ub`, so at 56K ctx the draft reserves 300-400 MiB of compute buffer on top of its
   KV. At `-c 57344` that is what makes the difference between OOM and running:
   `-b 256 -ub 128` shrinks it enough. (This also limits how much KV precision you can
   afford: q5_0 KV fits at 32K, q4_0 at 56K.)

## Spec sweep (8K ctx, 512 generated, temperature 0, 16K server ctx)

| # | Config | Decode tok/s | Acceptance |
| --- | --- | ---: | ---: |
| A | `draft-mtp`, n_max=3, draft KV q8_0, main KV q5_0 | 56.6 | 73% |
| B | `draft-mtp`, n_max=5 | **63.7** | ~70% |
| C | `draft-mtp,ngram-mod`, n_max=3 | 54.5 | 56-63% |
| D | `draft-mtp,ngram-mod`, n_max=5 | 68.8 / 52.0 | 48-65% |
| E | n_max=3, draft KV f16 | 57.0 | 73-74% |
| F | n_max=3, main KV q4_0 | 52.6 | 65% |
| G | no speculation | 25.7 | — |
| H | n_max=4 / n_max=6 / n_max=8 | 54.4 / 46.8 / 41.5 | collapsing |

Notes:
- Draft KV dtype (f16 vs q8_0) does not matter; main KV q4_0 costs ~7%.
- ngram-mod drafts occasionally win (68.8) but are unstable on prose; not adopted.
- n>=6 over-drafts: per-draft acceptance collapses (30% at n=8), total throughput drops.
- A temperature-0.7 pass (E2E-like sampling) lands at 41-51 tok/s on the same probe.

## Real coding-agent E2E (DeliverableBench ocr-dual-channel)

Same task text, same `pi` agent, same venv as the KVMem run (2026-09-17), thinking off
(`--reasoning off` — the first attempt had thinking silently ON and generated 26K
reasoning tokens; that run was discarded).

Three attempts were needed; **R3 is the headline run**:

| | R1 | R2 | **R3** |
| --- | ---: | ---: | ---: |
| Sampling | temp .7/top_p .8/k20, no presence | same | **same + presence_penalty 1.5** (matches KVMem) |
| Outcome | 100.0 | looping (295 bash calls), killed | **100.0** |
| Wall | 865 s (≈600 s lost to an agent-side catastrophic regex) | — | **433 s (7.2 min)** |
| Tool calls | 62 (bash 45) | — | 61 (bash 33 / read 14 / write 3 / edit 11) |
| Tokens in / out | 39.8K / 13.2K | — | 54.3K / 15.5K |
| Tests | 75 passed | — | 71 passed (full green; scorer ignores `test_model_paths.py`) |

Server-side during R3: **decode min 46.1 / p50 65.8 / max 79.7 tok/s**, draft acceptance
72.8% (12,678/17,420), mean accepted length **5.07** tokens per verification pass.

R3's server-side numbers turned out much better than R1's (~47 tok/s, 45-56% acceptance):
the `presence_penalty 1.5` (copied from KVMem's sampling profile) both suppresses the
agent's repetition loops **and** raises MTP draft acceptance (more predictable output),
which is why R3 is the configuration to keep.

## Comparison with KVMem (same card, same task, same agent, matched sampling)

| | KVMem v0.16.0-rc1 | Mainline + MTP-5 (R3) |
| --- | ---: | ---: |
| Context on 16GB | 262,144 (RAM-retrieved) | 57,344 (dense VRAM KV) |
| E2E score | 100.0 | 100.0 |
| **E2E wall** | 482 s | **433 s** |
| Server-side decode (p50) | 50.3 tok/s | **65.8 tok/s** |
| Draft acceptance | 83.8% | 72.8% |
| Mean tokens / verification pass | ~3.5 | **5.07** |
| Peak VRAM | 15,620 MiB | 15,640 MiB |

What decided the E2E: mainline drafts deeper (n_max=5) and, with the same sampling profile,
converts that into ~5 tokens per pass; KVMem has a higher per-draft hit rate but drafts only
3. Both spend ~68-70 ms per verification pass, so the deeper draft wins on wall time.

Where this leaves the roadmap:

1. **Per-pass time is the wall, and it is identical (~68-70 ms) in both schemes.** The
   weight-bandwidth floor is ~32 ms/pass (13.2 GiB / 448 GB/s), so both implementations
   spend ~36 ms/pass on attention, GDN state handling, draft forwards and framework
   overhead. Profiling that gap is the single highest-value next step.
2. **KVMem's high acceptance at n_max=3 comes with its per-turn retrieval machinery;
   mainline gets the same tokens/pass by drafting deeper.** They converge on the same
   throughput — there is no free lunch in either spec implementation.
3. **Mainline's hard wall is VRAM**: dense KV gives 57K tokens on this card (with q4_0 KV
   and a squeezed batch). Anything beyond that is exactly what a paging layer must fix.
4. **The behavioral variance between runs (stall, token counts) can exceed the speed
   difference between schemes.** Wall-time comparisons need either repeats or cleaned-up
   runs; scores are stable at 100.

## Next steps (the paging roadmap, not yet implemented)

- Profile the ~36 ms/pass gap (nsys/`--perf`, or add timers in the fork): candidates are
  the MTP draft forwards, GDN checkpoint/replay, and per-layer kernel launch overhead.
- KV paging module on top of mainline's cell/block KV: VRAM = fully-associative cache over
  RAM-resident blocks, score-driven replacement, async prefault. This is what turns 57K
  dense into 256K without KVMem's per-turn checkpoint ceremony.
- Low-bit KV (3-bit with per-channel K scales) to push the dense window toward 100K+.
- Page-sparse attention (Quest-style summaries + top-k page gather) as the long-context
  endgame: 256K with attention traffic comparable to a 32K dense window.

## Artifacts

- `~/coding/llama-tierkv/llama-next.bundle` — full git bundle (2 commits: upstream snapshot +
  recipe), `git clone llama-next.bundle`
- `~/coding/llama-tierkv/src/scripts-5060ti (git history)` — the recipe as a patch on top of upstream
- `~/coding/llama-tierkv/src/scripts-5060ti/` — `start-mtp5.sh` (tuned launcher), `probe.py`
- Remote: `/root/llama.cpp-next` (build: `build/bin/llama-server`)
- E2E results: `DeliverableBench-Live/results/llama-next-mtp5{,-r2}/`
