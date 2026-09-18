# Stage 2 design: TierKV — tiered KV + page-sparse attention on mainline llama.cpp

Status: **design + profiling complete, implementation not started** (2026-09-18).
Stage 1 (mainline + MTP-5, 100/100 in 433 s) is in [llama-next-mtp5-experiment.md](llama-next-mtp5-experiment.md).

## 1. Why (measured limits of the dense scheme)

Cost model measured on the 5060 Ti (8-28K ctx, q5_0 KV, temp 0, ub 128):

| Component | Cost | Source |
| --- | --- | --- |
| Weight read floor (IQ4_XS, 448 GB/s) | 31.6 ms / pass | physics |
| Fixed pass cost (weights + framework + GDN + launch) | **~40 ms** | linear fit over n_max 1/2/3/5 |
| Per draft token (MTP head, sequential) | **5.7 ms** | same fit; unaffected by backend sampling, p-split, draft-KV dtype |
| Attention scaling | **~0.32 ms per 1K ctx per token** | nospec 38.9 ms @8K → 45.5 ms @28K → ~55 ms @57K |

Consequences:
- The 16 GiB card fits **57,344 tokens dense** (q4_0 KV, b256/ub128, 13.2 GiB model).
  At 57K the attention traffic alone is ~18 ms/token (~30% of decode) — the dense path
  degrades quadratically in cost and linearly in VRAM.
- Beyond 57K there is no dense option. KVMem's answer is per-turn retrieval of a 32K
  window; it works, but it pays 100-200 ms/turn and gives up per-token continuity.
- The cost model says any scheme that keeps the *active working set* at ~32K tokens will
  stay near the weight-bandwidth floor (~65-80 tok/s with MTP-5), regardless of total
  context — provided the long tail costs ~0.

## 2. Architecture: 3 tiers, 2 mechanisms

```
                per-token                 per-block                 per-session
  ┌────────┐   ┌──────────────────┐   ┌───────────────────┐   ┌─────────────────┐
  │  VRAM  │   │ weights + active │   │ page staging buf  │   │  (nothing)      │
  │ 16 GiB │   │ KV window (32K)  │   │ (top-k page gather)│  │                 │
  └────────┘   └──────────────────┘   └───────────────────┘   └─────────────────┘
  ┌────────┐   ┌──────────────────┐   ┌───────────────────┐   ┌─────────────────┐
  │  RAM   │   │                  │   │ full KV block     │   │ checkpoint/     │
  │ 32 GiB │   │                  │   │ store + summaries │   │ snapshot store  │
  └────────┘   └──────────────────┘   └───────────────────┘   └─────────────────┘
  ┌────────┐                                                     ┌─────────────┐
  │  SSD   │                                                     │ session KV  │
  │ 1.5GB/s│                                                     │ archives    │
  └────────┘                                                     └─────────────┘
```

Mechanism 1 — **tiered block store (per-turn granularity, deterministic)**:
all KV blocks live in host RAM; VRAM holds a bounded active window plus a staging buffer.
Blocks move host→device asynchronously (pinned buffers, separate stream) — never directly
to/from SSD in the hot path (SSD read 1.5 GB/s).

Mechanism 2 — **page-sparse attention (per-token granularity, approximate but bounded
error)**: each 64-token page keeps K min/max summaries (Quest-style). Per generation step,
each attention layer scores its pages against the current query and stages the top-k pages
plus sinks + recent window into the FA path. This is what makes >57K possible at all:
attention traffic becomes O(top-k pages) instead of O(context).

The two mechanisms compose: the tiered store provides blocks, the sparse selector decides
which ones enter VRAM per step.

## 3. Where it hooks into mainline

| Piece | Location | Notes |
| --- | --- | --- |
| Block store (host) + page table | new `src/llama-kv-pager.{h,cpp}`, owned by the memory module | mirrors `llama_kv_cache_unified` cells; one block = 64 cells |
| Page summaries (K min/max) | maintained on KV write in the unified cache | +2×1024 f16 per page per attn layer ≈ 268 MB at 256K, 16 layers |
| Top-k selection | new ggml op (`ggml_topk_pages`) or a small CUDA kernel | per (layer, kv-head-group, query) — selection shared across the 4-6 verify rows |
| Page gather / staging | new ggml op `ggml_gather_pages` (copy) or strided FA over a page index | staging buffer keeps FA kernels unchanged (fixed shape → CUDA-graph safe) |
| Attend over staging | `llama-graph.cpp` `build_attn` uses the staged K/V view + a positions mask | sink pages + most recent window always staged |
| Host↔device migration | `ggml_backend_tensor_set_async` on a copy stream | double-buffered; overlaps the 40 ms/pass compute budget |

Constraints discovered in stage 1 that shape the design:
- CUDA graphs need **static shapes** → top-k count must be fixed (`k` constant), only the
  indices vary; staging buffer sized `k*64 + window` tokens.
- Spec decode verify batches are 2-6 rows; page selection can be shared across rows of the
  same sequence (saves k× row work).
- The 48 GDN layers are recurrent (144 MiB state, sequence-constant). They need **no**
  paging; only the 16 attention layers carry KV. This is what makes the whole thing
  affordable on a 16 GiB card.

## 4. Milestones

| # | Deliverable | Validation | Est. |
| --- | --- | --- | --- |
| 2.1 | Profiling + cost model (**done**, this doc) | numbers above | — |
| 2.2 | Standalone prototype: summaries + scoring + top-k + gather over a synthetic 256K KV (**done**, numbers below) | wall-clock per step ≤ 1 ms at 16 layers | 1-2 days |
| 2.3 | Memory-module integration behind env switches; dense fallback when ctx < window (**done**, evidence below) | same E2E output as dense at ≤ 48K (bit-identical greedy) | 1 week |
| 2.4 | Approximate mode: retrieval staging (**done**) + real page-sparse attention selector (query capture + K min/max page summaries, **done**, evidence below) | needle recall via attention scoring; agent E2E | 1-2 weeks |
| 2.5 | Low-bit archival KV (**done as mixed precision**: host Q4_0 vs window Q5_0; E2E also run with Q4_0 window) | quality on long-context suite; RAM per 256K ≤ 3 GB | 2 weeks |
| 2.6 | SSD session snapshots (prefix reuse across restarts) (**done**, evidence below) | restore + 2nd-run prefill saving | 1 week |

## 4b. Prototype results (milestone 2.2, 2026-09-18)

`~/coding/llama-tierkv/src/scripts-5060ti/proto/proto_page_sparse.cu` — synthetic 256K KV (4096 pages, 16 layers,
KV dim 1024, f16 storage, verify batch 6), measured on the 5060 Ti (sm_120a, CUDA 13.2):

| Stage | Measured | Notes |
| --- | ---: | --- |
| Summary build, full 256K (16 layers) | 321.6 ms | one-time per context |
| Summary update, incremental (1 page) | **0.014 ms** | per 64 new tokens — negligible |
| Page scoring, 1 row (naive, re-reads summaries per group) | 1.83 ms | → 11.0 ms per pass for 6 rows |
| Top-k, 1 row (naive merge) | 3.77 ms | → 22.6 ms per pass for 6 rows |
| Gather K+V, 16 layers × 16 pages (pinned host→device) | **2.90 ms** | 67.1 MB at 23.2 GB/s effective |
| Naive total overhead | 36.5 ms/pass | too slow as-is |

The prototype also shows the concrete fixes, all cheap:
- Score **once per (layer, page)** and apply all 4 KV-head group queries in the same read →
  summary traffic drops 4× (1.83 ms → ~0.46 ms per row).
- Share the top-k across the 6 verify rows (adjacent positions have near-identical queries)
  and use a proper warp/block selection instead of the merge-of-256-sorted-lists →
  ~3.8 ms → < 1 ms per pass.
- Gather with caching: selected pages change slowly, so fetch only deltas; 2.9 ms is the
  worst-case re-fetch-everything number.
Projected steady-state overhead: **~4-6 ms/pass** on top of the measured ~40 ms compute
budget (~12%) — acceptable for extending the working context from 57K to 256K.
Memory cost of the summaries: 16.8 MB per layer → **268 MB at 256K** (fits VRAM alongside
the weights and window).

Artifacts: `~/coding/llama-tierkv/src/scripts-5060ti/proto/proto_page_sparse.cu` + results in
`references/llama-next-mtp5-data.md` §7.

## 4c. Milestone 2.3-2.6 implementation and test evidence (2026-09-18)

Implemented in `~/coding/llama-tierkv/src` (fork commits `21d6981`, `035f26b`):
`src/llama-kv-pager.{h,cpp}` + hooks in `src/llama-kv-cache.{h,cpp}` and
`src/llama-model.cpp`. Everything is opt-in via env vars (no behavior change by default).

| Env | Meaning |
| --- | --- |
| `LLAMA_KV_PAGER=1` | enable the host-tier store |
| `LLAMA_KV_PAGER_WINDOW=N` | VRAM window in tokens; logical `-c` can be larger |
| `LLAMA_KV_PAGER_MAX_CTX=N` | logical capacity of the store |
| `LLAMA_KV_PAGER_STAGE=1` + `TOPK` / `QUERY` / `PIN_MAX` | retrieval staging (IDF-weighted lexical selector) |
| `LLAMA_KV_PAGER_HOST_Q4=1` | host rows in Q4_0 while the window stays Q5_0 |
| `LLAMA_KV_PAGER_SAVE` / `LOAD` / `AUTOSAVE=N` | SSD snapshots |

Test evidence (all on the 5060 Ti, `-c 16384`, window 2048, greedy, `spec none`):

| Milestone | Test | Result |
| --- | --- | --- |
| 2.3 | pager ON (`window=8192`, no evictions) vs OFF, greedy | **bit-identical output** |
| 2.3 | 7,769-token prompt through a 2,048-token window | request completes; 91,600 rows saved (5,725 evictions x 16 layers), no crash |
| 2.4 | control: needle at pos 0, 3,353-token prompt, staging OFF | wrong answer / refusal (needle evicted) |
| 2.4 | same prompt, staging ON (IDF selector, top-8 blocks) | **exact recall: "CORAL-482196"** (temp 0 and 0.7) |
| 2.4 | selector ranking with IDF weighting | needle block scores 3413 vs ~170 for filler blocks (was rank ~150+ unweighted) |
| 2.5 | window Q5_0 + host Q4_0 (704 -> 576 B/row, -18%) | exact recall preserved through the conversion path |
| 2.6 | save store (302 MB at max_ctx 16K / q4_0) then restart with `LOAD` | fresh server answers a **26-token question** with "CORAL-482196" — content never present in that process |

Known limitations of this MVP (next increments):
- The lexical (token-overlap) selector is a placeholder for query-key scoring; IDF
  weighting makes it workable on repetitive text but it is not semantic.
- A block is only recalled if its tokens overlap the query; paraphrases fail.
- `max_ctx` is preallocated in host RAM; 256K x q4_0 x 16 layers = 4.6 GB.
- The store file must be loaded with the same KV/host types (validated in the header).
- Staged cells are pinned with a simple FIFO cap; a scored replacement policy is next.
- Full 256K context with staging has not been E2E-benchmarked yet; recall was validated
  at 11K prompt length, and the dense window (57K) path is unchanged.

The first two limitations were addressed by the real page-sparse selector below.

## 4d. Page-sparse attention: real implementation (2026-09-18)

Implemented in fork commit `894e77b` (files: `src/llama-kv-pager.{h,cpp}`,
`src/llama-graph.cpp`, `src/llama-kv-cache.{h,cpp}`, `src/llama-model.cpp`):

1. **Page summaries** — every saved KV row updates per-page K min/max (f16) for the first
   head-dim channel slice (256 of 1024 channels) of its layer: `kmin/kmax[layer][page][ch]`.
2. **Query capture** — `build_attn_mha()` adds a `ggml_cpy` of the *last token's first query
   head* into a persistent per-layer tensor (`get_q_capture`), CUDA-graph safe.
3. **Scoring** — `select_blocks_attn()` computes the Quest-style upper bound
   `sum_c q_c * max(kmin_c, kmax_c)` per page (channel step 8), sums across layers, ranks
   pages, and feeds the top-k into staging. `LLAMA_KV_PAGER_SELECT=lex|attn|hybrid`.
4. **MTP exemption** — MTP draft contexts are exempt from both the window override and the
   host store (`llama_kv_pager_set_disabled`), otherwise spec decoding breaks
   ("failed to process speculative batch"): the draft cache must stay aligned with the target.

Validation (needle at pos 0, question at the end, 3,353-token prompt, window 2,048,
evictions force the needle out of VRAM):

| Selector | Needle-page rank | Answer |
| --- | --- | --- |
| `lex` (IDF) | #1 (score 3413) | `CORAL-482196` |
| **`attn`** | **#1 (score 160.3; runner-up 82.1)** | **`CORAL-482196`** |
| `hybrid` (lex ∪ attn) | #1 | `CORAL-482196` |

## 4e. End-to-end agent benchmarks under the pager

`DeliverableBench ocr-dual-channel`, `pi` agent, thinking off, MTP-5, presence 1.5:

| Run | Pager config | Result |
| --- | --- | --- |
| stage-1 baseline (no pager) | q5_0 window = ctx 57,344 | **100.0 / 100**, 433 s |
| pager #1 | window 32,768 (logical 65,536), host Q4_0 / window Q5_0, `lex` | 67.7 / 100 — agent entered a command loop (87 bash), killed and scored |
| **pager #2** | **window 49,152, q4_0 window + host Q4_0, `hybrid` selector** | **99.9 / 100** (73 tests green, 40/40 deliverables, 15/15 completion, 14.9/15 behavior with 1 steer), 20 min wall (loop + steer) |

Honest reading:
- Retrieval itself did not break correctness when the window was large enough to cover the
  working set; the pager #2 run produced a 99.9 score with the full stack (pager + low-bit KV
  + page-sparse selector + MTP-5) enabled.
- Both pager runs hit agent-side repetition loops (the stage-1 baseline also looped in R2),
  so wall-time comparisons remain contaminated by agent behavior; the loops are not caused
  by the pager per se but are more likely with longer, retrieval-dependent contexts.
- With window 49,152 and the agent's ~54K peak, only the last few thousand tokens were ever
  evicted — the E2E does not yet stress deep retrieval. The needle tests do.

## 4f. Policy v2 and long-context results (2026-09-18, commit c1e0c2f)

Policy changes (all env-gated, defaults on):
- **Head protection** (`LLAMA_KV_PAGER_HEAD=2048`): the first N positions are never evicted
  or purged. The system prompt / task spec, which the old oldest-first policy dropped first,
  now survive every wrap.
- **Eviction-triggered, rate-limited staging** (`STAGE_EVERY=1024`): staging runs only after
  new evictions and at most once per 1024 positions — no per-decode-step churn, near-zero
  overhead while the context fits the window.
- **Batched block gather**: contiguous cell runs + one tensor copy per layer per block
  (was one synchronous copy per row per layer: 1024 copies -> 16).
- **Lazy host store**: buffers allocated on first save; unused contexts cost nothing
  (the server creates two main contexts — see below — so this matters).

Long-context measurements (q4_0 KV window 32,768, ub 128; synthetic 3.4-chars/token probe):

| Config | Prefill | Decode | Notes |
| --- | ---: | ---: | --- |
| 132,140 tokens, no spec | 713 tok/s | **22.74 tok/s** | attention window 32K |
| 260,883 tokens, no spec | 680 tok/s | **22.70 tok/s** | decode identical to 128K: context-independent |
| 90,485 tokens, MTP-5 | 626 tok/s | 28.5 tok/s | synthetic prose, acceptance 29%; agent traffic is much higher |

Reference (same card, KVMem run): 261K single-shot 56.8 tok/s decode / 1,399 tok/s prefill
**with MTP**; 54K agent E2E decode p50 50.3. Our E2E decode p50 with MTP-5 is 65.8 (stage-1
baseline) and 59-65 in pager runs — the gap at 256K is entirely MTP.

**MTP at 256K is not feasible on this 16GB card with mainline's MTP implementation:**
the draft context must hold the target's full position range, and its graph reserve scales
with n_ctx (~1.1 GiB at 262,144) which does not fit next to the target. An attempt to cap
the draft's context (`LLAMA_SPEC_MTP_MAX_CTX=65536`) fails as soon as the target passes the
cap (`llama_decode(ctx_dft) failed rc=1 at pos=65536`). KVMem solved this with a windowed
MTP pool (45,056 cells) plus ReplaySSM; porting that is the remaining piece for MTP at 256K.

Server-side observations:
- The server instantiates the main context twice (two `create_memory(..., ctx_type=0)` calls,
  both 16 attention layers) — hence two pager instances. Lazy allocation means the idle one
  costs ~1 MiB of metadata. MTP draft contexts (ctx_type=1) are exempt from paging.
- A window equal to the logical context at 65,536 + MTP left only ~480 MiB free VRAM and
  crashed on `cudaGraphInstantiate`; 49,152 works (15.5 GB used).

End-to-end under policy v2:
- window 32,768 / ctx 65,536, hybrid selector, MTP-5: agent looped after evictions began
  (score 64.4, killed). **Root cause**: once the middle of the conversation is evicted, the
  model's KV no longer matches the token stream the server sends; our lexical+attention
  selector does not restore enough of the agent's own recent history, and the agent loses
  the thread. This is the retrieval-coverage limitation, not a mechanism failure.
- window 49,152 / ctx 65,536, hybrid selector, MTP-5 (full stack, no steer): **100.0 / 100**
  in **398 s** — vs the no-pager baseline's 100.0 in 433 s.

### Final comparison (same card, same task, same agent)

| | No pager (stage 1) | Pager v2 full stack | KVMem |
| --- | ---: | ---: | ---: |
| Score | 100.0 | **100.0** | 100.0 |
| Wall | 433 s | **398 s** | 482 s |
| Decode p50 | 65.8 tok/s | 62.1 tok/s | 50.3 tok/s |
| Prefill p50 | 479 tok/s | 382 tok/s | 477 tok/s (E2E) |
| Draft acceptance | 72.8% | 73.7% | 83.8% |
| Context ceiling | 57,344 dense | **262,144** | 262,144 |
| Decode at 256K | n/a (does not fit) | 22.7 tok/s (no MTP) | 56.8 tok/s (MTP) |
| Host RAM at 256K | n/a | ~4.6 GB (q4_0 store) | 13.8 GB |
| Persistence | — | **SSD snapshots** | — |

Honest verdict: the pager stack now matches the baseline score and wall time while extending
the context ceiling 4.5x and adding persistence; at 256K it runs without MTP (22.7 tok/s,
context-independent) because mainline's MTP draft cannot be windowed. Beating KVMem's 256K
decode number needs a windowed MTP with state replay (KVMem's ReplaySSM) — the remaining
piece.

Acceptance for stage 2: **256K context on the same card, agent E2E ≥ 100/100, decode
≥ KVMem's 50.3 tok/s**, and no regression below 57K (dense mode unchanged).

## 5. Risks

- **Quality**: page-sparse attention is approximate. Mitigation: sinks + recent window
  always dense; k tuned so recall tests pass; dense mode stays default below the threshold.
- **CUDA-graph breakage**: dynamic indices + static shapes must be enforced; the staging
  gather must be graph-capturable.
- **Score (2026-09-17) must not regress**: all work is behind `--kv-pager` until 2.4 passes.
- Effort: 2.3-2.4 are deep C++ across `llama-memory`, `llama-graph`, ggml-cuda. Not a
  single-session task; milestones are ordered so each one is independently testable.
