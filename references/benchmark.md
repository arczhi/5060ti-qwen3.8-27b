# SPDX-License-Identifier: Apache-2.0

# Reference Benchmark

This is an anonymized reference run on an RTX 5060 Ti 16GB with 32GB system RAM. It is a practical coding-agent result, not a synthetic tokens-per-second claim.

## Task result

The agent received a design document for a custom prompt feature in a medium-sized Go service. It inspected the repository, implemented the change, added tests, and ran scoped verification.

| Metric | Result |
| --- | --- |
| Agent wall time | About 14m 20s |
| Server generation requests | 14 |
| Files changed | 7 |
| New tests | 5 passed |
| Scoped build | Passed |
| Scoped vet | Passed |
| Formatting and diff check | Passed |
| Pre-existing tests | 2 failed; unrelated baseline failures |
| Real provider E2E | Not run; required credentials and a billable test account |
| Context compaction | Not observed in this run |

## Server measurements

Values below came from llama.cpp request logs. The prompt column is the server's request-level prompt field, not a claim that the complete agent context was only that size.

| Metric | Observed range |
| --- | ---: |
| Request prompt field | 19-401 tokens |
| Prefill | 22.18-326.36 tok/s |
| Decode | 11.06-12.95 tok/s |
| Generation time | 7.89-184.60s per request |
| MTP acceptance | 84.9%-100% |
| Short smoke test | 86.54 tok/s prefill, 17.77 tok/s decode, about 0.69s total |

The long generation outlier was 184.6s. That is why end-to-end agent time and task completion matter more than the short smoke decode rate.

## FFN offload comparison

This follow-up used the same Pi session prompt, repository checkout, design document, skill, and task. The 16GB GPU could not load the MTP configuration reliably at 96K during this test, so B and C used an 80K server context. A is the earlier completed no-offload run. “Weighted” tok/s is total prompt/output tokens divided by total server time for that phase; it is more useful than averaging tiny and large requests equally.

| Variant | Agent wall time | Requests | Weighted prefill | Weighted decode | Avg server request | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A: no `--n-cpu-ffn` | About 14m20s | 14 | 22.18-326.36 tok/s | 11.06-12.95 tok/s | 7.89-184.60s | Completed; 7 files, 5 new tests passed |
| B: `--n-cpu-ffn 4` | 14m04s | 53 | 656.86 tok/s | 19.40 tok/s | 15.16s | Completed; build/vet/gofmt and 5 new tests passed |
| C: `--n-cpu-ffn 8` | 25m30s until stopped | 112 | 565.51 tok/s | 18.36 tok/s | 12.78s | Code applied, but stopped in a repeated host test-environment loop |

The counts include agent tool calls and retries, so B/C are not synthetic single-request scores. FFN4 was selected as the practical default: it reduced GPU pressure and had the best observed decode/prefill balance. FFN8 did not improve decode and was not a completed coding-agent run. The test supports the community advice that CPU can usefully carry a small portion of FFN work while the MTP draft remains CPU-resident, but increasing the offload amount is not automatically faster.

This experiment was prompted by Reddit user **Square_Turn935**: [original discussion](https://www.reddit.com/r/LocalLLM/comments/1w509o5/comment/p7c2sim/).

## Earlier local baseline context

These older oMLX runs used different models, prompts, and workloads, so they are directional only:

| Setup | Agent result | Decode observation | Interpretation |
| --- | --- | ---: | --- |
| Qwen3.6 35B-A3B oQ2 + native MTP | About 99s, 7 rounds | 29.4 tok/s final | Faster local baseline, but weaker implementation fidelity in some design-to-code tasks |
| Qwen3.8 27B coder + oMLX | About 123s, 10 rounds | 13.8 tok/s final | MTP was inactive for that model; more tool loops increased total time |
| Ornith 35B oQ3e + oMLX native MTP | About 51s, 5 rounds | 43.4 tok/s final | Stronger speed baseline in a separate n8n task; not a controlled comparison |

## What to compare in future

For a fair local-model leaderboard, hold the repository, design document, clean-worktree state, tool allowlist, context limit, temperature, and warm/cold cache state constant. Report:

`first-success rate × average turns × total wall time × compactions × final test pass rate`

Tokens per second is a diagnostic metric. It is not the final coding metric.
