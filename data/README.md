# Evidence index

All headline tables come from the fresh Phase 2 cohort. Files here are unchanged copies of recorded analysis; absolute paths inside them identify the original environment and are not live links.

| Claim | Small evidence in this repository | Full evidence in the release |
|---|---|---|
| Latency and memory | [PERFORMANCE.json](phase2/performance/PERFORMANCE.json) | Phase 2 `raw/results/perf_*`, `mem_*`: commands, environment, request JSONL, worker logs and images |
| Paired quality | [Quality metrics](phase2/quality), [completed model-assisted reviews](phase2/visual-review) | Phase 2 `raw/results/quality_*`: all 120 scored original PNGs, five extra warmup PNGs and request records; `analysis/visual-review` |
| Quantization actually executed | [GPU audit](phase2/audits/GPU_AUDIT_REVIEW.json) | Phase 2 `raw/results/audit_*`, configs, module/dtype/scale observations |
| Kernel counts and quantization overhead | [Profile analysis](phase2/profiles/PROFILE_ANALYSIS.md), [JSON](phase2/profiles/profile_comparison.json), [independent audit](phase2/audits/ACTUAL_PROFILE_REVIEW.json) | Five profiler ZIPs, each with T2I/edit trace and memory snapshot; Phase 2 kernel-source provenance |
| Latest BF16 equals A0 anchors | [A0_ALIGNMENT_RECHECK.json](phase2/audits/A0_ALIGNMENT_RECHECK.json) | Phase 2 BF16 `t2i_teapot_s42.png` and `edit_bear_s42.png`; author originals at the pinned A0 revision |
| Historical n=2 A0 cross-check and 28 repeated images | [comparison.json](a0-cross/comparison.json), [review](a0-cross/final-review.json) | Separate `C1-A0-cross-20261010-evidence.zip`: nine run groups, including diagnostic-run images |

[Download the raw archives](https://github.com/Dmaner/qwen-image-21-bench/releases/tag/c1-h200-20261011) · [SHA256SUMS](SHA256SUMS) · [full member manifest](ARTIFACT_MANIFEST.json).

The A0 cross archive contains 16 T2I and 12 editing outputs, including warmup, feasibility and diagnostic outputs. These are repeated fixed inputs, not 28 independent prompt cases. Its old raw profiler payloads are omitted; the five Phase 2 profiler archives provide the ten traces analyzed in the main report.

The `human_review.csv` files under `quality/` are the analyzer's **unfilled review templates**, retained as original outputs. Completed observations are in [`visual-review/`](phase2/visual-review); these were **model-assisted**, not independent human acceptance. A frozen held-out split and PSNR/SSIM do not establish an agreed quality threshold.
