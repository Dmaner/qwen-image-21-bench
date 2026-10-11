# Evidence index

All headline tables come from the fresh Phase 2 cohort. Files here are unchanged copies of recorded analysis; absolute paths identify the original environment and are not live links.

| Claim | Evidence in this repository | Compact release / gallery |
|---|---|---|
| Latency and memory | [PERFORMANCE.json](phase2/performance/PERFORMANCE.json) | Raw request rows, environment/configuration and worker-memory records |
| Paired quality | [All quality metrics](phase2/quality), [completed model-assisted reviews](phase2/visual-review) | Request metadata; selected original PNGs in the [gallery](../GALLERY.md) |
| Quantization actually executed | [GPU audit](phase2/audits/GPU_AUDIT_REVIEW.json) | Frozen component configs; actual worker observations summarized in the linked audit |
| Kernel counts and quantization overhead | [Profile analysis](phase2/profiles/PROFILE_ANALYSIS.md), [JSON](phase2/profiles/profile_comparison.json), [independent audit](phase2/audits/ACTUAL_PROFILE_REVIEW.json) | Profile summaries; full raw traces and memory pickles remain local |
| Latest BF16 equals A0 anchors | [A0_ALIGNMENT_RECHECK.json](phase2/audits/A0_ALIGNMENT_RECHECK.json) | BF16 T2I/edit anchor images in the [gallery](../GALLERY.md); author originals referenced by revision |
| Historical n=2 A0 cross-check | [comparison.json](a0-cross/comparison.json), [review](a0-cross/final-review.json) | Comparison and run metadata; complete repeated-image corpus remains local |

[Single compact data package](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-H200-essential-data-20261011.zip) · [Release](https://github.com/Dmaner/qwen-image-21-bench/releases/tag/c1-h200-20261011) · [SHA256SUMS](SHA256SUMS) · [member manifest](ARTIFACT_MANIFEST.json).

The public package contains essential numerical records, not the full experiment archive. All quality metrics and selected original images are public; full raw traces, memory pickles and the complete image corpus remain local. The historical A0 cross-check's 28 outputs repeat fixed inputs, including diagnostic runs; they are not 28 independent prompt cases.

The `human_review.csv` files under `quality/` are the analyzer's **unfilled review templates**, retained as original outputs. Completed observations are in [`visual-review/`](phase2/visual-review); these were **model-assisted**, not independent human acceptance. A frozen held-out split and PSNR/SSIM do not establish an agreed quality threshold.
