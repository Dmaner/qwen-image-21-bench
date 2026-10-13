# Evidence index

The latest C1 supplement is indexed below. The **2026-10-11** C1 tables come from the archived Phase 2 cohort. The separate [A0 review](../A0_REVIEW.md) uses the 2026-10-10 BF16 cross-check. Recorded analysis files are unchanged copies; `a0-cross/review_metrics.json` and `publication_evidence.json` are newly derived report tables and a publication audit. Absolute paths in historical records identify the original environment and are not live links.

## Latest C1 supplement — 2026-10-13

[Supplement evidence index](supplement-20261013/README.md) · [Experimental design](../EXPERIMENT_DESIGN.md) · [Reproduction](../reproduce/supplement/README.md).

The supplement covers seven configurations and 36 CFG1 cases, plus an independent CFG4 prefix check. Its comparisons use the new same-GPU BF16 controls; the historical cohort remains separate.

## Historical C1 and A0 evidence

| Claim | Evidence in this repository | Compact release / gallery |
|---|---|---|
| Latency and memory | [PERFORMANCE.json](phase2/performance/PERFORMANCE.json) | Raw request rows, environment/configuration and worker-memory records |
| Paired quality | [Quality metrics](phase2/quality), [completed model-assisted reviews](phase2/visual-review) | Request metadata; selected original PNGs in the [gallery](../GALLERY.md) |
| Supplementary LPIPS | [Method, hashes and scores](phase2/quality/lpips/lpips_metrics.json), [per-image CSV](phase2/quality/lpips/lpips_pairs.csv) | Added on 2026-10-13 from archived images; separate from the original release ZIP |
| Quantization actually executed | [GPU audit](phase2/audits/GPU_AUDIT_REVIEW.json) | Frozen component configs; actual worker observations summarized in the linked audit |
| Kernel counts and quantization overhead | [Profile analysis](phase2/profiles/PROFILE_ANALYSIS.md), [JSON](phase2/profiles/profile_comparison.json), [independent audit](phase2/audits/ACTUAL_PROFILE_REVIEW.json) | Profile summaries; full raw traces and memory pickles remain local |
| 2026-10-11 BF16 equals A0 anchors | [A0_ALIGNMENT_RECHECK.json](phase2/audits/A0_ALIGNMENT_RECHECK.json) | BF16 T2I/edit anchor images in the [gallery](../GALLERY.md); author originals referenced by revision |
| Historical n=2 A0 cross-check | [comparison.json](a0-cross/comparison.json), [review](a0-cross/final-review.json) | Comparison and run metadata; complete repeated-image corpus remains local |
| Standalone A0 review | [Report tables](a0-cross/review_metrics.json), [publication and image audit](a0-cross/publication_evidence.json) | Same historical measurements; original rows/commands already in the compact package |
| A0 denoising and allocation | Trace summaries: [T2I](a0-cross/t2i_trace_analysis.json) / [edit](a0-cross/edit_trace_analysis.json); memory summaries: [T2I](a0-cross/t2i_memory_snapshot_analysis.json) / [edit](a0-cross/edit_memory_snapshot_analysis.json) | Four byte-identical analysis copies for direct browsing; full traces and memory pickles remain local |

[Single compact data package](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-H200-essential-data-20261011.zip) · [Release](https://github.com/Dmaner/qwen-image-21-bench/releases/tag/c1-h200-20261011) · [SHA256SUMS](SHA256SUMS) · [member manifest](ARTIFACT_MANIFEST.json).

The public package contains essential numerical records, not the full experiment archive. Quality metrics for the three reported FP8 configurations and selected original images are public; full raw traces, memory pickles and the complete image corpus remain local. The historical A0 cross-check's 28 outputs repeat fixed inputs, including diagnostic runs; they are not 28 independent prompt cases.

Completed observations are in [`visual-review/`](phase2/visual-review); these were **model-assisted**, not independent human acceptance. Empty review templates and standalone files for the omitted BF16-block configuration have been removed from this checkout. The release package and historical aggregate records remain unchanged.
