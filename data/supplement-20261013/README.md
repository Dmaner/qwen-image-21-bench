# C1 H200 supplement — evidence index

[C1 report](../../C1_REPORT.md) · [Experimental design](../../EXPERIMENT_DESIGN.md) · [Reproduction](../../reproduce/supplement/README.md).

Seven configurations share the 36-case CFG1 manifest and new same-GPU BF16 controls. Memory tables use the full-precision measured request fields; the frozen initial analysis remains in the compact raw archive. The single-case CFG4 prefix check is recorded separately. The [2026-10-11 records](../README.md#historical-c1-and-a0-evidence) remain historical.

| Evidence | Files |
|---|---|
| Analysis and table source | [analysis.json](analysis.json), [TABLES.md](TABLES.md) |
| Request latency | [performance.csv](performance.csv) |
| Worker allocated/reserved memory | [memory.csv](memory.csv) |
| Per-image scores and working gate | [quality_pairs.csv](quality_pairs.csv), [quality_gate.csv](quality_gate.csv) |
| Text, subject and edit review | [visual_review.csv](visual_review.csv), [coverage](visual_review_merge_summary.json) |
| Image validity and selected figures | [image_health.csv](image_health.csv), [selection and hashes](selected_figures.json) |
| Prefix storage and runtime audit | [cache_observations.csv](cache_observations.csv), [cache audit](cache_audit.json) |
| Profiler kernel inventory | [PREFIX_TRACE_SUMMARY.md](PREFIX_TRACE_SUMMARY.md), [prefix_trace_summary.json](prefix_trace_summary.json) |
| Independent T2I diagnostic repeat | [Recheck summary](recheck/REPORT.md), [separate-cohort data](recheck/analysis.json) |
| Compact request/environment/audit records | [rawrecords.zip](rawrecords.zip) |
| File hashes | [SHA256SUMS](SHA256SUMS) |

[Frozen inputs and manifests](../../reproduce/supplement/inputs) record prompts, seeds, ordered references and hashes. LPIPS ≤ 0.15 is the predeclared working numerical screen; human review and maintainer approval are tracked separately. Raw GPU traces and model weights are not included in this package.

`rawrecords.zip` contains the 60 primary cells and three independent T2I recheck cells, including their original JSON/log records, source hashes and timing-observer provenance. The two cohorts are not pooled. Extract the ZIP into a separate review directory; rerunning the published commands produces fresh results on the reviewer’s machine.
