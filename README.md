# Qwen-Image 2.1 — A0 review and C1 H200 experiments

Performance, paired image quality and torch-profiler evidence for the C1 FP8 work in [vLLM-Omni #8586](https://github.com/vllm-project/vllm-omni/issues/8586), following [luca-888's A0 baseline](https://github.com/luca-888/qwen-image-21-bench/blob/cf99e73c8fa1603718c47405730d94c9fbfe6e16/A0_REPORT.md).

**1× H200 SXM · 1024² · 50 steps · CFG 1 · eager · FA3 · batch/concurrency 1.** The latest C1 supplement covers seven configurations and 36 paired cases, with a separate CFG4 prefix check: BF16, DiT FP8, MLP BF16, Text Encoder FP8, combined FP8, Prefix V FP8 and Prefix K+V FP8.

| Start here | Contents |
|---|---|
| [A0 independent review](A0_REVIEW.md) | BF16 reproduction in A0's format; latency, serving, profiler, memory and differences from A0 |
| [C1 report](C1_REPORT.md) | Latest supplement: selective BF16, prefix storage and diverse editing inputs |
| [Supplement design](EXPERIMENT_DESIGN.md) | Frozen settings, 36 cases and predeclared working quality gate |
| [Supplement data](data/supplement-20261013/README.md) | New same-GPU measurements, paired quality, cache audits and profiler summaries |
| [Supplement image gallery](SUPPLEMENT_GALLERY.md) | New person, product, sign edits and selective-BF16 text comparison |
| [Original image gallery](GALLERY.md) | Same-input comparisons: T2I, editing, English and Chinese text |
| [Historical 2026-10-11 numerical records](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-H200-essential-data-20261011.zip) | Compact raw request/environment/configuration and worker-memory records; analysis and A0 cross-check metadata |
| [Data index](data/README.md) | Machine-readable results and evidence mapping |
| [Latest reproduction](reproduce/supplement/README.md) | Seven-configuration runner, fixed inputs and analysis |
| [Historical reproduction](reproduce/README.md#historical-2026-10-11-experiment) | Original four-configuration commands |
| [Historical issue comment](ISSUE_COMMENT.md) | Original draft with public links |
| [Provenance](PROVENANCE.md) | Source revisions, attribution and artifact boundaries |

## Historical 2026-10-11 experiment

DiT-only FP8 reduced offline T2I/edit latency by **17.1% / 14.7%** and peak allocated memory by **6.50 GiB**. Combined FP8 saved **12.97 GiB**, but both configurations omitted the requested title in one text-rendering case. Quality acceptance remains pending.

The historical C1 cohort contains four reported configurations from **2026-10-11**. Its archive and profiler records remain unchanged. The [A0 review](A0_REVIEW.md) covers the separate **2026-10-10 BF16 cross-check**.

Download the single compact [release asset](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-H200-essential-data-20261011.zip) and verify it from this repository checkout:

```sh
gh release download c1-h200-20261011 --repo Dmaner/qwen-image-21-bench \
  --pattern C1-H200-essential-data-20261011.zip --dir c1-data
(cd c1-data && sha256sum -c ../data/SHA256SUMS)
# macOS: replace sha256sum with shasum -a 256
```

See [ARTIFACT_MANIFEST.json](data/ARTIFACT_MANIFEST.json) for the included records and hashes. The historical records retain the four-configuration image gallery, its quality metrics and model-assisted reviews, plus the original performance/profile summaries. **Full raw traces, memory pickles and the complete image corpus remain local.** Retained evidence bytes were not edited; operational account files are excluded.
