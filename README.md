# Qwen-Image 2.1 — A0 review and C1 H200 experiments

Performance, paired image quality and torch-profiler evidence for the C1 FP8 work in [vLLM-Omni #8586](https://github.com/vllm-project/vllm-omni/issues/8586), following [luca-888's A0 baseline](https://github.com/luca-888/qwen-image-21-bench/blob/cf99e73c8fa1603718c47405730d94c9fbfe6e16/A0_REPORT.md).

**Requirement:** H200 BF16 versus FP8 quality, memory, and latency; extend only after the first comparison. BF16 versus DiT-only FP8 is the primary comparison; TE-only, joint and selective configurations are subsequent extensions.

**1× H200 SXM · 1024² · 50 steps · CFG 1 · eager · FA3 · batch/concurrency 1.** Five fresh arms: BF16, DiT FP8, language-encoder FP8, joint FP8, and selective DiT FP8 with BF16 blocks.

| Start here | Contents |
|---|---|
| [A0 independent review](A0_REVIEW.md) | BF16 reproduction in A0's format; latency, serving, profiler, memory and differences from A0 |
| [C1 report](C1_REPORT.md) | Setup, A0 cross-check, latency, memory, quality and profiler findings |
| [Original image gallery](GALLERY.md) | Same-input comparisons: T2I, editing, English and Chinese text |
| [Essential numerical records](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-H200-essential-data-20261011.zip) | Compact raw request/environment/configuration and worker-memory records; analysis and A0 cross-check metadata |
| [Data index](data/README.md) | Machine-readable results and evidence mapping |
| [Reproduction](reproduce/README.md) | Executed harness, configs, frozen prompts and analysis tools |
| [Issue comment](ISSUE_COMMENT.md) | Concise draft with public links |
| [Provenance](PROVENANCE.md) | Source revisions, attribution and artifact boundaries |

DiT-only FP8 reduced offline T2I/edit latency by **17.1% / 14.7%** and peak allocated memory by **6.50 GiB**. Joint FP8 saved **12.97 GiB**, but both lost the requested title in one text-rendering case. The selective configuration retained it with a small speed/memory cost. **Quality acceptance remains pending**; the review is model-assisted and the selected images are not a general quality pass.

The C1 report uses the **2026-10-11 Phase 2 five-arm cohort**. The [A0 review](A0_REVIEW.md) uses the separate **2026-10-10 BF16 cross-check**; the earlier Phase 1 pilot is not pooled into either report. No model weights are distributed.

Download the single compact [release asset](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-H200-essential-data-20261011.zip) and verify it from this repository checkout:

```sh
gh release download c1-h200-20261011 --repo Dmaner/qwen-image-21-bench \
  --pattern C1-H200-essential-data-20261011.zip --dir c1-data
(cd c1-data && sha256sum -c ../data/SHA256SUMS)
# macOS: replace sha256sum with shasum -a 256
```

See [ARTIFACT_MANIFEST.json](data/ARTIFACT_MANIFEST.json) for the included records and hashes. The repository retains the original-image gallery, all quality metrics, model-assisted reviews and profile summaries. **Full raw traces, memory pickles and the complete image corpus remain local.** Retained evidence bytes were not edited; operational account files are excluded.
