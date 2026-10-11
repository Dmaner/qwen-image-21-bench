# C1: Qwen-Image 2.1 FP8 performance and quality on H200

C1 update for [#8586](https://github.com/vllm-project/vllm-omni/issues/8586), following the [A0 report and pinned harness](https://github.com/luca-888/qwen-image-21-bench/blob/cf99e73c8fa1603718c47405730d94c9fbfe6e16/A0_REPORT.md).

**DiT-only FP8 reduced T2I latency by 17.1% and single-reference editing latency by 14.7%, with 6.50 GiB less peak allocated memory. Joint FP8 saved more memory. Both lost a requested title in a text-rendering case; selective DiT FP8 with BF16 blocks retained it, at a small latency cost versus DiT-only FP8. Overall FP8 quality acceptance remains pending.**

## Setup and A0 cross-check

| Item | Configuration |
|---|---|
| Hardware | 1× NVIDIA H200 SXM, Runpod; driver 595.91.07 |
| Code / model | vLLM-Omni `3dc35694b3d458fc1461a8fe71832ac12e8c8cea`; `Qwen/Qwen-Image-2.1@d26bb61231c349cf6b7896fa83353113880e1ba3` |
| Stack | Python 3.12.3; vLLM 0.31.0; PyTorch 2.13.0+cu130; Triton 3.7.1; Transformers 5.14.1; Diffusers 0.40.0 |
| Workload | 1024×1024, 50 steps, true CFG 1, batch/concurrency 1, eager; FA3 SM90 |
| Other controls | No CPU offload, VAE tiling, or CUDA graphs; native BF16 prefix KV reuse retained |
| Quantization | Online E4M3 weights with dynamic per-token activation quantization; actual execution checked by worker audits and traces |

Performance and profiler runs retain A0's **seed 42**, its ceramic-teapot prompt, and its moonlight-dancing-mascot editing prompt with the same `qwen_bear.png` input. A separate BF16 cross-check reproduced A0 at 7.379 vs 7.387 s T2I and 8.462 vs 8.363 s editing (n=2); all 28 saved repeated outputs matched A0's decoded RGB pixels. The author baseline and this cross-check used different physical H200s and drivers (580.95.05 vs 595.91.07), so this is a reproducibility check, not a paired hardware benchmark. The two BF16 seed-42 anchors in the latest cohort also match the author's original PNG files exactly.

The tables below use **fresh measurements of all five arms on one H200**, not mixed results from separate phases. DiT FP8 quantizes 160 block Linear modules; TE FP8 quantizes 252 language Linear modules. Selective BF16 retains DiT blocks **0, 1, 30, 31** in BF16 (20 Linears), quantizing the other 140, with TE in BF16. Vision, LM head, VAE, DiT boundary projections and attention remain unquantized. The selective configuration was fixed before this run, not identified by a sensitivity search.

## 1. Latency and memory

Per arm/task: **1 warmup + 1 feasibility + 10 measured requests**. Values are mean ± sample SD in seconds, from fixed requests repeated within one process. They do not estimate cross-start or multi-prompt variance. Torch profiler was disabled for these measurements; built-in stage timing remained enabled.

| Arm | T2I offline (s) | Edit, 1 reference (s) | T2I serving client (s) |
|---|---:|---:|---:|
| BF16 | 7.260 ± 0.016 | 8.332 ± 0.074 | 7.338 ± 0.025 |
| DiT FP8 | 6.018 ± 0.013 | 7.109 ± 0.081 | 6.103 ± 0.018 |
| TE FP8 (language only) | 7.284 ± 0.021 | 8.376 ± 0.081 | 7.372 ± 0.012 |
| DiT + TE FP8 | 6.013 ± 0.009 | 7.078 ± 0.062 | 6.131 ± 0.014 |
| Selective BF16 | 6.152 ± 0.009 | 7.243 ± 0.055 | 6.258 ± 0.019 |

Offline timing brackets `omni.generate(...)`, including sampling-parameter construction and model output processing, but excludes initialization and PNG encoding/writing. Serving timing ends after the response body is received; client JSON/base64/PNG decoding is separate.

Peak memory comes from separate **1 warmup + 1 measured** runs, using the measured request's worker counters. Each cell is **allocated / reserved GiB**; it is not sampled whole-device memory.

| Arm | T2I peak (GiB) | Edit peak (GiB) |
|---|---:|---:|
| BF16 | 36.86 / 38.12 | 36.87 / 39.28 |
| DiT FP8 | 30.36 / 31.60 | 30.37 / 32.77 |
| TE FP8 (language only) | 30.39 / 31.66 | 30.40 / 32.83 |
| DiT + TE FP8 | 23.89 / 25.14 | 23.90 / 26.31 |
| Selective BF16 | 31.17 / 32.43 | 31.18 / 33.60 |

TE-only FP8 reduced memory without a latency gain in this workload. Joint FP8 saved **12.97 GiB allocated** versus BF16 with latency close to DiT-only. Selective BF16 cost **2.2% T2I / 1.9% edit latency** and **0.81 GiB allocated** versus DiT-only. All 150 measured performance requests completed without recorded inference errors.

## 2. Paired quality

We kept A0's inputs and added text-rendering, spatial, and editing cases: **12 legacy cases** (four prompts × seeds 42/43/44) and **12 cases frozen before Phase 2 results** (four prompts × seeds 45/46/47). Every candidate uses the same prompt, seed and reference image as its BF16 pair. All 120 quality outputs completed without recorded inference errors; none were exactly black.

Mean per-image **PSNR (dB) / SSIM**, using full-resolution decoded RGB; each column has n=12:

| Candidate vs BF16 | Legacy | Held-out |
|---|---:|---:|
| DiT FP8 | 23.86 / 0.9071 | 26.96 / 0.9644 |
| TE FP8 (language only) | 24.54 / 0.9028 | 27.67 / 0.9609 |
| DiT + TE FP8 | 22.69 / 0.8759 | 24.14 / 0.9436 |
| Selective BF16 | 26.69 / 0.9178 | 27.76 / 0.9676 |

PSNR uses `data_range=255`; SSIM uses Gaussian weights, sigma 1.5, window 11, and population covariance. No candidate was RGB-identical to BF16 on any of its 24 pairs. These metrics have **no agreed acceptance threshold**.

Model-assisted review found:

- **DiT FP8 and joint FP8:** the requested `QWEN IMAGE 2.1` title disappeared at English seed 43; it was present in BF16.
- **DiT FP8:** Chinese seed 44 retained the characters but had ambiguous reading order.
- **Selective BF16:** the English seed-43 title was retained, and Chinese seed 44 was readable in a changed horizontal layout. TE-only also retained these targets.
- No clear new target violation was observed in the 12 held-out pairs per candidate. Existing BF16 errors were recorded separately.

This is model-assisted review, not independent human acceptance. Recovering one failure does not establish general quality preservation or prove that the protected blocks are the sensitive layers.

## 3. Torch profiler verification

We captured **10 complete traces**: one T2I and one editing request per arm, each after warmup, with all 50 denoising steps and shape/stack/memory collection.

| Arm | Actual DiT FP8 GEMMs / request | Actual language-encoder FP8 GEMMs / request |
|---|---:|---:|
| BF16 | 0 | 0 |
| DiT FP8 | 8,000 | 0 |
| TE FP8 | 0 | 252 |
| DiT + TE FP8 | 8,000 | 252 |
| Selective BF16 | 7,000 | 0 |

Counts agree for T2I and editing. Runtime audits and correlated CUDA launches verify CUTLASS SM90 E4M3 execution; DiT attention remains **BF16 FA3**. TE FP8 executes once per request, while DiT FP8 executes on every denoising step.

DiT-only activation quantization accumulated **599.67 ms T2I / 614.55 ms edit**, about **11.0% / 9.9% of denoising GPU-event time**. This is a meaningful overhead alongside the FP8 GEMMs. These are GPU-event duration sums, not E2E latency: events may overlap and host gaps are excluded.

A diagnostic adapter changed only post-capture summary aggregation; raw Chrome traces and memory snapshots were retained.

## Remaining work and artifacts

Agree on quality tolerances and independently cross-review the text failures and selective result with A0. Prefix-KV FP8, consumer GPUs, multi-GPU and concurrent workloads are **not covered** by this report.

Public evidence: [original comparison images](GALLERY.md), [machine-readable results and audit index](data/README.md), [executed harness/configs and reproduction guide](reproduce/README.md), and [SHA256 checksums](data/SHA256SUMS).

The [raw-data release](https://github.com/Dmaner/qwen-image-21-bench/releases/tag/c1-h200-20261011) contains the [Phase 2 request records, all 120 quality images and analysis](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-Phase2-20261011-results-public.zip), five profiler archives with ten raw traces/memory snapshots, and the [separate A0 cross-check evidence](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-A0-cross-20261010-evidence.zip). [Provenance and packaging boundaries](PROVENANCE.md) distinguish recorded originals, upstream references and omitted operational files.
