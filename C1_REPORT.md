# C1: Qwen-Image 2.1 BF16 versus FP8 quality, memory and latency (H200)

Item C1 of [vllm-project/vllm-omni#8586](https://github.com/vllm-project/vllm-omni/issues/8586), following the structure and baseline of [A0](https://github.com/luca-888/qwen-image-21-bench/blob/cf99e73c8fa1603718c47405730d94c9fbfe6e16/A0_REPORT.md).

**C1 requirement:**

> H200 BF16 versus FP8 quality, memory, and latency; extend only after the first comparison

The first comparison is **BF16 versus DiT-only FP8 on one H200**, with the same prompts, seeds, reference inputs and inference settings. It must report paired image quality and failures, peak memory, and latency before extending the experiment. Completing that comparison does not mean FP8 quality has been accepted.

The Text Encoder FP8, DiT + Text Encoder FP8, and DiT FP8 with BF16 blocks experiments followed the initial comparison as **extensions**. All tables below use fresh BF16/DiT-only controls and extension measurements from the same Phase 2 H200; earlier Phase 1 timings are not pooled into them. Frozen [configurations](reproduce/scripts/configs/arms.json) and [quality cases](reproduce/scripts/quality_manifest.json) define the workload.

## Setup

| Item | Configuration |
|---|---|
| Code | vLLM-Omni `3dc35694b3d458fc1461a8fe71832ac12e8c8cea`, clean checkout; exact container digest in the [reproduction guide](reproduce/README.md) |
| Packages | vLLM 0.31.0, torch 2.13.0+cu130, Transformers 5.14.1, Diffusers 0.40.0, Triton 3.7.1; Python 3.12.3 |
| Model | `Qwen/Qwen-Image-2.1@d26bb61231c349cf6b7896fa83353113880e1ba3` |
| Hardware | 1× NVIDIA H200 SXM (Runpod), driver 595.91.07 |
| Workload | 1024×1024, 50 steps, seed 42 for timing/profiling, `true_cfg_scale=1.0`, batch/concurrency 1, eager; no CUDA graphs, CPU offload or VAE tiling |
| Attention / cache | Actual BF16 FA3 SM90 execution; native BF16 prefix KV reuse retained, no prefix-KV FP8 |
| Quantization | Online E4M3 weights and dynamic per-token activation quantization; worker audits and traces check actual execution |
| T2I prompt | `A ceramic teapot on a wooden table` |
| Edit | One [reference image](figures/reference.png), A0's `qwen_bear.png` (514×556); `Let this mascot dance under the moon, surrounded by floating stars` |
| Timing budget | Per configuration/task: 1 warmup + 1 feasibility + 10 measured requests in one process; torch profiler disabled, synchronized stock stage timing enabled |
| Separate diagnostics | Memory: 1 warmup + 1 measured request. Profiler: 1 warmup + one full 50-step capture for T2I and editing per configuration |

| Scope | Configuration | DiT block Linears | Text Encoder Linears |
|---|---|---|---|
| Primary comparison | BF16 | BF16 | BF16 |
| Primary comparison | DiT FP8 | 160 FP8 | BF16 |
| Extension | Text Encoder FP8 | BF16 | 252 FP8 |
| Extension | DiT + Text Encoder FP8 | 160 FP8 | 252 FP8 |
| Extension | DiT FP8 with BF16 blocks | 140 FP8; 20 BF16 in blocks 0, 1, 30, 31 | BF16 |

Text Encoder FP8 quantizes only Linear modules in the language component. Vision, LM head, VAE, DiT boundary projections and attention are not quantized. DiT FP8 with BF16 blocks is a configuration fixed before this run, not a result of sensitivity search; compare it primarily with DiT-only FP8.

## 1. Latency: DiT-only FP8 reduces request time

Wall time per image, **mean ± sample SD**, measured n=10:

| Configuration | T2I | Edit (1 ref) |
|---|---:|---:|
| BF16 | 7.260 ± 0.016 s | 8.332 ± 0.074 s |
| DiT FP8 | 6.018 ± 0.013 s | 7.109 ± 0.081 s |
| Text Encoder FP8 | 7.284 ± 0.021 s | 8.376 ± 0.081 s |
| DiT + Text Encoder FP8 | 6.013 ± 0.009 s | 7.078 ± 0.062 s |
| DiT FP8 with BF16 blocks | 6.152 ± 0.009 s | 7.243 ± 0.055 s |

The primary comparison gives **17.1% lower T2I latency and 14.7% lower editing latency**. All 150 measured requests across offline T2I/editing and serving completed without recorded inference errors.

Measurement boundaries:

- Offline wall time brackets `omni.generate(...)`, including sampling-parameter construction and model output processing. Initialization, warmup and PNG encoding/writing are excluded.
- The SD describes repeated fixed requests within one process, not cross-start or multi-prompt variance.
- Memory diagnostics and torch-profiler captures are separate from headline timing.

### Stock stage accounting

Means from the same ten measured requests, in milliseconds. These are synchronized stage timers, not CUDA-event sums.

| Stage | BF16 T2I | DiT FP8 T2I | BF16 edit | DiT FP8 edit |
|---|---:|---:|---:|---:|
| `text_encoder.forward` | 24.94 | 25.65 | 73.94 | 69.18 |
| `vae.encode` | — | — | 32.74 | 32.72 |
| `diffuse` (50 steps) | 7,054.32 | 5,806.47 | 7,970.15 | 6,701.30 |
| `vae.decode` | 112.51 | 110.26 | 112.16 | 109.51 |

Most of the measured reduction is in `diffuse`. These four stock stages do not cover every pre/post-processing or scheduling operation; their sum is not the full request time. [Stage statistics and measurement summaries](data/phase2/performance/PERFORMANCE.json).

### A0 cross-check and outputs

The standalone [A0 review](A0_REVIEW.md) opens with the cross-review differences against A0, including torch-profiler measurements.

A separate earlier BF16 cross-check retained A0's exact prompts, seed 42, model and code revisions:

| Offline wall time, n=2 | A0 | Cross-check | Difference |
|---|---:|---:|---:|
| T2I | 7.387 s | 7.379 s | −0.11% |
| Edit (1 ref) | 8.363 s | 8.462 s | +1.18% |

The author baseline and cross-check used different physical H200s and drivers (580.95.05 vs 595.91.07). This is a reproducibility check; the performance table for the five configurations above uses its own fresh controls. All 28 saved repeated cross-check outputs matched A0's decoded RGB pixels, including warmup and diagnostic outputs—not 28 independent prompts. [Cross-check records](data/a0-cross/comparison.json).

The following are **Phase 2 quality-run outputs**, using the same A0 prompts and seed 42. They are not labeled as timing-run images. Both BF16 anchors match the author's PNG files exactly ([verification](data/phase2/audits/A0_ALIGNMENT_RECHECK.json)).

| Task | BF16 | DiT-only FP8 |
|---|---|---|
| T2I, seed 42 | [![BF16 T2I](figures/t2i_teapot_s42/bf16.png)](figures/t2i_teapot_s42/bf16.png) | [![DiT FP8 T2I](figures/t2i_teapot_s42/dit_fp8.png)](figures/t2i_teapot_s42/dit_fp8.png) |
| Edit, seed 42; [reference](figures/reference.png) | [![BF16 edit](figures/edit_bear_s42/bf16.png)](figures/edit_bear_s42/bf16.png) | [![DiT FP8 edit](figures/edit_bear_s42/dit_fp8.png)](figures/edit_bear_s42/dit_fp8.png) |

## 2. Serving path (T2I)

Sequential `/v1/images/generations` requests, `response_format=b64_json`, concurrency 1, n=10. Mean ± sample SD:

| Configuration | Client total | Engine generation |
|---|---:|---:|
| BF16 | 7.338 ± 0.025 s | 7.240 ± 0.021 s |
| DiT FP8 | 6.103 ± 0.018 s | 6.006 ± 0.020 s |
| Text Encoder FP8 | 7.372 ± 0.012 s | 7.276 ± 0.010 s |
| DiT + Text Encoder FP8 | 6.131 ± 0.014 s | 6.033 ± 0.012 s |
| DiT FP8 with BF16 blocks | 6.258 ± 0.019 s | 6.161 ± 0.015 s |

Client timing ends after the response body is received; subsequent client JSON/base64/PNG decoding is separate. Client minus engine time includes API handling, response encoding, hand-off and transfer, so it is not a measurement of encoding alone. [Serving measurements](data/phase2/performance/PERFORMANCE.json).

## 3. Where the time goes inside denoising (torch-profiler traces)

Ten separate captures cover all 50 steps: T2I and editing for each configuration, after warmup, with shapes, stacks and memory collection. Runtime audits and correlated launches verify actual CUTLASS SM90 E4M3 execution.

| Configuration | DiT FP8 GEMMs / request | Text Encoder FP8 GEMMs / request |
|---|---:|---:|
| BF16 | 0 | 0 |
| DiT FP8 | 8,000 | 0 |
| Text Encoder FP8 | 0 | 252 |
| DiT + Text Encoder FP8 | 8,000 | 252 |
| DiT FP8 with BF16 blocks | 7,000 | 0 |

Counts agree for T2I and editing. Text Encoder FP8 executes once per request; DiT FP8 executes on every denoising step. DiT attention remains BF16 FA3.

Selected denoising costs in the **DiT-only FP8** traces, summed across 50 steps:

| GPU-event category | T2I | Edit (1 ref) |
|---|---:|---:|
| FP8 GEMM | 1,940.62 ms | 1,983.43 ms |
| Activation quantization | 599.67 ms | 614.55 ms |
| Quantization share of denoising GPU-event time | 11.0% | 9.9% |

Activation quantization remains a measurable cost alongside FP8 GEMMs. Event-duration sums can double-count overlapping streams and exclude host gaps; they are not E2E latency, GPU utilization or an independent speedup estimate.

The export-only adapter changes post-capture summary aggregation, not trace collection or model inference. Full raw traces and memory snapshots remain local; [profile summaries](data/phase2/profiles/PROFILE_ANALYSIS.md), [event attribution](data/phase2/profiles/profile_comparison.json) and [independent audit](data/phase2/audits/ACTUAL_PROFILE_REVIEW.json) are public.

## 4. Memory

Binary units: 1 GiB = 1024³ bytes. Each separate diagnostic uses one warmup and one measured request; values below are the **measured request's worker peak counters**.

| Configuration | T2I allocated | T2I reserved | Edit allocated | Edit reserved |
|---|---:|---:|---:|---:|
| BF16 | 36.86 GiB | 38.12 GiB | 36.87 GiB | 39.28 GiB |
| DiT FP8 | 30.36 GiB | 31.60 GiB | 30.37 GiB | 32.77 GiB |
| Text Encoder FP8 | 30.39 GiB | 31.66 GiB | 30.40 GiB | 32.83 GiB |
| DiT + Text Encoder FP8 | 23.89 GiB | 25.14 GiB | 23.90 GiB | 26.31 GiB |
| DiT FP8 with BF16 blocks | 31.17 GiB | 32.43 GiB | 31.18 GiB | 33.60 GiB |

DiT-only FP8 saves **6.50 GiB allocated** versus BF16. DiT + Text Encoder FP8 saves **12.97 GiB**. DiT FP8 with BF16 blocks adds **0.81 GiB** versus DiT-only FP8. Allocated/reserved are allocator counters, not sampled whole-device usage. [Memory records](data/phase2/performance/PERFORMANCE.json).

## 5. Paired quality and failures

Each candidate is paired with BF16 using identical prompts, seeds and reference inputs: 12 legacy cases (four prompts × seeds 42/43/44), plus 12 held-out cases frozen before Phase 2 results (four prompts × seeds 45/46/47). All 120 scored outputs completed without recorded inference errors; none were exactly black.

Mean per-image **PSNR (dB) / SSIM**, full-resolution decoded RGB, n=12 per column:

| Candidate vs BF16 | Legacy | Held-out |
|---|---:|---:|
| DiT FP8 | 23.86 / 0.9071 | 26.96 / 0.9644 |
| Text Encoder FP8 | 24.54 / 0.9028 | 27.67 / 0.9609 |
| DiT + Text Encoder FP8 | 22.69 / 0.8759 | 24.14 / 0.9436 |
| DiT FP8 with BF16 blocks | 26.69 / 0.9178 | 27.76 / 0.9676 |

PSNR uses `data_range=255`; SSIM uses Gaussian weights, sigma 1.5, window 11 and population covariance. No candidate was RGB-identical to BF16 on any of its 24 pairs. No numerical acceptance threshold has been agreed.

Selected text cases from the labeled **model-assisted review**:

| Configuration | English title, seed 43 | Chinese text, seed 44 |
|---|---|---|
| BF16 | Title present; an unrequested portrait already appears | Clear vertical order |
| DiT FP8 | **Entire requested title missing** | All characters present; **reading order ambiguous** |
| Text Encoder FP8 | Title present | Clear vertical order |
| DiT + Text Encoder FP8 | **Entire requested title missing** | Clear vertical order |
| DiT FP8 with BF16 blocks | Title present | Clear horizontal order; layout changed |

[Original text comparisons](GALLERY.md#english-text-regression--seed-43) show the requested `QWEN IMAGE 2.1` title and Chinese `星河书店` sign. No clear new target violation was observed in the 12 held-out pairs per candidate; existing BF16 errors were recorded separately. This is not independent human acceptance. Recovering one failure does not establish general quality preservation or identify the sensitive layers. [Metrics](data/phase2/quality) and [completed observations](data/phase2/visual-review).

## 6. Conclusion

1. **Primary BF16/DiT-only comparison:** FP8 reduces T2I/edit latency by 17.1%/14.7% and peak allocated memory by 6.50 GiB, but introduces a clear text-rendering failure. Quality acceptance remains pending.
2. **Extensions:** Text Encoder FP8 saves memory without a latency gain here. DiT + Text Encoder FP8 saves more memory with latency close to DiT-only, but shares the title failure.
3. DiT FP8 with BF16 blocks retains that title at a cost of 2.2% T2I / 1.9% edit latency and 0.81 GiB versus DiT-only. Independent cross-review and agreed quality tolerances are still needed.
4. These results cover this H200 configuration. Prefix-KV FP8, consumer GPUs, multi-GPU and concurrent workloads are not covered.

## Artifacts

- [Original image gallery](GALLERY.md): representative T2I, editing and text cases across all five configurations.
- [Compact raw data](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-H200-essential-data-20261011.zip): request/environment/configuration records, worker memory and quantization audits, quality metrics/reviews, profiler summaries and separate A0 cross-check metadata.
- [Data index](data/README.md), [member manifest](data/ARTIFACT_MANIFEST.json) and [SHA256](data/SHA256SUMS).
- [Reproduction](reproduce/README.md): exact executed scripts, software pins, inputs and analysis commands; [source provenance](PROVENANCE.md).

The release contains one compact data attachment. Full raw traces, memory snapshots and the complete image corpus remain local.
