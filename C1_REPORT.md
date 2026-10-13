# C1: Qwen-Image 2.1 BF16 versus FP8 quality, memory and latency (H200)

For [vllm-project/vllm-omni#8586](https://github.com/vllm-project/vllm-omni/issues/8586). This report uses the 2026-10-13 supplement: seven configurations on the same H200, including the [#8099](https://github.com/vllm-project/vllm-omni/pull/8099) `img_mlp` BF16 configuration, FP8 prefix storage and additional editing inputs. [Previous 2026-10-11 results](reports/C1_20261011.md).

## Setup

| Item | Configuration |
|---|---|
| Code | vLLM-Omni `3dc35694b3d458fc1461a8fe71832ac12e8c8cea`, clean checkout |
| Packages | vLLM 0.31.0, torch 2.13.0+cu130, Transformers 5.14.1, Diffusers 0.40.0, Triton 3.7.1; Python 3.12.3 |
| Model | `Qwen/Qwen-Image-2.1@d26bb61231c349cf6b7896fa83353113880e1ba3` |
| Hardware | 1× NVIDIA H200 (Runpod), driver 580.95.05 |
| Workload | 1024×1024, 50 steps, `true_cfg_scale=1.0`, batch/concurrency 1; timing seed 42 |
| Execution | Eager, FlashAttention-3 (confirmed in traces); no CPU offload, VAE tiling or CUDA Graph |
| Timing | Per configuration/workload: 1 warmup + 1 feasibility + 10 measured requests; mean ± sample SD |
| Memory | Worker peak allocated/reserved GiB, one measured request after one warmup |
| Quality | 36 paired cases per candidate: 18 T2I and 18 edits; four distinct input images, including a two-reference composition |

| Configuration | DiT | Text Encoder | Prefix KV |
|---|---|---|---|
| BF16 | BF16 | BF16 | BF16 |
| DiT FP8 | 160 FP8 Linear layers | BF16 | BF16 |
| DiT FP8 + `img_mlp` BF16 | 64 FP8 + 96 BF16 Linear layers | BF16 | BF16 |
| Text Encoder FP8 | BF16 | 252 FP8 Linear layers | BF16 |
| DiT + Text Encoder FP8 | 160 FP8 Linear layers | 252 FP8 Linear layers | BF16 |
| Prefix V FP8 | BF16 | BF16 | K BF16, V FP8 |
| Prefix K+V FP8 | BF16 | BF16 | K/V FP8 |

Online linear quantization uses E4M3 per-tensor weights and dynamic per-token activations, as recorded by the H200 worker audit. The selective configuration preserves all `img_mlp` Linear layers. VAE, DiT boundary projections and text-encoder vision layers/LM head remain BF16; attention inputs are BF16. [Experiment design](EXPERIMENT_DESIGN.md) · [Configurations and commands](reproduce/supplement/README.md).

## 1. Latency

Offline wall time per image; initialization and PNG saving are excluded. T2I uses the ceramic-teapot prompt; one-reference editing uses the original bear prompt.

| Configuration | T2I | Δ vs BF16 | Edit (1 ref) | Δ vs BF16 |
|---|---:|---:|---:|---:|
| BF16 | 7.342 ± 0.016 s | — | 8.477 ± 0.041 s | — |
| DiT FP8 | 6.109 ± 0.005 s | −16.8% | 7.158 ± 0.052 s | −15.6% |
| DiT FP8 + `img_mlp` BF16 | 6.951 ± 0.015 s | −5.3% | 7.998 ± 0.036 s | −5.7% |
| Text Encoder FP8 | 7.455 ± 0.031 s | +1.5% | 8.512 ± 0.071 s | +0.4% |
| DiT + Text Encoder FP8 | 6.181 ± 0.008 s | −15.8% | 7.321 ± 0.267 s | −13.6% |
| Prefix V FP8 | 7.973 ± 0.568 s | +8.6% | 8.679 ± 0.088 s | +2.4% |
| Prefix K+V FP8 | 7.372 ± 0.016 s | +0.4% | 8.788 ± 0.035 s | +3.7% |

The two-reference task combines the astronaut and coffee inputs.

| Configuration | Edit (2 refs) | Δ vs BF16 |
|---|---:|---:|
| BF16 | 9.618 ± 0.064 s | — |
| Prefix V FP8 | 9.966 ± 0.038 s | +3.6% |
| Prefix K+V FP8 | 10.328 ± 0.030 s | +7.4% |

All 170 measured requests completed without recorded inference errors or OOMs.
Prefix V FP8 T2I and joint-FP8 editing showed elevated wall-time variance. A separate [T2I repeat](data/supplement-20261013/recheck/REPORT.md) measured **7.329 ± 0.006 s** for BF16, **7.341 ± 0.005 s (+0.17%)** for V FP8 and **7.340 ± 0.012 s (+0.15%)** for K+V FP8. The large V-only fluctuation did not recur; the primary table retains all original requests.

## 2. Memory

| Task | Configuration | Peak allocated | Δ vs BF16 | Peak reserved | Δ vs BF16 |
|---|---|---:|---:|---:|---:|
| T2I | BF16 | 36.86 GiB | — | 38.12 GiB | — |
| T2I | DiT FP8 | 30.36 GiB | −17.6% | 31.60 GiB | −17.1% |
| T2I | DiT FP8 + `img_mlp` BF16 | 34.86 GiB | −5.4% | 36.12 GiB | −5.2% |
| T2I | Text Encoder FP8 | 30.39 GiB | −17.5% | 31.66 GiB | −16.9% |
| T2I | DiT + Text Encoder FP8 | 23.89 GiB | −35.2% | 25.14 GiB | −34.0% |
| T2I | Prefix V FP8 | 36.86 GiB | 0.0% | 38.12 GiB | 0.0% |
| T2I | Prefix K+V FP8 | 36.86 GiB | 0.0% | 38.12 GiB | 0.0% |
| Edit (1 ref) | BF16 | 36.87 GiB | — | 39.28 GiB | — |
| Edit (1 ref) | DiT FP8 | 30.37 GiB | −17.6% | 32.77 GiB | −16.6% |
| Edit (1 ref) | DiT FP8 + `img_mlp` BF16 | 34.87 GiB | −5.4% | 37.28 GiB | −5.1% |
| Edit (1 ref) | Text Encoder FP8 | 30.40 GiB | −17.5% | 32.83 GiB | −16.4% |
| Edit (1 ref) | DiT + Text Encoder FP8 | 23.90 GiB | −35.2% | 26.31 GiB | −33.0% |
| Edit (1 ref) | Prefix V FP8 | 36.87 GiB | 0.0% | 38.73 GiB | −1.4% |
| Edit (1 ref) | Prefix K+V FP8 | 36.87 GiB | 0.0% | 38.75 GiB | −1.3% |
| Edit (2 refs) | BF16 | 36.88 GiB | — | 42.05 GiB | — |
| Edit (2 refs) | Prefix V FP8 | 36.87 GiB | 0.0% | 40.35 GiB | −4.0% |
| Edit (2 refs) | Prefix K+V FP8 | 36.87 GiB | 0.0% | 38.93 GiB | −7.4% |

Persistent prefix-cache storage, including quantization scales:

| Task | Prefix tokens | BF16 | V FP8 | Δ vs BF16 | K+V FP8 | Δ vs BF16 |
|---|---:|---:|---:|---:|---:|---:|
| T2I | 16 | 8.00 MiB | 6.06 MiB | −24.2% | 4.12 MiB | −48.4% |
| Edit (1 ref) | 4,118 | 2059.00 MiB | 1560.34 MiB | −24.2% | 1061.67 MiB | −48.4% |
| Edit (2 refs) | 8,272 | 4136.00 MiB | 3134.31 MiB | −24.2% | 2132.62 MiB | −48.4% |

[Per-request measurements and cache records](data/supplement-20261013/analysis.json).

## 3. Image Quality

Each candidate uses identical prompts, seeds and ordered reference images as BF16. The 36 cases include the original 24 cases plus person-background, product-color, sign-text and two-reference edits (three seeds each).

| Configuration vs BF16 | Mean PSNR ↑ | Mean SSIM ↑ | Mean LPIPS ↓ | LPIPS ≤ 0.15 |
|---|---:|---:|---:|---:|
| DiT FP8 | 26.69 dB | 0.9332 | 0.0629 | 32/36 |
| DiT FP8 + `img_mlp` BF16 | 28.97 dB | 0.9354 | 0.0546 | 33/36 |
| Text Encoder FP8 | 29.77 dB | 0.9401 | 0.0536 | 32/36 |
| DiT + Text Encoder FP8 | 26.12 dB | 0.9207 | 0.0719 | 32/36 |
| Prefix V FP8 | 43.73 dB | 0.9934 | 0.0037 | 36/36 |
| Prefix K+V FP8 | 35.06 dB | 0.9804 | 0.0151 | 36/36 |

LPIPS ≤ 0.15 is the predeclared working screen; maintainer acceptance and human assessment remain pending.
Model-assisted inspection found no clear new task failure across the 12 additional editing cases per candidate. The existing English-title failure remains in DiT FP8, joint FP8 and the `img_mlp` BF16 configuration.
[Per-case metrics](data/supplement-20261013/quality_pairs.csv) · [Category results and failures](data/supplement-20261013/quality_gate.csv).

### Separate CFG=4 prefix comparison

One teapot output per configuration: seed 42, 1024², 50 steps, eager, with an explicit empty negative prompt. These two quality pairs are excluded from the 36-case table; runtime cache audits above use CFG=1.

| Configuration vs BF16 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | LPIPS ≤ 0.15 |
|---|---:|---:|---:|---:|
| Prefix V FP8 | 41.26 dB | 0.9933 | 0.0059 | 1/1 |
| Prefix K+V FP8 | 30.11 dB | 0.9679 | 0.0406 | 1/1 |

### Selected examples

**English title, seed 43.** Prompt: `A poster with the exact text "QWEN IMAGE 2.1", clean typography on a plain background.` BF16 retains the title but adds an unrequested portrait. Both DiT FP8 and the `img_mlp` BF16 configuration omit the title.

| BF16 | DiT FP8 | DiT FP8 + `img_mlp` BF16 |
|---|---|---|
| ![BF16 English title, seed 43](figures/supplement-20261013/t2i_english_s43_bf16.png) | ![DiT FP8 missing English title, seed 43](figures/supplement-20261013/t2i_english_s43_dit_fp8.png) | ![MLP BF16 missing English title, seed 43](figures/supplement-20261013/t2i_english_s43_mlp_bf16.png) |

**Two-reference edit, seed 43.** Combine the [person](reproduce/supplement/inputs/astronaut.png) and [coffee set](reproduce/supplement/inputs/coffee.png) in a gray-background studio scene. This is the largest LPIPS difference among the prefix-storage two-reference cases: V-only **0.01571**, K+V **0.03701** versus BF16.

| BF16 | Prefix V FP8 | Prefix K+V FP8 |
|---|---|---|
| ![BF16 two-reference edit](figures/supplement-20261013/prefix_two_ref_s43_bf16.png) | ![Prefix V FP8 two-reference edit](figures/supplement-20261013/prefix_two_ref_s43_prefix_fp8_v.png) | ![Prefix K+V FP8 two-reference edit](figures/supplement-20261013/prefix_two_ref_s43_prefix_fp8.png) |

Model-assisted inspection: both prefix configurations retain the person, brown cup, white interior, coffee, saucer and spoon on the table. Fine face and patch details vary; no new clear edit failure is visible in this example.

[More examples: person, product color, sign editing and Chinese text](SUPPLEMENT_GALLERY.md).

## 4. Conclusion

- DiT FP8: T2I/edit latency −16.8%/−15.6% and peak allocated memory −17.6%/−17.6% versus BF16; 32/36 pairs pass the working LPIPS screen.
- DiT FP8 + `img_mlp` BF16: T2I/edit latency −5.3%/−5.7% and peak allocated memory −5.4%/−5.4% versus BF16; 33/36 pairs pass the working LPIPS screen. Preserving `img_mlp` does not fix the missing English title at seed 43.
- Prefix V FP8: 36/36 pairs pass the working LPIPS screen. On the two-reference task, latency changes by +3.6%, peak reserved memory by −4.0%, and persistent prefix storage by −24.2%; peak allocated memory is essentially unchanged.
- Prefix K+V FP8: 36/36 pairs pass the working LPIPS screen. On the two-reference task, latency changes by +7.4%, peak reserved memory by −7.4%, and persistent prefix storage by −48.4%; peak allocated memory is essentially unchanged.

Preserving `img_mlp` improves mean PSNR by 2.28 dB versus DiT FP8 on this set, but does not eliminate its text-rendering failures.

## Artifacts

- [Experiment design](EXPERIMENT_DESIGN.md) and [reproduction commands](reproduce/supplement/README.md).
- [Measurements, paired quality and runtime-cache evidence](data/supplement-20261013/analysis.json).
- [Per-image metrics](data/supplement-20261013/quality_pairs.csv), [gate results](data/supplement-20261013/quality_gate.csv) and [visual observations](data/supplement-20261013/visual_review.csv).
- [Historical 2026-10-11 report](reports/C1_20261011.md).
