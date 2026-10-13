# C1 supplement results

## Latency

| Case | Configuration | n | Mean ± SD (s) | Δ vs BF16 |
|---|---|---:|---:|---:|
| edit_bear_s42 | bf16 | 10/10 | 8.477 ± 0.041 | — |
| prefix_two_ref_s42 | bf16 | 10/10 | 9.618 ± 0.064 | — |
| t2i_teapot_s42 | bf16 | 10/10 | 7.342 ± 0.016 | — |
| edit_bear_s42 | dit_fp8 | 10/10 | 7.158 ± 0.052 | -15.6% |
| t2i_teapot_s42 | dit_fp8 | 10/10 | 6.109 ± 0.005 | -16.8% |
| edit_bear_s42 | joint_fp8 | 10/10 | 7.321 ± 0.267 | -13.6% |
| t2i_teapot_s42 | joint_fp8 | 10/10 | 6.181 ± 0.008 | -15.8% |
| edit_bear_s42 | mlp_bf16 | 10/10 | 7.998 ± 0.036 | -5.7% |
| t2i_teapot_s42 | mlp_bf16 | 10/10 | 6.951 ± 0.015 | -5.3% |
| edit_bear_s42 | prefix_fp8 | 10/10 | 8.788 ± 0.035 | +3.7% |
| prefix_two_ref_s42 | prefix_fp8 | 10/10 | 10.328 ± 0.030 | +7.4% |
| t2i_teapot_s42 | prefix_fp8 | 10/10 | 7.372 ± 0.016 | +0.4% |
| edit_bear_s42 | prefix_fp8_v | 10/10 | 8.679 ± 0.088 | +2.4% |
| prefix_two_ref_s42 | prefix_fp8_v | 10/10 | 9.966 ± 0.038 | +3.6% |
| t2i_teapot_s42 | prefix_fp8_v | 10/10 | 7.973 ± 0.568 | +8.6% |
| edit_bear_s42 | te_fp8 | 10/10 | 8.512 ± 0.071 | +0.4% |
| t2i_teapot_s42 | te_fp8 | 10/10 | 7.455 ± 0.031 | +1.5% |

## Memory

| Case | Configuration | Allocated (GiB) | Δ vs BF16 | Reserved (GiB) | Δ vs BF16 |
|---|---|---:|---:|---:|---:|
| edit_bear_s42 | bf16 | 36.87 | — | 39.28 | — |
| prefix_two_ref_s42 | bf16 | 36.88 | — | 42.05 | — |
| t2i_teapot_s42 | bf16 | 36.86 | — | 38.12 | — |
| edit_bear_s42 | dit_fp8 | 30.37 | -17.6% | 32.77 | -16.6% |
| t2i_teapot_s42 | dit_fp8 | 30.36 | -17.6% | 31.60 | -17.1% |
| edit_bear_s42 | joint_fp8 | 23.90 | -35.2% | 26.31 | -33.0% |
| t2i_teapot_s42 | joint_fp8 | 23.89 | -35.2% | 25.14 | -34.0% |
| edit_bear_s42 | mlp_bf16 | 34.87 | -5.4% | 37.28 | -5.1% |
| t2i_teapot_s42 | mlp_bf16 | 34.86 | -5.4% | 36.12 | -5.2% |
| edit_bear_s42 | prefix_fp8 | 36.87 | 0.0% | 38.75 | -1.3% |
| prefix_two_ref_s42 | prefix_fp8 | 36.87 | 0.0% | 38.93 | -7.4% |
| t2i_teapot_s42 | prefix_fp8 | 36.86 | 0.0% | 38.12 | 0.0% |
| edit_bear_s42 | prefix_fp8_v | 36.87 | 0.0% | 38.73 | -1.4% |
| prefix_two_ref_s42 | prefix_fp8_v | 36.87 | 0.0% | 40.35 | -4.0% |
| t2i_teapot_s42 | prefix_fp8_v | 36.86 | 0.0% | 38.12 | 0.0% |
| edit_bear_s42 | te_fp8 | 30.40 | -17.5% | 32.83 | -16.4% |
| t2i_teapot_s42 | te_fp8 | 30.39 | -17.5% | 31.66 | -16.9% |

## Prefix storage

| Case | Configuration | Branch | Prefix tokens | K / V dtype | KV + scale MiB | Δ vs BF16 |
|---|---|---|---:|---|---:|---:|
| edit_bear_s42 | bf16 | cond | 4118 | torch.bfloat16 / torch.bfloat16 | 2059.00 | — |
| prefix_two_ref_s42 | bf16 | cond | 8272 | torch.bfloat16 / torch.bfloat16 | 4136.00 | — |
| t2i_teapot_s42 | bf16 | cond | 16 | torch.bfloat16 / torch.bfloat16 | 8.00 | — |
| edit_bear_s42 | dit_fp8 | cond | 4118 | torch.bfloat16 / torch.bfloat16 | 2059.00 | 0.0% |
| edit_bear_s42 | joint_fp8 | cond | 4118 | torch.bfloat16 / torch.bfloat16 | 2059.00 | 0.0% |
| edit_bear_s42 | mlp_bf16 | cond | 4118 | torch.bfloat16 / torch.bfloat16 | 2059.00 | 0.0% |
| edit_bear_s42 | prefix_fp8 | cond | 4118 | torch.float8_e4m3fn / torch.float8_e4m3fn | 1061.67 | -48.4% |
| prefix_two_ref_s42 | prefix_fp8 | cond | 8272 | torch.float8_e4m3fn / torch.float8_e4m3fn | 2132.62 | -48.4% |
| t2i_teapot_s42 | prefix_fp8 | cond | 16 | torch.float8_e4m3fn / torch.float8_e4m3fn | 4.12 | -48.4% |
| edit_bear_s42 | prefix_fp8_v | cond | 4118 | torch.bfloat16 / torch.float8_e4m3fn | 1560.34 | -24.2% |
| prefix_two_ref_s42 | prefix_fp8_v | cond | 8272 | torch.bfloat16 / torch.float8_e4m3fn | 3134.31 | -24.2% |
| t2i_teapot_s42 | prefix_fp8_v | cond | 16 | torch.bfloat16 / torch.float8_e4m3fn | 6.06 | -24.2% |
| edit_bear_s42 | te_fp8 | cond | 4118 | torch.bfloat16 / torch.bfloat16 | 2059.00 | 0.0% |

## Image Quality: effective_manifest

| Category | Configuration | PSNR (dB) | SSIM | LPIPS | Pass / planned | Unassessed |
|---|---|---:|---:|---:|---:|---:|
| Chinese text | dit_fp8 | 23.48 | 0.9071 | 0.1081 | 5/6 | 0 |
| Chinese text | joint_fp8 | 20.33 | 0.8217 | 0.1905 | 3/6 | 0 |
| Chinese text | mlp_bf16 | 22.71 | 0.8765 | 0.1514 | 4/6 | 0 |
| Chinese text | prefix_fp8 | 29.72 | 0.9664 | 0.0376 | 6/6 | 0 |
| Chinese text | prefix_fp8_v | 36.49 | 0.9887 | 0.0112 | 6/6 | 0 |
| Chinese text | te_fp8 | 21.46 | 0.8532 | 0.1660 | 3/6 | 0 |
| Edit: bear | dit_fp8 | 25.79 | 0.9506 | 0.0519 | 6/6 | 0 |
| Edit: bear | joint_fp8 | 23.04 | 0.9359 | 0.0591 | 6/6 | 0 |
| Edit: bear | mlp_bf16 | 28.58 | 0.9557 | 0.0490 | 6/6 | 0 |
| Edit: bear | prefix_fp8 | 32.46 | 0.9834 | 0.0149 | 6/6 | 0 |
| Edit: bear | prefix_fp8_v | 46.37 | 0.9967 | 0.0020 | 6/6 | 0 |
| Edit: bear | te_fp8 | 27.23 | 0.9714 | 0.0335 | 6/6 | 0 |
| Edit: multi-reference | dit_fp8 | 23.99 | 0.8443 | 0.0958 | 2/3 | 0 |
| Edit: multi-reference | joint_fp8 | 27.02 | 0.9000 | 0.0439 | 3/3 | 0 |
| Edit: multi-reference | mlp_bf16 | 26.90 | 0.8999 | 0.0328 | 3/3 | 0 |
| Edit: multi-reference | prefix_fp8 | 32.69 | 0.9585 | 0.0157 | 3/3 | 0 |
| Edit: multi-reference | prefix_fp8_v | 39.79 | 0.9806 | 0.0059 | 3/3 | 0 |
| Edit: multi-reference | te_fp8 | 27.00 | 0.9036 | 0.0373 | 3/3 | 0 |
| Edit: person | dit_fp8 | 36.36 | 0.9911 | 0.0048 | 3/3 | 0 |
| Edit: person | joint_fp8 | 41.06 | 0.9938 | 0.0025 | 3/3 | 0 |
| Edit: person | mlp_bf16 | 40.15 | 0.9945 | 0.0023 | 3/3 | 0 |
| Edit: person | prefix_fp8 | 46.80 | 0.9966 | 0.0010 | 3/3 | 0 |
| Edit: person | prefix_fp8_v | 51.59 | 0.9982 | 0.0004 | 3/3 | 0 |
| Edit: person | te_fp8 | 43.45 | 0.9968 | 0.0012 | 3/3 | 0 |
| Edit: product | dit_fp8 | 29.41 | 0.9032 | 0.0329 | 3/3 | 0 |
| Edit: product | joint_fp8 | 29.05 | 0.9012 | 0.0351 | 3/3 | 0 |
| Edit: product | mlp_bf16 | 24.52 | 0.8428 | 0.0532 | 3/3 | 0 |
| Edit: product | prefix_fp8 | 38.81 | 0.9870 | 0.0041 | 3/3 | 0 |
| Edit: product | prefix_fp8_v | 46.98 | 0.9963 | 0.0010 | 3/3 | 0 |
| Edit: product | te_fp8 | 35.28 | 0.9288 | 0.0219 | 3/3 | 0 |
| Edit: text | dit_fp8 | 27.28 | 0.9741 | 0.0227 | 3/3 | 0 |
| Edit: text | joint_fp8 | 28.96 | 0.9754 | 0.0202 | 3/3 | 0 |
| Edit: text | mlp_bf16 | 39.51 | 0.9925 | 0.0031 | 3/3 | 0 |
| Edit: text | prefix_fp8 | 38.74 | 0.9933 | 0.0027 | 3/3 | 0 |
| Edit: text | prefix_fp8_v | 49.71 | 0.9983 | 0.0003 | 3/3 | 0 |
| Edit: text | te_fp8 | 42.70 | 0.9971 | 0.0008 | 3/3 | 0 |
| English text | dit_fp8 | 21.34 | 0.9090 | 0.1148 | 4/6 | 0 |
| English text | joint_fp8 | 22.40 | 0.9227 | 0.0784 | 5/6 | 0 |
| English text | mlp_bf16 | 24.12 | 0.9343 | 0.0647 | 5/6 | 0 |
| English text | prefix_fp8 | 30.55 | 0.9728 | 0.0202 | 6/6 | 0 |
| English text | prefix_fp8_v | 37.78 | 0.9917 | 0.0042 | 6/6 | 0 |
| English text | te_fp8 | 22.79 | 0.9269 | 0.0631 | 5/6 | 0 |
| Ordinary T2I | dit_fp8 | 31.03 | 0.9763 | 0.0243 | 6/6 | 0 |
| Ordinary T2I | joint_fp8 | 27.90 | 0.9587 | 0.0523 | 6/6 | 0 |
| Ordinary T2I | mlp_bf16 | 32.88 | 0.9809 | 0.0169 | 6/6 | 0 |
| Ordinary T2I | prefix_fp8 | 39.10 | 0.9920 | 0.0061 | 6/6 | 0 |
| Ordinary T2I | prefix_fp8_v | 47.70 | 0.9969 | 0.0014 | 6/6 | 0 |
| Ordinary T2I | te_fp8 | 32.93 | 0.9759 | 0.0285 | 6/6 | 0 |
| all | dit_fp8 | 26.69 | 0.9332 | 0.0629 | 32/36 | 0 |
| all | joint_fp8 | 26.12 | 0.9207 | 0.0719 | 32/36 | 0 |
| all | mlp_bf16 | 28.97 | 0.9354 | 0.0546 | 33/36 | 0 |
| all | prefix_fp8 | 35.06 | 0.9804 | 0.0151 | 36/36 | 0 |
| all | prefix_fp8_v | 43.73 | 0.9934 | 0.0037 | 36/36 | 0 |
| all | te_fp8 | 29.77 | 0.9401 | 0.0536 | 32/36 | 0 |

## Image Quality: cfg4_manifest

| Category | Configuration | PSNR (dB) | SSIM | LPIPS | Pass / planned | Unassessed |
|---|---|---:|---:|---:|---:|---:|
| Ordinary T2I | prefix_fp8 | 30.11 | 0.9679 | 0.0406 | 1/1 | 0 |
| Ordinary T2I | prefix_fp8_v | 41.26 | 0.9933 | 0.0059 | 1/1 | 0 |
| all | prefix_fp8 | 30.11 | 0.9679 | 0.0406 | 1/1 | 0 |
| all | prefix_fp8_v | 41.26 | 0.9933 | 0.0059 | 1/1 | 0 |

LPIPS ≤ 0.15 is the predeclared working screen; maintainer acceptance and requested-text/subject/edit review are recorded separately.
