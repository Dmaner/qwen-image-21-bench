# T2I reverse-order diagnostic repeat

Original and diagnostic cohorts are shown separately. All 10 measured requests per arm are retained.

| Cohort | Arm | Wall mean ± SD (s) | vs cohort BF16 | Denoising mean ± SD (s) | Unassigned stage interval mean / max (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| original | bf16 | 7.342 ± 0.016 | +0.00% | 7.152 ± 0.012 | 0.048 / 0.065 |
| original | prefix_fp8_v | 7.973 ± 0.568 | +8.59% | 7.189 ± 0.007 | 0.639 / 1.455 |
| original | prefix_fp8 | 7.372 ± 0.016 | +0.41% | 7.181 ± 0.015 | 0.046 / 0.050 |
| recheck | bf16 | 7.329 ± 0.006 | +0.00% | 7.142 ± 0.005 | 0.043 / 0.045 |
| recheck | prefix_fp8_v | 7.341 ± 0.005 | +0.17% | 7.154 ± 0.006 | 0.043 / 0.051 |
| recheck | prefix_fp8 | 7.340 ± 0.012 | +0.15% | 7.154 ± 0.014 | 0.042 / 0.046 |

## Observed engine breakdown

Diagnostic cohort only; mean / max in ms. Add-request-and-wait includes scheduling and execution.

| Arm | Preprocess | Add request and wait | Output-ready wait | Postprocess | Engine total |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 0.014 / 0.022 | 7290.463 / 7300.495 | 0.000 / 0.000 | 34.481 / 36.798 | 7325.017 / 7334.261 |
| prefix_fp8_v | 0.013 / 0.018 | 7303.936 / 7313.067 | 0.000 / 0.000 | 34.103 / 42.942 | 7338.105 / 7345.977 |
| prefix_fp8 | 0.014 / 0.027 | 7302.575 / 7320.856 | 0.000 / 0.000 | 33.928 / 36.682 | 7336.570 / 7355.915 |

Host `cpu.stat` and pressure counter changes are saved in `analysis.json`. They cover entire cells, including startup and warmup, and do not identify the cause of an individual slow request.

This report assigns no CPU-contention, scheduler or dequantization cause. The original means are not replaced or pooled with the diagnostic repeat.
