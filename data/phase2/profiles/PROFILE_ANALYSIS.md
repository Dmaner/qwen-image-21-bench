# C1 profile evidence

Only captured GPU-event duration sums are shown. They are not E2E latency, do not include all host gaps, and can overlap across streams. Use separate unprofiled runs for performance.

| Case | Steps | Whole GPU-event sum (ms) | Denoise FP8 GEMM (ms) | Denoise BF16 FA3 SM90 (ms) |
|---|---:|---:|---:|---:|
| bf16-t2i | 50 | 6881.070 | 0.000 | 653.988 |
| bf16-edit | 50 | 7851.993 | 0.000 | 1280.736 |
| dit_fp8-t2i | 50 | 5594.161 | 1940.619 | 597.664 |
| dit_fp8-edit | 50 | 6424.399 | 1983.434 | 1166.111 |
| te_fp8-t2i | 50 | 6872.200 | 0.000 | 653.646 |
| te_fp8-edit | 50 | 7822.891 | 0.000 | 1270.806 |
| joint_fp8-t2i | 50 | 5591.453 | 1939.828 | 597.766 |
| joint_fp8-edit | 50 | 6425.768 | 1982.328 | 1166.384 |
| selective_bf16-t2i | 50 | 5727.707 | 1701.400 | 598.884 |
| selective_bf16-edit | 50 | 6579.380 | 1744.234 | 1170.806 |

## bf16-t2i

| Denoise category | Count | GPU-event sum (ms) | Share of sum |
|---|---:|---:|---:|
| bf16_gemm | 8302 | 3636.619 | 53.83% |
| elementwise | 47096 | 1483.177 | 21.96% |
| fa3_sm90_bf16 | 1632 | 653.988 | 9.68% |
| copy_cat_gather | 10322 | 581.018 | 8.60% |
| dtype_cast | 6483 | 161.638 | 2.39% |
| reduction | 4009 | 132.021 | 1.95% |
| normalization | 3250 | 94.118 | 1.39% |
| memset | 8050 | 7.928 | 0.12% |
| memcpy | 1389 | 2.795 | 0.04% |
| other | 1207 | 1.948 | 0.03% |

Full names, operator input types, evidence tier, whole-trace distribution and attribution coverage are in this case JSON.

## bf16-edit

| Denoise category | Count | GPU-event sum (ms) | Share of sum |
|---|---:|---:|---:|
| bf16_gemm | 8302 | 3760.518 | 49.20% |
| elementwise | 47129 | 1518.697 | 19.87% |
| fa3_sm90_bf16 | 1696 | 1280.736 | 16.76% |
| copy_cat_gather | 10322 | 671.170 | 8.78% |
| dtype_cast | 6483 | 165.827 | 2.17% |
| reduction | 4109 | 136.299 | 1.78% |
| normalization | 3250 | 96.633 | 1.26% |
| memset | 8050 | 7.394 | 0.10% |
| memcpy | 1292 | 3.542 | 0.05% |
| other | 1209 | 2.266 | 0.03% |

Full names, operator input types, evidence tier, whole-trace distribution and attribution coverage are in this case JSON.

## dit_fp8-t2i

| Denoise category | Count | GPU-event sum (ms) | Share of sum |
|---|---:|---:|---:|
| fp8_gemm | 8000 | 1940.619 | 35.47% |
| elementwise | 47096 | 1411.882 | 25.81% |
| quantization | 8000 | 599.666 | 10.96% |
| fa3_sm90_bf16 | 1632 | 597.664 | 10.92% |
| copy_cat_gather | 10322 | 541.486 | 9.90% |
| dtype_cast | 6483 | 159.535 | 2.92% |
| reduction | 4009 | 123.031 | 2.25% |
| normalization | 3250 | 87.752 | 1.60% |
| bf16_gemm | 302 | 4.614 | 0.08% |
| memcpy | 1389 | 2.806 | 0.05% |
| other | 1207 | 1.759 | 0.03% |
| memset | 50 | 0.040 | 0.00% |

Full names, operator input types, evidence tier, whole-trace distribution and attribution coverage are in this case JSON.

## dit_fp8-edit

| Denoise category | Count | GPU-event sum (ms) | Share of sum |
|---|---:|---:|---:|
| fp8_gemm | 8000 | 1983.434 | 31.90% |
| elementwise | 47129 | 1438.630 | 23.14% |
| fa3_sm90_bf16 | 1696 | 1166.111 | 18.75% |
| copy_cat_gather | 10322 | 626.195 | 10.07% |
| quantization | 8000 | 614.549 | 9.88% |
| dtype_cast | 6483 | 164.046 | 2.64% |
| reduction | 4109 | 125.928 | 2.03% |
| normalization | 3250 | 89.092 | 1.43% |
| bf16_gemm | 302 | 4.673 | 0.08% |
| memcpy | 1292 | 3.489 | 0.06% |
| other | 1209 | 2.084 | 0.03% |
| memset | 50 | 0.041 | 0.00% |

Full names, operator input types, evidence tier, whole-trace distribution and attribution coverage are in this case JSON.

## te_fp8-t2i

| Denoise category | Count | GPU-event sum (ms) | Share of sum |
|---|---:|---:|---:|
| bf16_gemm | 8302 | 3632.632 | 53.84% |
| elementwise | 47096 | 1481.859 | 21.96% |
| fa3_sm90_bf16 | 1632 | 653.646 | 9.69% |
| copy_cat_gather | 10322 | 579.636 | 8.59% |
| dtype_cast | 6483 | 161.443 | 2.39% |
| reduction | 4009 | 131.711 | 1.95% |
| normalization | 3250 | 94.001 | 1.39% |
| memset | 8050 | 7.904 | 0.12% |
| memcpy | 1389 | 2.764 | 0.04% |
| other | 1207 | 1.923 | 0.03% |

Full names, operator input types, evidence tier, whole-trace distribution and attribution coverage are in this case JSON.

## te_fp8-edit

| Denoise category | Count | GPU-event sum (ms) | Share of sum |
|---|---:|---:|---:|
| bf16_gemm | 8302 | 3735.869 | 49.09% |
| elementwise | 47129 | 1518.408 | 19.95% |
| fa3_sm90_bf16 | 1696 | 1270.806 | 16.70% |
| copy_cat_gather | 10322 | 672.448 | 8.84% |
| dtype_cast | 6483 | 166.178 | 2.18% |
| reduction | 4109 | 135.026 | 1.77% |
| normalization | 3250 | 96.158 | 1.26% |
| memset | 8050 | 9.333 | 0.12% |
| memcpy | 1292 | 3.549 | 0.05% |
| other | 1209 | 2.243 | 0.03% |

Full names, operator input types, evidence tier, whole-trace distribution and attribution coverage are in this case JSON.

## joint_fp8-t2i

| Denoise category | Count | GPU-event sum (ms) | Share of sum |
|---|---:|---:|---:|
| fp8_gemm | 8000 | 1939.828 | 35.47% |
| elementwise | 47096 | 1411.790 | 25.81% |
| quantization | 8000 | 599.533 | 10.96% |
| fa3_sm90_bf16 | 1632 | 597.766 | 10.93% |
| copy_cat_gather | 10322 | 541.038 | 9.89% |
| dtype_cast | 6483 | 159.434 | 2.92% |
| reduction | 4009 | 122.660 | 2.24% |
| normalization | 3250 | 87.915 | 1.61% |
| bf16_gemm | 302 | 4.583 | 0.08% |
| memcpy | 1389 | 2.748 | 0.05% |
| other | 1207 | 1.760 | 0.03% |
| memset | 50 | 0.038 | 0.00% |

Full names, operator input types, evidence tier, whole-trace distribution and attribution coverage are in this case JSON.

## joint_fp8-edit

| Denoise category | Count | GPU-event sum (ms) | Share of sum |
|---|---:|---:|---:|
| fp8_gemm | 8000 | 1982.328 | 31.88% |
| elementwise | 47129 | 1439.193 | 23.15% |
| fa3_sm90_bf16 | 1696 | 1166.384 | 18.76% |
| copy_cat_gather | 10322 | 625.990 | 10.07% |
| quantization | 8000 | 614.356 | 9.88% |
| dtype_cast | 6483 | 163.699 | 2.63% |
| reduction | 4109 | 125.715 | 2.02% |
| normalization | 3250 | 89.344 | 1.44% |
| bf16_gemm | 302 | 4.673 | 0.08% |
| memcpy | 1292 | 3.509 | 0.06% |
| other | 1209 | 2.008 | 0.03% |
| memset | 50 | 0.040 | 0.00% |

Full names, operator input types, evidence tier, whole-trace distribution and attribution coverage are in this case JSON.

## selective_bf16-t2i

| Denoise category | Count | GPU-event sum (ms) | Share of sum |
|---|---:|---:|---:|
| fp8_gemm | 7000 | 1701.400 | 30.36% |
| elementwise | 47096 | 1412.912 | 25.21% |
| fa3_sm90_bf16 | 1632 | 598.884 | 10.69% |
| copy_cat_gather | 10322 | 547.961 | 9.78% |
| quantization | 7000 | 526.417 | 9.39% |
| bf16_gemm | 1302 | 439.805 | 7.85% |
| dtype_cast | 6483 | 160.210 | 2.86% |
| reduction | 4009 | 123.310 | 2.20% |
| normalization | 3250 | 87.831 | 1.57% |
| memcpy | 1389 | 2.736 | 0.05% |
| other | 1207 | 1.760 | 0.03% |
| memset | 1050 | 0.993 | 0.02% |

Full names, operator input types, evidence tier, whole-trace distribution and attribution coverage are in this case JSON.

## selective_bf16-edit

| Denoise category | Count | GPU-event sum (ms) | Share of sum |
|---|---:|---:|---:|
| fp8_gemm | 7000 | 1744.234 | 27.37% |
| elementwise | 47129 | 1446.228 | 22.69% |
| fa3_sm90_bf16 | 1696 | 1170.806 | 18.37% |
| copy_cat_gather | 10322 | 630.560 | 9.89% |
| quantization | 7000 | 544.064 | 8.54% |
| bf16_gemm | 1302 | 450.618 | 7.07% |
| dtype_cast | 6483 | 164.319 | 2.58% |
| reduction | 4109 | 126.589 | 1.99% |
| normalization | 3250 | 89.528 | 1.40% |
| memcpy | 1292 | 3.554 | 0.06% |
| other | 1209 | 2.066 | 0.03% |
| memset | 1050 | 0.951 | 0.01% |

Full names, operator input types, evidence tier, whole-trace distribution and attribution coverage are in this case JSON.

## Component execution

| Case | Component | FP8 GEMM count | FP8 GPU sum ms |
|---|---|---:|---:|
| bf16-t2i | te_vision | 0 | 0.000 |
| bf16-t2i | te_language | 0 | 0.000 |
| bf16-t2i | dit | 0 | 0.000 |
| bf16-t2i | te_other | 0 | 0.000 |
| bf16-t2i | unassigned | 0 | 0.000 |
| bf16-edit | te_vision | 0 | 0.000 |
| bf16-edit | te_language | 0 | 0.000 |
| bf16-edit | dit | 0 | 0.000 |
| bf16-edit | te_other | 0 | 0.000 |
| bf16-edit | unassigned | 0 | 0.000 |
| dit_fp8-t2i | te_vision | 0 | 0.000 |
| dit_fp8-t2i | te_language | 0 | 0.000 |
| dit_fp8-t2i | dit | 8000 | 1940.619 |
| dit_fp8-t2i | te_other | 0 | 0.000 |
| dit_fp8-t2i | unassigned | 0 | 0.000 |
| dit_fp8-edit | te_vision | 0 | 0.000 |
| dit_fp8-edit | te_language | 0 | 0.000 |
| dit_fp8-edit | dit | 8000 | 1983.434 |
| dit_fp8-edit | te_other | 0 | 0.000 |
| dit_fp8-edit | unassigned | 0 | 0.000 |
| te_fp8-t2i | te_vision | 0 | 0.000 |
| te_fp8-t2i | te_language | 252 | 3.335 |
| te_fp8-t2i | dit | 0 | 0.000 |
| te_fp8-t2i | te_other | 0 | 0.000 |
| te_fp8-t2i | unassigned | 0 | 0.000 |
| te_fp8-edit | te_vision | 0 | 0.000 |
| te_fp8-edit | te_language | 252 | 17.006 |
| te_fp8-edit | dit | 0 | 0.000 |
| te_fp8-edit | te_other | 0 | 0.000 |
| te_fp8-edit | unassigned | 0 | 0.000 |
| joint_fp8-t2i | te_vision | 0 | 0.000 |
| joint_fp8-t2i | te_language | 252 | 3.328 |
| joint_fp8-t2i | dit | 8000 | 1939.828 |
| joint_fp8-t2i | te_other | 0 | 0.000 |
| joint_fp8-t2i | unassigned | 0 | 0.000 |
| joint_fp8-edit | te_vision | 0 | 0.000 |
| joint_fp8-edit | te_language | 252 | 17.056 |
| joint_fp8-edit | dit | 8000 | 1982.328 |
| joint_fp8-edit | te_other | 0 | 0.000 |
| joint_fp8-edit | unassigned | 0 | 0.000 |
| selective_bf16-t2i | te_vision | 0 | 0.000 |
| selective_bf16-t2i | te_language | 0 | 0.000 |
| selective_bf16-t2i | dit | 7000 | 1701.400 |
| selective_bf16-t2i | te_other | 0 | 0.000 |
| selective_bf16-t2i | unassigned | 0 | 0.000 |
| selective_bf16-edit | te_vision | 0 | 0.000 |
| selective_bf16-edit | te_language | 0 | 0.000 |
| selective_bf16-edit | dit | 7000 | 1744.234 |
| selective_bf16-edit | te_other | 0 | 0.000 |
| selective_bf16-edit | unassigned | 0 | 0.000 |


