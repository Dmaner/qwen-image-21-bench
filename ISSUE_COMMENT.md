## C1 update: H200 BF16 versus FP8

Compared BF16 with DiT-only FP8, then added Text Encoder FP8 and combined DiT + Text Encoder FP8 on **1× H200 SXM**. All configurations used Omni `3dc35694`, model `d26bb612`, matched inputs/seeds, 1024×1024, 50 steps, CFG 1, eager FA3 and batch/concurrency 1.

| Configuration | T2I latency | Δ vs BF16 | Edit latency | Δ vs BF16 | T2I peak allocated | Δ vs BF16 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 7.260 s | — | 8.332 s | — | 36.86 GiB | — |
| DiT FP8 | 6.018 s | −17.1% | 7.109 s | −14.7% | 30.36 GiB | −17.6% |
| Text Encoder FP8 | 7.284 s | +0.3% | 8.376 s | +0.5% | 30.39 GiB | −17.6% |
| DiT + Text Encoder FP8 | 6.013 s | −17.2% | 7.078 s | −15.0% | 23.89 GiB | −35.2% |

Offline latency is the mean of 10 requests after warmup and feasibility checks. Memory is the worker's peak allocated value from one measured request after warmup. Negative percentages indicate reductions versus BF16.

- **Latency:** DiT FP8 reduces offline T2I/edit latency by 17.1%/14.7%; HTTP T2I client latency falls from 7.338 to 6.103 s (−16.8%). All 120 measured requests succeeded.
- **Memory:** DiT FP8 saves 6.50 GiB; combined FP8 saves 12.97 GiB. Text Encoder FP8 saves memory without reducing latency here.
- **Image quality:** 24 paired cases per FP8 configuration. DiT FP8 reaches 32.96 dB on teapot/seed 42; its initial/held-out means are 23.86/26.96 dB. DiT FP8 and combined FP8 omit the requested English title at seed 43; BF16 and Text Encoder FP8 retain it.

[Full report](https://github.com/Dmaner/qwen-image-21-bench/blob/main/C1_REPORT.md) · [Comparison images](https://github.com/Dmaner/qwen-image-21-bench/blob/main/GALLERY.md) · [Raw data](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-H200-essential-data-20261011.zip) · [Reproduction](https://github.com/Dmaner/qwen-image-21-bench/tree/main/reproduce)

Next: agree on quality tolerances and cross-review the FP8 evaluation and text regression with A0. Prefix-KV FP8 remains a separate follow-up with A3.
