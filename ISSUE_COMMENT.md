## C1 update: H200 FP8 performance, quality and profiler results

**C1 requirement:**

> H200 BF16 versus FP8 quality, memory, and latency; extend only after the first comparison

Completed the initial **BF16 versus DiT-only FP8** comparison on H200, covering paired quality/failures, peak memory and latency. Text Encoder FP8, DiT + Text Encoder FP8 and DiT FP8 with BF16 blocks followed as extensions. Completion of the comparison does not imply quality acceptance.

The results below use fresh controls and all three extensions on **1× H200 SXM** in Phase 2. DiT FP8 with BF16 blocks retains DiT blocks 0/1/30/31 in BF16; Phase 1 timings are not pooled into these values.

All configurations used the same pinned Omni/model revisions, 1024×1024, 50 steps, CFG 1, eager FA3, and batch/concurrency 1. Performance/profiler runs retained **A0's prompts, reference image and seed 42**. The latest BF16 anchor images match the author's PNGs exactly; the separate historical n=2 BF16 cross-check was within −0.11% T2I / +1.18% editing latency of A0 on a different H200.

| Configuration | T2I latency | Single-reference edit latency | T2I peak allocated |
|---|---:|---:|---:|
| BF16 | 7.260 s | 8.332 s | 36.86 GiB |
| DiT FP8 | 6.018 s | 7.109 s | 30.36 GiB |
| Text Encoder FP8 | 7.284 s | 8.376 s | 30.39 GiB |
| DiT + Text Encoder FP8 | 6.013 s | 7.078 s | 23.89 GiB |
| DiT FP8 with BF16 blocks | 6.152 s | 7.243 s | 31.17 GiB |

Latency is the mean of 10 measured requests after warmup/feasibility, with torch profiler disabled. Memory uses separate worker-peak measurements.

- **DiT FP8:** 17.1% T2I / 14.7% editing latency reduction and 6.50 GiB less allocated memory.
- **Text Encoder FP8:** memory savings without a latency benefit here. DiT + Text Encoder FP8 saved 12.97 GiB with latency close to DiT-only.
- **Quality:** 24 paired cases per candidate, including 12 frozen held-out cases. DiT FP8 and DiT + Text Encoder FP8 lost the requested English title at seed 43; Text Encoder FP8 and DiT FP8 with BF16 blocks retained it. The latter cost 2.2% T2I / 1.9% editing latency versus DiT-only. This is model-assisted review; overall quality acceptance remains pending.
- **Profiler:** ten complete 50-step traces verify actual E4M3 execution. DiT activation quantization took about 11.0% T2I / 9.9% editing of denoising GPU-event time in the DiT-only configuration; these event sums are not E2E latency.

Public artifacts:

- [Full report in A0 format, exact revisions and measurement definitions](https://github.com/Dmaner/qwen-image-21-bench/blob/main/C1_REPORT.md)
- [Original T2I/editing/text comparison images](https://github.com/Dmaner/qwen-image-21-bench/blob/c1-h200-20261011/GALLERY.md)
- [Compact raw numerical records, configuration and A0 cross-check metadata](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-H200-essential-data-20261011.zip)
- [Configs, frozen prompts and reproduction tools](https://github.com/Dmaner/qwen-image-21-bench/tree/c1-h200-20261011/reproduce)
- [Artifact SHA256 manifest](https://github.com/Dmaner/qwen-image-21-bench/blob/c1-h200-20261011/data/SHA256SUMS)

The [release](https://github.com/Dmaner/qwen-image-21-bench/releases/tag/c1-h200-20261011) contains this single compact package. Full raw traces, memory pickles and the complete image corpus remain local; profile summaries and quality metrics are public.

Next coordination: agree on quality tolerances and independently cross-review the text failures and the result with BF16 blocks with A0. Prefix-KV FP8, consumer GPUs, multi-GPU and concurrent workloads are not covered.
