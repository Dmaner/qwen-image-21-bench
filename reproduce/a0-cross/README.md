# A0 cross-check analysis source

`compare_a0.py` is the byte-identical analysis script used for the separate 2026-10-10 BF16 cross-check. SHA256: `7a0b7ca20e920c26a675099d9287e4444fa6b711a425165281d1b4bb9d7d76da`.

The [compact data package](https://github.com/Dmaner/qwen-image-21-bench/releases/download/c1-h200-20261011/C1-H200-essential-data-20261011.zip) contains `a0-cross/results/` request/environment records and `a0-cross/analysis/` recorded comparisons. The n=2 latency values use only measured rows from the T2I/edit `*_time` groups. Diagnostic groups remain separate.

The full 28-output image corpus is retained locally, so the compact public package cannot rerun that entire pixel comparison. Its recorded image hashes and comparison results remain available; the two latest Phase 2 anchor images are also in the public gallery.

To reuse the unmodified script with a complete corpus, explicitly supply `--reference-results`, `--candidate-results` and `--output-dir`; its original directory defaults do not apply here. The author reference is fixed at A0 revision `cf99e73c8fa1603718c47405730d94c9fbfe6e16`. Pillow is required for decoded RGB; inspect `read_errors` and image counts before interpreting an incomplete run.

## Report figures

[`plot_review.py`](plot_review.py) generates the two SVGs in [`figures/a0-review/`](../../figures/a0-review) from the published T2I/edit `trace_analysis.json` files. This is a new plotting script for the historical data; it does not run model inference or copy the author's charts.

From the repository root, with [uv](https://docs.astral.sh/uv/):

```sh
uv run --no-project --with matplotlib==3.9.4 python reproduce/a0-cross/plot_review.py
```

The line plot shows steps 0–49; its mean labels use steps 1–49. The stacked bars use whole-request category totals, combine `reduce` with `norm`, and put remaining categories (including editing's small `conv` entry) in `Other`. Both use summed GPU-event durations, not elapsed request time or GPU utilization.
