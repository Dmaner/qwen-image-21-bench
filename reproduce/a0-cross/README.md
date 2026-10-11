# Recompute the separate A0 cross-check

`compare_a0.py` is copied byte-for-byte from the original 2026-10-10 audit. Its SHA256 is:

```text
7a0b7ca20e920c26a675099d9287e4444fa6b711a425165281d1b4bb9d7d76da
```

This is a CPU-only reader for the **earlier A0 cross-check**, not the five-arm Phase 2 analysis. It keeps timing, memory, stage and trace groups separate. Pillow is required for decoded-RGB hashes/PSNR; scikit-image is optional for SSIM. No GPU package or model download is needed.

From the repository root, place the published `C1-A0-cross-20261010-evidence.zip` in `downloads/`, then run:

```bash
mkdir -p work/a0-cross
unzip downloads/C1-A0-cross-20261010-evidence.zip -d work/a0-cross
git clone https://github.com/luca-888/qwen-image-21-bench.git work/a0-upstream
git -C work/a0-upstream checkout --detach cf99e73c8fa1603718c47405730d94c9fbfe6e16
python3 reproduce/a0-cross/compare_a0.py \
  --reference-results work/a0-upstream/results \
  --candidate-results work/a0-cross/results \
  --output-dir work/a0-cross/recomputed
```

Use an existing Python environment with Pillow, or install Pillow in a local virtual environment first. The explicit paths are required because the unmodified script's original directory defaults do not apply to this publication layout. Outputs are `comparison.json` and `REPORT.md` in the requested output directory; archived evidence is not overwritten. The script records missing/decode errors in its output: inspect `read_errors`, cohort counts and RGB-match fields rather than treating an exit code alone as acceptance.

The archive contains 28 original repeated output PNGs: **16 T2I and 12 editing**, across nine successful run groups, including warmup, feasibility, memory/stage diagnostics and trace outputs. They are not 28 independent prompts or 28 headline measurements. The headline n=2 values come only from the two measured rows in `20261010_a0_t2i_h200_eager_time` and `20261010_a0_edit_h200_eager_time`.

Full Chrome trace `.gz` files and memory pickles are omitted from this earlier cross-check archive; available trace-run metadata/images remain. This does not prevent recomputing the n=2 timing or decoded-RGB comparisons. Original audit metadata may contain the original machine's paths; the explicit arguments above point to the extracted copies.

[verification.json](verification.json) records the script SHA, archive SHA/CRC check and all 28 PNG member hashes, each verified against the original local source bytes. This packaging check did not rerun inference or the comparison analysis. For the latest Phase 2 two-anchor PNG check, see `A0_ALIGNMENT_RECHECK.json` in the published Phase 2 evidence; that is a separate two-image claim.
