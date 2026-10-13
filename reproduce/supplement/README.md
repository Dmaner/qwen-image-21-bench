# Reproduce the C1 H200 supplement

Use a fresh single-H200 environment with the pinned base image:

```text
vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
```

The published files are under `reproduce/supplement/{scripts,inputs,reference,recheck}`. From the repository root, copy them to the path recorded in the manifests:

```sh
mkdir -p /workspace/c1-supplement
cp -a reproduce/supplement/. /workspace/c1-supplement/
cd /workspace/c1-supplement
bash scripts/setup.sh
```

Setup installs the pinned Omni/packages, downloads the fixed model revision and records the actual environment. The input PNGs and hashes are included; no reference-image download is required.

Prepare the CPU quality scorer's weights, then run the experiment and analysis:

```sh
cd /workspace/c1-supplement
TORCH_HOME=/workspace/c1-supplement/torch-cache CUDA_VISIBLE_DEVICES='' \
  python3 - <<'PY'
import lpips
lpips.LPIPS(net='alex', version='0.1').cpu().eval()
PY
bash scripts/execute_all.sh
python3 scripts/finalize_memory.py \
  --analysis analysis/completed/analysis.json \
  --results-root results --out-dir analysis/final
python3 scripts/analyze_prefix_traces.py \
  --results-root results --out-dir analysis/prefix_traces
```

## What runs

- **Seven configurations:** `bf16`, `dit_fp8`, `mlp_bf16`, `te_fp8`, `joint_fp8`, `prefix_fp8_v`, `prefix_fp8`.
- **Runtime audits:** verify the actual Linear configuration, attention inputs and prefix storage before the corresponding measurements.
- **Timing:** T2I and one-reference bear edit for all seven configurations, plus the two-reference edit for BF16 and both prefix FP8 configurations; one warmup, one feasibility and 10 measured requests per cell.
- **Memory:** one warmup and one measured request for each timing workload/configuration.
- **Quality:** 36 CFG1 cases per configuration from [effective_manifest.json](inputs/effective_manifest.json). A separate [CFG4 manifest](inputs/cfg4_manifest.json) runs only on BF16 and the two prefix configurations, with an explicit empty negative prompt.
- **Profiler:** one warmed two-reference capture for each prefix configuration.

`mlp_bf16` leaves all 96 DiT `img_mlp` Linears BF16 and quantizes the 64 attention Linears. Its [full-name glob configuration](scripts/configs/mlp_bf16.json) targets the same modules as #8099 on the pinned vLLM 0.31 environment.

The new edit inputs are a person photograph, an espresso cup and a synthetic text sign, each at seeds 42/43/44; the two-reference cases use astronaut then coffee in that order. [Input provenance](reference/INPUT_PROVENANCE.md) records permissions and hashes.

## Outputs

Each cell writes its command, environment, request records, images and audit under `results/`. The orchestrator resumes cells with successful `cell.ok` markers and stops for inspection if a partial cell exists.

- `analysis/final/TABLES.md` and `analysis.json`: final performance, full-precision memory, cache and quality summaries. The CPU finalization reads measured request memory from `runs.jsonl` and records source hashes; `analysis/completed` remains unchanged.
- `analysis/final/quality_pairs.csv` and `quality_gate.csv`: per-image scores and LPIPS ≤ 0.15 working-gate counts.
- `analysis/final/visual_review.csv`: requested-text, subject and edit checks to complete; human review and maintainer acceptance are recorded separately.
- `analysis/prefix_traces/PREFIX_TRACE_SUMMARY.md`: GPU kernel inventory and source trace hashes.

The 36-case comparisons use the newly measured BF16 outputs from this run. The CFG4 check has its own denominator. Raw outputs and failed cases remain available even when only a few comparisons are selected for the report.

## Independent T2I diagnostic repeat

After the primary experiment and CPU analysis finish, run the three-arm reverse-order diagnostic on the **same machine**. It checks its environment against the primary results just produced there; a new machine does not need to match the original report's GPU UUID.

```sh
cd /workspace/c1-supplement
python3 scripts/run_recheck.py
python3 scripts/run_recheck.py --execute
python3 scripts/analyze_recheck.py
```

This runs `prefix_fp8 → prefix_fp8_v → bf16`, each with 1 warmup, 1 feasibility and 10 measured requests. Execution requires complete primary analysis and no active primary pipeline. [Frozen diagnostic design](recheck/RECHECK_DESIGN.md), [runner diff](recheck/runner.diff) and [code hashes](recheck/CODE_HASHES.json) describe the added timing observation. Results are written under `recheck/results`, `recheck/diagnostics` and `recheck/analysis`; they stay separate from the primary measurements.

The compact [raw records](../../data/supplement-20261013/rawrecords.zip) preserve the same directory layout. Extract archived records into a separate review directory, not the fresh execution directory. They support numerical inspection; full-resolution quality rescoring and raw-trace analysis require outputs from a fresh run because images and large profiler traces are excluded from the ZIP.

To recompute only the archived recheck tables on CPU, from the repository root:

```sh
mkdir -p /tmp/c1-record-review
cp -a reproduce/supplement/. /tmp/c1-record-review/
unzip -n data/supplement-20261013/rawrecords.zip -d /tmp/c1-record-review
python3 /tmp/c1-record-review/scripts/analyze_recheck.py \
  --out-dir /tmp/c1-record-review/recomputed-recheck
```
