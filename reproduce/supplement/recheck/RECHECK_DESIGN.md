# T2I latency recheck — prepared, not executed

The primary matrix and its original measurements remain unchanged. Run this separate diagnostic cohort only after primary analysis and review of this design.

## Reason

The original `prefix_fp8_v` T2I mean wall time is **7.973 ± 0.568 s**, versus **7.342 ± 0.016 s** for BF16. The measured denoising interval is stable: **7.189 ± 0.0067 s**, versus **7.152 ± 0.0117 s**.

`stage_0_gen − (text_encoder.forward + diffuse + vae.decode)` rises from **0.048 to 0.639 s**. This unassigned interval explains **0.590 s of the 0.631 s** mean wall-time difference. Its ten original values are 44.7, 282.6, 1105.0, 1455.1, 1433.6, 404.7, 1004.0, 571.0, 42.4 and 43.1 ms. No sample is removed. This does not establish a dequantization cost or CPU-contention cause.

The original `joint_fp8` edit measured request at index 5 also has a **0.968 s** unassigned interval (including `vae.encode` in the named-stage sum). The variability is therefore not exclusive to V-only prefix storage. This minimal repeat remains limited to the three T2I arms.

The caller saves PNGs after stopping wall timing. `queue_wait_ms` is the orchestrator-admission interval, not diffusion-scheduler waiting. The existing stage profiler omits pipeline preparation, output postprocessing and transport; its initial synchronization is also outside each named interval. CPU scheduling, cgroup throttling, unprofiled preparation, output conversion and transport remain candidate explanations. The later host-load snapshot cannot identify conditions during the anomalous requests.

## Fixed comparison

- Reverse order: **prefix_fp8 → prefix_fp8_v → BF16**, one fresh engine per arm.
- Same frozen T2I case, seed 42, 1024×1024, 50 steps, CFG 1, eager, model/code/config pins and physical GPU as the primary run.
- **1 warmup + 1 feasibility + 10 measured requests** per arm. Preserve every request, failure and output.
- Results: `recheck/results/perf_<arm>_t2i`; diagnostics: `recheck/diagnostics/perf_<arm>_t2i`.
- All three arms use the same diagnostic runner hash and observer hash. Report original and recheck cohorts separately; do not pool them or replace the original mean with selected fast samples.

## Minimal observations

`scripts/c1_offline_recheck.py` is a copy of the primary runner. Only after wall timing stops, it serializes existing `o.metrics` and saves the already-measured monotonic wall interval. Dict, dataclass and Pydantic metadata are supported; tensors are not copied or reduced. The exact diff is in [runner.diff](runner.diff).

At the pinned source, `DiffusionEngine` creates internal timing fields, but `OmniBase` rebuilds final response metrics through `_merge_stage_metric_event`, which omits `diffusion_metrics`. Therefore saving `o.metrics` alone does **not** establish those internal timings.

An independent, file-hash-guarded observer enables DEBUG only for `vllm_omni.diffusion.diffusion_engine`, filtering it to the existing `step_streaming breakdown` message. It captures the original unrounded arguments: preprocessing, add-request-and-wait, output-ready wait, postprocessing and total. It does not wrap model functions, change tensors or add GPU synchronization. One small JSON line is written per engine request; this additional logging is part of the diagnostic cohort's wall time.

Each engine event must fall inside the corresponding caller's monotonic wall interval, with exactly 12 events matching the 12 sequential requests. Missing, duplicate or unmatched events fail diagnostic verification. The add-request-and-wait interval still combines internal scheduling and execution; no isolated scheduler-wait value is invented.

Read `/proc/loadavg`, host CPU/I/O/memory pressure and cgroup `cpu.stat`, `cpu.max` and pressure files before and after each cell. Preserve unavailable-file errors. These are cell-level observations, not per-request proof of CPU contention.

## Execution and review

Preview only, without importing Torch or starting a model:

```bash
python scripts/run_recheck.py
```

After the primary matrix and analysis finish and this design is reviewed:

```bash
python scripts/run_recheck.py --execute
```

The executable path requires both primary `experiment.ok` and `analysis.ok`, and checks that `analysis/completed/analysis.json` reports `complete_for_supplied_manifests_and_discovered_runs`. It also inspects Linux `/proc` and refuses execution while a primary matrix, inference or analysis script remains active. These gates run initially and before each cell, with the successful checks saved in that cell's diagnostics. This prevents overlap with primary CPU quality scoring even if a completion marker exists prematurely.

Each cell must match its original `results/perf_<arm>_t2i/env.json` in every recorded field except the diagnostic runner hash. Verification checks actual clean Omni checkout, model revision, package versions, GPU UUID/driver, topology, workload, cases/input hashes and runtime configuration; quantization and case-file hashes must also match the frozen files. Only the intended diagnostic harness and separately recorded timing observer differ. The executable verifies [CODE_HASHES.json](CODE_HASHES.json) and refuses existing output directories. No recheck has been started during preparation.

Compare wall, named stages, engine breakdown and host counter deltas. If the added time localizes to an observed interval, report that interval first. Attribute a specific mechanism only with supporting evidence; otherwise retain **unattributed end-to-end variability**.

After all three diagnostic cells complete, the CPU-only summary is generated by:

```bash
python scripts/analyze_recheck.py
```

It reads all ten measured rows in each original and diagnostic arm, rechecks provenance and engine-event mapping, and writes `recheck/analysis/analysis.json` plus `REPORT.md`. Means and sample standard deviations remain separate by cohort, with each arm compared to BF16 from that same cohort. Engine breakdown summaries preserve all samples; host counter deltas apply to entire cells and do not establish a cause.

## Source boundaries

- Caller timing and PNG saving: `scripts/c1_offline.py:384–405`.
- Named stage profiler and initial synchronization: `vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py:20–35`.
- Existing engine breakdown: `vllm_omni/diffusion/diffusion_engine.py:459–537`, SHA256 `9d5fbcc29c1166c1c45443e490c9764c5aa303dc456b467f6c28ded9c6a6037f` at Omni `3dc35694`.
- Final metric reconstruction: `vllm_omni/entrypoints/omni_base.py:731–767`; `vllm_omni/metrics/stats.py:557–660`.
- Admission queue measurement: `vllm_omni/engine/orchestrator.py:3016–3019`.
