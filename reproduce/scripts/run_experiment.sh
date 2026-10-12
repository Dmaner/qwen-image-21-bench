#!/usr/bin/env bash
set -euo pipefail
root=/workspace/c1-phase2
export HF_HOME="$root/hf" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
unset QWEN21_BENCH_STAGES VLLM_BATCH_INVARIANT C1_QUANT_AUDIT A0_PROFILE_EXPORT_ONLY PYTHONPATH
cd /opt/vllm-omni
common=(--model Qwen/Qwen-Image-2.1 --model-revision d26bb61231c349cf6b7896fa83353113880e1ba3 --vllm-omni-sha 3dc35694b3d458fc1461a8fe71832ac12e8c8cea --height 1024 --width 1024 --steps 50 --seed 42 --true-cfg-scale 1 --enforce-eager)
run_case() {
 local out="$1"; shift
 mkdir -p "$out"
 printf '%q ' "$@" > "$out/command.txt"; printf '\n' >> "$out/command.txt"
 printf '%s START %s\n' "$(date -u +%FT%TZ)" "$out" | tee -a "$root/progress.log"
 "$@" > "$out/engine.log" 2>&1
 cat "$out/summary.json" >> "$root/progress.log"
 printf '%s DONE %s\n' "$(date -u +%FT%TZ)" "$out" | tee -a "$root/progress.log"
}
arms=(bf16 dit_fp8 te_fp8 joint_fp8)
for arm in "${arms[@]}"; do
 q=(); if [[ "$arm" != bf16 ]]; then q=(--quant-config-file "$root/scripts/configs/$arm.json"); fi
 out="$root/results/audit_${arm}"
 run_case "$out" env C1_QUANT_AUDIT=1 PYTHONPATH="$root/scripts/quant_audit" python3 "$root/scripts/c1_offline.py" "${common[@]}" --arm "$arm" "${q[@]}" --mode edit --image "$root/reference/upstream/inputs/qwen_bear.png" --out "$out" --quant-audit --warmup 1 --feasibility 0 --measured 1
 python3 "$root/scripts/quant_audit/verify_audit.py" "$out/quant_audit" --arm "$arm" --task edit > "$out/validation.json"
done
touch "$root/quant-gates.ok"
for arm in "${arms[@]}"; do
 q=(); if [[ "$arm" != bf16 ]]; then q=(--quant-config-file "$root/scripts/configs/$arm.json"); fi
 for mode in t2i edit; do
  extra=(); if [[ "$mode" == edit ]]; then extra=(--image "$root/reference/upstream/inputs/qwen_bear.png"); fi
  out="$root/results/perf_${arm}_${mode}"
  run_case "$out" python3 "$root/scripts/c1_offline.py" "${common[@]}" --arm "$arm" "${q[@]}" --mode "$mode" --out "$out" --warmup 1 --feasibility 1 --measured 10 "${extra[@]}"
 done
 out="$root/results/perf_${arm}_serving"
 run_case "$out" python3 "$root/scripts/c1_serving.py" "${common[@]}" --arm "$arm" "${q[@]}" --out "$out" --warmup 1 --feasibility 1 --measured 10
 for mode in t2i edit; do
  extra=(); if [[ "$mode" == edit ]]; then extra=(--image "$root/reference/upstream/inputs/qwen_bear.png"); fi
  out="$root/results/mem_${arm}_${mode}"
  run_case "$out" python3 "$root/scripts/c1_offline.py" "${common[@]}" --arm "$arm" "${q[@]}" --mode "$mode" --out "$out" --warmup 1 --feasibility 0 --measured 1 --mem "${extra[@]}"
  python3 "$root/reference/parse_diagnostics.py" "$out" --mem > "$out/parse_memory.json"
 done
 out="$root/results/quality_${arm}"
 run_case "$out" python3 "$root/scripts/c1_offline.py" "${common[@]}" --arm "$arm" "${q[@]}" --out "$out" --warmup 1 --cases-file "$root/scripts/quality_manifest_effective.json"
done
touch "$root/perf-quality.ok"
for arm in "${arms[@]}"; do
 q=(); if [[ "$arm" != bf16 ]]; then q=(--quant-config-file "$root/scripts/configs/$arm.json"); fi
 for mode in t2i edit; do
  extra=(); if [[ "$mode" == edit ]]; then extra=(--image "$root/reference/upstream/inputs/qwen_bear.png"); fi
  out="$root/results/trace_${arm}_${mode}"
  run_case "$out" env A0_PROFILE_EXPORT_ONLY=1 PYTHONPATH="$root/reference/profiler_export_only" python3 "$root/scripts/c1_offline.py" "${common[@]}" --arm "$arm" "${q[@]}" --mode "$mode" --out "$out" --warmup 1 --feasibility 0 --measured 0 --trace "${extra[@]}"
 done
done
touch "$root/experiment.ok"
