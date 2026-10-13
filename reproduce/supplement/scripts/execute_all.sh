#!/usr/bin/env bash
set -euo pipefail
cd /workspace/c1-supplement
python3 scripts/run_matrix.py
CUDA_VISIBLE_DEVICES='' python3 scripts/analyze_supplement.py \
  --results-root results \
  --quality-manifest inputs/effective_manifest.json \
  --quality-manifest inputs/cfg4_manifest.json \
  --out-dir analysis/completed \
  --torch-home /workspace/c1-supplement/torch-cache \
  --threads 8
touch analysis.ok
