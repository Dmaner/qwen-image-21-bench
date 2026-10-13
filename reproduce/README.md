# Reproduce C1 experiments

## Latest H200 supplement

Use [supplement/README.md](supplement/README.md) for the seven-configuration, 36-case run and independent CFG4 prefix check. It includes all fixed inputs and the current setup, runtime-audit and analysis commands. [Design](../EXPERIMENT_DESIGN.md) · [Data](../data/supplement-20261013/README.md).

## Historical 2026-10-11 experiment

Run the four configurations from the historical 2026-10-11 cohort: `bf16`, `dit_fp8`, `te_fp8`, and `joint_fp8`. The runner is adapted from the archived experiment; [source-manifest.json](scripts/source-manifest.json) records original and current file hashes.

### Recorded environment

- 1× H200; recorded driver 595.91.07.
- Base image: `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`.
- Omni `3dc35694b3d458fc1461a8fe71832ac12e8c8cea`; model `Qwen/Qwen-Image-2.1@d26bb61231c349cf6b7896fa83353113880e1ba3`.
- Python 3.12.3, vLLM 0.31.0, Torch 2.13.0+cu130, Triton 3.7.1, Transformers 5.14.1, Diffusers 0.40.0, fa3-fwd 0.0.3, kernels 0.16.1.

From the repository root, start a fresh dedicated container. Setup installs into `/opt/vllm-omni` and the container's Python environment:

```sh
docker run --gpus 'device=0' --ipc=host -it \
  --mount type=bind,src="$PWD/reproduce",dst=/workspace/c1-phase2 \
  --entrypoint /bin/bash \
  vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
```

Inside the container, fetch and verify the reference image, then install the pinned environment and download the model:

```sh
cd /workspace/c1-phase2
mkdir -p reference/upstream/inputs
curl --fail --location \
  https://raw.githubusercontent.com/luca-888/qwen-image-21-bench/cf99e73c8fa1603718c47405730d94c9fbfe6e16/inputs/qwen_bear.png \
  --output reference/upstream/inputs/qwen_bear.png
printf '%s  %s\n' \
  69ac975d5408ebb8de85b1126ad5a37372621e416c4f1a9037cd2ba19b712817 \
  reference/upstream/inputs/qwen_bear.png | sha256sum --check
bash scripts/setup.sh
```

### GPU execution

Use an otherwise idle H200 with port 8091 free:

```sh
cd /workspace/c1-phase2
bash scripts/run_experiment.sh
```

The runner verifies each configuration's quantization audit before running:

- T2I/edit/serving: 1 warmup + 1 feasibility + 10 measured requests per configuration/task.
- T2I/edit memory: 1 warmup + 1 measured request.
- Quality: 1 warmup + 24 fixed cases per configuration.
- Profiler: 1 warmup + one complete 50-step request per configuration/task.

Prompts, seeds and reference inputs are in [quality_manifest_effective.json](scripts/quality_manifest_effective.json). Each run saves its command, environment, logs and results under `results/`.

### CPU analysis after successful execution

In the same container, calculate performance, PSNR/SSIM and profiler summaries:

```sh
cd /workspace/c1-phase2
python3 scripts/capture_kernel_sources.py
python3 scripts/analyze_performance.py --root . --out-dir analysis/performance
for arm in dit_fp8 te_fp8 joint_fp8; do
  python3 analysis/analyse_quality.py \
    --manifest scripts/quality_manifest_effective.json \
    --bf16-dir results/quality_bf16 --candidate-dir "results/quality_$arm" \
    --candidate-arm "$arm" --output-dir "analysis/quality/$arm"
done
python3 scripts/analyze_profiles.py --root . --out-dir analysis/profiles \
  --quant-source-dir supplementary/profiler-kernel-source
```

Then score LPIPS using an isolated CPU environment with the recorded scoring-package versions:

```sh
uv venv --python 3.13.13 .venv-lpips
uv pip install --python .venv-lpips/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.11.0 torchvision==0.26.0
uv pip install --python .venv-lpips/bin/python \
  lpips==0.1.4 numpy==2.4.6 Pillow==12.2.0 scipy==1.17.1
.venv-lpips/bin/python analysis/add_lpips.py \
  --metrics-root analysis/quality \
  --images-root results \
  --output-dir analysis/quality/lpips
```

LPIPS uses AlexNet v0.1 at the original image resolution and verifies PNG/RGB hashes against the newly generated quality records. It requires all 24 images per configuration; the selected public gallery is insufficient. The first run downloads the scoring weights; each run requires a fresh output directory. Recorded results: [LPIPS JSON](../data/phase2/quality/lpips/lpips_metrics.json) · [per-image CSV](../data/phase2/quality/lpips/lpips_pairs.csv).

Results are written to `analysis/performance/`, `analysis/quality/` and `analysis/profiles/`. Fill in the generated `human_review.csv` files for visual assessment. Original measurements remain in [`data/`](../data/README.md); new runs produce their own records.
