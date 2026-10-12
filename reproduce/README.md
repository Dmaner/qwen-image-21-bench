# Reproduce the C1 experiment with five configurations

These are the **exact executed scripts**, copied from the archived Phase 2 container. [source-manifest.json](scripts/source-manifest.json) maps every copied file to its original artifact path and SHA256. See [PROVENANCE.md](../PROVENANCE.md) for A0 derivation and third-party sources. No new blanket license is applied to these files.

## Recorded environment

- One H200; a CUDA-13-capable host driver (recorded driver595.91.07).
- Base image: `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`.
- Omni `3dc35694b3d458fc1461a8fe71832ac12e8c8cea`; model `Qwen/Qwen-Image-2.1@d26bb61231c349cf6b7896fa83353113880e1ba3`.
- Python3.12.3, vLLM0.31.0, Torch2.13.0+cu130, Triton3.7.1, Transformers5.14.1, Diffusers0.40.0, fa3-fwd0.0.3, kernels0.16.1.

Use a **new dedicated container**, not a general workstation installation. The recorded setup writes `/opt/vllm-omni` and `/usr/bin/python`, uses the image's Python/uv, and removes two image-specific OTEL remnants. It is not a general or idempotent installer. Model files and full traces require substantial disk space.

The exact scripts require the experiment root `/workspace/c1-phase2` and Omni checkout `/opt/vllm-omni`. For example, from the public repository root:

```sh
docker run --gpus 'device=0' --ipc=host -it \
  --mount type=bind,src="$PWD/reproduce",dst=/workspace/c1-phase2 \
  --entrypoint /bin/bash \
  vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b
```

Inside that container, fetch the original A0 input and verify its hash. No full A0 checkout is needed:

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

`setup.sh` pins the dependencies/source, checks the environment, and downloads the exact model before the runner enables offline mode. `quality_manifest_effective.json` preserves the recorded fixed input paths; its 24 IDs/prompts/seeds and six edit inputs are the frozen protocol. Do not replace the reference image or silently re-resolve a mutable model revision.

## GPU execution

Port8091 must be free. Run on an otherwise idle single H200:

```sh
cd /workspace/c1-phase2
bash scripts/run_experiment.sh
```

The runner first checks five separate GPU worker audits and aborts if the quantization gates fail. It then runs all configurations (`bf16`, `dit_fp8`, `te_fp8`, `joint_fp8`, `selective_bf16`):

- T2I/edit/serving: 1 warmup + 1 feasibility + 10 measured requests per configuration/task.
- Independent T2I/edit memory: 1 warmup + 1 measured request.
- Quality: 1 warmup + 24 frozen cases per configuration.
- Independent profiler: 1 warmup + one complete 50-step request for each configuration/task.

Quant audits and the profiler export-only adapter run in separate processes and stay off for headline/quality measurements. DiT FP8 with BF16 blocks keeps DiT blocks 0/1/30/31 and the text encoder in BF16; it is a preregistered structural hypothesis, not an identified sensitivity optimum. The serving harness additionally performs its existing three-prompt benchmark cross-check; those requests are not the n=10 headline sample.

This workflow performs a full GPU experiment. It does not provision a cloud machine, contact Feishu, or publish its outputs.

## CPU analysis after successful execution

Use the following commands instead of the old PID-specific postprocess runner, which is deliberately omitted:

```sh
cd /workspace/c1-phase2
python3 scripts/capture_kernel_sources.py
python3 scripts/analyze_performance.py --root . --out-dir analysis/performance
for arm in dit_fp8 te_fp8 joint_fp8 selective_bf16; do
  python3 analysis/analyse_quality.py \
    --manifest scripts/quality_manifest_effective.json \
    --bf16-dir results/quality_bf16 --candidate-dir "results/quality_$arm" \
    --candidate-arm "$arm" --output-dir "analysis/quality/$arm"
done
python3 scripts/analyze_profiles.py --root . --out-dir analysis/profiles \
  --quant-source-dir supplementary/profiler-kernel-source
```

Run the unchanged capture helper **only in the new experiment-only container**: it collects `/tmp/torchinductor_*` Python files plus installed `input_quant_fp8.py`. Preserve their original headers and source/hash manifest. It only copies local evidence; it does not authorize or automatically publish dependency source.

Performance/profile analyzers use the standard library. Quality analysis uses Pillow/NumPy/scikit-image (recorded analysis versions11.3.0/2.2.6/0.25.2). A newly generated review CSV is unfilled: PSNR/SSIM or successful execution do not constitute quality acceptance. Keep any new human/model-assisted observations separate from the frozen published review.

The recorded source/config hashes are evidence for the original run. If adapting paths, software, inputs or instrumentation, save a new version and hashes rather than overwriting that evidence. A different GPU or dependency stack is a new cohort, not a reproduction of the reported H200 numbers.
