# C1 H200 supplement: experimental design

This supplement evaluates the selective BF16 configuration from [#8099](https://github.com/vllm-project/vllm-omni/pull/8099), FP8 prefix storage, and editing with distinct reference inputs. Completed results are in the [C1 report](C1_REPORT.md).

## Setup

One Runpod H200 SXM; Omni `3dc35694b3d458fc1461a8fe71832ac12e8c8cea`; model `Qwen/Qwen-Image-2.1@d26bb61231c349cf6b7896fa83353113880e1ba3`; vLLM 0.31.0, Torch 2.13.0+cu130, Transformers 5.14.1, Diffusers 0.40.0 and Triton 3.7.1.

Main settings: 1024×1024, 50 steps, CFG 1, batch/concurrency 1, eager and FA3. No offload, VAE tiling, CUDA Graph or cross-step denoising cache. Each workload has a fresh BF16 control on the same GPU as its candidate runs; historical BF16 measurements are not used to calculate new deltas.

| Configuration | DiT block Linears | Text Encoder | Prefix storage |
|---|---|---|---|
| BF16 | BF16 | BF16 | BF16 |
| DiT FP8 | 160 FP8 | BF16 | BF16 |
| MLP BF16 | 64 attention FP8; 96 `img_mlp` BF16 | BF16 | BF16 |
| Text Encoder FP8 | BF16 | 252 language Linears FP8 | BF16 |
| DiT + Text Encoder FP8 | 160 FP8 | 252 language Linears FP8 | BF16 |
| Prefix V FP8 | BF16 | BF16 | K BF16; V FP8 |
| Prefix K+V FP8 | BF16 | BF16 | K and V FP8 |

FP8 uses E4M3. The MLP BF16 arm targets the same modules as #8099, using `transformer.transformer_blocks.*.img_mlp.*` to match full layer names in vLLM 0.31. Runtime inventory must confirm 64 FP8 attention Linears and 96 BF16 MLP Linears. Prefix-only arms keep weights and attention computation BF16; cache storage includes the scaling tensors. VAE, vision encoder, LM head and DiT boundary projections remain BF16.

## Inputs

The 36-case set preserves the original 24 cases and adds the following edits, each with seeds **42, 43 and 44**:

| Input | Fixed editing target |
|---|---|
| Person: NASA astronaut photograph | Replace only the background with light gray; preserve face, hair, spacesuit, helmet and pose. |
| Object: espresso cup photograph | Change only the brown cup exterior and saucer to cobalt blue. |
| Text: synthetic café sign | Replace `QWEN CAFE` with `IMAGE CAFE`; retain `OPEN 09:00 - 18:00`. |
| Two references: astronaut + espresso cup | Place the reference cup set on a table in front of the reference person, preserving both subjects. |

The original cases retain their prompts and seeds, including seeds 45/46/47 in the historical held-out subset. Every arm receives identical case inputs and image hashes. [Frozen manifest](reproduce/supplement/inputs/effective_manifest.json) · [Input sources and permissions](reproduce/supplement/reference/INPUT_PROVENANCE.md).

A separate prefix comparison uses the teapot prompt, seed 42 and **CFG 4** for BF16, Prefix V FP8 and Prefix K+V FP8. It explicitly supplies an empty negative prompt to activate CFG; #8099 does not disclose its full negative prompt. This single case is reported separately from the 36 CFG1 cases.

## Measurements

- **Latency:** all seven configurations run teapot T2I and the original bear edit; the three prefix configurations also run the two-reference edit. Each configuration/workload uses one warmup, one feasibility request and **10 measured requests**. Report mean ± sample SD, successful/failed counts and percentage differences from the matching BF16 control.
- **Memory:** worker peak allocated/reserved GiB after one warmup and one measured request. Prefix audits additionally record actual token lengths, K/V and scale dtypes, logical storage bytes and unique backing-storage bytes.
- **Image quality:** PSNR, SSIM and LPIPS for each paired output, category summaries, failure counts and worst cases. Review requested text, subject preservation and edit completion; record failures already present in BF16 separately.
- **Profiler:** one warmed two-reference capture for each prefix configuration. Report captured GPU kernel counts and durations, including identifiable quantization/dequantization/packing operations; preserve unresolved kernel names.

The predeclared working numerical gate is **LPIPS (AlexNet v0.1) ≤ 0.15 per image**. Maintainer approval of this gate and human quality confirmation remain pending. Missing outputs remain in the planned denominator, and selected effect images accompany the complete per-case data.
