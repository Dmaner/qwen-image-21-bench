# C1 supplement: frozen input fixtures

Prepared on 2026-10-13 before any supplement inference. The original 24 cases are copied unchanged from `inputs/legacy_quality_manifest.json`; their historical `heldout` labels do not imply that they remain unseen. The supplement adds three source images and nine single-reference editing cases (seeds 42, 43, 44). All photographs retain their original bytes; no generated image or edited user attachment is used as a new source photograph.

| Input | Size | Source / permission | Fixed edit |
|---|---|---|---|
| `astronaut.png` | 512 × 512 RGB | [scikit-image v0.19.3](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png), NASA photograph of Eileen Collins; [documented as public domain](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut) | Replace only the background with light-gray studio background; retain face, hair, spacesuit, helmet and pose. |
| `coffee.png` | 600 × 400 RGB | [scikit-image v0.19.3](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/coffee.png), Rachel Michetti / Pikolo Espresso Bar; [documented CC0](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coffee) | Change the brown cup exterior and saucer to cobalt blue; retain interior, coffee, spoon, table and geometry. |
| `synthetic_cafe_sign.png` | 1024 × 1024 RGB | Newly authored **synthetic** sign, CC0-1.0; drawn by `scripts/build_manifest.py` using geometric shapes and Pillow 11.3.0's bundled default font. It is not a model output. | Replace `QWEN CAFE` with `IMAGE CAFE`; retain `OPEN 09:00 - 18:00`, border, colors and layout. |
| `qwen_bear.png` | 514 × 556 RGB | Original archived A0/C1 input, byte-identical. Existing provenance is inherited; no additional license claim is made here. | Original six editing cases retained unchanged. |

## Frozen files

| File | SHA-256 |
|---|---|
| `astronaut.png` | `88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5` |
| `coffee.png` | `cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7` |
| `synthetic_cafe_sign.png` | `b979a34154682a0b00aacf69b02ac22b00e77615f90e2f2ba8274dc2d9848f32` |
| `qwen_bear.png` | `69ac975d5408ebb8de85b1126ad5a37372621e416c4f1a9037cd2ba19b712817` |
| `quality_manifest.json` | `a46def3217fbd8da483d393391a2b45edeb71fb618b33a0209ba5d1cb98e27a0` |
| `prefix_manifest.json` | `525e71b81359cc08c3d4d1575d3df0312623117a1ddb215dccb44ffae4848104` |

`inputs/asset_manifest.json` also records decoded RGB hashes, exact dimensions and source metadata. Use the stored PNGs as the authoritative fixtures; the generated sign does not depend on a font installed on the GPU server.

## Run inputs

- `inputs/quality_manifest.json`: 33 cases (24 original + 9 new). It records exact prompts, seeds and preservation targets. Retain per-case failures, including errors already present in BF16. Its candidate names are `bf16`, `dit_fp8`, `te_fp8`, `joint_fp8`, `selective_img_mlp_bf16`; the last name means all `img_mlp` modules remain BF16, not the old first/last-block experiment.
- `inputs/prefix_manifest.json`: 9 cases, three seeds per workload: short teapot T2I, one-reference coffee edit, two-reference composition using the astronaut first and coffee second. Keep model weights and attention inputs BF16; vary only prefix storage `auto`, `fp8_v`, `fp8`. Record actual prefix token lengths, cache tensor shapes, K/V and scale dtypes, and bytes. Two input images are not a substitute for measuring actual cache length.
- The new entries provide `input_images` as an ordered list of filenames relative to `inputs/`. Single-reference entries additionally provide the old `input_image` field. `expected_text` can be a string or a list; the sign case requires both title and subtitle to be checked.
- All listed cases use 1024 × 1024, 50 steps, CFG 1 and eager execution. If running the separate #8099 prefix check, use its CFG 4 teapot / seed 42 setting and keep that row separate from this CFG 1 table.

The fixed task checks cover requested text, subject preservation and edit completion. A numerical screen such as LPIPS ≤ 0.15 remains provisional until maintainers approve a Qwen-Image-2.1 gate. This input preparation does not claim quality acceptance or GPU execution.

Reproduce / validate the manifests with `python scripts/build_manifest.py` (Pillow 11.3.0 used for the original synthetic sign). The script validates photograph hashes and refuses to overwrite a manifest with different content.

## Effective run manifest

The source manifests above remain unchanged. `inputs/effective_manifest.json` is the authoritative runner input: schema 1, 36 CFG1 cases (33 above plus the three two-reference cases), seven arms `bf16`, `dit_fp8`, `mlp_bf16`, `te_fp8`, `joint_fp8`, `prefix_fp8_v`, `prefix_fp8`. Input paths are frozen under `/workspace/c1-supplement/inputs/`; input hashes are ordered lists. `inputs/cases/t2i.json`, `edit.json`, `multi_edit.json` select teapot seed 42, bear seed 42 and the two-reference seed 42 for performance runs.

The effective manifest **predeclares LPIPS (AlexNet v0.1) ≤ 0.15 per paired image as the experiment working gate** before supplement inference. Its numeric thresholds are explicit, not null. Maintainer approval and human text/subject/edit review remain pending; this distinction is separate from whether the experiment's numerical rule was declared in advance.

`inputs/cfg4_manifest.json` separately defines the teapot seed 42 / CFG4 check for `bf16`, `prefix_fp8_v`, `prefix_fp8`. It explicitly supplies `negative_prompt: ""` to enable CFG in Omni. The upstream recipe does not disclose its full negative prompt, so exact pixel reproduction of unknown upstream input is not claimed. All 36 main cases have `negative_prompt: null` and CFG1; the CFG4 case is not included in their denominator.

| Effective file | SHA-256 |
|---|---|
| `effective_manifest.json` | `8360ab5de58989c9b0ded6f23526b28961c68b22cc7c44bc566c71e4a469c5e3` |
| `cfg4_manifest.json` | `d211bb3891324123987bc4988743bfbd5a9236dbcf0d8bbf32d035ae1747418a` |
| `cases/t2i.json` | `11289f5674c014f04422862aa5b067bb03cf9d6609011215d227cba3ad1e91be` |
| `cases/edit.json` | `a907e4040df6889ddd45515e626b0cbdefa17b40e8d58fe14e4538c19bd375e0` |
| `cases/multi_edit.json` | `7d52898605f1d042ad6f72b4d5422f9bb13d93752366169c3e7f594a1edade03` |
