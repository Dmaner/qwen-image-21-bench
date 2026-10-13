#!/usr/bin/env python3
"""Freeze C1 supplement inputs; no model execution and no network requests.

Run from any directory with Python + Pillow. Existing photograph bytes and the
legacy 24 cases are preserved. The generated sign is a disclosed synthetic input,
not an inference output. Committed PNG/JSON files are the authoritative fixtures.
"""
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
SEEDS = (42, 43, 44)
PHOTO_HASHES = {
    "astronaut.png": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    "coffee.png": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    "qwen_bear.png": "69ac975d5408ebb8de85b1126ad5a37372621e416c4f1a9037cd2ba19b712817",
}
SOURCES = {
    "astronaut.png": {
        "source_url": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png",
        "license": "Public domain (NASA; as documented by scikit-image)",
        "license_source": "https://scikit-image.org/docs/0.19.x/api/skimage.data.html#skimage.data.astronaut",
        "description": "Unmodified scikit-image photograph of astronaut Eileen Collins; person editing fixture.",
        "transformation": "None: original PNG bytes copied directly.",
    },
    "coffee.png": {
        "source_url": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/coffee.png",
        "license": "CC0, photographer Rachel Michetti; courtesy of Pikolo Espresso Bar",
        "license_source": "https://scikit-image.org/docs/0.19.x/api/skimage.data.html#skimage.data.coffee",
        "description": "Unmodified photograph of an espresso cup, saucer and spoon; physical-object editing fixture.",
        "transformation": "None: original PNG bytes copied directly.",
    },
    "qwen_bear.png": {
        "source_url": "https://github.com/luca-888/qwen-image-21-bench/blob/main/inputs/qwen_bear.png",
        "license": "Inherited A0/C1 input; no new independent license claim",
        "description": "Byte-identical existing A0/C1 input, retained only to reproduce the legacy cases.",
        "transformation": "None: original archived C1 PNG bytes copied directly.",
    },
    "synthetic_cafe_sign.png": {
        "source_url": "Local reproducible construction: scripts/build_manifest.py",
        "license": "CC0-1.0 dedication for this newly authored synthetic test fixture",
        "description": "Synthetic flat sign: QWEN CAFE and OPEN 09:00 - 18:00; not a model output or a photograph.",
        "transformation": "Drawn from geometric shapes and Pillow 11.3.0's bundled default font (Aileron); no external image was altered.",
    },
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    encoded = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode()
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"Refusing to change an existing frozen manifest: {path}")
    path.write_bytes(encoded)


def draw_sign():
    path = INPUTS / "synthetic_cafe_sign.png"
    if path.exists():
        return
    image = Image.new("RGB", (1024, 1024), (227, 230, 228))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((72, 168, 952, 856), radius=28, fill=(15, 67, 66))
    draw.rectangle((104, 200, 920, 824), fill=(251, 248, 231))
    for label, y, size in (("QWEN CAFE", 354, 96), ("OPEN 09:00 - 18:00", 572, 48)):
        font = ImageFont.load_default(size=size)
        bounds = font.getbbox(label)
        mask = Image.new("L", (bounds[2]-bounds[0], bounds[3]-bounds[1]), 0)
        ImageDraw.Draw(mask).text((-bounds[0], -bounds[1]), label, font=font, fill=255)
        image.paste((15, 67, 66), ((1024-mask.width)//2, y), mask)
    image.save(path)


def make_cases(group, prompt, files, targets, expected_text=None):
    assets = [SOURCES[f] for f in files]
    return [dict(
        id=f"{group}_s{seed}", group=group,
        task="edit" if files else "t2i", prompt=prompt, seed=seed,
        width=1024, height=1024, num_inference_steps=50, true_cfg_scale=1.0,
        input_image=files[0] if len(files) == 1 else None,
        input_image_sha256=assets[0]["png_sha256"] if len(files) == 1 else None,
        input_images=files, input_images_sha256=[a["png_sha256"] for a in assets],
        expected_text=expected_text, split="supplement_frozen_20261013",
        targets=targets,
    ) for seed in SEEDS]


def build_effective_manifest():
    """Adapt source manifests to the frozen schema-1 GPU runner contract."""
    source_names = ("quality_manifest.json", "prefix_manifest.json", "asset_manifest.json")
    source_hashes = {name:digest(INPUTS / name) for name in source_names}
    quality = json.loads((INPUTS / source_names[0]).read_text())
    prefix = json.loads((INPUTS / source_names[1]).read_text())
    cases = copy.deepcopy(quality["cases"])
    cases += copy.deepcopy([c for c in prefix["cases"] if c["group"] == "prefix_two_ref"])
    assert len(cases) == 36
    remote_dir = "/workspace/c1-supplement/inputs/"
    for case in cases:
        filenames = case.pop("input_images", None)
        if filenames is None:
            original = case.get("input_image")
            filenames = original if isinstance(original, list) else [original] if original else []
        case.pop("input_images_sha256", None)
        paths = [remote_dir + Path(filename).name for filename in filenames]
        case["input_image"] = paths[0] if len(paths) == 1 else paths or None
        case["input_image_sha256"] = [SOURCES[Path(filename).name]["png_sha256"] for filename in filenames]
        case["negative_prompt"] = None

    effective = copy.deepcopy(quality)
    effective.update(schema_version=1, expected_cases=36, cases=cases,
        scope="C1 supplement: original 24 cases, nine new single-reference edits, and three two-reference edits. All 36 cases use CFG1.",
        expected_arms=["bf16", "dit_fp8", "mlp_bf16", "te_fp8", "joint_fp8", "prefix_fp8_v", "prefix_fp8"],
        source_manifest_sha256=source_hashes,
        frozen_at_utc="2026-10-13T13:20:15Z",
        frozen_before="Any supplement model inference",
        remote_inputs_directory=remote_dir.rstrip("/"),
        source_manifest_policy="Source manifests remain byte-identical. This effective schema-1 manifest is authoritative for the supplement run and its predeclared working quality screen.",
        multi_reference_schema="input_image is null, one absolute path string, or an ordered list of absolute paths; input_image_sha256 is always a matching ordered list.")
    effective["dataset_protocol"]["expected_pairs_per_candidate"] = 36
    effective["dataset_protocol"]["supplement_two_reference"] = {"cases":3,"seeds":list(SEEDS),"role":"New two-reference composition with ordered astronaut and coffee inputs"}
    metric = effective["metric_protocol"]
    metric["numerical_quality_thresholds"] = {
        "lpips": {"operator":"<=", "maximum":0.15, "network":"alex", "version":"0.1", "package":"lpips==0.1.4",
                  "input":"full-resolution RGB normalized to [-1,1]; no resizing, cropping or alignment", "application":"Each candidate image compared with same-run, same-case BF16 output"}
    }
    metric["threshold_status"] = "Predeclared experiment working gate before supplement inference: per-case LPIPS (AlexNet v0.1) <= 0.15. Qwen-Image-2.1 maintainer approval remains pending; human requested-text/subject/edit review remains pending."
    effective["quality_gate"] = {
        "predeclared_at_utc":"2026-10-13T13:20:15Z",
        "status":"predeclared_working_gate",
        "numeric_rule":"LPIPS (AlexNet v0.1) <= 0.15 for each paired image",
        "aggregation":"Report numerator/denominator, every failing case, category counts and worst case; do not replace per-case gating with an average.",
        "maintainer_approval":"pending",
        "human_review":"pending",
        "missing_output":"fail; retain the case in the denominator",
        "human_rule":"Check requested text exactly, including both sign strings; check subject preservation and requested edit completion. Distinguish existing BF16 errors from newly introduced candidate errors.",
        "formal_model_acceptance":False,
    }
    effective["qualitative_gate_protocol"]["status"] = "Predeclared operational checks; human review pending and maintainer approval pending"
    write_json(INPUTS / "effective_manifest.json", effective)
    cfg4 = copy.deepcopy(effective)
    case = copy.deepcopy(next(c for c in cases if c["id"] == "t2i_teapot_s42"))
    case.update(id="cfg4_teapot_s42", group="cfg4_upstream_prefix", true_cfg_scale=4.0, negative_prompt="",
                split="separate_cfg4_upstream_replication")
    cfg4.update(scope="Separate #8099 prefix-storage replication: teapot, seed42,1024x1024,50steps,CFG4,eager. Do not pool with the 36-case CFG1 quality set.",
                expected_cases=1, expected_arms=["bf16","prefix_fp8_v","prefix_fp8"], cases=[case],
                source_effective_manifest_sha256=digest(INPUTS / "effective_manifest.json"),
                negative_prompt_protocol="Empty-string negative_prompt is explicitly supplied so Omni enables CFG. The upstream recipe does not specify its full negative prompt; this implementation setting is disclosed and exact pixel reproduction of unknown upstream input is not claimed.",
                dataset_protocol={"expected_pairs_per_candidate":1,"comparison":"Separate same-session BF16 versus prefix-only candidates at CFG4; not part of the CFG1 36-case denominator."})
    write_json(INPUTS / "cfg4_manifest.json", cfg4)
    cases_dir = INPUTS / "cases"
    cases_dir.mkdir(exist_ok=True)
    for filename, case_id in (("t2i.json","t2i_teapot_s42"), ("edit.json","edit_bear_s42"), ("multi_edit.json","prefix_two_ref_s42")):
        write_json(cases_dir / filename, next(c for c in cases if c["id"] == case_id))


def main():
    for filename, expected in PHOTO_HASHES.items():
        assert digest(INPUTS / filename) == expected, filename
    draw_sign()
    for filename, metadata in SOURCES.items():
        image = Image.open(INPUTS / filename)
        metadata.update(png_sha256=digest(INPUTS / filename),
                        rgb_sha256=hashlib.sha256(image.convert("RGB").tobytes()).hexdigest(),
                        size=list(image.size), mode=image.mode)

    legacy_path = INPUTS / "legacy_quality_manifest.json"
    legacy = json.loads(legacy_path.read_text())
    original_cases = copy.deepcopy(legacy["cases"])
    assert len(original_cases) == 24
    added = []
    added += make_cases("edit_person_background", "Replace only the background with a plain light-gray studio backdrop. Keep the astronaut's face, hairstyle, facial expression, orange spacesuit, patches, helmet, body pose and framing unchanged.",
                        ["astronaut.png"], ["plain light-gray background", "same face, hair and expression", "same orange spacesuit and patches", "same helmet, pose and framing"])
    added += make_cases("edit_product_color", "Change only the brown exterior of the espresso cup and the brown saucer to cobalt blue. Keep the white cup interior, coffee, metal spoon, wooden table, lighting, object shapes and camera angle unchanged.",
                        ["coffee.png"], ["cobalt-blue cup exterior and saucer", "same white cup interior and coffee", "same metal spoon and wooden table", "same shapes, lighting and camera angle"])
    added += make_cases("edit_sign_text", 'Replace only the large title "QWEN CAFE" with the exact text "IMAGE CAFE". Keep the smaller text "OPEN 09:00 - 18:00" unchanged. Preserve the sign border, colors, layout and background; add no other text or objects.',
                        ["synthetic_cafe_sign.png"], ["exact title IMAGE CAFE", "exact unchanged subtitle OPEN 09:00 - 18:00", "unchanged border, colors, layout and background", "no extra text or objects"],
                        ["IMAGE CAFE", "OPEN 09:00 - 18:00"])

    quality = copy.deepcopy(legacy)
    quality.update(schema_version=2, expected_cases=33,
                   scope="C1 supplement: original 24 cases unchanged plus nine newly frozen single-reference editing cases.",
                   expected_arms=["bf16", "dit_fp8", "te_fp8", "joint_fp8", "selective_img_mlp_bf16"],
                   source_legacy_manifest_sha256=digest(legacy_path),
                   input_assets=SOURCES, cases=original_cases + added,
                   case_history="Legacy and heldout labels on the original 24 cases are historical; those cases are no longer unseen. The nine supplement cases were fixed before supplement inference.")
    quality["dataset_protocol"].update(expected_pairs_per_candidate=33,
        supplement_frozen_20261013={"role":"three new source images, one fixed edit per image, three seeds", "cases":9,"seeds":list(SEEDS)})
    quality["metric_protocol"]["threshold_status"] = "Maintainer-approved Qwen-Image-2.1 thresholds remain unconfirmed. Report per-case metrics and human requested-text/subject/edit checks; any LPIPS<=0.15 screen is provisional, not model acceptance."
    quality["qualitative_gate_protocol"]["subject_and_edit"] = "Preserve each case's specified subject and unchanged regions; complete the requested edit. Review every paired image and record pre-existing BF16 failures separately."
    quality["expected_text_schema"] = "null, one string, or a list of strings that must all be retained exactly"
    quality["multi_reference_schema"] = "For new cases, input_images is an ordered list relative to this inputs directory. Single-reference cases also populate legacy input_image."
    write_json(INPUTS / "quality_manifest.json", quality)

    prefix_cases = make_cases("prefix_short_t2i", "A ceramic teapot on a wooden table", [], ["ceramic teapot", "wooden table", "coherent spout/handle/lid"])
    prefix_cases += copy.deepcopy([case for case in added if case["group"] == "edit_product_color"])
    for case in prefix_cases[3:]:
        case["id"] = case["id"].replace("edit_product_color", "prefix_one_ref")
        case["group"] = "prefix_one_ref"
    prefix_cases += make_cases("prefix_two_ref", "Create a realistic studio photograph using both reference images. Use the astronaut in the first image as the person: preserve her face, hairstyle, orange spacesuit and patches. Place the espresso cup, saucer and spoon from the second image on a small wooden table in front of her. Preserve the cup's brown exterior, white interior and coffee, and the spoon's metallic appearance. Use a plain light-gray background. Keep the person and the cup clearly visible; add no other people, cups or text.",
        ["astronaut.png", "coffee.png"], ["person from first reference with preserved face, hair and orange spacesuit", "cup, saucer, spoon and coffee from second reference", "cup set on small wooden table in front of person", "light-gray background and no extra people, cups or text"])
    prefix = {key:copy.deepcopy(quality[key]) for key in ("schema_version", "code_revision", "model", "model_revision", "controls", "metric_protocol", "qualitative_gate_protocol", "input_assets", "hardware_target")}
    prefix.update(scope="C1 prefix-storage supplement: BF16 weights and attention inputs; only stored prefix KV dtype changes.", expected_cases=9,
        expected_arms=["prefix_auto", "prefix_fp8_v", "prefix_fp8"],
        cases=prefix_cases, warmup_per_arm=1,
        controls_override={"weights":"BF16", "attention_input_dtype":"bfloat16", "prefix_kv_cache_dtype":"auto | fp8_v | fp8"},
        cache_measurement="Record actual K/V shapes, prefix token lengths, per-layer stored dtype, scale dtype/shapes and summed storage bytes. Two references are a workload choice; do not infer actual cache length without recording it.",
        comparison="Compare storage candidates against the same-run auto result within each case, not absolute latency across different workloads.",
        upstream_cfg4_crosscheck="Optional separate #8099-matched teapot seed42,1024^2,50steps,true_cfg_scale=4.0,eager. Do not pool its scores with the CFG1 nine-case table.")
    write_json(INPUTS / "prefix_manifest.json", prefix)
    write_json(INPUTS / "asset_manifest.json", SOURCES)
    build_effective_manifest()
    print(json.dumps({"quality_cases":len(quality["cases"]),"prefix_cases":len(prefix_cases),"quality_manifest_sha256":digest(INPUTS/"quality_manifest.json"),"prefix_manifest_sha256":digest(INPUTS/"prefix_manifest.json"),"assets":{name:meta["png_sha256"] for name,meta in SOURCES.items()}},indent=2))


if __name__ == "__main__":
    main()
