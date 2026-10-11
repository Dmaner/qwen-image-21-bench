#!/usr/bin/env python3
"""CPU-only C1 Phase2 BF16/candidate quality analysis; 24 frozen cases.

Every manifest case is retained, including missing, duplicate, failed, or mismatched
records. Metrics use the decoded original RGB pixels; only contact sheets resize.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import io
import json
import math
from pathlib import Path
import platform
import statistics

from PIL import Image, ImageChops, ImageDraw, ImageStat


SSIM_SETTINGS = dict(data_range=255, gaussian_weights=True, sigma=1.5,
                     use_sample_covariance=False, win_size=11, channel_axis=-1)
FIELDS = ("group", "split", "task", "prompt", "seed", "width", "height", "num_inference_steps", "true_cfg_scale")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def hashes(value):
    return value if isinstance(value, list) else [value] if value else []


def load_run(directory):
    path = directory / "runs.jsonl"
    result = {"directory": str(directory.resolve()), "runs_path": str(path),
              "runs_sha256": sha(path) if path.is_file() else None,
              "parse_errors": [], "phase_counts": {}, "non_quality_failures": [],
              "env": None, "env_error": None}
    by_id, phases = defaultdict(list), Counter()
    if not path.is_file():
        result["parse_errors"].append({"error": "runs.jsonl missing"})
    else:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                phases[str(row.get("phase"))] += 1
                if row.get("phase") == "quality":
                    cid = row.get("case_id") or row.get("id")
                    by_id[str(cid)].append({"line": number, "row": row})
                elif row.get("error"):
                    result["non_quality_failures"].append({"line": number, "row": row})
            except (ValueError, TypeError) as exc:
                result["parse_errors"].append({"line": number, "error": str(exc)})
    result["phase_counts"] = dict(phases)
    env_path = directory / "env.json"
    if env_path.is_file():
        result["env_sha256"] = sha(env_path)
        try:
            result["env"] = json.loads(env_path.read_text())
            if not isinstance(result["env"], dict):
                result["env"] = None
                result["env_error"] = "env.json must be a JSON object"
        except ValueError as exc:
            result["env_error"] = str(exc)
    else:
        result["env_error"] = "env.json missing; runtime/source evidence not observed"
    return result, by_id


def image_info(path):
    with Image.open(path) as source:
        original_mode = source.mode
        rgb = source.convert("RGB")
        rgb.load()
    extrema = rgb.getextrema()
    return {"path": str(path.resolve()), "png_sha256": sha(path),
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "size": list(rgb.size), "original_mode": original_mode,
            "decoded_mode": "RGB", "dtype": "uint8", "rgb_extrema": extrema,
            "rgb_mean": ImageStat.Stat(rgb).mean,
            "exactly_black": all(high == 0 for _, high in extrema),
            "constant_rgb": all(low == high for low, high in extrema),
            "tensor_nonfinite_check": "not_observable_from_decoded_uint8_png"}, rgb


def inspect_arm(case, arm, directory, records):
    item = {"arm": arm, "record_count": len(records), "status": "missing_record",
            "issues": [], "image": None, "record": None}
    if not records:
        return item, None
    if len(records) != 1:
        item.update(status="duplicate_records", records=records)
        return item, None
    row = records[0]["row"]
    item.update(record=row, line=records[0]["line"], status="ok")
    for key in FIELDS:
        if key not in row:
            item["issues"].append(f"missing metadata: {key}")
        elif row[key] != case[key]:
            item["issues"].append(f"metadata mismatch: {key}")
    if row.get("arm") != arm:
        item["issues"].append("metadata mismatch: arm")
    if "input_image_sha256" not in row:
        item["issues"].append("missing metadata: input_image_sha256")
    elif hashes(row["input_image_sha256"]) != hashes(case.get("input_image_sha256")):
        item["issues"].append("metadata mismatch: input_image_sha256")
    if row.get("id") != case["id"] or row.get("case_id") != case["id"]:
        item["issues"].append("metadata mismatch: id/case_id")
    if row.get("error"):
        item["status"] = "inference_error"
    elif item["issues"]:
        item["status"] = "metadata_mismatch"
    # Copied remote absolute output paths are never followed outside the supplied run.
    expected = directory / "images" / (case["id"] + ".png")
    item["recorded_output_path"] = row.get("output_path")
    if not expected.is_file():
        item["issues"].append("output PNG missing")
        if item["status"] == "ok":
            item["status"] = "missing_image"
        return item, None
    try:
        info, rgb = image_info(expected)
        item["image"] = info
    except Exception as exc:
        item["issues"].append(f"image decode: {type(exc).__name__}: {exc}")
        if item["status"] == "ok":
            item["status"] = "image_decode_error"
        return item, None
    if info["size"] != [case["width"], case["height"]]:
        item["issues"].append("image dimensions differ from manifest")
    if row.get("image_sha256") and info["png_sha256"] != row["image_sha256"]:
        item["issues"].append("PNG hash differs from recorded image_sha256")
    if not row.get("image_sha256"):
        item["issues"].append("missing recorded image_sha256")
    if not row.get("output_path"):
        item["issues"].append("record has no output_path although local PNG exists")
    if item["status"] == "ok" and item["issues"]:
        item["status"] = "output_mismatch"
    return item, rgb


def compare_pixels(a, b):
    if a.size != b.size:
        return {"status": "shape_mismatch"}
    histogram = ImageChops.difference(a, b).histogram()
    count = a.width * a.height * 3
    mse = sum((i % 256) ** 2 * n for i, n in enumerate(histogram)) / count
    mae = sum((i % 256) * n for i, n in enumerate(histogram)) / count
    result = {"status": "computed", "rgb_equal": mse == 0, "rgb_mse": mse,
              "rgb_mae": mae, "psnr_db": "Infinity" if mse == 0 else 10 * math.log10(255 ** 2 / mse),
              "ssim": None, "ssim_status": None}
    try:
        import numpy as np
        from skimage.metrics import structural_similarity
        result["ssim"] = float(structural_similarity(np.asarray(a, dtype=np.uint8),
                                                     np.asarray(b, dtype=np.uint8), **SSIM_SETTINGS))
        result["ssim_status"] = "computed"
    except ImportError as exc:
        result["ssim_status"] = "unavailable_dependency"
        result["ssim_error"] = str(exc)
    except Exception as exc:
        result["ssim_status"] = "computation_error"
        result["ssim_error"] = f"{type(exc).__name__}: {exc}"
    return result


def summarise(pairs, candidate_arm):
    metrics = [p["metrics"] for p in pairs if p["status"] == "paired"]
    psnr = [m["psnr_db"] for m in metrics]
    finite = [x for x in psnr if isinstance(x, (int, float))]
    ssim = [m["ssim"] for m in metrics if m["ssim"] is not None]
    return {"expected_pairs": len(pairs), "paired": len(metrics),
            "pair_status_counts": dict(Counter(p["status"] for p in pairs)),
            "arm_status_counts": {arm: dict(Counter(p[arm]["status"] for p in pairs))
                                  for arm in ("bf16", candidate_arm)},
            "exact_black_cases": {arm: [p["case_id"] for p in pairs if (p[arm].get("image") or {}).get("exactly_black")]
                                  for arm in ("bf16", candidate_arm)},
            "rgb_equal_pairs": sum(m["rgb_equal"] for m in metrics),
            "mean_rgb_mse": statistics.mean(m["rgb_mse"] for m in metrics) if metrics else None,
            "mean_psnr_db": ("Infinity" if "Infinity" in psnr else statistics.mean(psnr)) if psnr else None,
            "psnr_n": len(psnr), "psnr_infinite_n": psnr.count("Infinity"),
            "finite_only_mean_psnr_db": statistics.mean(finite) if finite else None,
            "finite_only_psnr_n": len(finite),
            "mean_ssim": statistics.mean(ssim) if ssim else None, "ssim_n": len(ssim),
            "numeric_quality_gate": "not_defined_exploratory_only"}


def gpu_rows(env):
    """Parse the harness' nvidia-smi CSV, preserving UUID rather than model alone."""
    raw = env.get("nvidia_smi")
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        return [{str(k).strip(): str(v).strip() for k, v in row.items()}
                for row in csv.DictReader(io.StringIO(raw))]
    except (csv.Error, TypeError):
        return []


def env_audit(runs, manifest, candidate_arm):
    audit = {"evidence_boundary": "Manifest is intended protocol, env is requested/configuration evidence; actual kernels require independent worker audit/trace.",
             "arms": {}, "cross_arm_differences": {}, "missing_cross_arm_fields": [],
             "gpu_identity": {}, "comparison_validity": "unassessed"}
    for arm, run in runs.items():
        env = run.get("env") or {}
        checks = {}
        for key, expected in (("vllm_omni_sha", manifest["code_revision"]),
                              ("vllm_omni_checkout", manifest["code_revision"]),
                              ("model_revision", manifest["model_revision"]),
                              ("arm", arm), ("quality_manifest_mode", True)):
            checks[key] = {"expected": expected, "observed": env.get(key),
                           "status": "unobserved" if key not in env else "match" if env[key] == expected else "mismatch"}
        runtime = env.get("runtime_config") or {}
        for key, expected in (("enforce_eager", True), ("dtype", "bfloat16"), ("enable_cuda_graph_decode", False)):
            checks[key] = {"expected": expected, "observed": runtime.get(key),
                           "status": "unobserved" if key not in runtime else "match" if runtime[key] == expected else "mismatch"}
        for key in ("prefix_kv_cache_dtype", "diffusion_kv_cache_dtype"):
            checks[key] = {"expected": "native default (None/auto)", "observed": runtime.get(key),
                           "status": "unobserved" if key not in runtime else "match" if runtime[key] in (None, "auto") else "mismatch"}
        checks["source_dirty"] = {"expected": "clean checkout or separately reviewed changes", "observed": env.get("vllm_omni_dirty"),
                                  "status": "unobserved" if "vllm_omni_dirty" not in env else "match" if env["vllm_omni_dirty"] == "" else "requires_source_review"}
        audit["arms"][arm] = {"checks": checks, "runtime_config": runtime,
                              "workload": env.get("workload"), "source_checkout": env.get("vllm_omni_checkout"),
                              "source_dirty": env.get("vllm_omni_dirty")}
        gpu = gpu_rows(env)
        audit["gpu_identity"][arm] = {"devices": gpu, "h200_observed": bool(gpu) and all("H200" in r.get("name", "") for r in gpu),
                                       "uuids": [r.get("uuid") for r in gpu if r.get("uuid")]}
    left, right = (runs[a].get("env") or {} for a in ("bf16", candidate_arm))
    for key in ("vllm_omni_sha", "vllm_omni_checkout", "vllm_omni_dirty", "model_revision",
                "packages", "torch_cuda", "nvidia_smi", "hostname", "harness_sha256"):
        if key not in left or key not in right:
            audit["missing_cross_arm_fields"].append(key)
        elif left[key] != right[key]:
            audit["cross_arm_differences"][key] = {"bf16": left[key], candidate_arm: right[key]}
    runtime_left = {k: v for k, v in (left.get("runtime_config") or {}).items() if k != "quantization_config"}
    runtime_right = {k: v for k, v in (right.get("runtime_config") or {}).items() if k != "quantization_config"}
    if runtime_left != runtime_right:
        audit["cross_arm_differences"]["runtime_config_excluding_quantization"] = {"bf16": runtime_left, candidate_arm: runtime_right}
    guids = [audit["gpu_identity"][a]["uuids"] for a in ("bf16", candidate_arm)]
    audit["same_physical_gpu"] = "match" if all(guids) and guids[0] == guids[1] else "mismatch" if all(guids) else "unobserved"
    checks = [c for arm in audit["arms"].values() for c in arm["checks"].values()]
    if audit["cross_arm_differences"] or audit["same_physical_gpu"] == "mismatch" or any(c["status"] in {"mismatch", "requires_source_review"} for c in checks):
        audit["comparison_validity"] = "configuration_mismatch_requires_resolution"
    elif audit["missing_cross_arm_fields"] or audit["same_physical_gpu"] == "unobserved" or any(c["status"] != "match" for c in checks):
        audit["comparison_validity"] = "incomplete_configuration_evidence"
    elif not all(g["h200_observed"] for g in audit["gpu_identity"].values()):
        audit["comparison_validity"] = "requested_h200_not_observed"
    else:
        audit["comparison_validity"] = "recorded_controls_match_except_intended_quantization"
    return audit


def manual_template(path, pairs, candidate_arm):
    # Never replace human annotations on re-analysis.
    if path.exists():
        return "existing_manual_review_preserved"
    fields = ["case_id", "split", "task", "expected_text", "candidate_arm", "targets",
              "bf16_text_exact", "candidate_text_exact", "bf16_subject_ok", "candidate_subject_ok",
              "bf16_edit_goal_ok", "candidate_edit_goal_ok", "new_requested_text_error",
              "new_subject_or_edit_error", "bf16_existing_issues", "candidate_new_issues",
              "review_type", "reviewer", "notes", "review_status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({"case_id": pair["case_id"], "split": pair["split"], "task": pair["task"],
                             "candidate_arm": candidate_arm, "targets": "; ".join(pair["targets"]),
                             "expected_text": pair.get("expected_text") or "", "review_status": "pending"})
    return "created_pending_template"


def contact_sheets(directory, pairs, candidate_arm):
    directory.mkdir(exist_ok=True)
    groups, outputs = defaultdict(list), []
    for pair in pairs:
        groups[pair["group"]].append(pair)
    for group, group_pairs in groups.items():
        thumb, gap, header, rowheight = 448, 16, 48, 508
        canvas = Image.new("RGB", (thumb * 2 + gap * 3, header + rowheight * len(group_pairs)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((gap, 10), f"{group} | {group_pairs[0]['split']} | BF16 left / {candidate_arm} right | display only", fill="black")
        for row_index, pair in enumerate(group_pairs):
            y = header + rowheight * row_index
            for column, arm in enumerate(("bf16", candidate_arm)):
                x = gap + column * (thumb + gap)
                draw.text((x, y), f"{arm} seed {pair['seed']} | {pair[arm]['status']}", fill="black")
                info = pair[arm].get("image")
                if info:
                    with Image.open(info["path"]) as original:
                        display = original.convert("RGB")
                        display.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
                        canvas.paste(display, (x + (thumb-display.width)//2, y+24 + (thumb-display.height)//2))
                else:
                    draw.rectangle((x, y + 24, x + thumb, y + 24 + thumb), fill="#eeeeee")
                    draw.text((x + 15, y + 50), "MISSING / FAILED: see JSON record", fill="black")
            metric = pair.get("metrics") or {}
            draw.text((gap, y + 478), f"PSNR={metric.get('psnr_db')} dB; SSIM={metric.get('ssim')} | {pair['status']}", fill="black")
        output = directory / (group + ".png")
        canvas.save(output)
        outputs.append({"group": group, "path": str(output.resolve()), "sha256": sha(output),
                        "note": "448px display thumbnails; no metric uses these resized pixels"})
    return outputs


def report(path, result):
    s = result["summary"]
    candidate = result["candidate_arm"]
    lines = ["# C1 Phase 2 exploratory quality", "",
             f"同一协议的 BF16 / {candidate} 图像配对；数字容差尚未确认。定性检查覆盖文字新增错误、主体/编辑目标、失败和黑图；24 个样本不能认证全局质量。", "",
             f"图像可配对 {s['paired']}/{s['expected_pairs']}；RGB 相等 {s['rgb_equal_pairs']}；PSNR 均值 {s['mean_psnr_db']} dB (n={s['psnr_n']})；SSIM 均值 {s['mean_ssim']} (n={s['ssim_n']})。", "",
             f"配置比较状态：`{result['env_audit']['comparison_validity']}`。配置不一致或证据不全时，像素指标仅描述这些图片，不能归因于候选量化方案。", "",
             f"定性检查状态：`{result['qualitative_audit']['status']}`；不表示正式质量通过。", "",
             "PSNR 来自全部原尺寸 RGB uint8 像素的 MSE，data_range=255；完全相同记为 Infinity。SSIM 使用 Gaussian、sigma=1.5、win_size=11、population covariance、channel_axis=-1。缺库或计算失败保留 null；不得把不同有效样本数的均值视为完整配对。", "",
             "| split | n paired/expected | PSNR mean dB | SSIM mean |", "|---|---:|---:|---:|"]
    for split, data in result["splits"].items():
        lines.append(f"| {split} | {data['paired']}/{data['expected_pairs']} | {data['mean_psnr_db']} | {data['mean_ssim']} |")
    lines += ["", "legacy 已在 Phase1 看过；heldout 为本轮预先冻结的新输入/seed。须分组报告，查看后不能再次称为未见过的验证集。", "",
              f"| case | split | BF16 | {candidate} | PSNR dB | SSIM |", "|---|---|---|---|---:|---:|"]
    for p in result["pairs"]:
        m = p.get("metrics") or {}
        lines.append(f"| {p['case_id']} | {p['split']} | {p['bf16']['status']} | {p[candidate]['status']} | {m.get('psnr_db')} | {m.get('ssim')} |")
    lines += ["", "所有失败、缺失、重复、元数据不一致的 case 均保留在 quality_metrics.json；不从分母静默剔除。", "",
              "人工逐图文字准确性、主体和编辑目标检查见 human_review.csv，尚未填写不等于通过。BF16 自身错误也要记录。PNG 黑图检查只能检测全零 RGB，uint8 PNG 不能证明上游浮点张量没有 NaN/Inf。", "",
              "contact_sheets 仅用于并排人工观察，显示缩小不参与指标，原 PNG 不被修改。性能不从本批质量运行推断。", "",
              "来源与运行配置核验见 JSON env_audit：manifest 是预期协议，env 是请求/配置证据；FA3/SM90、实际 dtype 与量化覆盖需独立 worker audit/trace。", "",
              "SSIM 实现版本：" + json.dumps(result["implementation"], ensure_ascii=False), ""]
    path.write_text("\n".join(lines))


def qualitative_audit(pairs, candidate_arm, review_path=None):
    """Operational failures are objective; text/identity require explicit visual records."""
    result = {"status": "pending_visual_review", "new_observed_issues": [],
              "bf16_existing_or_shared_issues": [], "incomplete_pairs": [],
              "visual_review": {"source": str(review_path) if review_path else None,
                                "rows": [], "errors": [], "pending_case_ids": []},
              "numeric_threshold": None, "formal_quality_pass": False}
    for pair in pairs:
        cid = pair["case_id"]
        left, right = pair["bf16"], pair[candidate_arm]
        if pair["status"] != "paired":
            result["incomplete_pairs"].append(cid)
        for arm, side in (("bf16", left), (candidate_arm, right)):
            kinds = []
            if side["status"] == "inference_error":
                kinds.append("inference_error")
            if (side.get("image") or {}).get("exactly_black"):
                kinds.append("exact_all_zero_rgb")
            for kind in kinds:
                record = {"case_id": cid, "split": pair["split"], "arm": arm, "kind": kind}
                if arm == candidate_arm and left["status"] == "ok" and not (left.get("image") or {}).get("exactly_black"):
                    result["new_observed_issues"].append(record)
                else:
                    result["bf16_existing_or_shared_issues"].append(record)
    rows = defaultdict(list)
    if review_path:
        try:
            with review_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    rows[row.get("case_id", "")].append(row)
            result["visual_review"]["sha256"] = sha(review_path)
        except (OSError, csv.Error) as exc:
            result["visual_review"]["errors"].append(str(exc))
    expected_ids = {p["case_id"] for p in pairs}
    for cid in rows:
        if cid not in expected_ids:
            result["visual_review"]["errors"].append(f"unexpected visual case: {cid}")
    valid_flags = {"yes", "no", "ambiguous", "not_applicable"}
    for pair in pairs:
        cid = pair["case_id"]
        entries = rows.get(cid, [])
        if len(entries) != 1:
            result["visual_review"]["pending_case_ids"].append(cid)
            if len(entries) > 1:
                result["visual_review"]["errors"].append(f"duplicate visual case: {cid}")
            continue
        row = entries[0]
        result["visual_review"]["rows"].append(row)
        flags = [row.get(key, "") for key in ("new_requested_text_error", "new_subject_or_edit_error")]
        complete = (row.get("candidate_arm") == candidate_arm and row.get("split") == pair["split"]
                    and row.get("review_status") == "completed" and bool(row.get("reviewer"))
                    and row.get("review_type") in {"model_assisted", "human"}
                    and all(flag in valid_flags for flag in flags))
        if not complete or "ambiguous" in flags:
            result["visual_review"]["pending_case_ids"].append(cid)
        if complete:
            for key in ("new_requested_text_error", "new_subject_or_edit_error"):
                if row[key] == "yes":
                    result["new_observed_issues"].append({"case_id": cid, "split": pair["split"],
                                                         "arm": candidate_arm, "kind": key,
                                                         "review_type": row["review_type"], "notes": row.get("candidate_new_issues", "")})
        if row.get("bf16_existing_issues"):
            result["bf16_existing_or_shared_issues"].append({"case_id": cid, "kind": "visual_observation", "notes": row["bf16_existing_issues"]})
    if result["new_observed_issues"]:
        result["status"] = "observed_qualitative_violation"
    elif result["incomplete_pairs"] or result["visual_review"]["pending_case_ids"] or result["visual_review"]["errors"]:
        result["status"] = "incomplete_or_pending_visual_review"
    else:
        result["status"] = "no_new_violation_observed_in_reviewed_samples_not_certification"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "scripts/quality_manifest.json")
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", "--fp8-dir", dest="candidate_dir", type=Path, required=True)
    parser.add_argument("--candidate-arm", required=True, help="Raw row/env arm name, e.g. te_fp8 or selective_bf16")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--visual-review-csv", type=Path, help="Optional explicitly completed review, never inferred from PSNR/SSIM")
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"]
    if manifest.get("expected_cases") != 24 or len(cases) != 24:
        parser.error("Phase2 requires the frozen 24-case manifest")
    if not args.candidate_arm or args.candidate_arm == "bf16":
        parser.error("candidate arm must be a nonempty non-bf16 raw arm name")
    if Counter(c.get("split") for c in cases) != {"legacy": 12, "heldout": 12}:
        parser.error("Phase2 requires exactly 12 legacy and 12 heldout cases")
    ids = [c["id"] for c in cases]
    stems = ids + [c["group"] for c in cases]
    if len(ids) != len(set(ids)) or any(not cid or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in cid) for cid in stems):
        parser.error("manifest IDs must be unique and IDs/groups must be safe filename stems")
    candidate = args.candidate_arm
    dirs = {"bf16": args.bf16_dir, candidate: args.candidate_dir}
    runs, indices = {}, {}
    for arm, directory in dirs.items():
        runs[arm], indices[arm] = load_run(directory)
        runs[arm]["unexpected_quality_ids"] = {key: rows for key, rows in indices[arm].items() if key not in ids}
    pairs = []
    for case in cases:
        pair = {"case_id": case["id"], "group": case["group"], "task": case["task"],
                "split": case["split"], "targets": case.get("targets", []),
                "seed": case["seed"], "prompt": case["prompt"], "expected_text": case.get("expected_text"),
                "status": "unpaired", "metrics": None}
        pixels = {}
        for arm, directory in dirs.items():
            pair[arm], pixels[arm] = inspect_arm(case, arm, directory, indices[arm].get(case["id"], []))
        if all(pair[arm]["status"] == "ok" for arm in dirs):
            pair["metrics"] = compare_pixels(pixels["bf16"], pixels[candidate])
            pair["status"] = "paired" if pair["metrics"]["status"] == "computed" else "metric_error"
        pairs.append(pair)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = args.output_dir / "quality_metrics.json"
    if existing.exists() and json.loads(existing.read_text()).get("candidate_arm") != candidate:
        parser.error("output directory belongs to another candidate; choose a distinct directory")
    result = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "reference_arm": "bf16", "candidate_arm": candidate,
              "scope": "exploratory; no global quality certification; no numeric gate agreed",
              "manifest_sha256": sha(args.manifest), "analyser_sha256": sha(__file__),
              "metric_protocol": manifest.get("metric_protocol"), "ssim_call_parameters": SSIM_SETTINGS,
              "implementation": {"python": platform.python_version(), "Pillow": version("Pillow"),
                                 "numpy": version("numpy"), "scikit-image": version("scikit-image")},
              "runs": runs, "env_audit": env_audit(runs, manifest, candidate), "pairs": pairs,
              "qualitative_audit": qualitative_audit(pairs, candidate, args.visual_review_csv),
              "summary": summarise(pairs, candidate),
              "splits": {split: summarise([p for p in pairs if p["split"] == split], candidate) for split in ("legacy", "heldout")},
              "groups": {g: summarise([p for p in pairs if p["group"] == g], candidate) for g in sorted({p["group"] for p in pairs})}}
    result["manual_review_file_status"] = manual_template(args.output_dir / "human_review.csv", pairs, candidate)
    result["contact_sheets"] = [] if args.no_contact_sheets else contact_sheets(args.output_dir / "contact_sheets", pairs, candidate)
    json_write(args.output_dir / "quality_metrics.json", result)
    fields = ["case_id", "group", "split", "candidate_arm", "seed", "status", "bf16_status", "candidate_status", "rgb_equal", "rgb_mse", "rgb_mae", "psnr_db", "ssim", "ssim_status"]
    with (args.output_dir / "quality_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in pairs:
            row = {key: p[key] for key in ("case_id", "group", "split", "seed", "status")}
            row.update(candidate_arm=candidate, bf16_status=p["bf16"]["status"], candidate_status=p[candidate]["status"])
            row.update({key: (p.get("metrics") or {}).get(key) for key in fields if key not in row})
            writer.writerow(row)
    report(args.output_dir / "QUALITY_REPORT.md", result)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "summary": result["summary"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
