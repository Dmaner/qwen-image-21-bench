#!/usr/bin/env python3
"""CPU-only A0 audit. Read original runs; never pool timing and diagnostics.

python3 analysis/compare_a0.py --candidate-results results
Outputs analysis/comparison.json and analysis/REPORT.md (not the root report).
Pillow enables RGB hashes/PSNR; optional scikit-image enables SSIM. No installs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
CODE_SHA = "3dc35694b3d458fc1461a8fe71832ac12e8c8cea"
MODEL_SHA = "d26bb61231c349cf6b7896fa83353113880e1ba3"
UNITS = {
    "wall_s": "s", "client_total_s": "s", "time_to_headers_s": "s",
    "body_transfer_s": "s", "client_json_parse_s": "s",
    "client_b64_decode_s": "s", "client_png_decode_s": "s",
    "device_used_peak_mib": "MiB (sampled device usage, not allocated)",
    "engine_peak_memory_mb": "MiB (reserved in the pinned A0 implementation)",
    "torch_max_memory_allocated_gib": "GiB allocated",
    "torch_max_memory_reserved_gib": "GiB reserved",
}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def stats(values):
    values = [float(x) for x in values if number(x)]
    return {"n": len(values), "values": values,
            "mean": statistics.fmean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None}


def delta(reference, candidate):
    a, b = reference.get("mean"), candidate.get("mean")
    return {"reference": reference, "candidate": candidate,
            "absolute_difference": b - a if a is not None and b is not None else None,
            "relative_percent": (b / a - 1) * 100 if a not in (None, 0) and b is not None else None}


def classify(path, env):
    w = env.get("workload", {})
    mode = w.get("mode", "unknown")
    if "serving" in mode or "serving" in path.name:
        return mode.replace("-serving", "") + "/serving"
    category = next((s for s in ("trace", "mem", "stages")
                     if w.get(s) or ("_" + s) in path.name), "time")
    return mode + "/" + category


def stage_values(row):
    # The upstream object contains *_ms keys despite its name stage_durations_s.
    values = row.get("stage_durations_s")
    if values is None:
        values = row.get("response_meta", {}).get("metrics", {}).get("stage_durations", {})
    return {k: v / 1000 if k.endswith("_ms") else v
            for k, v in (values or {}).items() if number(v)}


def image_info(path):
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    result["png_sha256"] = digest(path)
    try:
        from PIL import Image
        with Image.open(path) as im:
            result.update(original_mode=im.mode, size=list(im.size),
                          rgb_sha256=hashlib.sha256(im.convert("RGB").tobytes()).hexdigest())
    except Exception as exc:
        result["decode_error"] = str(exc)
    return result


def runtime_evidence(path):
    """Record available command/log provenance; absence of a log is not a config value."""
    artifacts, backends, commands = [], [], []
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    pattern = re.compile(r"Resolved diffusion attention backend '([^']+)' for role='([^']+)'(.*)")
    for name in ("command.txt", "server_cmd.txt", "benchmark_cmd.txt", "engine.log", "server.log"):
        source = path / name
        if not source.exists():
            continue
        artifacts.append({"path": str(source), "sha256": digest(source), "bytes": source.stat().st_size})
        if source.suffix == ".txt":
            commands.append(source.read_text(errors="replace"))
        if source.suffix == ".log":
            for lineno, line in enumerate(source.read_text(errors="replace").splitlines(), 1):
                match = pattern.search(ansi.sub("", line))
                if match:
                    backends.append({"backend": match[1], "role": match[2], "resolution": match[3].strip(),
                                     "source": str(source), "line": lineno})
    adapters = [{"path": str(f), "sha256": digest(f), "marker": read_json(f)}
                for f in sorted(path.glob("trace/*/a0_export_only_adapter_rank*.json"))]
    return {"artifacts": artifacts, "resolved_attention_backends": backends,
            "trace_export_only_adapters": adapters,
            "serving_metadata_wrapper_in_command": any("run_serving_compat.py" in c for c in commands),
            "boundary": "Only exact backend-resolution log messages interpreted. Commands/logs are hashed for source audit; missing precision, cache or offload logs do not establish disabled/default values."}


def pixel_compare(reference, candidate):
    result = {"reference": reference, "candidate": candidate}
    if not reference.get("rgb_sha256") or not candidate.get("rgb_sha256"):
        return dict(result, status="unavailable", reason="missing image or RGB decode dependency/error")
    if reference["size"] != candidate["size"]:
        return dict(result, status="shape_mismatch")
    from PIL import Image, ImageChops
    with Image.open(reference["path"]) as a, Image.open(candidate["path"]) as b:
        a, b = a.convert("RGB"), b.convert("RGB")
        histogram = ImageChops.difference(a, b).histogram()
        count = a.width * a.height * 3
        mse = sum((i % 256) ** 2 * n for i, n in enumerate(histogram)) / count
        mae = sum((i % 256) * n for i, n in enumerate(histogram)) / count
        result.update(status="compared", rgb_identical=reference["rgb_sha256"] == candidate["rgb_sha256"],
                      mse_uint8=mse, mae_uint8=mae,
                      psnr_db="Infinity" if mse == 0 else 10 * math.log10(255 ** 2 / mse))
        try:
            import numpy as np
            from skimage.metrics import structural_similarity
            # Explicit standard 7x7 uniform-window parameters, not a release gate.
            result["ssim"] = float(structural_similarity(
                np.asarray(a), np.asarray(b), channel_axis=2, data_range=255,
                win_size=7, gaussian_weights=False, use_sample_covariance=True))
            result["ssim_method"] = "skimage: RGB, data_range=255, win_size=7, uniform, sample covariance"
        except ImportError:
            result.update(ssim=None, ssim_unavailable_reason="numpy/scikit-image not installed")
        except Exception as exc:
            result.update(ssim=None, ssim_unavailable_reason=str(exc))
    return result


def load_run(path):
    env = read_json(path / "env.json") or {}
    summary = read_json(path / "summary.json")
    rows, warnings, parse_errors = [], [], []
    runs = path / "runs.jsonl"
    if runs.exists():
        for lineno, line in enumerate(runs.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("request row is not a JSON object")
                rows.append(row)
            except (json.JSONDecodeError, ValueError) as exc:
                parse_errors.append({"line": lineno, "error": str(exc)})
    else:
        warnings.append("runs.jsonl missing: possibly startup failure or incomplete transfer")
    selected = [i for i, r in enumerate(rows) if r.get("phase") == "measured" and not r.get("error")]
    measured = [rows[i] for i in selected]
    failures = [r for r in rows if r.get("error")]
    metrics = {k: stats(r.get(k) for r in measured) for k in UNITS
               if any(k in r for r in measured)}
    stages = [stage_values(r) for r in measured]
    stage_stats = {k: stats(r.get(k) for r in stages) for k in sorted({k for r in stages for k in r})}
    # Read worker logs independently; align only with an unambiguous full request sequence.
    extra = {}
    for filename, key in [("stage_log.json", "stage_log_s"), ("engine_peak_memory.json", "engine_peak_gib")]:
        data = read_json(path / filename)
        if data is not None:
            aligned = isinstance(data, list) and len(data) == len(rows) and not failures and not parse_errors
            extra[key] = {"alignment": "request order; full counts agree" if aligned else "unaligned; no measured aggregate",
                          "raw": data,
                          "measured": {k: stats(data[i].get(k) for i in selected)
                                       for k in sorted({k for i in selected for k in data[i]})} if aligned else {}}
            if not aligned:
                warnings.append(filename + " cannot be safely aligned to request phases")
    for kind in ("allocated", "reserved"):
        field = "torch_max_memory_" + kind + "_gib"
        if field in metrics:
            if any(r.get(field) == 0 for r in measured):
                warnings.append(field + " includes zero; caller-process counter may not measure the worker")
            metrics[field] = stats(r.get(field) for r in measured if number(r.get(field)) and r[field] > 0)
    memory = {"allocated_gib": metrics.get("torch_max_memory_allocated_gib", stats([])),
              "reserved_gib": metrics.get("torch_max_memory_reserved_gib", stats([])),
              "source": "positive request-row torch peak counters; worker placement must be verified"}
    # Rounded engine logs are a separate source, never silently substituted for exact counters.
    memory["engine_log_measured"] = extra.get("engine_peak_gib", {}).get("measured", {})
    if "engine_peak_memory_mb" in metrics:
        memory["engine_reported_reserved_gib"] = stats(x / 1024 for x in metrics["engine_peak_memory_mb"]["values"])
    if "device_used_peak_mib" in metrics:
        memory["sampled_device_used_gib"] = stats(x / 1024 for x in metrics["device_used_peak_mib"]["values"])
    discrepancies = []
    if summary:
        for key, actual in metrics.items():
            published = summary.get(key)
            if not isinstance(published, dict):
                continue
            for a, s in [("n", "n"), ("mean", "mean"), ("sample_sd", "stdev")]:
                if actual[a] is not None and number(published.get(s)) and not math.isclose(actual[a], published[s], rel_tol=1e-9, abs_tol=1e-9):
                    discrepancies.append({"metric": key, "field": s, "recomputed": actual[a], "summary": published[s]})
        if summary.get("failures", 0) != len(failures):
            discrepancies.append({"metric": "failures", "recomputed": len(failures), "summary": summary.get("failures")})
        for key, actual in stage_stats.items():
            published = summary.get("stages_s", {}).get(key, {})
            factor = 0.001 if key.endswith("_ms") else 1
            for a, s in [("mean", "mean"), ("sample_sd", "stdev")]:
                if actual[a] is not None and number(published.get(s)) and not math.isclose(actual[a], published[s] * factor, rel_tol=1e-9, abs_tol=1e-9):
                    discrepancies.append({"stage": key, "field": s, "recomputed_s": actual[a], "summary_s": published[s] * factor})
    images = []
    for r in rows:
        phase, index = r.get("phase"), r.get("index")
        info = image_info(path / "images" / f"{phase}_{index}.png")
        info.update(phase=phase, index=index, run_error=r.get("error"), recorded_png_sha256=r.get("image_sha256"))
        info["recorded_png_hash_matches"] = (info["png_sha256"] == r["image_sha256"]
                                              if info.get("png_sha256") and r.get("image_sha256") else None)
        images.append(info)
    hashes = [x["rgb_sha256"] for x in images if x.get("rgb_sha256")]
    expected = env.get("workload", {}).get("measured", 2)
    if classify(path, env).endswith("/trace"):
        expected = 0
    if len(measured) != expected:
        warnings.append(f"expected {expected} successful measured rows, found {len(measured)}")
    if len({r.get("index") for r in measured}) != len(measured):
        warnings.append("duplicate measured indices: do not treat these as independent runs")
    for key, value in [("vllm_omni_sha", CODE_SHA), ("model_revision", MODEL_SHA)]:
        if env.get(key) != value:
            warnings.append(key + " differs from/missing pinned A0 revision")
    if env.get("vllm_omni_dirty"):
        warnings.append("checkout has local changes; inspect env provenance")
    if parse_errors:
        warnings.append(f"{len(parse_errors)} JSONL parse errors: artifact may be partial")
    if discrepancies:
        warnings.append(f"{len(discrepancies)} supplied-summary discrepancies; use recomputed values")
    if any(i["recorded_png_hash_matches"] is False for i in images):
        warnings.append("saved PNG hash differs from recorded runs.jsonl hash")
    inputs = [image_info(f) for f in sorted((path / "inputs").glob("*")) if f.is_file()]
    return {"directory": str(path), "category": classify(path, env), "env": env,
            "runtime_evidence": runtime_evidence(path),
            "measured_success_count": len(measured), "measured_row_count": sum(r.get("phase") == "measured" for r in rows),
            "request_count": len(rows), "failure_count": len(failures), "failures": failures,
            "jsonl_parse_errors": parse_errors, "metrics": metrics, "metric_units": UNITS,
            "stage_seconds": stage_stats, "memory": memory, "diagnostics": extra,
            "supplied_summary": summary, "summary_discrepancies": discrepancies,
            "images": images, "inputs": inputs,
            "rgb_repeatability": {"decoded": len(hashes), "requests": len(rows),
                                  "all_present_and_identical": bool(rows) and len(hashes) == len(rows) and len(set(hashes)) == 1},
            "warnings": warnings,
            "source_sha256": {str(f): digest(f) for f in [path / n for n in
                              ("env.json", "runs.jsonl", "summary.json", "stage_log.json", "engine_peak_memory.json")] if f.exists()}}


def compare(reference, candidate):
    a, b = reference, candidate
    fields = ("vllm_omni_sha", "vllm_omni_checkout", "model_revision", "packages", "torch_cuda", "python", "nvidia_smi", "hostname", "cpu", "vllm_omni_dirty")
    env_diff = {k: {"reference": a["env"].get(k), "candidate": b["env"].get(k)}
                for k in fields if a["env"].get(k) != b["env"].get(k)}
    wa, wb = a["env"].get("workload", {}), b["env"].get("workload", {})
    workload_diff = {k: {"reference": wa.get(k), "candidate": wb.get(k)}
                     for k in sorted(set(wa) | set(wb)) if k != "image" and wa.get(k) != wb.get(k)}
    input_a, input_b = [i.get("rgb_sha256") for i in a["inputs"]], [i.get("rgb_sha256") for i in b["inputs"]]
    full_a = a["diagnostics"].get("stage_log_s", {}).get("measured", {})
    full_b = b["diagnostics"].get("stage_log_s", {}).get("measured", {})
    backend_a = sorted({(x["role"], x["backend"]) for x in a["runtime_evidence"]["resolved_attention_backends"]})
    backend_b = sorted({(x["role"], x["backend"]) for x in b["runtime_evidence"]["resolved_attention_backends"]})
    pairs = []
    for ai in a["images"]:
        if ai["phase"] != "measured" or ai["run_error"]:
            continue
        matches = [bi for bi in b["images"] if bi["phase"] == "measured" and bi["index"] == ai["index"] and not bi["run_error"]]
        if len(matches) == 1:
            pairs.append(dict(pixel_compare(ai, matches[0]), measured_index=ai["index"]))
    return {"category": b["category"], "reference_directory": a["directory"], "candidate_directory": b["directory"],
            "environment_differences": env_diff, "workload_differences": workload_diff,
            "runtime_attention": {"reference": backend_a, "candidate": backend_b,
                                  "equal_when_both_observed": backend_a == backend_b if backend_a and backend_b else None},
            "configuration_interpretation": [
                "Repository/model SHA and workload fields are reported separately from observed runtime backend.",
                "Different GPU UUID/driver/CPU/package entries are environmental confounders, not optimization evidence.",
                "No precision/offload/cache equality inferred from absent runtime log entries; audit command/source provenance."],
            "input_rgb_hashes_match": (input_a == input_b if input_a and input_b and all(input_a + input_b) else None),
            "metrics": {k: delta(a["metrics"].get(k, stats([])), b["metrics"].get(k, stats([])))
                        for k in sorted(set(a["metrics"]) | set(b["metrics"]))},
            "stage_seconds": {k: delta(a["stage_seconds"].get(k, stats([])), b["stage_seconds"].get(k, stats([])))
                              for k in sorted(set(a["stage_seconds"]) | set(b["stage_seconds"]))},
            "full_stage_log_seconds": {k: delta(full_a.get(k, stats([])), full_b.get(k, stats([])))
                                       for k in sorted(set(full_a) | set(full_b))},
            "memory_gib": {k: delta(a["memory"].get(k, stats([])), b["memory"].get(k, stats([])))
                           for k in ("allocated_gib", "reserved_gib", "engine_reported_reserved_gib", "sampled_device_used_gib")},
            "pixel_pairs": pairs, "candidate_failure_count": b["failure_count"],
            "candidate_summary_discrepancies": b["summary_discrepancies"], "candidate_warnings": b["warnings"]}


def fmt(x, digits=4):
    return "—" if x is None else f"{x:.{digits}f}"


def report(data):
    lines = ["# A0 复跑离线对比片段", "", "本文件由原始 JSONL 重算；不是 GPU 执行记录或质量验收结论。", "",
             "仅成功的 measured 请求进入均值；± 为样本标准差（n−1），n<2 不给标准差。warmup、feasibility、trace、初始化不进入 headline；失败保留。", "",
             "| 类别 | A0 mean ± sample SD (s) | 本次 mean ± sample SD (s) | n(A0/本次) | 相对 A0 |", "|---|---:|---:|---:|---:|"]
    for c in data["comparisons"]:
        metric = "client_total_s" if c["category"].endswith("/serving") else "wall_s"
        d = c["metrics"].get(metric)
        if not d:
            continue
        a, b = d["reference"], d["candidate"]
        lines.append(f"| {c['category']} | {fmt(a['mean'])} ± {fmt(a['sample_sd'])} | {fmt(b['mean'])} ± {fmt(b['sample_sd'])} | {a['n']}/{b['n']} | {fmt(d['relative_percent'], 2)}% |")
    if not data["comparisons"]:
        lines.extend(["", "尚无可配对本次结果。以下 A0 基线已由原始记录独立重算："])
        for r in data["reference_runs"]:
            if r["category"].endswith("/time"):
                s = r["metrics"].get("wall_s", stats([]))
                lines.append(f"- {r['category']}: {fmt(s['mean'], 6)} ± {fmt(s['sample_sd'], 6)} s, n={s['n']}。")
    adapted = [r for r in data["candidate_runs"] if r["runtime_evidence"]["trace_export_only_adapters"]]
    wrapped = [r for r in data["candidate_runs"] if r["runtime_evidence"]["serving_metadata_wrapper_in_command"]]
    if adapted:
        lines.extend(["", f"诊断适配：{len(adapted)} 个成功 trace 组使用已留 marker 的 export-only adapter，只替换 `_on_stop_hook`，跳过 CPU 聚合表、stack 文本、XLSX 和 profiler_out 表；原始采集参数、Chrome trace/gzip 与 finally 显存快照路径保留。这些是适配后的诊断，不是全部原 harness 不变；不用于 headline 计时。原未完成 stop_profile 尝试保存在 setup-attempts，未纳入本次 9 组结果。"])
    if wrapped:
        lines.extend(["", "Serving 的 run_serving_compat.py 仅给原 harness 的 parse_args 补 `mem=False/stages=False` 元数据；服务器、请求与测量逻辑未改。源码审计见 reference/run_serving_compat.py。"])
    if data["output_rgb_cohorts"]:
        lines.extend(["", "全部保存输出（含 warmup/feasibility/trace，逐张重新解码 RGB）："])
        for mode, cohort in data["output_rgb_cohorts"].items():
            lines.append(f"- {mode}: 本次 {cohort['candidate_images']} 张，A0 {cohort['reference_images']} 张；本次解码完整={cohort['candidate_decode_complete']}；全部本次 RGB hash 属于相应 A0 集合={cohort['all_candidate_rgb_match_reference']}。")
    for c in data["comparisons"]:
        lines.extend(["", f"## {c['category']} · {Path(c['candidate_directory']).name}", "",
                      f"失败 {c['candidate_failure_count']}；配置差异 {len(c['workload_differences'])} 项，环境差异 {len(c['environment_differences'])} 项（详见 JSON，不自动解释为优化收益）。"])
        backend = c["runtime_attention"]
        lines.append(f"- attention 日志：A0 {backend['reference'] or '本组未记录'}；本次 {backend['candidate'] or '本组未记录'}。缺日志不代表配置相同。")
        for k, d in c["memory_gib"].items():
            if d["reference"]["n"] or d["candidate"]["n"]:
                lines.append(f"- {k}: A0 {fmt(d['reference']['mean'])} → 本次 {fmt(d['candidate']['mean'])} GiB。")
        for pair in c["pixel_pairs"]:
            lines.append(f"- measured_{pair['measured_index']}: RGB {'相同' if pair.get('rgb_identical') else '不同/不可比'}，PSNR={pair.get('psnr_db', '未计算')} dB，SSIM={pair.get('ssim') if pair.get('ssim') is not None else '未计算'}。")
        lines.extend(["", "| stage（均已换算为秒，不相加嵌套阶段） | A0 mean | 本次 mean |", "|---|---:|---:|"])
        for k, d in c["stage_seconds"].items():
            lines.append(f"| {k} | {fmt(d['reference']['mean'], 6)} | {fmt(d['candidate']['mean'], 6)} |")
        if c["full_stage_log_seconds"]:
            lines.extend(["", "完整 stage_log 诊断（与上述阶段重叠，不相加）：", "",
                          "| 阶段 | A0 mean (s) | 本次 mean (s) |", "|---|---:|---:|"])
            for k, d in c["full_stage_log_seconds"].items():
                lines.append(f"| {k} | {fmt(d['reference']['mean'], 6)} | {fmt(d['candidate']['mean'], 6)} |")
        for w in c["candidate_warnings"]:
            lines.append("- 注意：" + w)
    lines.extend(["", "GiB=2³⁰ bytes；allocated、reserved 与设备采样 used 分列。pinned A0 的 engine_peak_memory_mb 实际是 reserved MiB，不能冒充 allocated。诊断组只与同类比较；stage 含嵌套关系，不求和归因。",
                  "", "RGB hash/PSNR 由解码后 uint8 RGB 计算，与 PNG 文件 hash 区分；不评估 alpha。SSIM 仅安装 scikit-image 时计算，缺库不补造。微小延迟差不代表跨物理 GPU 的确定优化收益。",
                  "", f"候选结果 {len(data['candidate_runs'])} 组；未配对 {len(data['unmatched_candidates'])}；读取错误 {len(data['read_errors'])}。原始路径、文件 SHA256、逐次值和完整差异见 comparison.json。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-results", type=Path, default=ROOT / "reference/upstream/results")
    parser.add_argument("--candidate-results", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis")
    args = parser.parse_args()
    if args.output_dir.resolve() == ROOT:
        parser.error("output-dir cannot be experiment root: protect root REPORT.md")
    errors = []
    def safe_discover(path):
        result = []
        # Include env-only startup failures, not merely completed summaries.
        dirs = {f.parent for name in ("env.json", "runs.jsonl") for f in path.rglob(name)} if path.exists() else set()
        for d in sorted(dirs):
            try:
                result.append(load_run(d.resolve()))
            except Exception as exc:
                errors.append({"directory": str(d), "error": str(exc)})
        return result
    reference, candidate = safe_discover(args.reference_results), safe_discover(args.candidate_results)
    comparisons, unmatched = [], []
    for run in candidate:
        matches = [r for r in reference if r["category"] == run["category"]]
        if len(matches) == 1:
            comparisons.append(compare(matches[0], run))
        else:
            unmatched.append({"directory": run["directory"], "category": run["category"], "reference_matches": len(matches)})
    packages = {}
    for package in ("Pillow", "numpy", "scikit-image"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    data = {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "reference_results": str(args.reference_results.resolve()), "candidate_results": str(args.candidate_results.resolve()),
            "pinned_code_sha": CODE_SHA, "pinned_model_sha": MODEL_SHA,
            "analysis_packages": packages, "analysis_script_sha256": digest(Path(__file__)),
            "reference_runs": reference, "candidate_runs": candidate, "comparisons": comparisons,
            "unmatched_candidates": unmatched, "read_errors": errors,
            "notes": ["No GPU execution, network, dependency installation or source-data mutation.",
                      "No pooling across physical devices, retries, or diagnostic categories.",
                      "RGB comparison is not perceptual/human quality acceptance; no automatic quality gate."]}
    cohorts = {}
    for mode in sorted({r["category"].split("/")[0] for r in reference + candidate}):
        images_a = [im for r in reference if r["category"].split("/")[0] == mode for im in r["images"]]
        images_b = [im for r in candidate if r["category"].split("/")[0] == mode for im in r["images"]]
        hashes_a = {im["rgb_sha256"] for im in images_a if im.get("rgb_sha256")}
        hashes_b = {im["rgb_sha256"] for im in images_b if im.get("rgb_sha256")}
        complete = bool(images_b) and all(im.get("rgb_sha256") for im in images_b)
        cohorts[mode] = {"reference_images": len(images_a), "candidate_images": len(images_b),
                         "reference_rgb_hashes": sorted(hashes_a), "candidate_rgb_hashes": sorted(hashes_b),
                         "candidate_decode_complete": complete,
                         "all_candidate_rgb_match_reference": complete and hashes_b.issubset(hashes_a)}
    data["output_rgb_cohorts"] = cohorts
    source = ROOT / "reference/api/report-content.json"
    if source.exists() and args.reference_results.resolve() == (ROOT / "reference/upstream/results").resolve():
        metadata = read_json(source)
        data["reference_benchmark_provenance"] = {"report_url": metadata.get("html_url"),
                                                 "report_blob_sha": metadata.get("sha"),
                                                 "metadata_file": str(source), "metadata_sha256": digest(source)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output_dir / "REPORT.md").write_text(report(data))
    print(json.dumps({"reference_runs": len(reference), "candidate_runs": len(candidate),
                      "comparisons": len(comparisons), "read_errors": errors, "output_dir": str(args.output_dir)}))
    return 1 if errors or not reference else 0


if __name__ == "__main__":
    raise SystemExit(main())
