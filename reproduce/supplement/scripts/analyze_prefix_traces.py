#!/usr/bin/env python3
"""Small, streaming, CPU-only inventory of C1 prefix-storage GPU traces.

Usage:
  python scripts/analyze_prefix_traces.py --results-root results \
      --out-dir analysis/prefix_traces
  python scripts/analyze_prefix_traces.py --self-test

Only Chrome trace complete events with cat=kernel are timed. CPU operations,
launch calls, memcpy and memset are excluded. No torch import or GPU work.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

ARMS = ("bf16", "prefix_fp8_v", "prefix_fp8")
BLOCK = 65536
MAX_EVENT = 16 * 1024 * 1024
CATEGORIES = (
    "fa3_sm90_name", "dequantization_name", "quantization_name",
    "packing_name", "copy_concat_name", "unattributed_triton", "other",
)
BOUNDARIES = [
    "Kernel durations are microseconds in the Chrome trace and reported as sums in milliseconds. Overlapping GPU streams can be counted twice.",
    "Whole captured trace, not a denoising-only attribution. These sums are not request latency, GPU utilization, idle time, or speedup; use the independent unprofiled benchmark for latency.",
    "Name categories are an inventory, not causal attribution. A quant/dequant/pack/copy name alone does not prove that its work belongs to the prefix cache.",
    "Generic Triton names are preserved without guessing their operation; absence of an explicit dequantization name does not imply zero dequantization cost.",
    "FA3 is counted only for explicit FlashAttnFwdSm90/FlashAttnBwdSm90 name markers. Name matching does not establish tensor dtype or correctness; consult the separate worker audit.",
]


def iter_events(path):
    """Decode traceEvents one object at a time; validate trailing JSON/gzip CRC."""
    opener = gzip.open if str(path).endswith(".gz") else open
    decoder = json.JSONDecoder()
    with opener(path, "rt", encoding="utf-8") as stream:
        buffer = ""
        while True:
            chunk = stream.read(BLOCK)
            if not chunk:
                raise ValueError("Missing traceEvents array")
            buffer += chunk
            match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
            if match:
                prefix = buffer[:match.end() - 1]
                position = match.end()
                break
            if len(buffer) > 1024 * 1024:
                raise ValueError("Trace header exceeds 1 MiB")
        need_value = True
        may_end = True
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                chunk = stream.read(BLOCK)
                if not chunk:
                    raise ValueError("Truncated traceEvents array")
                buffer = chunk
                position = 0
            char = buffer[position]
            if char == "]":
                if need_value and not may_end:
                    raise ValueError("Trailing comma in traceEvents")
                suffix = buffer[position + 1:]
                while True:
                    chunk = stream.read(BLOCK)
                    if not chunk:
                        break
                    suffix += chunk
                    if len(suffix) > 1024 * 1024:
                        raise ValueError("Trace footer exceeds 1 MiB")
                json.loads(prefix + "[]" + suffix)
                return
            if not need_value:
                if char != ",":
                    raise ValueError("Missing traceEvents comma")
                position += 1
                need_value, may_end = True, False
                continue
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                chunk = stream.read(BLOCK)
                if not chunk:
                    raise ValueError("Truncated or invalid trace event") from None
                buffer = buffer[position:] + chunk
                position = 0
                if len(buffer) > MAX_EVENT:
                    raise ValueError("Single trace event exceeds 16 MiB")
                continue
            if not isinstance(event, dict):
                raise ValueError("traceEvents contains a non-object")
            position = end
            need_value, may_end = False, True
            yield event
            if position > BLOCK:
                buffer = buffer[position:]
                position = 0


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def category(name):
    lower = name.lower()
    if "FlashAttnFwdSm90" in name or "FlashAttnBwdSm90" in name:
        return "fa3_sm90_name", "Explicit FlashAttnFwdSm90/FlashAttnBwdSm90 marker"
    if "dequant" in lower:
        return "dequantization_name", "Literal dequant substring; component attribution unresolved"
    if "quant" in lower:
        return "quantization_name", "Literal quant substring; component attribution unresolved"
    if re.search(r"(?:^|[^a-z])(?:un)?pack(?:ing|ed)?(?:[^a-z]|$)", lower):
        return "packing_name", "Literal pack/unpack token; component attribution unresolved"
    if any(token in lower for token in ("catarray", "cat_kernel", "concat", "copy_kernel", "copy_")):
        return "copy_concat_name", "Explicit copy/concatenation name; component attribution unresolved"
    if "triton" in lower:
        return "unattributed_triton", "Triton name retained; no operation or prefix-cache attribution"
    return "other", "Unclassified original kernel name"


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def analyze(path, arm):
    names = {}
    observed_categories = collections.Counter()
    skipped_kernel_reasons = collections.Counter()
    device_counts = collections.Counter()
    for event in iter_events(path):
        cat = event.get("cat", "")
        observed_categories[str(cat)] += 1
        if cat != "kernel":
            continue
        if event.get("ph") != "X":
            skipped_kernel_reasons["not_complete_event_X"] += 1
            continue
        duration, timestamp = event.get("dur"), event.get("ts")
        if not number(duration) or duration < 0 or not number(timestamp):
            skipped_kernel_reasons["invalid_or_missing_duration_timestamp"] += 1
            continue
        name = event.get("name")
        if not isinstance(name, str) or not name:
            skipped_kernel_reasons["missing_kernel_name"] += 1
            continue
        label, evidence = category(name)
        item = names.setdefault(name, {"name":name, "category":label,
            "evidence":evidence, "count":0, "duration_sum_us":0.0})
        item["count"] += 1
        item["duration_sum_us"] += duration
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        device_counts[str(args.get("device", "unrecorded"))] += 1
    rows = sorted(names.values(), key=lambda item:(-item["duration_sum_us"], item["name"]))
    totals = {label:{"kernel_count":0,"duration_sum_ms":0.0} for label in CATEGORIES}
    for item in rows:
        item["duration_sum_ms"] = item.pop("duration_sum_us") / 1000
        totals[item["category"]]["kernel_count"] += item["count"]
        totals[item["category"]]["duration_sum_ms"] += item["duration_sum_ms"]
    total_count = sum(item["count"] for item in rows)
    total_ms = sum(item["duration_sum_ms"] for item in rows)
    for item in rows:
        item["pct_of_kernel_duration_sum"] = item["duration_sum_ms"] / total_ms * 100 if total_ms else None
    warnings = []
    if skipped_kernel_reasons:
        warnings.append("Some kernel-category events were invalid or incomplete and were excluded; see skipped_kernel_events.")
    if not total_count:
        warnings.append("No valid GPU kernel events found.")
    if len(device_counts) > 1:
        warnings.append("Multiple device labels observed; duration sums combine them. Do not interpret as single-device busy time.")
    return {"arm":arm,"trace":str(path),"trace_sha256":sha256(path),"trace_bytes":Path(path).stat().st_size,
        "gpu_kernel_count":total_count,"gpu_kernel_duration_sum_ms":total_ms,
        "event_categories_observed":dict(observed_categories),"device_event_counts":dict(device_counts),
        "skipped_kernel_events":dict(skipped_kernel_reasons),"categories":totals,
        "all_kernel_names":rows,"warnings":warnings}


def md_escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("`", "'")


def render_markdown(result):
    lines = ["# Prefix-storage profiler kernel inventory", "",
             "Captured GPU kernel durations only; use the separate unprofiled benchmark for request latency.", "",
             "| Arm | Trace | GPU kernels | Sum of kernel durations (ms) | FA3 kernels | FA3 duration (ms) |",
             "|---|---|---:|---:|---:|---:|"]
    for trace in result["traces"]:
        fa3 = trace["categories"]["fa3_sm90_name"]
        lines.append(f"| {trace['arm']} | {md_escape(Path(trace['trace']).name)} | {trace['gpu_kernel_count']:,} | {trace['gpu_kernel_duration_sum_ms']:.3f} | {fa3['kernel_count']:,} | {fa3['duration_sum_ms']:.3f} |")
    if not result["traces"]:
        lines.append("| — | No trace files found | — | — | — | — |")
    for trace in result["traces"]:
        lines += ["", f"## {trace['arm']} — {md_escape(Path(trace['trace']).name)}", "",
                  "| Name class | GPU kernels | Duration sum (ms) |", "|---|---:|---:|"]
        for label, item in trace["categories"].items():
            lines.append(f"| {label} | {item['kernel_count']:,} | {item['duration_sum_ms']:.3f} |")
        lines += ["", "Explicit FA3 and quant/dequant/pack/copy names; then up to 10 largest unresolved Triton names:", "",
                  "| Class | Original kernel name | Count | Duration sum (ms) |", "|---|---|---:|---:|"]
        selected = [r for r in trace["all_kernel_names"] if r["category"] not in ("other", "unattributed_triton")]
        selected += [r for r in trace["all_kernel_names"] if r["category"] == "unattributed_triton"][:10]
        for row in selected:
            lines.append(f"| {row['category']} | `{md_escape(row['name'])}` | {row['count']:,} | {row['duration_sum_ms']:.3f} |")
        if not selected:
            lines.append("| — | No explicit matching names | — | — |")
        if trace["warnings"]:
            lines += ["", *["- " + warning for warning in trace["warnings"]]]
    lines += ["", "All original kernel names, source trace hashes and excluded event counts are retained in `prefix_trace_summary.json`.", "", "## Evidence boundary", ""]
    lines += ["- " + text for text in BOUNDARIES]
    if result["warnings"]:
        lines += ["", "## Coverage", "", *["- " + x for x in result["warnings"]]]
    return "\n".join(lines) + "\n"


def self_test():
    events = [
        {"cat":"cpu_op","ph":"X","name":"_dequantize_prefix_kv_fp8","ts":0,"dur":999999},
        {"cat":"cuda_runtime","ph":"X","name":"cudaLaunchKernel","ts":0,"dur":999999},
        {"cat":"gpu_memcpy","ph":"X","name":"Memcpy HtoD","ts":0,"dur":900000},
        {"cat":"gpu_memset","ph":"X","name":"Memset","ts":0,"dur":900000},
    ]
    for name, duration in (("void flash::FlashAttnFwdSm90<bfloat16>(Params)",1000),
                           ("prefix_dequant_kernel",200),("per_token_quant_fp8",100),
                           ("pack_prefix_kernel",50),("CatArrayBatchedCopy",40),
                           ("triton_poi_fused_mul_add_0",300),("nvjet_sm90_kernel",600)):
        events.append({"cat":"kernel","ph":"X","name":name,"ts":10,"dur":duration,"args":{"device":0,"stream":7}})
    events += [{"cat":"kernel","ph":"B","name":"not_completed","ts":0,"dur":123},
               {"cat":"kernel","ph":"X","name":"bad_duration","ts":0,"dur":-1}]
    payload = {"schemaVersion":1,"traceEvents":events,"displayTimeUnit":"ms"}
    with tempfile.TemporaryDirectory(prefix="c1-prefix-traces-") as temporary:
        path = Path(temporary)/"trace_rank0.json.gz"
        with gzip.open(path,"wt",encoding="utf-8") as stream:
            json.dump(payload,stream)
        result = analyze(path,"synthetic")
        assert result["gpu_kernel_count"] == 7
        assert math.isclose(result["gpu_kernel_duration_sum_ms"],2.29)
        assert result["categories"]["fa3_sm90_name"]["kernel_count"] == 1
        assert result["categories"]["dequantization_name"]["duration_sum_ms"] == 0.2
        assert result["categories"]["quantization_name"]["duration_sum_ms"] == 0.1
        assert result["categories"]["packing_name"]["duration_sum_ms"] == 0.05
        assert result["categories"]["unattributed_triton"]["duration_sum_ms"] == 0.3
        assert result["skipped_kernel_events"] == {"not_complete_event_X":1,"invalid_or_missing_duration_timestamp":1}
        assert any(r["name"]=="triton_poi_fused_mul_add_0" and r["category"]=="unattributed_triton" for r in result["all_kernel_names"])
        # Force a valid event to cross the 64-KiB input boundary.
        events[0]["args"] = {"fixture_padding":"x"*(BLOCK+50)}
        plain=Path(temporary)/"boundary.json"
        plain.write_text(json.dumps({"traceEvents":events}))
        assert analyze(plain,"synthetic")["gpu_kernel_count"] == 7
        malformed=Path(temporary)/"invalid.json"
        malformed.write_text('{"traceEvents": [{"name":"x"},]}')
        try:list(iter_events(malformed))
        except ValueError:pass
        else:raise AssertionError("trailing comma accepted")
        truncated=Path(temporary)/"truncated.json.gz"
        truncated.write_bytes(path.read_bytes()[:-8])
        try:list(iter_events(truncated))
        except (EOFError,OSError,ValueError):pass
        else:raise AssertionError("truncated gzip accepted")
    return {"status":"pass","fixture_only":True,"gpu_execution":False,
            "tests":["kernel-only 7 events / 2.29 ms", "exclude CPU/launch/memcpy/memset", "explicit FA3 names", "quant/dequant/pack/copy name classes", "preserve unresolved Triton original names", "reject invalid kernel events", "bounded streaming across chunk boundary", "reject invalid JSON and truncated gzip"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root",type=Path,default=Path("results"))
    parser.add_argument("--out-dir",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        result=self_test()
        if args.out_dir:
            args.out_dir.mkdir(parents=True,exist_ok=True)
            (args.out_dir/"CPU_SYNTHETIC_TEST.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
        return
    result={"schema_version":1,"analyzed_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "measurement_boundary":BOUNDARIES,"traces":[],"warnings":[]}
    for arm in ARMS:
        directory=args.results_root/f"trace_{arm}_multi_edit"/"trace"
        paths=sorted(directory.glob("**/trace_rank*.json.gz"))
        if len(paths)!=1:
            result["warnings"].append(f"{arm}: expected one trace, found {len(paths)}; files are reported separately, never silently merged.")
        for path in paths:
            print(f"Streaming {arm}: {path}",flush=True)
            result["traces"].append(analyze(path,arm))
    result["coverage_status"]="complete" if len(result["traces"])==3 and not result["warnings"] else "incomplete_or_ambiguous"
    out_dir=args.out_dir or Path("analysis/prefix_traces")
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"prefix_trace_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    (out_dir/"PREFIX_TRACE_SUMMARY.md").write_text(render_markdown(result))
    print(json.dumps({"coverage_status":result["coverage_status"],"traces":len(result["traces"]),"out_dir":str(out_dir)}))


if __name__=="__main__":
    main()
