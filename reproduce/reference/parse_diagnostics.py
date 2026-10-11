"""Parse a captured A0 log with the same regex/accounting as upstream modal_app.py.

No GPU execution and no log modification. Stage nesting is preserved, not summed.
"""
import argparse
import json
import re
from pathlib import Path

PEAK_LINE = re.compile(r"Peak GPU memory \(this request\): ([\d.]+) GB reserved, ([\d.]+) GB allocated")
STAGE_LINE = re.compile(r"\[DiffusionPipelineProfiler\] QwenImage21Pipeline\.(\S+) took ([\d.]+)s")


def parse(text):
    peaks, requests, hooked = [], [{}], False
    for line in text.splitlines():
        hooked = hooked or "[stagehook] patched" in line
        if m := PEAK_LINE.search(line):
            peaks.append({"reserved_gib": float(m[1]), "allocated_gib": float(m[2])})
        if m := STAGE_LINE.search(line):
            name, seconds = m[1], float(m[2])
            row = requests[-2] if name == "post_process" and len(requests) > 1 else requests[-1]
            row[name] = row.get(name, 0.0) + seconds
            if name == "forward":
                requests.append({})
    return peaks, [r for r in requests if r], hooked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--mem", action="store_true")
    p.add_argument("--stages", action="store_true")
    args = p.parse_args()
    peaks, stages, hooked = parse((args.out / "engine.log").read_text())
    if args.mem:
        if not peaks:
            raise SystemExit("No runner peak lines: memory diagnostic is incomplete")
        (args.out / "engine_peak_memory.json").write_text(json.dumps(peaks, indent=2) + "\n")
    if args.stages:
        if not hooked or not stages:
            raise SystemExit("Stage import hook did not run, or no stage lines: diagnostic is incomplete")
        (args.out / "stage_log.json").write_text(json.dumps(stages, indent=2) + "\n")
    print(json.dumps({"peak_rows": len(peaks), "stage_requests": len(stages), "stage_hooked": hooked}))


if __name__ == "__main__":
    main()
