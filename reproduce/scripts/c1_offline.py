"""C1 Phase2: paired five-arm component quantization, minimally extended from A0 cf99e73c.

Runs inside one process / one engine so warmups and measured runs share the same
prepared state. Writes:
    env.json        pinned revisions, package versions, GPU identity
    runs.jsonl      one row per request (phase, wall time, stage durations, memory)
    summary.json    mean/min/max/stdev over measured runs
    images/         every output image (+ sha256 in runs.jsonl)
    trace/          torch-profiler artifacts (only with --trace)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from importlib import metadata
from pathlib import Path

TEAPOT = "A ceramic teapot on a wooden table"
EDIT_PROMPT = "Let this mascot dance under the moon, surrounded by floating stars"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--vllm-omni-sha", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["t2i", "edit"], default="t2i")
    p.add_argument("--prompt", default=None)
    p.add_argument("--image", nargs="*", default=[])
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--true-cfg-scale", type=float, default=1.0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--feasibility", type=int, default=1)
    p.add_argument("--measured", type=int, default=10)
    p.add_argument("--quant-config-file", default=None,
                   help="Component JSON passed to Omni; omit for matching BF16 baseline.")
    p.add_argument("--arm", choices=["bf16", "dit_fp8", "te_fp8", "joint_fp8", "selective_bf16"], required=True)
    p.add_argument("--cases-file", default=None,
                   help="Quality manifest: one engine, warmup first case then each case once.")
    p.add_argument("--quant-audit", action="store_true",
                   help="Separate worker diagnostic; requires the gated quant_audit sitecustomize.")
    p.add_argument("--trace", action="store_true",
                   help="Diagnostic run: torch-profile one request after warmup. Not for timing.")
    p.add_argument("--stages", action="store_true",
                   help="Diagnostic run: time every step of a request (bench/stagehook patches the "
                        "pipeline's profiler targets; modal_app parses the log into stage_log.json). "
                        "A local modification of the pinned checkout, so not a timing baseline.")
    p.add_argument("--mem", action="store_true",
                   help="Diagnostic run: engine DEBUG logging, so each request logs its reserved and "
                        "allocated peaks (parsed into engine_peak_memory.json by modal_app). Not for timing.")
    return p.parse_args()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_c1(args):
    if args.vllm_omni_sha != "3dc35694b3d458fc1461a8fe71832ac12e8c8cea":
        raise ValueError("C1 Phase2 requires the exact A0 Omni 3dc35694 source")
    if not args.enforce_eager:
        raise ValueError("C1 Phase2 requires --enforce-eager in all arms")
    if min(args.warmup, args.feasibility, args.measured) < 0:
        raise ValueError("Run counts must be non-negative")
    arms = json.loads((Path(__file__).parent / "configs/arms.json").read_text())["arms"]
    spec = next(x for x in arms if x["arm"] == args.arm)
    frozen = Path(__file__).parent / spec["quant_config_file"]
    if file_sha256(frozen) != spec["quant_config_sha256"]:
        raise ValueError("Frozen arm config hash mismatch")
    expected = json.loads(frozen.read_text())
    supplied = json.loads(Path(args.quant_config_file).read_text()) if args.quant_config_file else None
    if args.arm == "bf16" and supplied is None:
        args.quantization_config = None
    elif supplied != expected:
        raise ValueError("Arm config differs from frozen manifest")
    else:
        args.quantization_config = supplied
    active_audit = os.environ.get("C1_QUANT_AUDIT") == "1"
    if args.quant_audit != active_audit:
        raise ValueError("--quant-audit and C1_QUANT_AUDIT=1 must be enabled together; never on headline/quality")
    if args.quant_audit:
        if any((args.trace, args.mem, args.stages, args.cases_file)):
            raise ValueError("Quant audit is a separate diagnostic, not a trace/memory/quality run")
        if not getattr(sys.modules.get("sitecustomize"), "C1_QUANT_AUDIT_INSTALLED", False):
            raise RuntimeError("quant_audit/sitecustomize.py was not installed at interpreter startup")
        os.environ["C1_AUDIT_OUT"] = str(Path(args.out).resolve() / "quant_audit")
        os.environ["C1_AUDIT_ARM"] = args.arm
    if args.cases_file and any((args.trace, args.mem, args.stages)):
        raise ValueError("Quality manifest mode does not enable diagnostic hooks")
    if not args.trace and os.environ.get("A0_PROFILE_EXPORT_ONLY") == "1":
        raise ValueError("Export-only profiler adapter is for the independent --trace run only")


def load_cases(args):
    if args.cases_file:
        manifest = json.loads(Path(args.cases_file).read_text())
        if manifest.get("schema_version") != 1 or not manifest.get("cases"):
            raise ValueError("Expected schema_version=1 and a nonempty cases array")
        cases = manifest["cases"]
    else:
        cases = [{"id": "a0_" + args.mode, "task": args.mode, "prompt": args.prompt,
                  "seed": args.seed, "width": args.width, "height": args.height,
                  "num_inference_steps": args.steps, "true_cfg_scale": args.true_cfg_scale,
                  "input_image": args.image[0] if len(args.image) == 1 else args.image or None}]
    ids = set()
    for c in cases:
        cid = c["id"]
        if not isinstance(cid, str) or not cid or any(x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for x in cid):
            raise ValueError("Case IDs must be safe unique filename stems")
        if cid in ids:
            raise ValueError("Duplicate case ID: " + cid)
        ids.add(cid)
        if c["task"] not in {"t2i", "edit"} or not c.get("prompt"):
            raise ValueError("Each case needs task=t2i/edit and a prompt")
        if (c["width"], c["height"], c["num_inference_steps"], c["true_cfg_scale"]) != (1024, 1024, 50, 1.0):
            raise ValueError("Phase2 fixes 1024x1024, 50 steps, true_cfg_scale=1.0")
        images = c.get("input_image")
        images = images if isinstance(images, list) else [images] if images else []
        if c["task"] == "edit" and len(images) != 1:
            raise ValueError("Phase2 edit requires exactly one reference image")
        if c["task"] == "t2i" and images:
            raise ValueError("T2I must not contain an input image")
        c["input_image_sha256"] = [file_sha256(path) for path in images]
    return cases


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def pkg(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_env(args: argparse.Namespace) -> dict:
    import torch

    return {
        "vllm_omni_sha": args.vllm_omni_sha,
        "vllm_omni_checkout": sh("git -C /opt/vllm-omni rev-parse HEAD"),
        "vllm_omni_dirty": sh("git -C /opt/vllm-omni status --porcelain"),
        "model": args.model,
        "model_revision": args.model_revision,
        "packages": {n: pkg(n) for n in
                     ["vllm", "vllm-omni", "torch", "transformers", "diffusers", "flash-attn", "triton"]},
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "nvidia_smi": sh("nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,"
                         "memory.total,clocks.max.sm,power.limit --format=csv"),
        "nvidia_smi_topo": sh("nvidia-smi topo -m"),
        "hostname": platform.node(),
        "cpu": sh("lscpu | grep 'Model name'"),
        # Requested values, not resolved ones: neither script passes these, so the engine uses its
        # defaults (platform-default attention backend, prefix KV stored in the native dtype).
        "runtime_config": {"evidence_kind": "requested_only; worker quant_audit contains runtime observations",
                           "diffusion_attention_config": None, "prefix_kv_cache_dtype": None,
                           "diffusion_kv_cache_dtype": None, "dtype": "bfloat16",
                           "enforce_eager": True, "enable_cuda_graph_decode": False,
                           "quantization_config": args.quantization_config},
        "arm": args.arm,
        "harness_sha256": file_sha256(__file__),
        "source_harness": "luca-888/qwen-image-21-bench@cf99e73c/bench/a0_offline.py",
        "quant_config_file_sha256": file_sha256(args.quant_config_file) if args.quant_config_file else None,
        "cases_file_sha256": file_sha256(args.cases_file) if args.cases_file else None,
        "cases": args.cases,
        "worker_quant_audit_enabled": args.quant_audit,
        "quality_manifest_mode": bool(args.cases_file),
        "profiler_export_only_enabled": os.environ.get("A0_PROFILE_EXPORT_ONLY") == "1",
        "workload": {k: getattr(args, k) for k in
                     ["mode", "prompt", "image", "height", "width", "steps", "seed",
                      "true_cfg_scale", "enforce_eager", "warmup", "feasibility", "measured", "trace", "mem", "stages"]},
    }


class GpuMemSampler:
    """Polls device-used memory via nvidia-smi (whole device, including the CUDA context)."""

    def __init__(self, interval: float = 0.1):
        self.interval, self.peak_mib, self._stop = interval, 0, threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
            try:
                self.peak_mib = max(self.peak_mib, max(int(x) for x in out.split()))
            except ValueError:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def main() -> None:
    args = parse_args()
    prepare_c1(args)
    if args.mem:
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"  # before vllm is imported; inherited by engine workers
    if args.prompt is None:
        args.prompt = TEAPOT if args.mode == "t2i" else EDIT_PROMPT
    args.cases = load_cases(args)
    out = Path(args.out).resolve()
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(collect_env(args), indent=2))

    import torch
    from PIL import Image

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras.qwen_image_21 import build_image_to_image_prompt

    omni_kwargs = {
        "model": args.model,
        "revision": args.model_revision,
        "enable_diffusion_pipeline_profiler": True,
        "dtype": torch.bfloat16,
        "quantization_config": args.quantization_config,
        "enable_cuda_graph_decode": False,
        "diffusion_kv_cache_dtype": None,
        "extras": {"prefix_kv_cache_dtype": None},
    }
    if args.enforce_eager:
        omni_kwargs["enforce_eager"] = True
    if args.trace:
        omni_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(out / "trace"),
            "torch_profiler_use_gzip": True,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_stack": True,
            "torch_profiler_with_memory": True,
            "active_iterations": 1,
        }

    t0 = time.perf_counter()
    with GpuMemSampler() as init_mem:
        omni = Omni(**omni_kwargs)
    init_s = time.perf_counter() - t0
    # vllm does not configure log output for the offline entrypoint, so surface the two loggers
    # the diagnostic runs read: the runner's per-request peak line (DEBUG) and the profiler's
    # per-stage lines (INFO), which include `forward` and `_decode_latents`.
    surfaced = ([("vllm_omni.diffusion.worker.diffusion_model_runner", logging.DEBUG)] if args.mem else []) + \
               ([("vllm_omni.diffusion.profiler.diffusion_pipeline_profiler", logging.INFO)] if args.stages else [])
    for name, level in surfaced:
        log = logging.getLogger(name)
        log.setLevel(level)
        if not log.hasHandlers():
            log.addHandler(logging.StreamHandler())

    def make_prompt(case):
        if case["task"] == "t2i":
            return {"prompt": case["prompt"], "modalities": ["image"]}
        paths = case["input_image"]
        paths = paths if isinstance(paths, list) else [paths]
        imgs = [Image.open(path).convert("RGB") for path in paths]
        return build_image_to_image_prompt(
            prompt=case["prompt"], negative_prompt=None, input_image=imgs[0],
            height=case["height"], width=case["width"],
        )

    def make_params(case):
        return OmniDiffusionSamplingParams(
            height=case["height"], width=case["width"], seed=case["seed"],
            generator=torch.Generator(device="cuda").manual_seed(case["seed"]),
            true_cfg_scale=case["true_cfg_scale"],
            num_inference_steps=case["num_inference_steps"], num_outputs_per_prompt=1,
        )

    runs_f = (out / "runs.jsonl").open("w")
    if args.cases_file:
        jobs = [("warmup", args.cases[0])] * args.warmup + [("quality", c) for c in args.cases]
    else:
        phases = (["warmup"] * args.warmup
                  + ([] if args.trace else ["feasibility"] * args.feasibility + ["measured"] * args.measured)
                  + (["trace"] if args.trace else []))
        jobs = [(phase, args.cases[0]) for phase in phases]
    counters: dict[str, int] = {}

    for phase, case in jobs:
        prompt = make_prompt(case)
        idx = counters.get(phase, 0)
        counters[phase] = idx + 1
        row: dict = {"phase": phase, "index": idx, "arm": args.arm,
                     "id": case["id"], "case_id": case["id"], **case,
                     "output_path": None}
        if phase == "trace":
            omni.start_profile()
        try:
            with GpuMemSampler() as mem:
                t = time.perf_counter()
                outputs = omni.generate(prompt, sampling_params_list=[make_params(case)])
                row["wall_s"] = time.perf_counter() - t
            row["device_used_peak_mib"] = mem.peak_mib
            o = outputs[0]
            row["stage_durations_s"] = dict(getattr(o, "stage_durations", {}) or {})
            row["engine_peak_memory_mb"] = getattr(o, "peak_memory_mb", None)
            # The runner resets torch's peak counters before each request; non-zero here only if
            # the diffusion worker shares this process.
            row["torch_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            row["torch_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
            row["error"] = getattr(o, "error", None)
            images = getattr(o, "images", None) or []
            if images:
                buf = io.BytesIO()
                images[0].save(buf, format="PNG")
                row["image_sha256"] = hashlib.sha256(buf.getvalue()).hexdigest()
                filename = f"{case['id']}.png" if phase == "quality" else f"{phase}_{idx}.png"
                image_path = out / "images" / filename
                image_path.write_bytes(buf.getvalue())
                row["output_path"] = str(image_path)
            elif not row.get("error"):
                row["error"] = "MissingImageOutput: engine returned no image"
        except Exception as e:  # noqa: BLE001 - failures/OOMs stay in the record
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()
        if phase == "trace":
            row["profile_result"] = str(omni.stop_profile())
        print(json.dumps(row), flush=True)
        runs_f.write(json.dumps(row) + "\n")
        runs_f.flush()

    measured = [json.loads(l) for l in (out / "runs.jsonl").read_text().splitlines()]
    measured = [r for r in measured if r["phase"] == "measured" and not r.get("error")]

    def stats(xs):
        if not xs:
            return None
        return {"n": len(xs), "mean": statistics.fmean(xs), "min": min(xs), "max": max(xs),
                "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0, "median": statistics.median(xs)}

    stage_names = sorted({k for r in measured for k in r["stage_durations_s"]})
    all_rows = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    summary = {
        "arm": args.arm,
        "quality_cases": sum(r["phase"] == "quality" for r in all_rows),
        "successful_quality_cases": sum(r["phase"] == "quality" and not r.get("error") for r in all_rows),
        "statistics_scope": "within-process repeated fixed requests; not cross-start variability",
        "engine_init_s": init_s,
        "engine_init_device_used_peak_mib": init_mem.peak_mib,
        "wall_s": stats([r["wall_s"] for r in measured]),
        "device_used_peak_mib": stats([r["device_used_peak_mib"] for r in measured]),
        "stages_s": {s: stats([r["stage_durations_s"].get(s, 0.0) for r in measured]) for s in stage_names},
        "image_sha256_identical": len({r.get("image_sha256") for r in measured}) <= 1,
        "failures": sum(1 for l in (out / "runs.jsonl").read_text().splitlines() if json.loads(l).get("error")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.quant_audit:
        audit_dir = out / "quant_audit"
        required = ("worker_inventory_*.json", "prefix_runtime_*.json", "attention_runtime_*.json")
        missing = [pattern for pattern in required if not list(audit_dir.glob(pattern))]
        if missing:
            raise RuntimeError("Quant audit incomplete: missing actual worker evidence " + repr(missing))
        import importlib.util
        verifier_path = Path(__file__).parent / "quant_audit" / "verify_audit.py"
        spec = importlib.util.spec_from_file_location("c1_verify_audit", verifier_path)
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        verification = verifier.verify(audit_dir, args.arm, task=args.mode)
        (audit_dir / "verification.json").write_text(json.dumps(verification, indent=2))
        if not verification["passed"]:
            raise RuntimeError("Worker quant audit failed: " + repr(verification["errors"]))


if __name__ == "__main__":
    main()
