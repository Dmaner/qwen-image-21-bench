"""Preview by default; execute three independent reverse-order T2I diagnostics."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RECHECK = ROOT / "recheck"
ORDER = ("prefix_fp8", "prefix_fp8_v", "bf16")
ENGINE_SOURCE_SHA256 = "9d5fbcc29c1166c1c45443e490c9764c5aa303dc456b467f6c28ded9c6a6037f"
OMNI_SHA = "3dc35694b3d458fc1461a8fe71832ac12e8c8cea"
MODEL_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
PRIMARY_RUNNER_SHA256 = "2f7107aec535ff5f8e2ae1f195274e0dc3cfd6b368730239a0bf4fbb119e6be5"
COMPLETE_ANALYSIS_STATUS = "complete_for_supplied_manifests_and_discovered_runs"
PRIMARY_PROCESS_SCRIPTS = frozenset((
    "execute_all.sh", "run_matrix.py", "c1_offline.py",
    "analyze_supplement.py", "analyze_prefix_traces.py",
))
REQUIRED_ENV_FIELDS = frozenset((
    "vllm_omni_sha", "vllm_omni_checkout", "vllm_omni_dirty", "model", "model_revision",
    "packages", "torch_cuda", "python", "nvidia_smi", "nvidia_smi_topo", "hostname", "cpu",
    "runtime_config", "arm", "harness_sha256", "source_harness", "quant_config_file_sha256",
    "case_file_sha256", "cases_file_sha256", "cases", "worker_quant_audit_enabled",
    "quality_manifest_mode", "profiler_export_only_enabled", "workload",
))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primary_process_snapshot(proc_root=Path("/proc")):
    """Fail closed when active process command lines cannot be inspected."""
    if not (proc_root / "self/cmdline").is_file():
        raise RuntimeError("Linux /proc process inspection is required before execution")
    matches, inspected = [], 0
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = path.joinpath("cmdline").read_bytes().decode("utf-8", "replace").split("\0")
        except (FileNotFoundError, ProcessLookupError):
            continue  # A process exited during this read.
        except OSError as exc:
            raise RuntimeError("Cannot inspect process " + path.name) from exc
        inspected += 1
        scripts = sorted({Path(arg).name for arg in argv if arg} & PRIMARY_PROCESS_SCRIPTS)
        if scripts:
            matches.append({"pid": int(path.name), "scripts": scripts})
    return {"inspected_processes": inspected, "primary_processes": matches,
            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}


def check_primary_ready(root=ROOT, proc_root=Path("/proc")):
    for name in ("experiment.ok", "analysis.ok"):
        if not (root / name).is_file():
            raise RuntimeError("Primary completion marker missing: " + name)
    path = root / "analysis/completed/analysis.json"
    analysis = json.loads(path.read_text())
    if analysis.get("analysis_status") != COMPLETE_ANALYSIS_STATUS:
        raise RuntimeError("Primary analysis is incomplete; a marker alone is insufficient")
    processes = primary_process_snapshot(proc_root)
    if processes["primary_processes"]:
        raise RuntimeError("Primary pipeline is still active: " + json.dumps(processes["primary_processes"]))
    return {"analysis_status": analysis["analysis_status"], "analysis_sha256": sha256(path),
            "completion_markers": ["experiment.ok", "analysis.ok"], "process_snapshot": processes}


def compare_environment(actual, original, arm):
    """Every recorded field must agree except the explicitly changed harness."""
    for label, env in (("original", original), ("recheck", actual)):
        if REQUIRED_ENV_FIELDS - env.keys():
            raise RuntimeError(label + " environment is missing required provenance fields")
        if (env["vllm_omni_checkout"] != OMNI_SHA or env["vllm_omni_sha"] != OMNI_SHA
                or env["vllm_omni_dirty"] != "" or env["model"] != "Qwen/Qwen-Image-2.1"
                or env["model_revision"] != MODEL_REVISION or env["arm"] != arm):
            raise RuntimeError(label + " actual code/model/arm does not match the frozen clean runtime")
    if original["harness_sha256"] != PRIMARY_RUNNER_SHA256:
        raise RuntimeError("Original cohort has an unexpected harness hash")
    differences = [key for key in sorted(set(actual) | set(original))
                   if key != "harness_sha256" and (key not in actual or key not in original
                                                  or actual[key] != original[key])]
    if differences:
        raise RuntimeError("Original/recheck environment mismatch: " + ", ".join(differences))
    gpu_rows = list(csv.DictReader(actual["nvidia_smi"].splitlines(), skipinitialspace=True))
    if len(gpu_rows) != 1 or not gpu_rows[0].get("uuid", "").startswith("GPU-") or not gpu_rows[0].get("driver_version"):
        raise RuntimeError("Missing single-GPU UUID/driver provenance")
    if actual["quant_config_file_sha256"] != sha256(ROOT / "scripts/configs" / (arm + ".json")):
        raise RuntimeError("Runtime quantization config hash differs from the reviewed file")
    if actual["case_file_sha256"] != sha256(ROOT / "inputs/cases/t2i.json"):
        raise RuntimeError("Runtime case hash differs from the reviewed file")
    if actual["cases_file_sha256"] is not None or len(actual["cases"]) != 1:
        raise RuntimeError("Recheck requires the single frozen T2I case")
    return {"matched": True, "compared_fields": sorted(set(actual) - {"harness_sha256"}),
            "allowed_differences": ["harness_sha256"],
            "gpu_uuid": gpu_rows[0]["uuid"], "driver_version": gpu_rows[0]["driver_version"],
            "actual_omni_checkout": actual["vllm_omni_checkout"], "actual_omni_dirty": actual["vllm_omni_dirty"],
            "model_revision": actual["model_revision"], "original_runner_sha256": original["harness_sha256"]}


def host_snapshot():
    paths = [
        "/proc/loadavg", "/proc/pressure/cpu", "/proc/pressure/io", "/proc/pressure/memory",
        "/proc/self/cgroup", "/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpu.pressure", "/sys/fs/cgroup/io.pressure", "/sys/fs/cgroup/memory.pressure",
    ]
    result = {"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "perf_counter": time.perf_counter(), "files": {}, "errors": {}}
    for name in paths:
        try:
            result["files"][name] = Path(name).read_text()
        except OSError as exc:
            result["errors"][name] = str(exc)
    return result


def command(arm, out):
    return [sys.executable, str(ROOT / "scripts/c1_offline_recheck.py"),
            "--model", "Qwen/Qwen-Image-2.1",
            "--model-revision", "d26bb61231c349cf6b7896fa83353113880e1ba3",
            "--vllm-omni-sha", "3dc35694b3d458fc1461a8fe71832ac12e8c8cea",
            "--arm", arm, "--enforce-eager", "--case-file", str(ROOT / "inputs/cases/t2i.json"),
            "--warmup", "1", "--feasibility", "1", "--measured", "10", "--out", str(out)]


def verify_cell(out, diag, arm):
    rows = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines() if line]
    summary = json.loads((out / "summary.json").read_text())
    if [r["phase"] for r in rows] != ["warmup", "feasibility"] + ["measured"] * 10:
        raise RuntimeError("Unexpected recheck request count/order")
    if any(r.get("error") for r in rows) or summary["failures"] or summary["wall_s"]["n"] != 10:
        raise RuntimeError("Recheck contains failures or incomplete measured data")
    expected_runner = sha256(ROOT / "scripts/c1_offline_recheck.py")
    actual_env = json.loads((out / "env.json").read_text())
    if actual_env["harness_sha256"] != expected_runner or actual_env["arm"] != arm:
        raise RuntimeError("Recheck runner provenance mismatch")
    original_path = ROOT / "results" / ("perf_" + arm + "_t2i") / "env.json"
    original_env = json.loads(original_path.read_text())
    environment_match = compare_environment(actual_env, original_env, arm)
    case = actual_env["cases"][0]
    for row in rows:
        if (row.get("arm") != arm or row.get("case_id") != case["id"]
                or any(key not in row or row[key] != value for key, value in case.items())
                or row.get("prefix_kv_cache_dtype") != actual_env["runtime_config"]["prefix_kv_cache_dtype"]):
            raise RuntimeError("Completed row does not match the frozen case/input/runtime")
    event_files = list(diag.glob("engine_breakdown_*.jsonl"))
    if len(event_files) != 1:
        raise RuntimeError("Expected one actual engine breakdown stream")
    events = [json.loads(line) for line in event_files[0].read_text().splitlines() if line]
    if len(events) != len(rows):
        raise RuntimeError("Engine breakdown must cover every completed request")
    mapping = []
    for ordinal, (row, event) in enumerate(zip(rows, events), 1):
        interval = row["diagnostic_wall_interval"]
        if (event["arm"] != arm or event["engine_event_ordinal"] != ordinal
                or event["source_module_sha256"] != ENGINE_SOURCE_SHA256
                or not interval["start_perf_counter"] <= event["perf_counter"] <= interval["end_perf_counter"]):
            raise RuntimeError("Engine timing event is not within the corresponding caller wall interval")
        mapping.append({"ordinal": ordinal, "phase": row["phase"], "index": row["index"],
                        "case_id": row["case_id"], "timings_ms": event["timings_ms"]})
    observer_sha = sha256(RECHECK / "timing_observer/sitecustomize.py")
    manifests = [json.loads(p.read_text()) for p in diag.glob("observer_manifest_*.json")]
    if not manifests or any(x["observer_sha256"] != observer_sha or x["arm"] != arm
                            or x["source_module_sha256"] != ENGINE_SOURCE_SHA256 for x in manifests):
        raise RuntimeError("Observer provenance missing/mismatched")
    if not {event["pid"] for event in events} <= {manifest["pid"] for manifest in manifests}:
        raise RuntimeError("Engine timing emitter has no matching source-guard manifest")
    result = {"passed": True, "arm": arm, "request_count": len(rows), "measured_count": 10,
              "runner_sha256": expected_runner, "observer_sha256": observer_sha,
              "environment_match": environment_match,
              "original_env_sha256": sha256(original_path), "recheck_env_sha256": sha256(out / "env.json"),
              "engine_breakdown_file": str(event_files[0]), "request_mapping": mapping}
    (diag / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Only after primary experiment and analysis finish, and review of the recheck design")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"mode": "preview_only_no_gpu", "order": ORDER,
                          "commands": [shlex.join(command(a, RECHECK / "results" / ("perf_" + a + "_t2i"))) for a in ORDER]}, indent=2))
        return
    check_primary_ready()
    manifest = json.loads((RECHECK / "CODE_HASHES.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError("Reviewed recheck code/config changed: " + name)
    os.chdir(ROOT)
    env = os.environ.copy()
    for key in ("C1_QUANT_AUDIT", "C1_AUDIT_ARM", "C1_AUDIT_OUT", "C1_AUDIT_PREFIX_DTYPE", "A0_PROFILE_EXPORT_ONLY", "PYTHONPATH"):
        env.pop(key, None)
    env.update(HF_HOME=str(ROOT / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", VLLM_WORKER_MULTIPROC_METHOD="spawn",
               C1_RECHECK_TIMING="1", PYTHONPATH=str(RECHECK / "timing_observer"))
    for arm in ORDER:
        readiness = check_primary_ready()
        name = "perf_" + arm + "_t2i"
        out = RECHECK / "results" / name
        diag = RECHECK / "diagnostics" / name
        if out.exists() or diag.exists():
            raise RuntimeError("Fresh recheck directories required; never overwrite an attempt: " + name)
        diag.mkdir(parents=True)
        (diag / "primary_readiness.json").write_text(json.dumps(readiness, indent=2) + "\n")
        cmd = command(arm, out)
        cell_env = {**env, "C1_RECHECK_ARM": arm, "C1_RECHECK_TIMING_OUT": str(diag)}
        (diag / "command.txt").write_text(shlex.join(cmd) + "\n")
        (diag / "diagnostic_flags.json").write_text(json.dumps({k:cell_env[k] for k in ("C1_RECHECK_TIMING", "C1_RECHECK_ARM", "C1_RECHECK_TIMING_OUT", "PYTHONPATH")}, indent=2) + "\n")
        (diag / "host_before.json").write_text(json.dumps(host_snapshot(), indent=2) + "\n")
        print("START recheck", arm, flush=True)
        try:
            with (diag / "engine.log").open("w") as log:
                completed = subprocess.run(cmd, env=cell_env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        finally:
            (diag / "host_after.json").write_text(json.dumps(host_snapshot(), indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError("Recheck failed; preserve evidence: " + name)
        verify_cell(out, diag, arm)
        (out / "cell.ok").write_text(datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
        print("DONE recheck", arm, flush=True)
    (RECHECK / "experiment.ok").write_text(datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")


if __name__ == "__main__":
    main()
