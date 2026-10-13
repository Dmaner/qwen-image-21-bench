"""Capture one existing engine timing log per request; no tensor/model hooks."""
import hashlib
import importlib.abc
import importlib.machinery
import json
import logging
import os
from pathlib import Path
import threading
import time

MODULE = "vllm_omni.diffusion.diffusion_engine"
SOURCE_SHA256 = "9d5fbcc29c1166c1c45443e490c9764c5aa303dc456b467f6c28ded9c6a6037f"
MESSAGE = (
    "DiffusionEngine.step_streaming breakdown: preprocess=%.2f ms, "
    "add_req_and_wait=%.2f ms, output_ready_wait=%.2f ms, "
    "postprocess=%.2f ms, total=%.2f ms"
)
FIELDS = ("preprocess_ms", "add_req_and_wait_ms", "output_ready_wait_ms", "postprocess_ms", "total_ms")


class _ModuleFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.INFO or record.msg == MESSAGE


class _BreakdownHandler(logging.Handler):
    def __init__(self, directory):
        super().__init__(logging.DEBUG)
        self.directory = Path(directory)
        self.ordinal = 0

    def emit(self, record):
        if record.msg != MESSAGE:
            return
        values = record.args
        if not isinstance(values, tuple) or len(values) != len(FIELDS):
            raise RuntimeError("Pinned engine breakdown log signature changed")
        self.ordinal += 1
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "arm": os.environ["C1_RECHECK_ARM"],
            "engine_event_ordinal": self.ordinal,
            "perf_counter": time.perf_counter(),
            "time_unix": record.created,
            "source_module_sha256": SOURCE_SHA256,
            "timings_ms": dict(zip(FIELDS, map(float, values))),
            "measurement_source": "unrounded arguments of existing step_streaming DEBUG log",
        }
        with (self.directory / f"engine_breakdown_{os.getpid()}.jsonl").open("a") as out:
            out.write(json.dumps(payload, allow_nan=False) + "\n")


def _install(module):
    source_path = Path(module.__file__)
    actual_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual_sha256 != SOURCE_SHA256:
        raise RuntimeError("C1 recheck engine source hash mismatch: " + actual_sha256)
    directory = Path(os.environ["C1_RECHECK_TIMING_OUT"])
    directory.mkdir(parents=True, exist_ok=True)
    logger = module.logger
    original_level = logger.level
    logger.addFilter(_ModuleFilter())
    logger.addHandler(_BreakdownHandler(directory))
    logger.setLevel(logging.DEBUG)
    manifest = {
        "schema_version": 1, "pid": os.getpid(), "arm": os.environ["C1_RECHECK_ARM"],
        "source_module": MODULE, "source_path": str(source_path),
        "source_module_sha256": actual_sha256,
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "logger_name": logger.name, "original_logger_level": original_level,
        "enabled_logger_level": logging.DEBUG,
        "scope": "This module only; DEBUG filtered to one existing breakdown message; INFO and higher preserved",
        "adds_gpu_work_or_synchronization": False,
    }
    (directory / f"observer_manifest_{os.getpid()}.json").write_text(json.dumps(manifest, indent=2) + "\n")


if os.environ.get("C1_RECHECK_TIMING") == "1":
    import sys

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != MODULE:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load guarded C1 timing module")
            original_exec = spec.loader.exec_module

            def exec_module(module):
                original_exec(module)
                _install(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
