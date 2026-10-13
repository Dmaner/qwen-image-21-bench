"""Optional A0 diagnostic adapter: skip only Omni's CPU summary/Excel stop hook.

Inactive unless A0_PROFILE_EXPORT_ONLY=1. Preserves the pinned profiler creation,
Chrome trace export, gzip launch, and finally memory snapshot code. No inference
code or upstream files are changed. Not a clean upstream profiling run.
"""
import os
import sys

TARGET = 'vllm_omni.profiler.omni_torch_profiler'
PINNED_SOURCE_SHA256 = 'c6fdde889400d5ed45f890fbb616a456e098c175b5eeda364205d2fe30e746fb'


def _patch(module):
    import hashlib
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if actual != PINNED_SOURCE_SHA256:
        raise RuntimeError(f'A0 export-only adapter requires exact Omni 3dc profiler source; got {actual}')
    wrapper = module.OmniTorchProfilerWrapper
    if getattr(wrapper, '_a0_export_only_adapter', False):
        return

    def export_only_stop_hook(self):
        marker = Path(self._ensure_session_dir()) / f'a0_export_only_adapter_rank{self._rank()}.json'
        marker.write_text(json.dumps({
            'adapter': 'A0_PROFILE_EXPORT_ONLY',
            'time_utc': datetime.now(timezone.utc).isoformat(),
            'pid': os.getpid(),
            'omni_revision': '3dc35694b3d458fc1461a8fe71832ac12e8c8cea',
            'original_profiler_source_sha256': actual,
            'skipped': ['summary/shape/stack key_averages', 'CPU/CUDA text export_stacks', 'ops Excel workbook', 'profiler_out table'],
            'unchanged': ['profiler creation and collection options', 'profiler.stop', 'Chrome trace and background gzip', 'finally memory snapshot'],
        }, indent=2) + '\n')
        self._artifact_paths['a0_export_only_adapter'] = str(marker)
        print(f'[a0-export-only] skipped CPU aggregation; memory snapshot follows; marker={marker}', file=sys.stderr, flush=True)

    wrapper._on_stop_hook = export_only_stop_hook
    wrapper._a0_export_only_adapter = True
    print(f'[a0-export-only] patched only _on_stop_hook in pid {os.getpid()}; source sha256={actual}', file=sys.stderr, flush=True)


if os.environ.get('A0_PROFILE_EXPORT_ONLY') == '1':
    import importlib.abc
    import importlib.util

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                raise ImportError(f'Cannot find profiler module {name}')
            original_exec = spec.loader.exec_module

            def exec_module(module):
                original_exec(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
