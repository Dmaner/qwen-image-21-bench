"""Sequential C1 H200 supplement. Resume only cells with validated success markers."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['bf16', 'dit_fp8', 'mlp_bf16', 'te_fp8', 'joint_fp8', 'prefix_fp8_v', 'prefix_fp8']
PREFIX = ['bf16', 'prefix_fp8_v', 'prefix_fp8']


def check_audit_backend(out):
    """Require FLASH_ATTN selection; FA3 GPU execution is checked in traces."""
    files = list((out/'quant_audit').glob('attention_runtime_*.json'))
    if len(files) != 1:
        raise RuntimeError(f'{out.name}: expected one attention audit')
    observations = json.loads(files[0].read_text()).get('observations', {})
    for phase in ('cond_prefill', 'cond_decode'):
        layers = observations.get(phase, {})
        if len(layers) != 32 or any('FlashAttentionBackend' not in layer.get('backend', '') for layer in layers.values()):
            raise RuntimeError(f'{out.name}: {phase} FLASH_ATTN backend unconfirmed')


def cells():
    # Metadata-only hooks must validate before any benchmark or quality inference.
    for arm in ARMS:
        yield 'audit', arm, 'edit'
    for task in ['t2i', 'multi_edit']:
        for arm in PREFIX:
            yield 'audit', arm, task
    for arm in ARMS:
        for task in ['t2i', 'edit']:
            yield 'perf', arm, task
    for arm in PREFIX:
        yield 'perf', arm, 'multi_edit'
    for arm in ARMS:
        for task in ['t2i', 'edit']:
            yield 'mem', arm, task
    for arm in PREFIX:
        yield 'mem', arm, 'multi_edit'
    for arm in ARMS:
        yield 'quality', arm, None
    for arm in PREFIX:
        yield 'quality_cfg4', arm, None
    for arm in PREFIX:
        yield 'trace', arm, 'multi_edit'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    if not (ROOT/'setup.ok').exists():
        raise RuntimeError('setup.ok missing')
    base_env = os.environ.copy()
    base_env.update(HF_HOME=str(ROOT/'hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                    TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                    VLLM_WORKER_MULTIPROC_METHOD='spawn')
    for key in ['C1_QUANT_AUDIT', 'C1_AUDIT_ARM', 'C1_AUDIT_OUT', 'C1_AUDIT_PREFIX_DTYPE',
                'A0_PROFILE_EXPORT_ONLY', 'PYTHONPATH']:
        base_env.pop(key, None)
    results = ROOT/'results'
    results.mkdir(exist_ok=True)
    for kind, arm, task in cells():
        if args.audit_only and kind != 'audit':
            continue
        name = '_'.join(x for x in (kind, arm, task) if x)
        out = results/name
        if (out/'cell.ok').exists():
            if kind == 'audit':
                check_audit_backend(out)
            print('SKIP validated', name, flush=True)
            continue
        if out.exists() and any(out.iterdir()):
            raise RuntimeError(f'Incomplete cell {name}; preserve and inspect it before a fresh attempt')
        out.mkdir(exist_ok=True)
        env = base_env.copy()
        cmd = [sys.executable, str(ROOT/'scripts/c1_offline.py'),
               '--model', 'Qwen/Qwen-Image-2.1',
               '--model-revision', 'd26bb61231c349cf6b7896fa83353113880e1ba3',
               '--vllm-omni-sha', '3dc35694b3d458fc1461a8fe71832ac12e8c8cea',
               '--arm', arm, '--enforce-eager', '--out', str(out), '--warmup', '1']
        if kind.startswith('quality'):
            manifest = 'cfg4_manifest.json' if kind == 'quality_cfg4' else 'effective_manifest.json'
            cmd += ['--cases-file', str(ROOT/'inputs'/manifest)]
        else:
            cmd += ['--case-file', str(ROOT/'inputs/cases'/f'{task}.json')]
            cmd += ['--feasibility', '1' if kind == 'perf' else '0',
                    '--measured', '10' if kind == 'perf' else '0' if kind == 'trace' else '1']
        if kind == 'audit':
            env.update(C1_QUANT_AUDIT='1', C1_AUDIT_ARM=arm,
                       C1_AUDIT_OUT=str(out/'quant_audit'),
                       PYTHONPATH=str(ROOT/'scripts/quant_audit'))
            cmd += ['--quant-audit']
        elif kind == 'mem':
            cmd += ['--mem']
        elif kind == 'trace':
            env.update(A0_PROFILE_EXPORT_ONLY='1', PYTHONPATH=str(ROOT/'reference/profiler_export_only'))
            cmd += ['--trace']
        (out/'command.txt').write_text(shlex.join(cmd)+'\n')
        (out/'environment-overrides.json').write_text(json.dumps({k:env[k] for k in env if k not in base_env or env[k] != base_env[k]},indent=2))
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        print(started, 'START', name, flush=True)
        (ROOT/'progress.json').write_text(json.dumps({'cell':name,'started':started,'status':'running'},indent=2))
        with (out/'engine.log').open('w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}; see engine.log')
        summary = json.loads((out/'summary.json').read_text())
        if summary['failures']:
            raise RuntimeError(f'{name} has {summary["failures"]} recorded failures; stop for inspection')
        if kind == 'perf' and summary['wall_s']['n'] != 10:
            raise RuntimeError('Wrong measured count')
        rows = [json.loads(line) for line in (out/'runs.jsonl').read_text().splitlines() if line.strip()]
        if any(row.get('error') for row in rows):
            raise RuntimeError(f'{name} raw request failure is not acceptable')
        if kind.startswith('quality'):
            expected_cases = json.loads((ROOT/'inputs'/manifest).read_text())['cases']
            expected_ids = [case['id'] for case in expected_cases]
            quality_rows = [row for row in rows if row['phase'] == 'quality']
            if ([row['case_id'] for row in quality_rows] != expected_ids
                    or summary.get('quality_cases') != len(expected_ids)
                    or summary.get('successful_quality_cases') != len(expected_ids)):
                raise RuntimeError(f'{name} missing, duplicated, reordered or unsuccessful quality cases')
            if any(not row.get('output_path') or not Path(row['output_path']).is_file()
                   or hashlib.sha256(Path(row['output_path']).read_bytes()).hexdigest() != row.get('image_sha256')
                   for row in quality_rows):
                raise RuntimeError(f'{name} missing or altered quality output image')
        if kind == 'audit' and not json.loads((out/'quant_audit/verification.json').read_text())['passed']:
            raise RuntimeError('Runtime audit did not pass')
        if kind == 'audit':
            check_audit_backend(out)
        if kind == 'mem':
            subprocess.run([sys.executable,str(ROOT/'reference/parse_diagnostics.py'),str(out),'--mem'],check=True,env=base_env)
            peaks = json.loads((out/'engine_peak_memory.json').read_text())
            if len(peaks) != 2 or any(not (0 < row['allocated_gib'] <= row['reserved_gib']) for row in peaks):
                raise RuntimeError(f'{name} must have exactly warmup + measured worker memory peaks')
        if kind == 'trace':
            # The pinned profiler spawns gzip asynchronously. Its input is
            # removed only after a successful completed compression.
            deadline = time.monotonic() + 120
            while True:
                traces = list((out/'trace').rglob('trace_rank*.json.gz'))
                ready = traces and not list((out/'trace').rglob('trace_rank*.json'))
                if ready and all(path.stat().st_size > 0 for path in traces):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'{name} missing/incomplete profiler trace compression')
                time.sleep(2)
        (out/'cell.ok').write_text(dt.datetime.now(dt.timezone.utc).isoformat()+'\n')
        print('DONE', name, flush=True)
    if not args.audit_only:
        (ROOT/'experiment.ok').write_text(dt.datetime.now(dt.timezone.utc).isoformat()+'\n')
    (ROOT/'progress.json').write_text(json.dumps({'status':'audits_complete' if args.audit_only else 'complete'},indent=2))

if __name__ == '__main__':
    main()
