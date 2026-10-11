#!/usr/bin/env python3
"""Read synchronized C1 results; write audited statistics without running models.

  python3 scripts/analyze_performance.py --root . --arm-manifest scripts/configs/arms.json

Success ratios require complete ten-measured-request protocols, consistent core
runtime/source/GPU metadata and matched pair workloads. Partial raw statistics
remain visible with validation errors. Only the second independent memory sample
is used. No Torch import, SSH, cloud request or input/result modification.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import math
import statistics
from pathlib import Path

SOURCE = '3dc35694b3d458fc1461a8fe71832ac12e8c8cea'
MODEL_REVISION = 'd26bb61231c349cf6b7896fa83353113880e1ba3'
MODEL = 'Qwen/Qwen-Image-2.1'
QUANT = {'transformer': {'method': 'fp8'}, 'text_encoder': None, 'vae': None, 'default': None}
EXPECTED_PACKAGES = {'vllm': '0.31.0', 'torch': '2.13.0+cu130', 'triton': '3.7.1',
                     'transformers': '5.14.1', 'diffusers': '0.40.0'}
ARMS = ('bf16', 'dit_fp8', 'te_fp8', 'joint_fp8', 'selective_bf16')
ARM_SPECS = {}
OFFLINE_HARNESS = 'c1_offline.py'
SERVING_HARNESS = 'c1_serving.py'
MODES = ('t2i', 'edit', 'serving')


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path, errors, optional=False):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        if not optional:
            errors.append(f'{path.name}: {type(e).__name__}: {e}')
        return None


def read_rows(path, errors):
    rows = []
    try:
        lines = path.read_text().splitlines()
    except OSError as e:
        errors.append(f'{path.name}: {e}')
        return rows
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('row is not an object')
            rows.append(row)
        except ValueError as e:
            errors.append(f'{path.name} line {i}: {e}; raw={line[:300]!r}')
    return rows


def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def stats(values):
    xs = [float(x) for x in values if number(x)]
    return {'n': len(xs), 'mean': statistics.fmean(xs) if xs else None,
            'sample_sd': statistics.stdev(xs) if len(xs) > 1 else None,
            'min': min(xs) if xs else None, 'max': max(xs) if xs else None,
            'median': statistics.median(xs) if xs else None, 'values': xs}


def stage0(row):
    stages = ((row.get('response_meta') or {}).get('metrics') or {}).get('stage_metrics') or {}
    if isinstance(stages, list):
        return stages[0] if stages and isinstance(stages[0], dict) else {}
    return stages.get('0', stages.get(0, {})) if isinstance(stages, dict) else {}


def engine_seconds(row):
    v = stage0(row).get('stage_gen_time_ms')
    return v / 1000 if number(v) else None


def gpu_metadata(env, errors):
    raw = env.get('nvidia_smi')
    if not isinstance(raw, str) or not raw.strip() or '<error:' in raw:
        errors.append('Missing/invalid nvidia_smi evidence')
        return None
    data = list(csv.DictReader(io.StringIO(raw.strip())))
    if len(data) != 1:
        errors.append(f'Expected single-GPU metadata row, got {len(data)}')
        return None
    result = {str(k).strip(): str(v).strip() for k, v in data[0].items()}
    for key in ('name', 'uuid', 'driver_version'):
        if not result.get(key) or result[key] in ('None', 'N/A'):
            errors.append(f'Missing GPU metadata {key}')
    if 'H200' not in result.get('name', ''):
        errors.append('GPU name is not the planned H200 workload')
    if not result.get('uuid', '').startswith('GPU-'):
        errors.append('GPU UUID was not captured as a valid GPU-* value')
    return result


def env_audit(env, arm, mode, root, errors):
    if not isinstance(env, dict):
        errors.append('env.json missing or invalid')
        return {}, {}
    for key, expected in [('vllm_omni_sha', SOURCE), ('vllm_omni_checkout', SOURCE),
                          ('vllm_omni_dirty', ''), ('model', MODEL), ('model_revision', MODEL_REVISION),
                          ('arm', arm), ('torch_cuda', '13.0'), ('python', '3.12.3')]:
        if env.get(key) != expected:
            errors.append(f'Environment {key}: {env.get(key)!r}, expected {expected!r}')
    packages = env.get('packages') or {}
    for key, expected in EXPECTED_PACKAGES.items():
        if packages.get(key) != expected:
            errors.append(f'Package {key}: {packages.get(key)!r}, expected {expected!r}')
    if not packages.get('vllm-omni') or 'flash-attn' not in packages:
        errors.append('Incomplete package evidence (vllm-omni or flash-attn field absent)')
    for key in ('hostname', 'cpu', 'nvidia_smi_topo'):
        if not isinstance(env.get(key), str) or not env[key].strip():
            errors.append(f'Missing core environment field {key}')
    runtime = env.get('runtime_config') or {}
    requested_quant = env.get('component_quant_config_requested') if mode == 'serving' else runtime.get('quantization_config')
    spec = ARM_SPECS.get(arm, {})
    relative_config = spec.get('quant_config_file')
    quant_path = root / 'scripts' / relative_config if relative_config else None
    expected_quant = read_json(quant_path, errors) if quant_path else None
    if quant_path and spec.get('quant_config_sha256') and quant_path.exists() and file_sha(quant_path) != spec['quant_config_sha256']:
        errors.append('Config file hash disagrees with frozen arm manifest')
    bf16_omitted_equivalent = arm == 'bf16' and requested_quant is None and (
        expected_quant is None or isinstance(expected_quant, dict)
        and set(expected_quant) == {'transformer', 'text_encoder', 'vae', 'default'}
        and all(v is None for v in expected_quant.values()))
    if requested_quant != expected_quant and not bf16_omitted_equivalent:
        errors.append(f'Unexpected requested component quantization for {arm}: {requested_quant!r}')
    workload = env.get('workload') or {}
    for key, expected in [('height', 1024), ('width', 1024), ('steps', 50), ('seed', 42),
                          ('true_cfg_scale', 1.0), ('enforce_eager', True)]:
        if workload.get(key) != expected:
            errors.append(f'Workload {key}: {workload.get(key)!r}, expected {expected!r}')
    expected_mode = 't2i-serving' if mode == 'serving' else mode
    if workload.get('mode') != expected_mode:
        errors.append(f'Unexpected workload mode {workload.get("mode")!r}')
    harness = root / 'scripts' / (SERVING_HARNESS if mode == 'serving' else OFFLINE_HARNESS)
    if not env.get('harness_sha256'):
        errors.append('Harness SHA256 absent')
    elif not harness.exists():
        errors.append(f'Synchronized harness missing: {harness}')
    elif env['harness_sha256'] != file_sha(harness):
        errors.append('Saved harness hash does not match the synchronized script')
    quant_hash = env.get('quant_config_sha256') if mode == 'serving' else env.get('quant_config_file_sha256')
    if bf16_omitted_equivalent and quant_hash is not None:
        errors.append('BF16 omitted quantization unexpectedly records a config file hash')
    if quant_path and not bf16_omitted_equivalent and (not quant_hash or not quant_path.exists() or quant_hash != file_sha(quant_path)):
        errors.append('FP8 component configuration hash absent or differs from synchronized file')
    core = {k: env.get(k) for k in ('vllm_omni_checkout', 'vllm_omni_dirty', 'model', 'model_revision',
                                   'packages', 'python', 'torch_cuda', 'hostname', 'cpu', 'nvidia_smi_topo')}
    core['gpu'] = gpu_metadata(env, errors)
    # Deliberately omit mem/stages/trace/warmup counts and per-entrypoint harness IDs.
    # They are protocol-specific and must not be compared across timing and memory.
    pair = {k: workload.get(k) for k in ('mode', 'prompt', 'height', 'width', 'steps', 'seed', 'true_cfg_scale', 'enforce_eager')}
    if mode == 'edit':
        cases = env.get('cases') or []
        pair['input_image_sha256'] = cases[0].get('input_image_sha256') if len(cases) == 1 else None
        if not pair['input_image_sha256']:
            errors.append('Edit input image content hash absent')
    return core, pair


def image_readback(directory, row, errors):
    expected = row.get('image_sha256')
    image = directory / 'images' / f"{row.get('phase')}_{row.get('index')}.png"
    if not expected:
        errors.append(f"{row.get('phase')}[{row.get('index')}]: missing image SHA256")
        return None
    if not image.exists():
        errors.append(f'{image.relative_to(directory)} not synchronized or absent')
        return None
    actual = file_sha(image)
    if actual != expected:
        errors.append(f'{image.relative_to(directory)} SHA256 mismatch')
    return {'path': str(image.resolve()), 'recorded_sha256': expected, 'actual_sha256': actual, 'matches': actual == expected}


def load_group(root, kind, arm, mode):
    directory = root / 'results' / f'{kind}_{arm}_{mode}'
    errors, notes = [], []
    env = read_json(directory / 'env.json', errors)
    if not isinstance(env, dict):
        env = {}
    rows = read_rows(directory / 'runs.jsonl', errors)
    summary = read_json(directory / 'summary.json', errors)
    if not isinstance(summary, dict):
        errors.append('summary.json missing/invalid completion evidence')
        summary = {}
    environment_errors = []
    core, workload = env_audit(env, arm, mode, root, environment_errors)
    errors.extend(environment_errors)
    expected_n = 10 if kind == 'perf' else 1
    expected_phases = {'warmup': 1, 'measured': expected_n}
    if kind == 'perf':
        expected_phases['feasibility'] = 1
    phase_counts = dict(collections.Counter(r.get('phase') for r in rows))
    if phase_counts != expected_phases:
        errors.append(f'Phase counts {phase_counts}, expected {expected_phases}')
    row_errors = [{'phase': r.get('phase'), 'index': r.get('index'), 'error': r.get('error'),
                   'traceback': r.get('traceback')} for r in rows if r.get('error')]
    if row_errors:
        errors.append(f'{len(row_errors)} request error(s), retained in row_errors')
    measured = [r for r in rows if r.get('phase') == 'measured']
    success = [r for r in measured if not r.get('error')]
    indices = [r.get('index') for r in measured]
    if not all(isinstance(i, int) and not isinstance(i, bool) for i in indices) or sorted(indices) != list(range(expected_n)):
        errors.append('Measured indices are incomplete, duplicated or unexpected')
    if len(success) != expected_n:
        errors.append(f'Expected {expected_n} successful measured rows, got {len(success)}')
    env_workload = (env or {}).get('workload') or {}
    for k, expected in [('warmup', 1), ('feasibility', 1 if kind == 'perf' else 0), ('measured', expected_n)]:
        if env_workload.get(k) != expected:
            errors.append(f'Saved protocol {k}={env_workload.get(k)!r}, expected {expected}')
    # Check eligibility per protocol, without treating intended memory flags as cross-group drift.
    if env_workload.get('trace') or env_workload.get('stages') or kind == 'perf' and env_workload.get('mem'):
        errors.append('Unexpected diagnostic instrumentation in this protocol')
    if kind == 'mem' and env_workload.get('mem') is not True:
        errors.append('Independent memory diagnostic lacks mem=True metadata')
    if (env or {}).get('worker_quant_audit_enabled') or (env or {}).get('profiler_export_only_enabled'):
        errors.append('Worker audit/profile adapter was enabled in a performance/memory group')
    metrics = {}
    fields = {'serving_client_s': lambda r: r.get('client_total_s'), 'serving_engine_s': engine_seconds} if mode == 'serving' else {'offline_wall_s': lambda r: r.get('wall_s')}
    for name, getter in fields.items():
        values = [getter(r) for r in success]
        invalid = [r.get('index') for r, v in zip(success, values) if not number(v) or v <= 0]
        if invalid:
            errors.append(f'{name}: missing/nonpositive value for successful measured indices {invalid}')
        metrics[name] = stats(v for v in values if number(v) and v > 0)
    stage_stats = {}
    if mode != 'serving':
        keys = sorted({k for r in success for k in (r.get('stage_durations_s') or {})})
        for key in keys:
            raw = [(r.get('stage_durations_s') or {}).get(key) for r in success]
            factor = .001 if key.endswith('_ms') else 1
            stage_stats[key] = {'source_unit': 'ms' if factor == .001 else 's',
                                'missing_or_invalid_rows': [r.get('index') for r, v in zip(success, raw) if not number(v)],
                                'seconds': stats(v * factor for v in raw if number(v))}
    images = [x for row in rows if not row.get('error') and (x := image_readback(directory, row, errors))]
    if summary.get('failures') != len(row_errors):
        errors.append('summary.failures does not match raw request errors')
    initialization = {'source': 'summary.json', 'engine_init_s': summary.get('engine_init_s'),
                      'process_to_ready_s': summary.get('process_to_ready_s'),
                      'boundary': 'Separate startup observation, excluded from measured latency; not a cross-start distribution.'}
    if mode == 'serving' and kind == 'perf' and summary.get('benchmark_c1_returncode') != 0:
        notes.append('Additional random-prompt serving benchmark failed or is incomplete; it is excluded from fixed-prompt n=10 statistics.')
    memory = None
    if kind == 'mem':
        peaks = read_json(directory / 'engine_peak_memory.json', errors)
        if not isinstance(peaks, list) or len(peaks) != 2:
            errors.append('Memory log must contain exactly two entries: warmup then measured; no entry selected')
        elif not all(isinstance(item, dict) for item in peaks):
            errors.append('Memory peak log entries must be JSON objects')
        elif len(rows) != 2 or [r.get('phase') for r in rows] != ['warmup', 'measured'] or len(success) != 1:
            errors.append('Cannot align peak-memory entry 2 with exactly one successful measured request')
        else:
            peak, row = peaks[1], success[0]
            allocated, reserved, used = peak.get('allocated_gib'), peak.get('reserved_gib'), row.get('device_used_peak_mib')
            if not all(number(x) and x > 0 for x in (allocated, reserved, used)):
                errors.append('Measured allocated/reserved/device memory values missing or invalid')
            else:
                memory = {'sample_count': 1, 'peak_log_zero_based_index': 1, 'phase': 'measured',
                          'allocated_gib': allocated, 'reserved_gib': reserved,
                          'device_used_peak_mib': used, 'device_used_peak_gib': used / 1024,
                          'boundary': 'One independent measured request after one warmup; do not average both log rows. Allocated/reserved/device-used are different counters.'}
                for key, expected in [('torch_max_memory_allocated_gib', allocated), ('torch_max_memory_reserved_gib', reserved)]:
                    v = row.get(key)
                    if number(v) and abs(v - expected) > .006:
                        errors.append(f'Memory {key} disagrees with rounded runner peak line: {v} vs {expected}')
                v = row.get('engine_peak_memory_mb')
                if number(v) and abs(v / 1024 - reserved) > .006:
                    errors.append('engine_peak_memory_mb/1024 disagrees with reserved GiB (it is not allocated memory)')
    files = {f.name: {'sha256': file_sha(f), 'bytes': f.stat().st_size} for f in
             [directory / 'env.json', directory / 'runs.jsonl', directory / 'summary.json', directory / 'engine_peak_memory.json'] if f.exists()}
    return {'id': directory.name, 'path': str(directory.resolve()), 'kind': kind, 'arm': arm, 'mode': mode,
            'expected_measured': expected_n, 'successful_measured_rows': len(success), 'phase_counts': phase_counts,
            'metrics': metrics, 'stage_statistics': stage_stats, 'initialization': initialization, 'memory': memory,
            'core_environment': core, 'environment_errors': environment_errors, 'paired_workload': workload, 'row_errors': row_errors, 'errors': errors,
            'notes': notes, 'image_readback': images, 'source_files': files}


def compare_metric(base, candidate, field, global_errors):
    reasons = list(global_errors) + base['errors'] + candidate['errors']
    if base['paired_workload'] != candidate['paired_workload']:
        reasons.append('Baseline/candidate paired workload mismatch')
    a, b = base['metrics'].get(field, {}), candidate['metrics'].get(field, {})
    if a.get('n') != 10 or b.get('n') != 10:
        reasons.append('Not exactly 10 successful measured observations per arm')
    if not number(a.get('mean')) or not number(b.get('mean')) or a.get('mean', 0) <= 0 or b.get('mean', 0) <= 0:
        reasons.append('Missing/nonpositive means')
    valid = not reasons
    return {'metric': field, 'baseline_arm': base['arm'], 'candidate_arm': candidate['arm'],
            'baseline': a, 'candidate': b, 'valid_comparison': valid,
            'baseline_over_candidate': a['mean'] / b['mean'] if valid else None,
            'latency_reduction_percent': 100 * (1 - b['mean'] / a['mean']) if valid else None,
            'invalid_reasons': reasons,
            'interpretation': 'selective versus dit_fp8 isolates the selected DiT BF16 protections; joint comparison also changes TE quantization' if candidate['arm'] == 'selective_bf16' else 'Paired workload; quality is a separate gate.'}


def load_arm_specs(path, arms):
    raw = json.loads(path.read_text())
    data = raw.get('arms', raw) if isinstance(raw, dict) else raw
    if isinstance(data, list):
        data = {x.get('arm', x.get('name')): x for x in data}
    missing = set(arms) - set(data)
    if missing:
        raise ValueError(f'Arm manifest missing {sorted(missing)}')
    return {k: data[k] for k in arms}


def comparisons_for(arms):
    pairs = [('bf16', arm) for arm in arms if arm != 'bf16']
    for base in ('dit_fp8', 'joint_fp8'):
        if base in arms and 'selective_bf16' in arms:
            pairs.append((base, 'selective_bf16'))
    return pairs


def analyze(root, arms=ARMS, modes=MODES):
    groups = {}
    for kind in ('perf', 'mem'):
        for arm in arms:
            for mode in modes if kind == 'perf' else [m for m in modes if m != 'serving']:
                g = load_group(root, kind, arm, mode)
                groups[g['id']] = g
    global_errors = [f"{g['id']}: {e}" for g in groups.values() for e in g['environment_errors']]
    reference_name = f'perf_bf16_{modes[0]}'
    reference = groups[reference_name]['core_environment']
    for name, group in groups.items():
        for key in sorted(set(reference) | set(group['core_environment'])):
            if reference.get(key) != group['core_environment'].get(key):
                global_errors.append(f'{name} core field {key} differs from {reference_name}')
    comparisons, stage_comparisons, memory_comparisons = {}, {}, {}
    for base_arm, candidate_arm in comparisons_for(arms):
        pair = f'{base_arm}_vs_{candidate_arm}'
        for mode in modes:
            a, b = groups[f'perf_{base_arm}_{mode}'], groups[f'perf_{candidate_arm}_{mode}']
            for field in ('serving_client_s', 'serving_engine_s') if mode == 'serving' else ('offline_wall_s',):
                comparisons[f'{pair}.{mode}.{field}'] = compare_metric(a, b, field, global_errors)
            if mode == 'serving':
                continue
            for stage in sorted(set(a['stage_statistics']) | set(b['stage_statistics'])):
                x = {**a, 'metrics': {'stage_s': a['stage_statistics'].get(stage, {}).get('seconds', {})}}
                y = {**b, 'metrics': {'stage_s': b['stage_statistics'].get(stage, {}).get('seconds', {})}}
                stage_comparisons[f'{pair}.{mode}.{stage}'] = compare_metric(x, y, 'stage_s', global_errors)
            x, y = groups[f'mem_{base_arm}_{mode}'], groups[f'mem_{candidate_arm}_{mode}']
            reasons = list(global_errors) + x['errors'] + y['errors']
            if x['paired_workload'] != y['paired_workload']:
                reasons.append('Memory paired workload mismatch')
            if not x['memory'] or not y['memory']:
                reasons.append('Missing measured memory sample')
            memory_comparisons[f'{pair}.{mode}'] = {
                'baseline_arm': base_arm, 'candidate_arm': candidate_arm,
                'baseline': x['memory'], 'candidate': y['memory'],
                'valid_comparison': not reasons, 'invalid_reasons': reasons,
                'reduction_gib': {k: x['memory'][k] - y['memory'][k] for k in ('allocated_gib', 'reserved_gib', 'device_used_peak_gib')} if not reasons else None}
    return {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'root': str(root.resolve()), 'phase': 'Phase2', 'arms': list(arms), 'modes': list(modes),
            'arm_specs': ARM_SPECS, 'groups': groups, 'comparisons': comparisons,
            'stage_comparisons': stage_comparisons, 'memory_comparisons': memory_comparisons,
            'environment_consistency': {'valid': not global_errors, 'reference_group': reference_name, 'errors': global_errors,
                'comparison_scope': 'source/model/packages/Python/CUDA/GPU UUID+driver/CPU/host/topology; intentional per-arm quant configs and diagnostic flags excluded'},
            'complete_and_valid': not global_errors and all(not g['errors'] for g in groups.values()) and all(c['valid_comparison'] for c in comparisons.values()) and all(c['valid_comparison'] for c in memory_comparisons.values()),
            'statistics_scope': 'Only successful measured rows; mean and sample SD. n=10 fixed repeated requests per process, not cross-start or independent-prompt replication. No historical values fill missing Phase2 data.'}


def render(report):
    def fmt(v):
        if v.get('mean') is None:
            return f"missing (n={v.get('n', 0)})"
        sd = f"{v['sample_sd']:.6f}" if v.get('sample_sd') is not None else 'undefined'
        return f"{v['mean']:.6f} ± {sd} (n={v['n']})"
    lines = ['# Phase 2 performance audit', '', report['statistics_scope'], '',
             '| Comparison / metric | Baseline seconds | Candidate seconds | Baseline/candidate | Time reduction |', '|---|---:|---:|---:|---:|']
    for name, c in report['comparisons'].items():
        ratio = f"{c['baseline_over_candidate']:.6f}" if c['valid_comparison'] else 'INVALID'
        reduction = f"{c['latency_reduction_percent']:.4f}%" if c['valid_comparison'] else '—'
        lines.append(f"| {name} | {fmt(c['baseline'])} | {fmt(c['candidate'])} | {ratio} | {reduction} |")
    lines += ['', 'Serving client_total includes headers+body; JSON/base64/PNG decoding is separate. Engine metrics are ms/1000. Initialization is separate in each group; no profile wall time or GPU event sums are performance samples.', '',
              'Selective versus dit_fp8 isolates the selected DiT BF16 protections; joint comparison also changes TE quantization. A performance gain does not imply passing quality.', '',
              '## Independent memory', '', '| Group | Allocated GiB | Reserved GiB | Device-used GiB | Valid |', '|---|---:|---:|---:|---|']
    for g in report['groups'].values():
        if g['kind'] != 'mem':
            continue
        m = g['memory']
        lines.append(f"| {g['id']} | {m['allocated_gib']:.3f} | {m['reserved_gib']:.3f} | {m['device_used_peak_gib']:.3f} | {not g['errors'] and report['environment_consistency']['valid']} |" if m else f"| {g['id']} | — | — | — | incomplete |")
    lines += ['', 'Only entry 2 of exactly [warmup, measured] is reported. One sample per memory group; counters are distinct. Pairwise reductions, stage observations, initialization, raw values and source hashes are in PERFORMANCE.json.', '', '## Validation gaps', '']
    lines += [f'- ENV: {x}' for x in report['environment_consistency']['errors']]
    for g in report['groups'].values():
        lines += [f"- {g['id']}: {x}" for x in g['errors']]
        lines += [f"- {g['id']} note: {x}" for x in g['notes']]
    if report['complete_and_valid']:
        lines += ['- All requested groups and pairing checks passed.']
    return '\n'.join(lines) + '\n'


def main():
    global ARM_SPECS, OFFLINE_HARNESS, SERVING_HARNESS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
    p.add_argument('--modes', nargs='+', default=list(MODES), choices=MODES)
    p.add_argument('--arm-manifest', type=Path)
    p.add_argument('--offline-harness', default=OFFLINE_HARNESS)
    p.add_argument('--serving-harness', default=SERVING_HARNESS)
    p.add_argument('--fixture-only', action='store_true', help='Mark output as tool testing, never Phase2 evidence')
    a = p.parse_args()
    if 'bf16' not in a.arms or len(set(a.arms)) != len(a.arms) or len(set(a.modes)) != len(a.modes):
        p.error('Require bf16 and unique arms/modes')
    manifest = a.arm_manifest or a.root / 'scripts/configs/arms.json'
    ARM_SPECS = load_arm_specs(manifest, a.arms)
    OFFLINE_HARNESS, SERVING_HARNESS = a.offline_harness, a.serving_harness
    report = analyze(a.root, a.arms, a.modes)
    report['arm_manifest'] = {'path': str(manifest.resolve()), 'sha256': file_sha(manifest)}
    report['fixture_only_not_phase2_evidence'] = a.fixture_only
    out = a.out_dir or a.root / 'analysis/performance'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'PERFORMANCE.json').write_text(json.dumps(report, indent=2) + '\n')
    text = render(report)
    if a.fixture_only:
        text = '# TOOL FIXTURE ONLY — NOT PHASE 2 RESULTS\n\n' + text
    (out / 'PERFORMANCE.md').write_text(text)
    print(json.dumps({'complete_and_valid': report['complete_and_valid'], 'output': str(out), 'groups': len(report['groups'])}))
    raise SystemExit(0 if report['complete_and_valid'] else 2)


if __name__ == '__main__':
    main()

