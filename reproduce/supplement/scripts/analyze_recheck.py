"""CPU-only summary of separate original/recheck cohorts; no causal attribution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import run_recheck

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('bf16', 'prefix_fp8_v', 'prefix_fp8')
BREAKDOWN_FIELDS = ('preprocess_ms', 'add_req_and_wait_ms', 'output_ready_wait_ms', 'postprocess_ms', 'total_ms')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def stats(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError('Missing or non-finite measurement')
    return {'n': len(values), 'mean': statistics.mean(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values), 'max': max(values), 'values': values}


def measured_rows(cell):
    if not (cell / 'cell.ok').is_file():
        raise RuntimeError('Cell completion marker missing: ' + str(cell))
    rows = [json.loads(line) for line in (cell / 'runs.jsonl').read_text().splitlines() if line]
    if ([row.get('phase') for row in rows] != ['warmup', 'feasibility'] + ['measured'] * 10
            or any(row.get('error') for row in rows)):
        raise RuntimeError('Incomplete/failed cell: ' + str(cell))
    measured = [row for row in rows if row['phase'] == 'measured']
    if [row['index'] for row in measured] != list(range(10)):
        raise RuntimeError('Measured request indices changed: ' + str(cell))
    return measured


def performance(rows):
    values = {name: [] for name in ('wall_s', 'stage_0_gen_s', 'text_encoder_s', 'diffuse_s',
                                    'vae_decode_s', 'unassigned_stage_0_s', 'outside_stage_0_s',
                                    'queue_admission_ms')}
    for row in rows:
        stages = row['stage_durations_s']
        te = stages['QwenImage21Pipeline.text_encoder.forward']
        diffuse = stages['QwenImage21Pipeline.diffuse']
        vae = stages['QwenImage21Pipeline.vae.decode']
        gen = stages['stage_0_gen_ms'] / 1000.0
        for name, value in {'wall_s': row['wall_s'], 'stage_0_gen_s': gen,
                            'text_encoder_s': te, 'diffuse_s': diffuse, 'vae_decode_s': vae,
                            'unassigned_stage_0_s': gen - te - diffuse - vae,
                            'outside_stage_0_s': row['wall_s'] - gen,
                            'queue_admission_ms': stages['queue_wait_ms']}.items():
            values[name].append(value)
    return {name: stats(series) for name, series in values.items()}


def parse_counters(text):
    return {key: int(value) for key, value in (line.split() for line in text.splitlines() if line)}


def parse_pressure(text):
    return {parts[0]: {key: int(value) if key == 'total' else float(value)
                       for key, value in (part.split('=', 1) for part in parts[1:])}
            for parts in (line.split() for line in text.splitlines() if line)}


def host_delta(before, after):
    result = {'scope': 'whole cell, including engine startup/warmup/feasibility; not per-request evidence',
              'snapshot_duration_s': after['perf_counter'] - before['perf_counter'],
              'before_time_utc': before['time_utc'], 'after_time_utc': after['time_utc'],
              'cpu_stat': None, 'pressure': {},
              'loadavg_before': before['files'].get('/proc/loadavg'),
              'loadavg_after': after['files'].get('/proc/loadavg'),
              'cpu_max_before': before['files'].get('/sys/fs/cgroup/cpu.max'),
              'cpu_max_after': after['files'].get('/sys/fs/cgroup/cpu.max'),
              'unavailable_before': before.get('errors', {}), 'unavailable_after': after.get('errors', {})}
    cpu_path = '/sys/fs/cgroup/cpu.stat'
    if cpu_path in before['files'] and cpu_path in after['files']:
        first, last = parse_counters(before['files'][cpu_path]), parse_counters(after['files'][cpu_path])
        keys = sorted(first.keys() & last.keys())
        result['cpu_stat'] = {'before': first, 'after': last, 'delta': {key: last[key] - first[key] for key in keys}}
        result['cpu_stat']['counter_reset_detected'] = any(last[key] < first[key] for key in keys)
    for path in sorted(before['files'].keys() & after['files'].keys()):
        if '/pressure/' not in path and not path.endswith('.pressure'):
            continue
        first, last = parse_pressure(before['files'][path]), parse_pressure(after['files'][path])
        result['pressure'][path] = {kind: {'before': first[kind], 'after': last[kind],
                                         'total_delta_us': last[kind]['total'] - first[kind]['total'],
                                         'counter_reset_detected': last[kind]['total'] < first[kind]['total']}
                                    for kind in sorted(first.keys() & last.keys())}
    return result


def percent(value, baseline):
    return (value / baseline - 1.0) * 100.0 if baseline else None


def analyze(root):
    recheck = root / 'recheck'
    if not (recheck / 'experiment.ok').is_file():
        raise RuntimeError('Recheck is not complete')
    result = {'schema_version': 1, 'analysis_status': 'complete',
              'cohorts_pooled': False, 'samples_removed': 0,
              'recheck_order': list(run_recheck.ORDER),
              'summary_method': '10 measured requests per arm; sample standard deviation; baseline within each cohort',
              'cohorts': {'original': {}, 'recheck': {}}, 'original_vs_recheck': {},
              'host_observations': {}, 'source_sha256': {}}
    for arm in ARMS:
        name = 'perf_' + arm + '_t2i'
        original, current = root / 'results' / name, recheck / 'results' / name
        diag = recheck / 'diagnostics' / name
        original_env, current_env = read_json(original / 'env.json'), read_json(current / 'env.json')
        # This function only reads; do not rewrite the existing verification evidence.
        environment_match = run_recheck.compare_environment(current_env, original_env, arm)
        verification = read_json(diag / 'verification.json')
        if (not verification.get('passed') or verification.get('arm') != arm
                or verification.get('environment_match') != environment_match
                or verification.get('original_env_sha256') != sha256(original / 'env.json')
                or verification.get('recheck_env_sha256') != sha256(current / 'env.json')
                or verification.get('runner_sha256') != current_env['harness_sha256']
                or verification.get('runner_sha256') != sha256(root / 'scripts/c1_offline_recheck.py')
                or verification.get('observer_sha256') != sha256(recheck / 'timing_observer/sitecustomize.py')):
            raise RuntimeError('Missing/changed diagnostic provenance: ' + arm)
        old_rows, rows = measured_rows(original), measured_rows(current)
        case = current_env['cases'][0]
        for row in old_rows + rows:
            if row.get('arm') != arm or any(row.get(k) != value for k, value in case.items()):
                raise RuntimeError('Measured workload/case changed: ' + arm)
        for cohort, source_rows in (('original', old_rows), ('recheck', rows)):
            result['cohorts'][cohort][arm] = {'performance': performance(source_rows),
                                           'output_sha256': [row['image_sha256'] for row in source_rows]}
        mapping = verification['request_mapping']
        if [entry['phase'] for entry in mapping] != ['warmup', 'feasibility'] + ['measured'] * 10:
            raise RuntimeError('Incomplete engine timing mapping: ' + arm)
        entries = [entry for entry in mapping if entry['phase'] == 'measured']
        event_files = list(diag.glob('engine_breakdown_*.jsonl'))
        if len(event_files) != 1:
            raise RuntimeError('Expected one engine timing stream: ' + arm)
        events = [json.loads(line) for line in event_files[0].read_text().splitlines() if line]
        all_rows = [json.loads(line) for line in (current / 'runs.jsonl').read_text().splitlines() if line]
        if len(events) != len(mapping) or len(all_rows) != len(mapping):
            raise RuntimeError('Raw engine timing coverage changed: ' + arm)
        for ordinal, (row, event, entry) in enumerate(zip(all_rows, events, mapping), 1):
            interval = row['diagnostic_wall_interval']
            if (event['arm'] != arm or event['engine_event_ordinal'] != ordinal
                    or event['source_module_sha256'] != run_recheck.ENGINE_SOURCE_SHA256
                    or not interval['start_perf_counter'] <= event['perf_counter'] <= interval['end_perf_counter']
                    or entry != {'ordinal': ordinal, 'phase': row['phase'], 'index': row['index'],
                                 'case_id': row['case_id'], 'timings_ms': event['timings_ms']}):
                raise RuntimeError('Raw engine timing no longer matches verified request: ' + arm)
        for row, entry in zip(rows, entries):
            if entry['index'] != row['index'] or entry['case_id'] != row['case_id']:
                raise RuntimeError('Engine timing mapping is not aligned: ' + arm)
        breakdown = {field: stats([entry['timings_ms'][field] for entry in entries]) for field in BREAKDOWN_FIELDS}
        result['cohorts']['recheck'][arm]['engine_breakdown_ms'] = breakdown
        result['cohorts']['recheck'][arm]['environment_match'] = environment_match
        result['host_observations'][arm] = host_delta(read_json(diag / 'host_before.json'), read_json(diag / 'host_after.json'))
        result['original_vs_recheck'][arm] = {
            metric: {'original_mean': old['mean'], 'recheck_mean': result['cohorts']['recheck'][arm]['performance'][metric]['mean'],
                     'mean_change': result['cohorts']['recheck'][arm]['performance'][metric]['mean'] - old['mean'],
                     'mean_change_percent': percent(result['cohorts']['recheck'][arm]['performance'][metric]['mean'], old['mean'])}
            for metric, old in result['cohorts']['original'][arm]['performance'].items()}
        sources = [original / 'env.json', original / 'runs.jsonl', current / 'env.json', current / 'runs.jsonl',
                   diag / 'verification.json', diag / 'host_before.json', diag / 'host_after.json', diag / 'primary_readiness.json']
        sources += sorted(diag.glob('engine_breakdown_*.jsonl')) + sorted(diag.glob('observer_manifest_*.json'))
        for path in sources:
            result['source_sha256'][str(path.relative_to(root))] = sha256(path)
    for cohort in result['cohorts'].values():
        baseline = cohort['bf16']['performance']
        for arm in ARMS:
            for metric, values in cohort[arm]['performance'].items():
                values['mean_change_percent_vs_same_cohort_bf16'] = percent(values['mean'], baseline[metric]['mean'])
    return result


def markdown(result):
    lines = ['# T2I reverse-order diagnostic repeat', '',
             'Original and diagnostic cohorts are shown separately. All 10 measured requests per arm are retained.', '',
             '| Cohort | Arm | Wall mean ± SD (s) | vs cohort BF16 | Denoising mean ± SD (s) | Unassigned stage interval mean / max (s) |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for name, cohort in result['cohorts'].items():
        for arm in ARMS:
            metrics = cohort[arm]['performance']
            wall, diffuse, residual = (metrics[k] for k in ('wall_s', 'diffuse_s', 'unassigned_stage_0_s'))
            lines.append(f"| {name} | {arm} | {wall['mean']:.3f} ± {wall['stdev']:.3f} | {wall['mean_change_percent_vs_same_cohort_bf16']:+.2f}% | {diffuse['mean']:.3f} ± {diffuse['stdev']:.3f} | {residual['mean']:.3f} / {residual['max']:.3f} |")
    lines += ['', '## Observed engine breakdown', '',
              'Diagnostic cohort only; mean / max in ms. Add-request-and-wait includes scheduling and execution.', '',
              '| Arm | Preprocess | Add request and wait | Output-ready wait | Postprocess | Engine total |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        data = result['cohorts']['recheck'][arm]['engine_breakdown_ms']
        cells = [f"{data[field]['mean']:.3f} / {data[field]['max']:.3f}" for field in BREAKDOWN_FIELDS]
        lines.append('| ' + arm + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Host `cpu.stat` and pressure counter changes are saved in `analysis.json`. They cover entire cells, including startup and warmup, and do not identify the cause of an individual slow request.', '',
              'This report assigns no CPU-contention, scheduler or dequantization cause. The original means are not replaced or pooled with the diagnostic repeat.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    result = analyze(args.root.resolve())
    out = args.out_dir or args.root / 'recheck/analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (out / 'REPORT.md').write_text(markdown(result))
    print(json.dumps({'analysis_status': result['analysis_status'], 'out_dir': str(out), 'gpu_executed': False}))


if __name__ == '__main__':
    main()
