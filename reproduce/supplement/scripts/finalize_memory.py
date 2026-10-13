#!/usr/bin/env python3
"""Derive full-precision memory tables from successful measured raw requests.

Preserves the frozen scorer and completed analysis. This CPU-only step reads
runs.jsonl; it never launches inference or rescores image quality.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

VERSION = 1
ARMS = ('bf16', 'dit_fp8', 'mlp_bf16', 'te_fp8', 'joint_fp8', 'prefix_fp8_v', 'prefix_fp8')
PREFIX_ARMS = ('bf16', 'prefix_fp8_v', 'prefix_fp8')
WORKLOADS = {'t2i': ('t2i_teapot_s42', 't2i'), 'edit': ('edit_bear_s42', 'edit'),
             'multi_edit': ('prefix_two_ref_s42', 'edit')}
EXPECTED = {f'mem_{arm}_{workload}': (arm, *WORKLOADS[workload])
            for workload in WORKLOADS
            for arm in (PREFIX_ARMS if workload == 'multi_edit' else ARMS)}
CASE_FIELDS = ('id', 'case_id', 'group', 'task', 'prompt', 'negative_prompt', 'seed',
               'width', 'height', 'num_inference_steps', 'true_cfg_scale',
               'input_image', 'input_image_sha256', 'cfg_requested_active', 'reference_count')
RAW_FIELDS = {'allocated_gib': 'torch_max_memory_allocated_gib',
              'reserved_gib': 'torch_max_memory_reserved_gib'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def load_formatter():
    path = Path(__file__).with_name('analyze_supplement.py')
    spec = importlib.util.spec_from_file_location('frozen_supplement_scorer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def finalize(analysis_path, results_root, out_dir):
    analysis_path, results_root, out_dir = (Path(p).resolve() for p in (analysis_path, results_root, out_dir))
    require(not out_dir.exists(), 'Use a fresh output directory; preserve prior analysis.')
    require(out_dir != analysis_path.parent, 'Cannot overwrite the source analysis directory.')
    original = read(analysis_path)
    require(original.get('analysis_status') == 'complete_for_supplied_manifests_and_discovered_runs',
            'The original CPU analysis must be complete.')
    require(not original.get('run_errors'), 'Original analysis contains run errors.')
    require('memory_finalization' not in original, 'Input is already a finalized analysis.')
    formatter, formatter_path = load_formatter()
    require(original.get('script_sha256') == sha(formatter_path), 'Frozen scorer/source-analysis hash mismatch.')
    cells = original.get('memory', [])
    require(len(cells) == 17, 'Expected exactly 17 memory cells.')
    names = [Path(c['directory']).name for c in cells]
    require(len(set(names)) == 17 and set(names) == set(EXPECTED), 'Unexpected, duplicate or missing memory cell.')
    actual_names = {p.name for p in results_root.glob('mem_*') if p.is_dir()}
    require(actual_names == set(EXPECTED), 'Raw memory cell set differs from the frozen 17-cell design.')
    result = copy.deepcopy(original)
    derived, evidence = [], {}
    envs, measured = {}, {}
    for old in cells:
        name = Path(old['directory']).name
        arm, case_id, task = EXPECTED[name]
        directory = results_root / name
        require((directory / 'cell.ok').is_file(), f'{name}: cell.ok missing.')
        paths = {k: directory / k for k in ('env.json', 'runs.jsonl', 'engine_peak_memory.json')}
        env = read(paths['env.json'])
        rows = [json.loads(s) for s in paths['runs.jsonl'].read_text().splitlines() if s.strip()]
        require([r.get('phase') for r in rows] == ['warmup', 'measured'], f'{name}: must retain one warmup and one measured row.')
        require(all(r.get('index') == 0 and r.get('error') is None for r in rows), f'{name}: invalid index or failed request.')
        row = rows[1]
        require(old.get('arm') == env.get('arm') == arm and all(r.get('arm') == arm for r in rows), f'{name}: arm mismatch.')
        require(old.get('case_id') == row.get('case_id') == case_id and old.get('task') == row.get('task') == task,
                f'{name}: task/case mismatch.')
        require(old.get('cfg') == row.get('true_cfg_scale') == 1.0, f'{name}: unexpected CFG.')
        require(old.get('expected_n') == old.get('observed_n') == old.get('peak_n') == 1,
                f'{name}: source analysis count mismatch.')
        require(not old.get('issues') and old.get('baseline_comparison', {}).get('status') == 'matched',
                f'{name}: source memory comparison is not verified.')
        require(len(env.get('cases', [])) == 1, f'{name}: expected one environment case.')
        case = env['cases'][0]
        require(case.get('id') == case_id and all(row.get(k) == v for k, v in case.items()), f'{name}: measured input differs from env.')
        require(all(rows[0].get(k) == row.get(k) for k in CASE_FIELDS), f'{name}: warmup/measured case differs.')
        require(row.get('prefix_kv_cache_dtype') == env.get('runtime_config', {}).get('prefix_kv_cache_dtype'),
                f'{name}: prefix configuration differs from env.')
        workload = env.get('workload', {})
        require(workload.get('mem') is True and workload.get('warmup') == 1 and workload.get('measured') == 1,
                f'{name}: unexpected memory workload.')
        require(workload.get('trace') is False and workload.get('feasibility') == 0, f'{name}: unexpected extra measurement mode.')
        peaks = read(paths['engine_peak_memory.json'])
        require(len(peaks) == 2, f'{name}: expected one warmup and one measured rounded engine peak.')
        values = {}
        for field, raw_field in RAW_FIELDS.items():
            value = row.get(raw_field)
            require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0,
                    f'{name}: missing/invalid full-precision {raw_field}.')
            require(old.get(field) == peaks[1].get(field), f'{name}: old table differs from measured rounded engine peak.')
            require(abs(value - peaks[1][field]) <= .005 + 1e-12,
                    f'{name}: {raw_field} differs by more than rounding precision from the engine log.')
            values[field] = value
        require(values['allocated_gib'] <= values['reserved_gib'], f'{name}: allocated exceeds reserved.')
        item = copy.deepcopy(old)
        item.update(values)
        item['rounded_engine_peak_source'] = old['peak_source']
        item['peak_source'] = f'results/{name}/runs.jsonl#phase=measured,index=0'
        item['peak_fields'] = dict(RAW_FIELDS)
        item['memory_precision'] = 'full-precision measured request values; GiB'
        derived.append(item)
        envs[name], measured[name] = env, row
        evidence[name] = {'arm': arm, 'case_id': case_id, 'task': task,
                         'source_files': {f'results/{name}/{k}': sha(p) for k, p in paths.items()},
                         'selected_phase': 'measured', 'selected_index': 0,
                         'raw_fields': dict(RAW_FIELDS), 'rounded_engine_values': peaks[1],
                         'full_precision_values': values}
    for item in derived:
        name = Path(item['directory']).name
        matches = [b for b in derived if b['arm'] == 'bf16' and b['case_id'] == item['case_id'] and b['task'] == item['task']]
        require(len(matches) == 1, f'{name}: BF16 denominator not unique.')
        baseline = matches[0]; bname = Path(baseline['directory']).name
        require(not formatter.environment_comparison(envs[name], envs[bname]), f'{name}: runtime/GPU/environment differs from BF16 denominator.')
        require(formatter.case_key(measured[name]) == formatter.case_key(measured[bname]), f'{name}: BF16 prompt/seed/CFG/reference denominator mismatch.')
        require(all(measured[name].get(k) == measured[bname].get(k) for k in CASE_FIELDS), f'{name}: BF16 full case mismatch.')
        require(item['baseline_comparison']['directory'] == baseline['directory'], f'{name}: old BF16 denominator mismatch.')
        for field in RAW_FIELDS:
            item['delta_' + field.removesuffix('_gib') + '_pct'] = (
                None if item['arm'] == 'bf16' else (item[field] / baseline[field] - 1) * 100)
        evidence[name]['bf16_denominator'] = bname
        evidence[name]['delta_allocated_pct'] = item['delta_allocated_pct']
        evidence[name]['delta_reserved_pct'] = item['delta_reserved_pct']
    result['memory'] = derived
    copied = {p.name: sha(p) for p in sorted(analysis_path.parent.glob('*.csv')) if p.name != 'memory.csv'}
    result['memory_finalization'] = {
        'version': VERSION, 'kind': 'CPU derivation from measured raw request memory; no inference rerun',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha(Path(__file__)),
        'source_analysis_sha256': sha(analysis_path), 'source_analysis_path': str(analysis_path),
        'source_frozen_scorer_sha256': sha(formatter_path), 'results_root': str(results_root),
        'rounded_log_tolerance_gib': .005, 'cells': evidence, 'copied_csv_sha256': copied,
        'preserved_sections': [k for k in original if k != 'memory'],
    }
    require(all(result[k] == original[k] for k in original if k != 'memory'), 'Unexpected non-memory mutation.')
    out_dir.mkdir(parents=True)
    (out_dir / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    formatter.write_csv(out_dir / 'memory.csv', derived)
    for name in copied:
        shutil.copyfile(analysis_path.parent / name, out_dir / name)
    # Reuse the frozen scorer's table layout; avoid the misleading rendered -0.0%.
    tables = formatter.markdown(result).replace('+0.0%', '0.0%').replace('-0.0%', '0.0%')
    (out_dir / 'TABLES.md').write_text(tables)
    return {'output': str(out_dir), 'memory_cells': len(derived), 'version': VERSION,
            'source_analysis_sha256': sha(analysis_path), 'final_analysis_sha256': sha(out_dir / 'analysis.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        info = finalize(args.analysis, args.results_root, args.out_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'Memory finalization refused: {exc}') from exc
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
