"""CPU-only synthetic tests; never imports a GPU runtime or changes primary evidence."""
import ast
import copy
import dataclasses
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
TESTS = {}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(name, value=True):
    assert value, name
    TESTS[name] = True


def rejects(name, callback):
    try:
        callback()
    except (RuntimeError, FileNotFoundError):
        check(name)
    else:
        raise AssertionError(name)


def main():
    os.environ.pop('C1_RECHECK_TIMING', None)
    wrapper = load('recheck_test_wrapper', ROOT / 'scripts/run_recheck.py')
    sys.modules['run_recheck'] = wrapper
    analyzer = load('recheck_test_analyzer', ROOT / 'scripts/analyze_recheck.py')
    observer = load('recheck_test_observer', ROOT / 'recheck/timing_observer/sitecustomize.py')
    for name in ('scripts/c1_offline_recheck.py', 'scripts/run_recheck.py', 'scripts/analyze_recheck.py', 'recheck/timing_observer/sitecustomize.py'):
        compile((ROOT / name).read_text(), name, 'exec')
        check('syntax:' + name)
    tree = ast.parse((ROOT / 'scripts/c1_offline_recheck.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'metrics_to_json')
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<serializer>', 'exec'), namespace)
    serialize = namespace['metrics_to_json']
    @dataclasses.dataclass
    class Meta:
        value: float
    class Pydantic2:
        def model_dump(self, mode):
            assert mode == 'python'
            return {'pydantic': Meta(1.25)}
    class Pydantic1:
        def dict(self):
            return {'old': 2}
    check('dict_dataclass_pydantic_serialization', serialize({'a': Pydantic2(), 'b': Pydantic1()}) == {'a': {'pydantic': {'value': 1.25}}, 'b': {'old': 2}})
    class TensorLike:
        def cpu(self):
            raise AssertionError('tensor read')
        def item(self):
            raise AssertionError('tensor read')
    check('no_tensor_read', 'unsupported_metadata_type' in serialize(TensorLike()))
    record = logging.LogRecord('test', logging.DEBUG, '', 1, observer.MESSAGE, (1.23456, 2, 3, 4, 10.23456), None)
    other = logging.LogRecord('test', logging.DEBUG, '', 1, 'other', (), None)
    check('single_exact_debug_message_only', observer._ModuleFilter().filter(record) and not observer._ModuleFilter().filter(other))
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        os.environ['C1_RECHECK_ARM'] = 'bf16'
        handler = observer._BreakdownHandler(base)
        handler.emit(record)
        event = json.loads(next(base.glob('engine_breakdown_*.jsonl')).read_text())
        check('unrounded_existing_timing_arguments', event['timings_ms']['preprocess_ms'] == 1.23456)
        wrong = base / 'wrong.py'
        wrong.write_text('not the pinned source')
        rejects('source_hash_mismatch_rejected', lambda: observer._install(types.SimpleNamespace(__file__=str(wrong))))
        primary = base / 'primary'
        proc = base / 'proc'
        primary.mkdir()
        (proc / 'self').mkdir(parents=True)
        (proc / 'self/cmdline').write_bytes(b'python\0run_recheck.py\0')
        (proc / '101').mkdir()
        (proc / '101/cmdline').write_bytes(b'python\0run_recheck.py\0')
        rejects('missing_primary_markers_rejected', lambda: wrapper.check_primary_ready(primary, proc))
        (primary / 'experiment.ok').touch()
        rejects('experiment_marker_alone_rejected', lambda: wrapper.check_primary_ready(primary, proc))
        (primary / 'analysis.ok').touch()
        (primary / 'analysis/completed').mkdir(parents=True)
        analysis_path = primary / 'analysis/completed/analysis.json'
        analysis_path.write_text(json.dumps({'analysis_status': 'incomplete'}))
        rejects('incomplete_analysis_despite_markers_rejected', lambda: wrapper.check_primary_ready(primary, proc))
        analysis_path.write_text(json.dumps({'analysis_status': wrapper.COMPLETE_ANALYSIS_STATUS}))
        check('complete_idle_primary_accepted', wrapper.check_primary_ready(primary, proc)['analysis_status'] == wrapper.COMPLETE_ANALYSIS_STATUS)
        for script in sorted(wrapper.PRIMARY_PROCESS_SCRIPTS):
            (proc / '102').mkdir(exist_ok=True)
            (proc / '102/cmdline').write_bytes(('python\0/workspace/c1-supplement/scripts/' + script + '\0').encode())
            rejects('active_primary_rejected:' + script, lambda: wrapper.check_primary_ready(primary, proc))
        (proc / '102/cmdline').unlink()
        (proc / '102').rmdir()
        rejects('missing_proc_inspection_rejected', lambda: wrapper.check_primary_ready(primary, base / 'unavailable'))
        original = json.loads((ROOT / 'results/perf_bf16_t2i/env.json').read_text())
        actual = copy.deepcopy(original)
        actual['harness_sha256'] = wrapper.sha256(ROOT / 'scripts/c1_offline_recheck.py')
        check('same_environment_except_diagnostic_harness_accepted', wrapper.compare_environment(actual, original, 'bf16')['matched'])
        mutations = {
            'different_gpu_uuid': ('nvidia_smi', original['nvidia_smi'].replace('GPU-000d1196', 'GPU-111d1196')),
            'different_driver': ('nvidia_smi', original['nvidia_smi'].replace('580.95.05', '581.00.00')),
            'dirty_checkout': ('vllm_omni_dirty', ' M source.py'),
            'wrong_actual_checkout': ('vllm_omni_checkout', '0' * 40),
            'different_packages': ('packages', {**original['packages'], 'torch': 'other'}),
            'different_model_revision': ('model_revision', '0' * 40),
            'different_case_inputs': ('cases', [{**original['cases'][0], 'input_image_sha256': ['wrong']}]),
            'different_workload': ('workload', {**original['workload'], 'seed': 43}),
            'different_runtime_config': ('runtime_config', {**original['runtime_config'], 'prefix_kv_cache_dtype': 'fp8'}),
            'different_quant_config_hash': ('quant_config_file_sha256', '0' * 64),
            'different_case_file_hash': ('case_file_sha256', '0' * 64),
            'unexpected_environment_field': ('unreviewed_flag', True),
        }
        for name, (key, value) in mutations.items():
            bad = copy.deepcopy(actual)
            bad[key] = value
            rejects(name + '_rejected', lambda bad=bad: wrapper.compare_environment(bad, original, 'bf16'))
        missing = copy.deepcopy(actual)
        del missing['packages']
        rejects('missing_required_provenance_rejected', lambda: wrapper.compare_environment(missing, original, 'bf16'))
        out, diag = base / 'out', base / 'diag'
        out.mkdir(); diag.mkdir()
        (out / 'env.json').write_text(json.dumps(actual))
        rows = [json.loads(line) for line in (ROOT / 'results/perf_bf16_t2i/runs.jsonl').read_text().splitlines()]
        events = []
        for ordinal, row in enumerate(rows, 1):
            start = ordinal * 100.0
            row['diagnostic_wall_interval'] = {'start_perf_counter': start, 'end_perf_counter': start + row['wall_s']}
            events.append({'pid': 999, 'arm': 'bf16', 'engine_event_ordinal': ordinal,
                           'source_module_sha256': wrapper.ENGINE_SOURCE_SHA256, 'perf_counter': start + 1,
                           'timings_ms': dict(zip(observer.FIELDS, (1, 2, 3, 4, 10)))})
        (out / 'runs.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
        (out / 'summary.json').write_text(json.dumps({'failures': [], 'wall_s': {'n': 10}}))
        event_path = diag / 'engine_breakdown_999.jsonl'
        event_path.write_text(''.join(json.dumps(x) + '\n' for x in events))
        (diag / 'observer_manifest_999.json').write_text(json.dumps({'pid': 999, 'arm': 'bf16', 'observer_sha256': wrapper.sha256(ROOT / 'recheck/timing_observer/sitecustomize.py'), 'source_module_sha256': wrapper.ENGINE_SOURCE_SHA256}))
        check('synthetic_12_request_event_binding', wrapper.verify_cell(out, diag, 'bf16')['passed'])
        events[0]['perf_counter'] = 1
        event_path.write_text(''.join(json.dumps(x) + '\n' for x in events))
        rejects('wrong_request_interval_rejected', lambda: wrapper.verify_cell(out, diag, 'bf16'))
        events[0]['perf_counter'] = 101
        event_path.write_text(''.join(json.dumps(x) + '\n' for x in events[:-1]))
        rejects('missing_engine_event_rejected', lambda: wrapper.verify_cell(out, diag, 'bf16'))
        event_path.write_text(''.join(json.dumps(x) + '\n' for x in events))
        rows[0]['seed'] = 99
        (out / 'runs.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
        rejects('completed_row_case_mismatch_rejected', lambda: wrapper.verify_cell(out, diag, 'bf16'))
        fixture = base / 'analysis_fixture'
        for relative in ('scripts/c1_offline_recheck.py', 'recheck/timing_observer/sitecustomize.py'):
            target = fixture / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        (fixture / 'recheck/experiment.ok').touch()
        before = {'perf_counter': 100.0, 'time_utc': '2026-10-13T00:00:00Z', 'errors': {},
                  'files': {'/sys/fs/cgroup/cpu.stat': 'usage_usec 100\nnr_throttled 1\n',
                            '/proc/pressure/cpu': 'some avg10=1.20 avg60=1.00 avg300=0.50 total=100\n',
                            '/proc/loadavg': '1.0 2.0 3.0 1/100 123\n'}}
        after = copy.deepcopy(before)
        after.update(perf_counter=200.0, time_utc='2026-10-13T00:01:40Z')
        after['files']['/sys/fs/cgroup/cpu.stat'] = 'usage_usec 300\nnr_throttled 3\n'
        after['files']['/proc/pressure/cpu'] = 'some avg10=2.20 avg60=2.00 avg300=0.70 total=500\n'
        for arm in analyzer.ARMS:
            name = 'perf_' + arm + '_t2i'
            old_cell = fixture / 'results' / name
            cell = fixture / 'recheck/results' / name
            diagnostics = fixture / 'recheck/diagnostics' / name
            for directory in (old_cell, cell, diagnostics):
                directory.mkdir(parents=True)
            for filename in ('env.json', 'runs.jsonl'):
                shutil.copyfile(ROOT / 'results' / name / filename, old_cell / filename)
            (old_cell / 'cell.ok').touch()
            (cell / 'cell.ok').touch()
            env = json.loads((old_cell / 'env.json').read_text())
            env['harness_sha256'] = wrapper.sha256(ROOT / 'scripts/c1_offline_recheck.py')
            (cell / 'env.json').write_text(json.dumps(env))
            diag_rows = [json.loads(line) for line in (old_cell / 'runs.jsonl').read_text().splitlines()]
            diag_events = []
            for ordinal, row in enumerate(diag_rows, 1):
                # Deliberate, synthetic 10% wall change; every sample remains present.
                row['wall_s'] *= 1.1
                start = ordinal * 100.0
                row['diagnostic_wall_interval'] = {'start_perf_counter': start, 'end_perf_counter': start + row['wall_s']}
                diag_events.append({'pid': 999, 'arm': arm, 'engine_event_ordinal': ordinal,
                                    'source_module_sha256': wrapper.ENGINE_SOURCE_SHA256,
                                    'perf_counter': start + 1,
                                    'timings_ms': dict(zip(observer.FIELDS, (1, 2, 3, 4, 10)))})
            (cell / 'runs.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in diag_rows))
            (cell / 'summary.json').write_text(json.dumps({'failures': [], 'wall_s': {'n': 10}}))
            (diagnostics / 'engine_breakdown_999.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in diag_events))
            (diagnostics / 'observer_manifest_999.json').write_text(json.dumps({'pid': 999, 'arm': arm, 'observer_sha256': wrapper.sha256(ROOT / 'recheck/timing_observer/sitecustomize.py'), 'source_module_sha256': wrapper.ENGINE_SOURCE_SHA256}))
            (diagnostics / 'host_before.json').write_text(json.dumps(before))
            (diagnostics / 'host_after.json').write_text(json.dumps(after))
            (diagnostics / 'primary_readiness.json').write_text(json.dumps({'analysis_status': wrapper.COMPLETE_ANALYSIS_STATUS}))
            wrapper.verify_cell(cell, diagnostics, arm)
        report = analyzer.analyze(fixture)
        check('analysis_keeps_original_recheck_separate', abs(report['original_vs_recheck']['bf16']['wall_s']['mean_change_percent'] - 10.0) < 1e-10 and not report['cohorts_pooled'])
        check('analysis_retains_all_60_measured_samples', sum(len(cell['performance']['wall_s']['values']) for cohort in report['cohorts'].values() for cell in cohort.values()) == 60 and report['samples_removed'] == 0)
        check('analysis_same_cohort_bf16_baseline', abs(report['cohorts']['recheck']['prefix_fp8_v']['performance']['wall_s']['mean_change_percent_vs_same_cohort_bf16'] - report['cohorts']['original']['prefix_fp8_v']['performance']['wall_s']['mean_change_percent_vs_same_cohort_bf16']) < 1e-10)
        check('analysis_host_counter_and_pressure_deltas', report['host_observations']['bf16']['cpu_stat']['delta'] == {'usage_usec': 200, 'nr_throttled': 2} and report['host_observations']['bf16']['pressure']['/proc/pressure/cpu']['some']['total_delta_us'] == 400)
        check('analysis_breakdown_mean_max', report['cohorts']['recheck']['bf16']['engine_breakdown_ms']['postprocess_ms']['mean'] == 4 and report['cohorts']['recheck']['bf16']['engine_breakdown_ms']['postprocess_ms']['max'] == 4)
        check('analysis_markdown_render', '| original | bf16 |' in analyzer.markdown(report) and '| recheck | bf16 |' in analyzer.markdown(report))
        altered = fixture / 'recheck/diagnostics/perf_bf16_t2i/engine_breakdown_999.jsonl'
        original_text = altered.read_text()
        altered.write_text(original_text.replace('"postprocess_ms": 4', '"postprocess_ms": 5', 1))
        rejects('analysis_changed_raw_engine_mapping_rejected', lambda: analyzer.analyze(fixture))
        altered.write_text(original_text)
        (fixture / 'recheck/results/perf_bf16_t2i/cell.ok').unlink()
        rejects('analysis_incomplete_cell_rejected', lambda: analyzer.analyze(fixture))
    check('primary_runner_unchanged', wrapper.sha256(ROOT / 'scripts/c1_offline.py') == wrapper.PRIMARY_RUNNER_SHA256)
    stats_path = ROOT.parents[1] / 'vllm-omni-src/vllm_omni/metrics/stats.py'
    stats_tree = ast.parse(stats_path.read_text())
    projection = next(n for n in ast.walk(stats_tree) if isinstance(n, ast.FunctionDef) and n.name == '_merge_stage_metric_event')
    check('pinned_response_projection_omits_diffusion_breakdown', 'diffusion_metrics' not in ast.unparse(projection))
    result = {'passed': all(TESTS.values()), 'tests': TESTS, 'gpu_executed': False}
    (ROOT / 'recheck/CPU_SELFTEST.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'tests': len(TESTS), 'gpu_executed': False}))


if __name__ == '__main__':
    main()
