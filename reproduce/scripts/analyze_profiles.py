#!/usr/bin/env python3
"""CPU-only streaming audit of four pinned Qwen21 C1 profiles.

Example:
  python3 analyze_profiles.py --case bf16-t2i=/path/to/trace.json.gz \
    --case dit-fp8-t2i=/path/to/trace.json.gz --case bf16-edit=/path/to/trace.json.gz \
    --case dit-fp8-edit=/path/to/trace.json.gz --out-dir analysis/profiles

Times are sums of captured GPU-event durations, not E2E/request latency. No
PyTorch import, model loading, CUDA operation, or profiler key_averages call.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import datetime
import gzip
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

OMNI_REVISION = '3dc35694b3d458fc1461a8fe71832ac12e8c8cea'
STEP = re.compile(r'qwen_image_21_transformer\.py\(1021\): forward$')
STEP_LOOSE = re.compile(r'qwen_image_21_transformer\.py\(\d+\): forward$')
PHASES = {
    'encode_prompt': re.compile(r'pipeline_qwen_image_21\.py\(\d+\): encode_prompt$'),
    'diffuse': re.compile(r'cfg_parallel\.py\(\d+\): diffuse$'),
    'decode_latents': re.compile(r'pipeline_qwen_image_21\.py\(\d+\): _decode_latents$'),
    'pipeline_forward': re.compile(r'pipeline_qwen_image_21\.py\(\d+\): forward$'),
}
FP8 = re.compile(r'float8|float_e[45]m[23]|e[45]m[23]|\bfp8\b|(?:^|_)f8(?:_|$)', re.I)
BF16 = re.compile(r'bfloat16|bf16', re.I)
ATTN = re.compile(r'flash|fmha|attn|attention', re.I)
GEMM = re.compile(r'gemm|nvjet|xmma|scaled_mm|cutlass.*(?:device_kernel|Kernel2)', re.I)
INTERESTING_OP = re.compile(r'cutlass|scaled_mm|fp8|gemm|flash|attn|attention|aten::(?:mm|addmm|bmm)$', re.I)
GPU_CATS = {'kernel', 'gpu_memcpy', 'gpu_memset'}
CLASSIFICATION_OVERRIDES = {}


def iter_events(path: Path):
    """Stream a Chrome trace's top-level traceEvents array with bounded JSON buffer."""
    opener = gzip.open if path.suffix == '.gz' else open
    decoder = json.JSONDecoder()
    with opener(path, 'rt', encoding='utf-8') as stream:
        buf = ''
        while True:
            chunk = stream.read(65536)
            if not chunk:
                raise ValueError('incomplete Chrome trace header')
            buf += chunk
            match = re.search(r'"traceEvents"\s*:\s*\[', buf)
            if match:
                prefix = buf[:match.end() - 1]
                pos = match.end()
                break
            if len(buf) > 1024 * 1024 or not buf:
                raise ValueError('traceEvents array missing from trace header')
            if len(buf) < 65536:
                raise ValueError('incomplete Chrome trace header')
        while True:
            while pos < len(buf) and buf[pos] in ' \r\n\t,':
                pos += 1
            if pos < len(buf) and buf[pos] == ']':
                # Drain gzip to EOF for CRC/truncation detection; validate the outer JSON.
                suffix = buf[pos + 1:] + stream.read()
                json.loads(prefix + '[]' + suffix)
                return
            try:
                event, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                chunk = stream.read(65536)
                if not chunk:
                    raise ValueError('truncated or invalid traceEvents JSON')
                buf = buf[pos:] + chunk
                pos = 0
                if len(buf) > 16 * 1024 * 1024:
                    raise ValueError('single trace event exceeds 16 MiB or JSON is invalid')
                continue
            pos = end
            if not isinstance(event, dict):
                raise ValueError('traceEvents member is not an object')
            yield event
            if pos >= 65536:
                buf = buf[pos:]
                pos = 0


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dtype_evidence(types, expression, count):
    """Require the first count tensor inputs, not an unrelated BF16 output/scalar."""
    return isinstance(types, list) and len(types) >= count and all(expression.search(str(t)) for t in types[:count])


def classify(name, cat, op=None):
    """Mutually exclusive heuristic categories plus explicit execution evidence tier."""
    if cat == 'kernel' and name in CLASSIFICATION_OVERRIDES:
        return CLASSIFICATION_OVERRIDES[name]
    lower = name.lower()
    op_name, types = op if op else ('', None)
    if cat == 'gpu_memcpy':
        return 'memcpy', None
    if cat == 'gpu_memset':
        return 'memset', None
    if ATTN.search(name):
        if 'FlashAttnFwdSm90' in name:
            if FP8.search(name):
                return 'fa3_sm90_fp8_or_mixed', 'kernel_name'
            if BF16.search(name):
                return 'fa3_sm90_bf16', 'kernel_name'
            if dtype_evidence(types, BF16, 3):
                return 'fa3_sm90_bf16', 'correlated_operator_input_types'
            return 'fa3_sm90_dtype_unresolved', None
        return 'other_attention', None
    # Convolution can have cutlass GEMM implementation names; keep explicit conv first.
    if 'conv' in lower or 'implicit_convolve' in lower:
        return 'convolution', None
    if any(s in lower for s in ('marlin', 'w8a16', 'weight_only')) and (GEMM.search(name) or 'marlin' in lower):
        return 'weight_only_gemm_not_native_fp8_proof', 'name_heuristic'
    if GEMM.search(name):
        if FP8.search(name):
            return 'fp8_gemm', 'kernel_name'
        if dtype_evidence(types, FP8, 2):
            return 'fp8_gemm', 'correlated_operator_input_types'
        if BF16.search(name):
            return 'bf16_gemm', 'kernel_name'
        if dtype_evidence(types, BF16, 2):
            return 'bf16_gemm', 'correlated_operator_input_types'
        return 'gemm_dtype_unresolved', None
    if 'dequant' in lower:
        return 'dequantization', 'name_heuristic'
    if 'quant' in lower or ('fp8' in lower and ('cvt' in lower or 'convert' in lower)):
        return 'quantization', 'name_heuristic'
    # PyTorch generic gpu_kernel_impl_nocast is NOT a cast indicator.
    if ('copy' in lower and ('bfloat16' in lower or 'float8' in lower)) or any(s in lower for s in ('cast_kernel', 'convert_kernel', 'conversion_kernel')):
        return 'dtype_cast', 'name_heuristic'
    if any(s in lower for s in ('copy', 'catarray', 'cat_kernel', 'concatenate', 'gather', 'scatter')):
        return 'copy_cat_gather', 'name_heuristic'
    if 'norm' in lower:
        return 'normalization', 'name_heuristic'
    if any(s in lower for s in ('reduce', 'reduction', 'amax')):
        return 'reduction', 'name_heuristic'
    if any(s in lower for s in ('elementwise', 'pointwise', 'vectorized', 'unrolled_elementwise')):
        return 'elementwise', 'name_heuristic'
    return 'other', 'unclassified'


def stats(rows):
    total_us = sum(v[1] for v in rows.values())
    return {'gpu_event_sum_ms': total_us / 1000, 'event_count': sum(v[0] for v in rows.values()),
            'categories': {k: {'event_count': n, 'gpu_event_sum_ms': us / 1000,
                               'pct_of_gpu_event_sum': 100 * us / total_us if total_us else 0}
                           for k, (n, us) in sorted(rows.items(), key=lambda item: -item[1][1])}}


def analyze(path, label, expected_steps):
    launches, ambiguous_corr, operators, ambiguous_external = {}, set(), {}, set()
    gpu, frames, loose_step_names = [], collections.defaultdict(list), collections.Counter()
    cats = collections.Counter()
    for e in iter_events(path):
        cat, args = e.get('cat', ''), e.get('args') or {}
        cats[cat] += 1
        if cat in ('cuda_runtime', 'cuda_driver') and 'correlation' in args and 'ts' in e:
            corr = args['correlation']
            item = (e['ts'], args.get('External id'), e.get('pid'), e.get('tid'))
            if corr in launches and launches[corr] != item:
                ambiguous_corr.add(corr)
            else:
                launches[corr] = item
        elif cat == 'cpu_op' and 'External id' in args and INTERESTING_OP.search(e.get('name', '')):
            ext = args['External id']
            item = (sys.intern(e['name']), args.get('Input type'))
            if ext in operators and operators[ext] != item:
                ambiguous_external.add(ext)
            else:
                operators[ext] = item
        elif cat in GPU_CATS and 'dur' in e and 'ts' in e:
            gpu.append((e['ts'], e['dur'], sys.intern(e.get('name', '')), cat, args.get('correlation'),
                        args.get('External id'), args.get('device'), args.get('stream')))
        elif cat == 'python_function' and 'dur' in e and 'ts' in e:
            name = e.get('name', '')
            interval = (e['ts'], e['ts'] + e['dur'], e.get('pid'), e.get('tid'))
            if STEP.search(name):
                frames['steps'].append(interval)
            if STEP_LOOSE.search(name):
                loose_step_names[name] += 1
            for phase, pattern in PHASES.items():
                if pattern.search(name):
                    frames[phase].append(interval)
    for fr in frames.values():
        fr.sort()
    steps = frames['steps']
    step_starts = [x[0] for x in steps]
    per_step = [collections.defaultdict(lambda: [0, 0.0]) for _ in steps]
    totals = collections.defaultdict(lambda: [0, 0.0])
    denoise_totals = collections.defaultdict(lambda: [0, 0.0])
    phase_totals = {p: collections.defaultdict(lambda: [0, 0.0]) for p in PHASES}
    names = collections.defaultdict(lambda: {'event_count': 0, 'gpu_event_sum_ms': 0.0,
                                           'denoise_event_count': 0, 'denoise_gpu_event_sum_ms': 0.0,
                                           'evidence_tiers': set(), 'operators': set()})
    uncorrelated, matched = 0, 0
    for _, dur, name, cat, corr, ext, device, stream in gpu:
        launch = launches.get(corr) if corr not in ambiguous_corr else None
        op = operators.get(ext) if ext not in ambiguous_external else None
        if not op and launch and launch[1] not in ambiguous_external:
            op = operators.get(launch[1])
        category, tier = classify(name, cat, op)
        row = totals[category]; row[0] += 1; row[1] += dur
        key = (name, category)
        item = names[key]; item['event_count'] += 1; item['gpu_event_sum_ms'] += dur / 1000
        if tier: item['evidence_tiers'].add(tier)
        if op: item['operators'].add((op[0], json.dumps(op[1], sort_keys=True)))
        if not launch:
            uncorrelated += 1
            continue
        matched += 1
        ts, _, pid, tid = launch
        idx = bisect.bisect_right(step_starts, ts) - 1
        if idx >= 0 and ts <= steps[idx][1] and (pid, tid) == steps[idx][2:]:
            row = per_step[idx][category]; row[0] += 1; row[1] += dur
            row = denoise_totals[category]; row[0] += 1; row[1] += dur
            item['denoise_event_count'] += 1; item['denoise_gpu_event_sum_ms'] += dur / 1000
        for phase in PHASES:
            # Each phase has only one/few frames. Require the launching thread as well as interval.
            if any(a <= ts <= b and (pid, tid) == (p, t) for a, b, p, t in frames[phase]):
                row = phase_totals[phase][category]; row[0] += 1; row[1] += dur
    named = []
    for (name, category), item in sorted(names.items(), key=lambda x: -x[1]['gpu_event_sum_ms']):
        named.append({'name': name, 'category': category, **item,
                      'evidence_tiers': sorted(item['evidence_tiers']),
                      'operators': [{'name': n, 'input_types': json.loads(t)} for n, t in sorted(item['operators'])]})
    step_rows = [stats(r) for r in per_step]
    measured_steps = [r['gpu_event_sum_ms'] for r in step_rows]
    warnings = []
    if len(steps) != expected_steps:
        warnings.append(f'Expected {expected_steps} exact pinned transformer-forward frames; found {len(steps)}. Do not claim full denoise coverage.')
    if uncorrelated:
        warnings.append(f'{uncorrelated} GPU events lack an unambiguous launching runtime correlation; included in whole-trace totals but excluded from step/phase attribution.')
    if ambiguous_corr or ambiguous_external:
        warnings.append('Duplicate correlation/external IDs detected; affected attribution or dtype inference omitted.')
    if not totals.get('fp8_gemm') and 'fp8' in label.lower():
        warnings.append('No explicit FP8 GEMM execution evidence found. A requested FP8 label or generic GEMM name is not proof; inspect kernel names/inventory and report unresolved.')
    if not denoise_totals.get('fa3_sm90_bf16'):
        warnings.append('No BF16 FA3 SM90 kernel attributed to exact denoise steps; backend/dtype remains unverified for that scope.')
    artifact_dir = path.parent
    markers = list(artifact_dir.glob('a0_export_only_adapter_rank*.json'))
    adapters = [json.loads(p.read_text()) for p in markers]
    return {
        'case': label, 'trace': str(path.resolve()), 'trace_sha256': sha256(path), 'trace_bytes': path.stat().st_size,
        'analyzed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_attribution_revision': OMNI_REVISION,
        'event_categories': dict(cats), 'warnings': warnings,
        'measurement_boundary': 'Profile diagnostic only. gpu_event_sum_ms sums captured kernel/memcpy/memset durations and can double-count overlapping streams. It is neither E2E latency nor a speedup estimate. No profiled wall time is presented as a performance result.',
        'classification_boundary': 'Named categories are mutually exclusive heuristics. Quant/cast/copy costs can be fused into another kernel. Explicit FP8/BF16 evidence tiers use full kernel templates or directly correlated operator input types; generic CUTLASS/nvjet names remain unresolved.',
        'correlation': {'matched_gpu_events': matched, 'unmatched_gpu_events': uncorrelated,
                        'ambiguous_correlation_ids': len(ambiguous_corr), 'ambiguous_external_ids': len(ambiguous_external)},
        'whole_trace': stats(totals), 'denoise': stats(denoise_totals),
        'phases': {p: {'python_frame_count': len(frames[p]), **stats(phase_totals[p])} for p in PHASES},
        'steps': {'expected': expected_steps, 'count': len(steps), 'complete_count': len(steps) == expected_steps,
                  'frame_pattern': STEP.pattern, 'other_transformer_forward_names': dict(loose_step_names),
                  'first': step_rows[0] if step_rows else None,
                  'steps_1_plus_mean_gpu_event_sum_ms': statistics.fmean(measured_steps[1:]) if len(measured_steps) > 1 else None,
                  'per_step': step_rows},
        'execution_evidence': {
            'fp8_gemm': [x for x in named if x['category'] == 'fp8_gemm'],
            'bf16_fa3_sm90': [x for x in named if x['category'] == 'fa3_sm90_bf16'],
            'unresolved_gemm': [x for x in named if x['category'] == 'gemm_dtype_unresolved'],
            'unresolved_fa3_sm90': [x for x in named if x['category'] == 'fa3_sm90_dtype_unresolved'],
        },
        'all_kernel_names': named, 'export_only_adapter_markers': adapters,
    }


def markdown(cases):
    lines = ['# C1 profile evidence', '',
             'Only captured GPU-event duration sums are shown. They are not E2E latency, do not include all host gaps, and can overlap across streams. Use separate unprofiled runs for performance.', '',
             '| Case | Steps | Whole GPU-event sum (ms) | Denoise FP8 GEMM (ms) | Denoise BF16 FA3 SM90 (ms) |',
             '|---|---:|---:|---:|---:|']
    for c in cases:
        cats = c['denoise']['categories']
        get = lambda n: cats.get(n, {}).get('gpu_event_sum_ms', 0)
        lines.append(f"| {c['case']} | {c['steps']['count']} | {c['whole_trace']['gpu_event_sum_ms']:.3f} | {get('fp8_gemm'):.3f} | {get('fa3_sm90_bf16'):.3f} |")
    for c in cases:
        lines += ['', f"## {c['case']}", '', '| Denoise category | Count | GPU-event sum (ms) | Share of sum |', '|---|---:|---:|---:|']
        for k, v in c['denoise']['categories'].items():
            lines.append(f"| {k} | {v['event_count']} | {v['gpu_event_sum_ms']:.3f} | {v['pct_of_gpu_event_sum']:.2f}% |")
        lines += ['', 'Full names, operator input types, evidence tier, whole-trace distribution and attribution coverage are in this case JSON.']
        if c['warnings']:
            lines += [''] + [f'- {w}' for w in c['warnings']]
    return '\n'.join(lines) + '\n'


def self_test():
    assert classify('cutlass::Kernel2<Gemm<float_e4m3_t, bfloat16_t>>', 'kernel')[0] == 'fp8_gemm'
    assert classify('nvjet_sm90_gemm', 'kernel')[0] == 'gemm_dtype_unresolved'
    assert classify('nvjet_sm90_gemm', 'kernel', ('aten::mm', ['c10::Float8_e4m3fn', 'c10::Float8_e4m3fn']))[0] == 'fp8_gemm'
    assert classify('nvjet_sm90_gemm', 'kernel', ('aten::mm', ['BFloat16', 'BFloat16']))[0] == 'bf16_gemm'
    assert classify('flash::FlashAttnFwdSm90<cutlass::bfloat16_t>', 'kernel')[0] == 'fa3_sm90_bf16'
    assert classify('flash::FlashAttnFwdSm90<float>', 'kernel')[0] == 'fa3_sm90_dtype_unresolved'
    assert classify('flash::FlashAttnFwdSm90<float_e4m3_t, bfloat16_t>', 'kernel')[0] == 'fa3_sm90_fp8_or_mixed'
    assert classify('marlin_gemm_fp8_w8a16', 'kernel')[0] == 'weight_only_gemm_not_native_fp8_proof'
    assert classify('at::native::elementwise_kernel<gpu_kernel_impl_nocast>', 'kernel')[0] == 'elementwise'
    assert classify('scaled_fp8_quant_kernel', 'kernel')[0] == 'quantization'
    assert classify('bfloat16_copy_kernel_cuda', 'kernel')[0] == 'dtype_cast'
    print('classification self-tests passed')


COMPONENT_PATTERNS = {
    'te_vision': re.compile(r'^nn\.Module: Qwen3VLVisionModel_\d+$'),
    'te_language': re.compile(r'^nn\.Module: Qwen3VLTextModel_\d+$'),
    'dit': re.compile(r'^nn\.Module: QwenImage21Transformer2DModel_\d+$'),
    'te_other': re.compile(r'^nn\.Module: Qwen3VLForConditionalGeneration_\d+$'),
}
QUANT_FRAME = re.compile(r'vllm/model_executor/layers/quantization/input_quant_fp8\.py\(\d+\): forward_native$')
COMPILED_FRAME = re.compile(r'^(.*/torchinductor_[^/]+/.*\.py)\(\d+\): call$')
MODEL_SOURCE_FRAME = re.compile(r'(?:modeling_qwen3_vl|qwen_image_21_transformer)\.py\(\d+\): forward$')


def load_quant_sources(directory):
    """Read captured source as text only; never import generated CUDA code."""
    if not directory or not (directory / 'manifest.json').exists():
        return {}, {'available': False, 'reason': 'current-run source manifest not captured'}
    manifest = json.loads((directory / 'manifest.json').read_text())
    sources, provenance = {}, []
    for item in manifest['files']:
        local = directory / Path(item['copied_path']).name
        if sha256(local) != item['sha256']:
            raise ValueError(f'Captured source hash mismatch: {local}')
        sources[item['source_path']] = local.read_text()
        provenance.append({**item, 'local_path': str(local.resolve())})
    native = [v for k, v in sources.items() if k.endswith('/quantization/input_quant_fp8.py')]
    verified_native = len(native) == 1 and hashlib.sha256(native[0].encode()).hexdigest() == '7ab6276f1e2d9adf248fc28bcd41031374942db7e5e88dbb15e4f90a34330202'
    return sources, {'available': True, 'pinned_quantizer_source_verified': verified_native, 'files': provenance}


def component_execution(path, sources, source_proof):
    launches, ambiguous, gpu, operators, shapes = {}, set(), [], {}, {}
    frames = collections.defaultdict(lambda: collections.defaultdict(list))
    frame_counts = collections.Counter()
    for e in iter_events(path):
        cat, args, name = e.get('cat'), e.get('args') or {}, e.get('name', '')
        if cat in ('cuda_runtime', 'cuda_driver') and 'correlation' in args:
            value = (e['ts'], args.get('External id'), e.get('pid'), e.get('tid'))
            corr = args['correlation']
            if corr in launches and launches[corr] != value:
                ambiguous.add(corr)
            launches[corr] = value
        elif cat == 'cpu_op' and 'External id' in args and INTERESTING_OP.search(name):
            operators[args['External id']] = (name, args.get('Input type'))
            shapes[args['External id']] = args.get('Input Dims', args.get('Input Shapes'))
        elif cat in GPU_CATS and 'dur' in e:
            gpu.append((name, cat, e['dur'], args.get('correlation'), args.get('External id')))
        elif cat == 'python_function' and 'dur' in e:
            thread = (e.get('pid'), e.get('tid'))
            interval = (e['ts'], e['ts'] + e['dur'], name)
            for component, pattern in COMPONENT_PATTERNS.items():
                if pattern.search(name):
                    frames[component][thread].append(interval)
                    frame_counts[component] += 1
            if QUANT_FRAME.search(name):
                frames['quantizer'][thread].append(interval)
            if COMPILED_FRAME.search(name):
                frames['compiled'][thread].append(interval)
            if MODEL_SOURCE_FRAME.search(name):
                frames['model_source'][thread].append(interval)
    starts, maximum_ends = {}, {}
    for kind, threads in frames.items():
        starts[kind] = {}
        maximum_ends[kind] = {}
        for thread, rows in threads.items():
            rows.sort()
            starts[kind][thread] = [x[0] for x in rows]
            ends, last = [], float('-inf')
            for row in rows:
                last = max(last, row[1]); ends.append(last)
            maximum_ends[kind][thread] = ends

    def enclosing(kind, thread, ts):
        times = starts.get(kind, {}).get(thread, [])
        idx = bisect.bisect_right(times, ts) - 1
        while idx >= 0 and maximum_ends[kind][thread][idx] >= ts:
            row = frames[kind][thread][idx]
            if ts <= row[1]:
                return row[2]
            idx -= 1
        return None

    kernel_counts = collections.Counter(name for name, cat, *_ in gpu if cat == 'kernel')
    verified_quant, candidate_quant, pending_requests = collections.Counter(), collections.Counter(), {}
    attributed = []
    for name, cat, dur, corr, ext in gpu:
        launch = launches.get(corr) if corr not in ambiguous else None
        component, comp_frame, quant_frame, code_frame, code_path, model_source = 'unassigned', None, None, None, None, None
        if launch:
            ts, launch_ext, pid, tid = launch
            thread = (pid, tid)
            for c in ('te_vision', 'te_language', 'dit', 'te_other'):
                matched = enclosing(c, thread, ts)
                if matched:
                    component, comp_frame = c, matched
                    break
            quant_frame = enclosing('quantizer', thread, ts)
            code_frame = enclosing('compiled', thread, ts)
            model_source = enclosing('model_source', thread, ts)
            if code_frame:
                code_path = COMPILED_FRAME.search(code_frame)[1]
            op = operators.get(ext) or operators.get(launch_ext)
            op_shapes = shapes.get(ext) or shapes.get(launch_ext)
        else:
            op, op_shapes = operators.get(ext), shapes.get(ext)
        # A generated name alone is never sufficient to turn copy into quantization.
        if cat == 'kernel' and quant_frame and code_path:
            candidate_quant[name] += 1
            code = sources.get(code_path, '')
            verified = source_proof.get('pinned_quantizer_source_verified') and f'def {name}(' in code and 'torch.float8_e4m3fn' in code and 'tl.float8e4nv' in code and 'reciprocal' in code
            if verified:
                verified_quant[name] += 1
            else:
                pending_requests[(name, code_path)] = {'kernel_name': name, 'compiled_source_path': code_path,
                    'quantizer_frame': quant_frame, 'needed': 'capture exact generated .py and installed input_quant_fp8.py with hashes in current-run source manifest'}
        attributed.append((name, cat, dur, component, comp_frame, model_source, op, op_shapes))
    overrides = {name: ('quantization', 'installed_source_generated_code_and_same_thread_launch')
                 for name, count in verified_quant.items() if count == kernel_counts[name]}
    rows = {c: collections.defaultdict(lambda: [0, 0.0]) for c in (*COMPONENT_PATTERNS, 'unassigned')}
    evidence = collections.defaultdict(lambda: {'event_count': 0, 'gpu_event_sum_ms': 0.0, 'source_module_frames': set(), 'model_source_functions': set(), 'operator_shapes': set()})
    saved_overrides = dict(CLASSIFICATION_OVERRIDES)
    CLASSIFICATION_OVERRIDES.clear(); CLASSIFICATION_OVERRIDES.update(overrides)
    for name, cat, dur, component, comp_frame, model_source, op, dims in attributed:
        category, tier = classify(name, cat, op)
        rows[component][category][0] += 1; rows[component][category][1] += dur
        if category in ('fp8_gemm', 'quantization'):
            item = evidence[(component, name, category)]
            item['event_count'] += 1; item['gpu_event_sum_ms'] += dur / 1000
            if comp_frame:
                item['source_module_frames'].add(comp_frame)
            if model_source:
                item['model_source_functions'].add(model_source)
            if op:
                item['operator_shapes'].add(json.dumps({'name': op[0], 'input_types': op[1], 'input_dims': dims}, sort_keys=True))
    CLASSIFICATION_OVERRIDES.clear(); CLASSIFICATION_OVERRIDES.update(saved_overrides)
    named = [{'component': c, 'kernel_name': name, 'category': cat, **v,
              'source_module_frames': sorted(v['source_module_frames']),
              'model_source_functions': sorted(v['model_source_functions']),
              'operator_shapes': [json.loads(x) for x in sorted(v['operator_shapes'])]}
             for (c, name, cat), v in evidence.items()]
    return {'components': {c: stats(v) for c, v in rows.items()},
            'component_root_frame_counts': dict(frame_counts), 'execution_evidence': named,
            'quantization_source_requests': list(pending_requests.values()),
            'verified_quantization_kernel_names': sorted(overrides),
            'quantization_candidates': dict(candidate_quant), 'source_proof': source_proof,
            'boundary': 'GPU -> CUDA launch correlation -> enclosing same CPU pid/tid module frame. Vision takes precedence over enclosing TE root. Shapes are recorded as supporting evidence, never the sole component inference. Unattributed FP8 stays unassigned.'}, overrides


def execution_gates(result, components, spec, mode):
    errors = list(result['warnings'])
    denoise = result['denoise']['categories']
    actual = denoise.get('fp8_gemm', {}).get('event_count', 0)
    expected = spec['expected_dit_fp8_linears'] * result['steps']['expected']
    if actual != expected:
        errors.append(f'DiT FP8 GEMM count {actual}, expected {spec["expected_dit_fp8_linears"]} linears × {result["steps"]["expected"]} steps = {expected}')
    step_counts = [row['categories'].get('fp8_gemm', {}).get('event_count', 0) for row in result['steps']['per_step']]
    wrong_steps = [i for i, n in enumerate(step_counts) if n != spec['expected_dit_fp8_linears']]
    if wrong_steps:
        errors.append(f'DiT FP8 GEMM count differs from eligible Linear count in steps {wrong_steps}')
    for category in ('fa3_sm90_fp8_or_mixed', 'fa3_sm90_dtype_unresolved'):
        if denoise.get(category, {}).get('event_count', 0):
            errors.append(f'Denoise contains non-BF16/unresolved FA3 execution: {category}')
    component_counts = {c: x['categories'].get('fp8_gemm', {}).get('event_count', 0) for c, x in components['components'].items()}
    if component_counts['dit'] != actual:
        errors.append('DiT module-frame FP8 attribution disagrees with exact denoise-frame attribution')
    te_count = component_counts['te_language']
    if (spec['expected_te_fp8_linears'] > 0) != (te_count > 0):
        errors.append('Observed TE-language FP8 GEMM presence disagrees with planned arm')
    for c in ('te_vision', 'te_other', 'unassigned'):
        if component_counts[c]:
            errors.append(f'Unexpected/unresolved FP8 GEMM in {c}: {component_counts[c]}')
    if components['component_root_frame_counts'].get('te_language', 0) < 1:
        errors.append('TE-language root module frame absent; cannot validate component attribution')
    if mode == 'edit' and components['component_root_frame_counts'].get('te_vision', 0) < 1:
        errors.append('Edit trace lacks actual vision root module frame')
    if components['quantization_source_requests']:
        errors.append('Current-run exact generated quantization source is missing/unverified; capture requested files and rerun')
    return {'passed': not errors, 'errors': errors, 'expected_dit_fp8_gemm_count': expected,
            'observed_dit_fp8_gemm_count': actual, 'observed_component_fp8_gemm_counts': component_counts,
            'observed_dit_fp8_gemm_count_per_step': step_counts,
            'expected_te_fp8_linears': spec['expected_te_fp8_linears'],
            'te_boundary': 'Trace proves execution and component attribution. Full eligible/executed TE Linear coverage is verified independently by quant_audit; do not assume every Linear emits exactly one GEMM from only this count.'}


def main():
    import importlib.util
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument('--case', action='append', default=[], metavar='ARM-MODE=TRACE')
    p.add_argument('--arms', nargs='+', default=['bf16', 'dit_fp8', 'te_fp8', 'joint_fp8', 'selective_bf16'])
    p.add_argument('--arm-manifest', type=Path)
    p.add_argument('--quant-source-dir', type=Path)
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--expected-steps', type=int, default=50)
    p.add_argument('--fixture-only', action='store_true')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        self_test()
        if not a.case:
            return
    modspec = importlib.util.spec_from_file_location('perf', Path(__file__).with_name('analyze_performance.py'))
    perf = importlib.util.module_from_spec(modspec); modspec.loader.exec_module(perf)
    manifest = a.arm_manifest or a.root / 'scripts/configs/arms.json'
    specs = perf.load_arm_specs(manifest, a.arms)
    perf.ARM_SPECS = specs
    source_dir = a.quant_source_dir or a.root / 'supplementary/profiler-kernel-source'
    sources, source_proof = load_quant_sources(source_dir)
    out = a.out_dir or a.root / 'analysis/profiles'
    out.mkdir(parents=True, exist_ok=True)
    items, errors = [], []
    if a.case:
        for entry in a.case:
            label, path = entry.split('=', 1)
            arm, mode = label.rsplit('-', 1)
            if arm not in specs or mode not in ('t2i', 'edit'):
                p.error('case label must be a selected ARM followed by -t2i or -edit')
            items.append((arm, mode, Path(path)))
    else:
        for arm in a.arms:
            for mode in ('t2i', 'edit'):
                paths = list((a.root / f'results/trace_{arm}_{mode}').rglob('trace_rank0.json.gz'))
                if len(paths) != 1:
                    errors.append(f'{arm}-{mode}: expected one trace, found {len(paths)}')
                else:
                    items.append((arm, mode, paths[0]))
    cases, requests, env_reference, workload_references = [], [], None, {}
    for arm, mode, path in items:
        label = f'{arm}-{mode}'
        print(f'Component audit {label}', file=sys.stderr, flush=True)
        try:
            components, overrides = component_execution(path, sources, source_proof)
            CLASSIFICATION_OVERRIDES.clear(); CLASSIFICATION_OVERRIDES.update(overrides)
            result = analyze(path, label, a.expected_steps)
            result['arm'], result['mode'] = arm, mode
            result['component_execution'] = components
            result['execution_gates'] = execution_gates(result, components, specs[arm], mode)
            result['fixture_only_not_phase2_evidence'] = a.fixture_only
            if not a.fixture_only:
                env_errors = []
                env_path = a.root / f'results/trace_{arm}_{mode}/env.json'
                env = perf.read_json(env_path, env_errors) or {}
                core, workload = perf.env_audit(env, arm, mode, a.root, env_errors)
                if env_reference is None:
                    env_reference = core
                elif env_reference != core:
                    env_errors.append('Trace core environment differs across arms/workloads')
                if mode not in workload_references:
                    workload_references[mode] = workload
                elif workload_references[mode] != workload:
                    env_errors.append('Paired trace workload differs across arms for this mode')
                for key, expected in [('trace', True), ('warmup', 1), ('feasibility', 0), ('measured', 0), ('mem', False), ('stages', False)]:
                    if (env.get('workload') or {}).get(key) != expected:
                        env_errors.append(f'Unexpected trace protocol {key}')
                if env.get('profiler_export_only_enabled') is not True:
                    env_errors.append('Missing export-only diagnostic adaptation metadata')
                result['profile_environment_audit'] = {'core': core, 'workload': workload, 'errors': env_errors,
                    'env_sha256': sha256(env_path) if env_path.exists() else None}
                result['execution_gates']['errors'] += env_errors
                result['execution_gates']['passed'] = not result['execution_gates']['errors']
            requests += components['quantization_source_requests']
            (out / f'{label}.json').write_text(json.dumps(result, indent=2) + '\n')
            cases.append(result)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{label}: {type(exc).__name__}: {exc}')
    expected_cases = len(a.case) if a.case else len(a.arms) * 2
    valid = len(cases) == expected_cases and not errors and all(c['execution_gates']['passed'] for c in cases)
    report = {'cases': cases, 'case_count': len(cases), 'expected_case_count': expected_cases,
              'complete_and_valid': valid, 'errors': errors, 'not_e2e_latency': True,
              'all_expected_steps': len(cases) == expected_cases and all(c['steps']['complete_count'] for c in cases),
              'fixture_only_not_phase2_evidence': a.fixture_only, 'arm_manifest_sha256': sha256(manifest)}
    (out / 'profile_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'quantization_source_requests.json').write_text(json.dumps(requests, indent=2) + '\n')
    text = markdown(cases)
    text += '\n## Component execution\n\n| Case | Component | FP8 GEMM count | FP8 GPU sum ms |\n|---|---|---:|---:|\n'
    for c in cases:
        for comp, row in c['component_execution']['components'].items():
            v = row['categories'].get('fp8_gemm', {})
            text += f"| {c['case']} | {comp} | {v.get('event_count', 0)} | {v.get('gpu_event_sum_ms', 0):.3f} |\n"
        text += ''.join(f"\n- {c['case']}: {e}\n" for e in c['execution_gates']['errors'])
    text += '\n' + '\n'.join('- ' + e for e in errors) + '\n'
    if a.fixture_only:
        text = '# TOOL FIXTURE ONLY — NOT PHASE 2 RESULTS\n\n' + text
    (out / 'PROFILE_ANALYSIS.md').write_text(text)
    print(json.dumps({'complete_and_valid': valid, 'cases': len(cases), 'source_requests': len(requests), 'output': str(out)}))
    raise SystemExit(0 if valid else 2)


if __name__ == '__main__':
    main()

