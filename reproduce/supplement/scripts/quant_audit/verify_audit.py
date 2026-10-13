"""Supplement worker audit gate; CPU-only reader, no torch imports."""
import argparse
import hashlib
import json
from pathlib import Path
from audit_protocol import (ARMS, DTYPE, OMNI_SHA, TOPOLOGY, SCHEMA_VERSION,
                            expected_names, expected_prefix_dtype,
                            expected_prefix_tensors, gpu_meta, inventory_errors)

ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / 'source_guards.json').read_text())


def request_evidence_errors(data, directory, arm):
    errors = []
    marker = data.get('request_start', {})
    case = marker.get('case', {})
    ordinal = marker.get('request_ordinal')
    if not isinstance(ordinal, int) or ordinal < 1:
        return ['Missing positive actual-request ordinal; initialization evidence is invalid']
    filename = f'request_start_{ordinal:04d}.json'
    if data.get('request_start_file') != filename:
        errors.append('Unexpected immutable request marker filename')
    path = directory / filename
    if not path.exists():
        errors.append('Missing preserved actual-request marker')
    else:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != data.get('request_start_sha256') or json.loads(raw) != marker:
            errors.append('Request marker hash/content mismatch')
    fingerprint = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    if marker.get('case_fingerprint_sha256') != fingerprint or marker.get('case_id') != case.get('id'):
        errors.append('Actual-request case fingerprint mismatch')
    if (marker.get('schema_version') != 1 or marker.get('arm') != arm
            or marker.get('engine_initialization_complete') is not True
            or marker.get('request_phase') not in {'warmup', 'feasibility', 'measured'}):
        errors.append('Invalid real-request marker identity/phase')
    env_path = directory.parent / 'env.json'
    rows_path = directory.parent / 'runs.jsonl'
    if not env_path.exists() or not rows_path.exists():
        errors.append('Missing caller environment/request rows to bind runtime to the case')
        return errors
    env = json.loads(env_path.read_text())
    if env.get('cases') != [case] or env.get('arm') != arm:
        errors.append('Runtime case differs from caller env.cases')
    if env.get('harness_sha256') != marker.get('harness_sha256') or env.get('case_file_sha256') != marker.get('case_file_sha256'):
        errors.append('Caller harness/case-file provenance mismatch')
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if ordinal > len(rows):
        errors.append('Actual-request ordinal is absent from completed caller rows')
    else:
        row = rows[ordinal - 1]
        if (row.get('case_id') != case.get('id') or row.get('phase') != marker.get('request_phase')
                or row.get('error') or any(row.get(k) != v for k, v in case.items())):
            errors.append('Actual-request marker does not match its successful caller row')
    return errors


def prefix_observation_errors(obs, arm):
    """Validate actual stored tensors and byte accounting without reading values."""
    import math
    errors = []
    blocks = obs.get('blocks', [])
    if len(blocks) != 32 or {x.get('block') for x in blocks} != set(range(32)):
        errors.append('Missing complete unique 32-block cache')
    if obs.get('configured_prefix_dtype') != expected_prefix_dtype(arm):
        errors.append('Unexpected prefix configuration')
    wanted = expected_prefix_tensors(arm)
    logical = {name: 0 for name in wanted}
    storages = {}
    bf16_payload = 0
    shapes = set()
    for block in blocks:
        tensors = block.get('tensors', {})
        if set(tensors) != set(wanted):
            errors.append(f'block {block.get("block")}: cache tensor/scale names mismatch')
        keyshape = tensors.get('key', {}).get('shape', [])
        if len(keyshape) != 4 or not all(isinstance(n, int) and n > 0 for n in keyshape):
            errors.append(f'block {block.get("block")}: invalid BSHD cache shape')
            continue
        shapes.add(tuple(keyshape))
        for name, dtype in wanted.items():
            tensor = tensors.get(name, {})
            if not gpu_meta(tensor, dtype):
                errors.append(f'block {block.get("block")}: {name} dtype/device mismatch')
            expected_shape = keyshape[:-1] + [1] if name.endswith('_scale') else keyshape
            element_size = 4 if name.endswith('_scale') else (1 if dtype == 'torch.float8_e4m3fn' else 2)
            nbytes = math.prod(expected_shape) * element_size
            if (tensor.get('shape') != expected_shape or tensor.get('element_size') != element_size
                    or tensor.get('numel') != math.prod(expected_shape) or tensor.get('logical_nbytes') != nbytes):
                errors.append(f'block {block.get("block")}: {name} shape/byte metadata mismatch')
            index = tensor.get('storage_index')
            storage_nbytes = tensor.get('storage_nbytes')
            if not isinstance(index, int) or not isinstance(storage_nbytes, int) or storage_nbytes < nbytes:
                errors.append(f'block {block.get("block")}: {name} invalid storage metadata')
            else:
                if index in storages and storages[index] != storage_nbytes:
                    errors.append('Conflicting sizes for one storage')
                storages[index] = storage_nbytes
            logical[name] += nbytes
            if name in ('key', 'value'):
                bf16_payload += math.prod(expected_shape) * 2
    if len(shapes) != 1:
        errors.append('Unexpected cross-block cache shape mismatch')
    totals = obs.get('cache_bytes', {})
    if totals.get('logical_by_name') != logical or totals.get('logical_total') != sum(logical.values()):
        errors.append('Cache logical-byte totals mismatch')
    if totals.get('unique_storage_total') != sum(storages.values()) or totals.get('unique_storage_count') != len(storages):
        errors.append('Cache unique-storage-byte totals mismatch')
    if totals.get('bf16_equivalent_kv_payload') != bf16_payload:
        errors.append('BF16-equivalent KV payload mismatch')
    return errors


def verify(directory, arm, task='t2i'):
    directory = Path(directory)
    errors = []
    workers = []
    inventories = sorted(directory.glob('worker_inventory_*.json'))
    if len(inventories) != 1:
        errors.append('Expected exactly one TP1 worker inventory (fresh output directory)')
    expected_adapter = hashlib.sha256((ROOT / 'sitecustomize.py').read_bytes()).hexdigest()
    required_modules = {k for k in SOURCES if k.startswith('vllm_omni.') and not k.endswith('fp8_config')}
    if expected_names(arm)['fp8']:
        required_modules = set(SOURCES)
    for path in inventories:
        inv = json.loads(path.read_text()); pid = inv['pid']; workers.append(pid)
        errors.extend(f'pid {pid}: {e}' for e in inventory_errors(inv, arm))
        if inv.get('schema_version') != SCHEMA_VERSION or inv.get('arm') != arm:
            errors.append(f'pid {pid}: schema/arm mismatch')
        if inv.get('adapter_source_sha256') != expected_adapter:
            errors.append(f'pid {pid}: hook hash differs from verifier-bundled hook')
        local_protocol = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('audit_protocol.py','expected_topology.json','source_guards.json')}
        if inv.get('audit_protocol_sha256') != local_protocol:
            errors.append(f'pid {pid}: protocol/helper hash mismatch')
        observed_guards = inv.get('source_guards', {})
        if not required_modules <= set(observed_guards) or any(observed_guards.get(k) != SOURCES[k] for k in required_modules):
            errors.append(f'pid {pid}: missing/mismatched actual exact-source guards')
        pins = {'vllm':'0.31.0','torch':'2.13.0+cu130','transformers':'5.14.1','diffusers':'0.40.0'}
        if any(inv.get('packages',{}).get(k) != v for k,v in pins.items()):
            errors.append(f'pid {pid}: core package mismatch')
        files = {}
        for kind in ('prefix_runtime','attention_runtime','text_encoder_runtime'):
            p = directory / f'{kind}_{pid}.json'
            if not p.exists():
                errors.append(f'pid {pid}: missing {kind} actual execution')
                continue
            data = json.loads(p.read_text()); files[kind] = data
            if data.get('schema_version') != SCHEMA_VERSION or data.get('pid') != pid or data.get('arm') != arm or data.get('omni_sha') != OMNI_SHA:
                errors.append(f'pid {pid}: {kind} identity mismatch')
            errors.extend(f'pid {pid}: {kind}: {e}' for e in request_evidence_errors(data, directory, arm))
        request_hashes = {data.get('request_start_sha256') for data in files.values()}
        if len(request_hashes) != 1 or None in request_hashes:
            errors.append(f'pid {pid}: runtime modules do not refer to the same real request')
        prefix = files.get('prefix_runtime', {}).get('observations', {})
        for phase in ('cond_prefill','cond_decode'):
            obs = prefix.get(phase, {})
            errors.extend(f'pid {pid}: {phase}: {e}' for e in prefix_observation_errors(obs, arm))
        for phase in ('cond_prefill', 'cond_decode'):
            attention = files.get('attention_runtime', {}).get('observations', {}).get(phase, {})
            if set(attention) != {str(i) for i in range(32)}:
                errors.append(f'pid {pid}: incomplete 32-block {phase} attention observations')
            for index, layer in attention.items():
                if not all(gpu_meta(layer.get(k),DTYPE) for k in ('query','key','value')):
                    errors.append(f'pid {pid}: {phase} attention {index} QKV not CUDA BF16')
        te = files.get('text_encoder_runtime', {}); linears = te.get('linears', {})
        expected_language = set(TOPOLOGY['te_language_linears'])
        expected_vision = set(TOPOLOGY['te_vision_linears'])
        if not expected_language <= set(linears):
            errors.append(f'pid {pid}: incomplete actual 36-layer/252-Linear language execution')
        if task == 'edit' and not expected_vision <= set(linears):
            errors.append(f'pid {pid}: edit missing actual full 116-Linear vision execution')
        if set(linears) - (expected_language | expected_vision | set(TOPOLOGY['te_lm_head'])):
            errors.append(f'pid {pid}: unknown text encoder runtime layer')
        if not gpu_meta(te.get('last_hidden_state'),DTYPE):
            errors.append(f'pid {pid}: text encoder output hidden state not CUDA BF16')
        for name, layer in linears.items():
            if not all(gpu_meta(layer.get(k),DTYPE) for k in ('input','output')):
                errors.append(f'pid {pid}: {name} input/output not CUDA BF16')
    return {'passed':not errors,'schema_version':SCHEMA_VERSION,'arm':arm,'task':task,'worker_pids':workers,'errors':errors,
            'scope':'Actual worker weights/scales, selected online method, executed BF16 module I/O, prefill/decode prefix storage precision/bytes and BF16 attention. Not executed-GEMM proof, peak VRAM, quality or performance proof.',
            'vision_execution_required':task=='edit','expected_hook_sha256':expected_adapter}


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--arm',choices=ARMS,required=True);parser.add_argument('--task',choices=('t2i','edit'),default='t2i');args=parser.parse_args()
    result=verify(args.directory,args.arm,args.task);print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
