"""Fail-closed Phase2 worker audit gate; CPU-only reader, no torch imports."""
import argparse
import hashlib
import json
from pathlib import Path
from audit_protocol import ARMS, DTYPE, OMNI_SHA, TOPOLOGY, gpu_meta, inventory_errors

ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / 'source_guards.json').read_text())


def verify(directory, arm, task='t2i'):
    directory = Path(directory)
    errors = []
    workers = []
    inventories = sorted(directory.glob('worker_inventory_*.json'))
    if len(inventories) != 1:
        errors.append('Expected exactly one TP1 worker inventory (fresh output directory)')
    expected_adapter = hashlib.sha256((ROOT / 'sitecustomize.py').read_bytes()).hexdigest()
    required_modules = {k for k in SOURCES if k.startswith('vllm_omni.') and not k.endswith('fp8_config')}
    if arm != 'bf16':
        required_modules = set(SOURCES)
    for path in inventories:
        inv = json.loads(path.read_text()); pid = inv['pid']; workers.append(pid)
        errors.extend(f'pid {pid}: {e}' for e in inventory_errors(inv, arm))
        if inv.get('schema_version') != 2 or inv.get('arm') != arm:
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
            if data.get('pid') != pid or data.get('arm') != arm or data.get('omni_sha') != OMNI_SHA:
                errors.append(f'pid {pid}: {kind} identity mismatch')
        prefix = files.get('prefix_runtime', {}).get('observations', {})
        for phase in ('cond_prefill','cond_decode'):
            obs = prefix.get(phase, {}); blocks = obs.get('blocks', [])
            if len(blocks) != 32 or {x.get('block') for x in blocks} != set(range(32)):
                errors.append(f'pid {pid}: missing complete unique 32-block {phase}')
            if obs.get('configured_prefix_dtype') is not None:
                errors.append(f'pid {pid}: {phase} prefix config not native')
            for block in blocks:
                tensors = block.get('tensors', {})
                if set(tensors) != {'key','value'} or not all(gpu_meta(x,DTYPE) for x in tensors.values()):
                    errors.append(f'pid {pid}: {phase} block {block.get("block")} KV not native CUDA BF16/no scale')
        attention = files.get('attention_runtime', {}).get('blocks', {})
        if set(attention) != {str(i) for i in range(32)}:
            errors.append(f'pid {pid}: incomplete 32-block attention observations')
        for index, layer in attention.items():
            if not all(gpu_meta(layer.get(k),DTYPE) for k in ('query','key','value')):
                errors.append(f'pid {pid}: attention {index} QKV not CUDA BF16')
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
    return {'passed':not errors,'arm':arm,'task':task,'worker_pids':workers,'errors':errors,
            'scope':'Actual worker topology/weights/scales, selected online method, executed module BF16 input/output, native KV/attention metadata. Not executed-GEMM kernel proof or quality/performance proof.',
            'vision_execution_required':task=='edit','expected_hook_sha256':expected_adapter}


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('--arm',choices=ARMS,required=True);parser.add_argument('--task',choices=('t2i','edit'),default='t2i');args=parser.parse_args()
    result=verify(args.directory,args.arm,args.task);print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
