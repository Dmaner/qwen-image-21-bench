"""Frozen Phase2 topology and expected precision; no torch/GPU dependencies."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ('bf16', 'dit_fp8', 'te_fp8', 'joint_fp8', 'selective_bf16')
OMNI_SHA = '3dc35694b3d458fc1461a8fe71832ac12e8c8cea'
TOPOLOGY = json.loads((ROOT / 'expected_topology.json').read_text())
DTYPE = 'torch.bfloat16'
FP8 = 'torch.float8_e4m3fn'


def expected_names(arm):
    if arm not in ARMS:
        raise ValueError('Unknown audit arm: ' + str(arm))
    dit = set(TOPOLOGY['dit_block_linears']) if arm in ('dit_fp8', 'joint_fp8', 'selective_bf16') else set()
    if arm == 'selective_bf16':
        dit = {n for n in dit if int(n.split('.')[2]) not in (0, 1, 30, 31)}
    te = set(TOPOLOGY['te_language_linears']) if arm in ('te_fp8', 'joint_fp8') else set()
    all_names = set().union(*(set(v) for v in TOPOLOGY.values() if isinstance(v, list)))
    return {'dit_fp8': dit, 'te_fp8': te, 'fp8': dit | te, 'bf16': all_names - dit - te, 'all': all_names}


def gpu_meta(meta, dtype):
    return bool(meta and meta.get('dtype') == dtype and meta.get('device', '').startswith('cuda') and not meta.get('is_meta', False))


def inventory_errors(inventory, arm):
    errors = []
    expected = expected_names(arm)
    layers = inventory.get('linears', [])
    names = [x['name'] for x in layers]
    if len(names) != len(set(names)) or set(names) != expected['all']:
        errors.append('Linear inventory differs from frozen 537-name topology')
    for layer in layers:
        name = layer['name']; is_fp8 = name in expected['fp8']
        if not gpu_meta(layer.get('weight'), FP8 if is_fp8 else DTYPE):
            errors.append(name + ': unexpected weight dtype/device')
        method = layer.get('quant_method') or ''
        if is_fp8:
            if 'Fp8PerTensorOnlineLinearMethod' not in method:
                errors.append(name + ': not expected online FP8 method')
            if not gpu_meta(layer.get('scales', {}).get('weight_scale'), 'torch.float32'):
                errors.append(name + ': missing CUDA FP32 weight scale')
            if layer.get('postprocess_flag') is not True:
                errors.append(name + ': post-load processing not observed')
            if 'CutlassFP8ScaledMMLinearKernel' not in (layer.get('fp8_linear_kernel_class') or ''):
                errors.append(name + ': unexpected FP8 kernel selection for H200')
            if layer.get('use_marlin') is True:
                errors.append(name + ': unexpected Marlin path')
        elif method and not method.endswith('UnquantizedLinearMethod'):
            errors.append(name + ': unexpected quant method on preserved BF16 layer')
    # Check ALL component parameters, not only named Linear weights. The only
    # non-BF16 tensors allowed are intended FP8 weights and their FP32 scales.
    for component, values in inventory.get('components', {}).items():
        params = values.get('parameters', [])
        if not params:
            errors.append(component + ': missing parameter inventory')
        for param in params:
            name = param['name']
            expected_dtype = DTYPE
            if name.endswith('.weight') and name[:-7] in expected['fp8']:
                expected_dtype = FP8
            elif name.endswith('.weight_scale') and name[:-13] in expected['fp8']:
                expected_dtype = 'torch.float32'
            if not gpu_meta(param, expected_dtype):
                errors.append(name + ': unexpected component parameter dtype/device')
    if set(inventory.get('components', {})) != {'transformer', 'text_encoder', 'vae'}:
        errors.append('Missing component inventory')
    if inventory.get('model_architecture') != {'dit_blocks':32,'te_language_layers':36,'te_vision_blocks':27}:
        errors.append('Actual model config is not DiT32 / language36 / vision27')
    runtime = inventory.get('resolved_runtime', {})
    if runtime.get('dtype') != DTYPE or runtime.get('enforce_eager') is not True or runtime.get('enable_cuda_graph_decode') is not False:
        errors.append('Expected BF16/eager/no CUDA graph runtime')
    if runtime.get('prefix_kv_cache_dtype') is not None or runtime.get('diffusion_kv_cache_dtype') not in (None, 'auto'):
        errors.append('Expected native BF16 prefix/attention configuration')
    if inventory.get('omni_sha') != OMNI_SHA:
        errors.append('Wrong Omni source pin')
    return errors
