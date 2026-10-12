"""CPU synthetic guard tests, never runtime support evidence."""
import copy
import fnmatch
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from audit_protocol import ARMS, TOPOLOGY, OMNI_SHA, expected_names, inventory_errors
from verify_audit import verify, SOURCES
import hashlib

ROOT=Path(__file__).resolve().parent

def meta(dtype='torch.bfloat16'):
    return {'dtype':dtype,'device':'cuda:0','shape':[4,4],'is_meta':False}

def synthetic(arm):
    expected=expected_names(arm); layers=[];components={k:{'parameters':[]} for k in ['transformer','text_encoder','vae']}
    for name in sorted(expected['all']):
        fp8=name in expected['fp8'];w=meta('torch.float8_e4m3fn' if fp8 else 'torch.bfloat16')
        layer={'name':name,'weight':w,'quant_method':'test.Fp8PerTensorOnlineLinearMethod' if fp8 else 'test.UnquantizedLinearMethod','scales':{'weight_scale':meta('torch.float32') if fp8 else None},'postprocess_flag':fp8,'fp8_linear_kernel_class':'test.CutlassFP8ScaledMMLinearKernel' if fp8 else None,'use_marlin':False}
        layers.append(layer);component=components[name.split('.')[0]]['parameters'];component.append({'name':name+'.weight',**w})
        if fp8:component.append({'name':name+'.weight_scale',**meta('torch.float32')})
    components['vae']['parameters']=[{'name':'vae.fake.weight',**meta()}]
    return {'schema_version':2,'pid':123,'arm':arm,'omni_sha':OMNI_SHA,'linears':layers,'components':components,
            'adapter_source_sha256':hashlib.sha256((ROOT/'sitecustomize.py').read_bytes()).hexdigest(),
            'audit_protocol_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('audit_protocol.py','expected_topology.json','source_guards.json')},
            'source_guards':SOURCES,'packages':{'vllm':'0.31.0','torch':'2.13.0+cu130','transformers':'5.14.1','diffusers':'0.40.0'},
            'model_architecture':{'dit_blocks':32,'te_language_layers':36,'te_vision_blocks':27},
            'resolved_runtime':{'dtype':'torch.bfloat16','enforce_eager':True,'enable_cuda_graph_decode':False,'prefix_kv_cache_dtype':None,'diffusion_kv_cache_dtype':None}}

checks=[]
for arm in ARMS:
    inv=synthetic(arm);assert not inventory_errors(inv,arm)
    with tempfile.TemporaryDirectory() as temp:
        d=Path(temp);(d/'worker_inventory_123.json').write_text(json.dumps(inv));ident={'pid':123,'arm':arm,'omni_sha':OMNI_SHA}
        prefix={'observations':{phase:{'configured_prefix_dtype':None,'blocks':[{'block':i,'tensors':{'key':meta(),'value':meta()}} for i in range(32)]} for phase in ['cond_prefill','cond_decode']}}
        att={'blocks':{str(i):{k:meta() for k in ['query','key','value']} for i in range(32)}}
        te={'linears':{n:{'input':meta(),'output':meta()} for n in TOPOLOGY['te_language_linears']+TOPOLOGY['te_vision_linears']},'last_hidden_state':meta()}
        for name,v in [('prefix_runtime',prefix),('attention_runtime',att),('text_encoder_runtime',te)]:
            (d/f'{name}_123.json').write_text(json.dumps({**ident,**v}))
        assert verify(d,arm,'edit')['passed'],verify(d,arm,'edit')
        te['linears'].pop(TOPOLOGY['te_vision_linears'][0]);(d/'text_encoder_runtime_123.json').write_text(json.dumps({**ident,**te}))
        assert not verify(d,arm,'edit')['passed'];assert verify(d,arm,'t2i')['passed']
    checks.append(arm+': full synthetic inventory/runtime accepted; missing edit vision execution rejected')
    bad=copy.deepcopy(inv);bad['linears'][0]['weight']['device']='cpu';assert inventory_errors(bad,arm)
    bad=copy.deepcopy(inv);bad['linears'].pop();assert inventory_errors(bad,arm)
    bad=copy.deepcopy(inv);bad['model_architecture']['te_language_layers']=35;assert inventory_errors(bad,arm)
    bad=copy.deepcopy(inv);bad['resolved_runtime']['enable_cuda_graph_decode']=True;assert inventory_errors(bad,arm)
inv=synthetic('bf16')
for layer in inv['linears']:
    if layer['name']=='transformer.transformer_blocks.1.attn.to_qkv':layer['weight']=meta('torch.float8_e4m3fn')
assert inventory_errors(inv,'bf16');checks.append('BF16 baseline accidentally quantized DiT block rejected')
inv=synthetic('te_fp8');next(x for x in inv['linears'] if x['name'].startswith('text_encoder.model.visual.'))['weight']=meta('torch.float8_e4m3fn');assert inventory_errors(inv,'te_fp8');checks.append('vision accidentally quantized rejected')
local_vision=[n.removeprefix('text_encoder.') for n in TOPOLOGY['te_vision_linears']]
assert sum(fnmatch.fnmatch(n,'model.visual') for n in local_vision)==0
assert sum(fnmatch.fnmatch(n,'model.visual.*') for n in local_vision)==116
checks.append('explicit vision glob covers 116; bare prefix covers zero')
# Importing diagnostic with C1_QUANT_AUDIT absent must install no hooks.
os.environ.pop('C1_QUANT_AUDIT',None)
spec=importlib.util.spec_from_file_location('phase2_hook_test',ROOT/'sitecustomize.py');hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook);assert not hook.C1_QUANT_AUDIT_INSTALLED
checks.append('diagnostic hook inert by default')
print(json.dumps({'passed':True,'scope':'CPU synthetic guard tests only; no GPU execution or support/quality proof','checks':checks},indent=2))
