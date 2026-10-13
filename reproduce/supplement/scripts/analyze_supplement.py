#!/usr/bin/env python3
"""Summarize C1 supplement runs, without running model inference.

Usage: python scripts/analyze_supplement.py --results-root raw/results \
  --quality-manifest inputs/effective_manifest.json \
  --quality-manifest inputs/cfg4_manifest.json --out-dir analysis/supplement
Quality scoring uses full-resolution RGB and CPU AlexNet LPIPS v0.1. Pre-cache
alexnet-owt-7be5be79.pth under TORCH_HOME/hub/checkpoints, or explicitly permit
its official torchvision download with --allow-weight-download. No image resize.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import statistics

ARMS = ('bf16', 'dit_fp8', 'mlp_bf16', 'te_fp8', 'joint_fp8', 'prefix_fp8_v', 'prefix_fp8')
ALIASES = {'selective_img_mlp_bf16': 'mlp_bf16', 'prefix_auto': 'bf16'}
CASE_FIELDS = ('task','prompt','negative_prompt','seed','width','height','num_inference_steps','true_cfg_scale')
ENV_FIELDS = ('vllm_omni_sha','vllm_omni_checkout','vllm_omni_dirty','model','model_revision',
              'packages','python','torch_cuda','hostname','harness_sha256','nvidia_smi')
PEAK_LINE = re.compile(r'Peak GPU memory \(this request\): ([\d.]+) GB reserved, ([\d.]+) GB allocated')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def as_list(value):
    return value if isinstance(value, list) else [value] if value else []


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity' if value < 0 else None
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [clean(v) for v in value]
    return value


def stats(values):
    values = list(values)
    return dict(n=len(values),mean=statistics.fmean(values),stdev=statistics.stdev(values) if len(values)>1 else 0.,
                min=min(values),max=max(values),median=statistics.median(values)) if values else None


def delta(value, reference):
    return (value/reference-1)*100 if value is not None and reference not in (None,0) else None


def percent(value):
    return '—' if value is None else f'{value:+.1f}%'


def number(value, digits=3):
    return '—' if value is None else (value if isinstance(value,str) else f'{value:.{digits}f}')


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else clean(v) for k,v in r.items()})


def gpu_uuids(env):
    try:
        return tuple(r['uuid'].strip() for r in ({k.strip():v for k,v in r.items()} for r in csv.DictReader(io.StringIO(env.get('nvidia_smi','')))))
    except (KeyError,TypeError,csv.Error): return ()


def environment_comparison(a,b):
    issues=[]
    for key in ENV_FIELDS:
        if key not in a or key not in b:issues.append('unobserved:'+key)
        elif a[key]!=b[key]:issues.append('mismatch:'+key)
    if not gpu_uuids(a) or not gpu_uuids(b):issues.append('unobserved:gpu_uuid')
    elif gpu_uuids(a)!=gpu_uuids(b):issues.append('mismatch:gpu_uuid')
    for key in ('dtype','enforce_eager','enable_cuda_graph_decode','diffusion_attention_config','diffusion_kv_cache_dtype'):
        av=a.get('runtime_config',{});bv=b.get('runtime_config',{})
        if key not in av or key not in bv:issues.append('unobserved:runtime.'+key)
        elif av[key]!=bv[key]:issues.append('mismatch:runtime.'+key)
    return issues


def case_key(row):
    # Different CFG, seed or ordered reference bytes never share a denominator.
    return json.dumps([row.get(k) for k in CASE_FIELDS]+[as_list(row.get('input_image_sha256'))],ensure_ascii=False,sort_keys=True)


def read_run(directory):
    item={'directory':str(directory),'rows':[],'errors':[],'env':{},'summary':{}}
    for name in ('env','summary'):
        p=directory/(name+'.json')
        try:item[name]=read_json(p)
        except (OSError,ValueError) as e:item['errors'].append(name+': '+str(e))
    path=directory/'runs.jsonl'
    try:
        for i,line in enumerate(path.read_text().splitlines(),1):
            if not line.strip():continue
            try:
                row=json.loads(line)
                if not isinstance(row,dict):raise ValueError('row is not an object')
                item['rows'].append(row)
            except ValueError as e:item['errors'].append(f'runs.jsonl:{i}: {e}')
    except OSError as e:item['errors'].append(str(e))
    item['arm']=item['env'].get('arm') or item['summary'].get('arm')
    if not item['arm'] and item['rows']:item['arm']=item['rows'][0].get('arm')
    item['arm']=ALIASES.get(item['arm'],item['arm'])
    item['runs_sha256']=sha(path) if path.is_file() else None
    return item


def archived(path, root):
    """Failed/provisional experiments retained under archive are not new runs."""
    return any(part.lower() in {'archive', 'archives'} for part in path.relative_to(root).parts)


def discover(root):
    if not root.is_dir():raise ValueError('Results root missing: '+str(root))
    if (root/'raw/results').is_dir():root=root/'raw/results'
    elif (root/'results').is_dir():root=root/'results'
    dirs=sorted({p.parent for p in root.rglob('runs.jsonl') if not archived(p,root)
                 and p.parent.name.startswith(('perf_','mem_','quality_','audit_','trace_'))})
    return [read_run(p) for p in dirs]


def fixed_run_issues(run,kind):
    """Validate measured requests against the saved, unchanged runner protocol."""
    issues=[];env=run['env'];work=env.get('workload') or {};rows=run['rows']
    cases=env.get('cases') or []
    if len(cases)!=1:issues.append('fixed run requires exactly one env case')
    else:
        case=cases[0]
        for row in rows:
            if any(row.get(k)!=v for k,v in case.items()):issues.append('raw request differs from saved env case/input');break
    if run['arm'] not in ARMS:issues.append('unknown arm')
    if any(row.get('arm')!=run['arm'] for row in rows):issues.append('raw request arm/env mismatch')
    expected_phases=[]
    for phase,key in (('warmup','warmup'),('feasibility','feasibility'),('measured','measured')):
        count=work.get(key)
        if not isinstance(count,int) or isinstance(count,bool) or count<0:issues.append('invalid planned count: '+key)
        else:expected_phases.extend([phase]*count)
        indices=[row.get('index') for row in rows if row.get('phase')==phase]
        if indices!=list(range(len(indices))):issues.append('duplicate/nonconsecutive request index: '+phase)
    if [row.get('phase') for row in rows]!=expected_phases:issues.append('request phases/counts/order differ from planned warmup/feasibility/measured protocol')
    if any(row.get('error') for row in rows):issues.append('at least one request failed (including warmup/feasibility)')
    if any(work.get(k) is not False for k in ('trace','stages')):issues.append('timing/memory run enabled trace/stage hook or flags unobserved')
    if work.get('mem') is not (kind=='mem'):issues.append('memory diagnostic flag differs from run type')
    if any(env.get(k) is not False for k in ('worker_quant_audit_enabled','quality_manifest_mode','profiler_export_only_enabled')):
        issues.append('timing/memory run has audit/quality/profiler adapter enabled or flags unobserved')
    return issues


def performance(runs):
    results=[]
    for run in runs:
        if not Path(run['directory']).name.startswith('perf_'):continue
        measured=[r for r in run['rows'] if r.get('phase')=='measured']
        valid=[r for r in measured if not r.get('error') and isinstance(r.get('wall_s'),(int,float)) and math.isfinite(r['wall_s']) and r['wall_s']>0]
        expected=run['env'].get('workload',{}).get('measured')
        observed_keys={case_key(r) for r in measured}
        issues=list(run['errors'])+fixed_run_issues(run,'perf')
        if len(observed_keys)!=1:issues.append('expected one fixed workload in this timing run')
        if expected is None:issues.append('expected measured count unobserved')
        elif expected!=len(measured):issues.append('measured row count differs from planned count')
        if len(valid)!=len(measured):issues.append('failed or invalid measured requests')
        case=measured[0] if measured else (run['env'].get('cases') or [{}])[0]
        results.append({'directory':run['directory'],'arm':run['arm'],'case_id':case.get('case_id',case.get('id')),
                        'task':case.get('task'),'cfg':case.get('true_cfg_scale'),'expected_n':expected,
                        'observed_n':len(measured),'successful_n':len(valid),'failed_n':len(measured)-len(valid),
                        'wall_s':stats(r['wall_s'] for r in valid),'stage_metrics':stage_metrics(valid),
                        'issues':issues,'baseline_comparison':None,'delta_wall_pct':None,
                        '_key':case_key(case),'_env':run['env']})
    add_comparisons(results,['wall_s'])
    return results


def stage_metrics(rows):
    """The historical stage_durations_s container mixes seconds and *_ms values."""
    result={}
    for name in sorted({k for r in rows for k in (r.get('stage_durations_s') or {})}):
        values=[(r.get('stage_durations_s') or {}).get(name) for r in rows]
        valid=[v for v in values if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0]
        unit='ms' if name.endswith('_ms') else 's' if name.startswith('QwenImage21Pipeline.') or name.endswith('_s') else 'unverified'
        result[name]={'source_unit':unit,'raw_values':stats(valid),
                      'seconds':stats(v/1000 if unit=='ms' else v for v in valid) if unit!='unverified' else None,
                      'missing_or_invalid_n':len(values)-len(valid)}
    return result


def memory(runs):
    results=[]
    for run in runs:
        directory=Path(run['directory'])
        if not directory.name.startswith('mem_'):continue
        issues=list(run['errors'])+fixed_run_issues(run,'mem');peak_source=directory/'engine_peak_memory.json'
        try:
            if peak_source.exists():peaks=read_json(peak_source)
            else:
                peak_source=directory/'engine.log'
                peaks=[dict(reserved_gib=float(m[0]),allocated_gib=float(m[1])) for m in PEAK_LINE.findall(peak_source.read_text())]
            if not isinstance(peaks,list):raise ValueError('peak-memory JSON is not a list')
        except (OSError,ValueError) as e:peaks=[];issues.append(str(e))
        rows=run['rows'];measured=[r for r in rows if r.get('phase')=='measured'];valid=[]
        if len(peaks)!=len(rows):issues.append('peak/row count mismatch; refusing ambiguous alignment')
        else:
            valid=[p for r,p in zip(rows,peaks) if r.get('phase')=='measured' and not r.get('error')]
        if any(r.get('error') for r in measured):issues.append('failed measured request')
        expected=run['env'].get('workload',{}).get('measured')
        if expected!=len(measured) or not measured:issues.append('measured count absent or incomplete')
        if len({case_key(r) for r in measured})!=1:issues.append('expected one fixed workload in memory run')
        for p in valid:
            if not isinstance(p,dict) or any(not isinstance(p.get(k),(int,float)) or isinstance(p[k],bool) or not math.isfinite(p[k]) or p[k]<=0 for k in ('allocated_gib','reserved_gib')):
                issues.append('invalid worker peak values');valid=[];break
            if p['allocated_gib']>p['reserved_gib']:
                issues.append('allocated worker peak exceeds reserved peak');valid=[];break
        case=measured[0] if measured else (run['env'].get('cases') or [{}])[0]
        results.append({'directory':str(directory),'arm':run['arm'],'case_id':case.get('case_id',case.get('id')),
            'task':case.get('task'),'cfg':case.get('true_cfg_scale'),'expected_n':expected,'observed_n':len(measured),'peak_n':len(valid),
            'allocated_gib':max((p['allocated_gib'] for p in valid),default=None),
            'reserved_gib':max((p['reserved_gib'] for p in valid),default=None),
            'peak_source':str(peak_source),'issues':issues,'baseline_comparison':None,
            'delta_allocated_pct':None,'delta_reserved_pct':None,'_key':case_key(case),'_env':run['env']})
    add_comparisons(results,['allocated_gib','reserved_gib'])
    return results


def add_comparisons(rows,metrics):
    for row in rows:
        matches=[b for b in rows if b['arm']=='bf16' and b['_key']==row['_key']]
        if len(matches)!=1:row['baseline_comparison']={'status':'missing_or_duplicate_bf16','matches':len(matches)};continue
        base=matches[0];issues=environment_comparison(base['_env'],row['_env'])
        if base['issues'] or row['issues']:issues.append('incomplete_or_invalid_run')
        row['baseline_comparison']={'status':'matched' if not issues else 'unverified','directory':base['directory'],'issues':issues}
        if not issues and row['arm']!='bf16':
            for metric in metrics:
                if metric=='wall_s':row['delta_wall_pct']=delta((row[metric] or {}).get('mean'),(base[metric] or {}).get('mean'))
                else:row['delta_'+metric.replace('_gib','')+'_pct']=delta(row[metric],base[metric])
    for row in rows:row.pop('_env',None);row.pop('_key',None)


def category(case):
    if case['task']=='edit':
        group=case.get('group','')
        return 'Edit: person' if 'person' in group else 'Edit: product' if any(x in group for x in ('product','one_ref')) else 'Edit: text' if 'sign' in group else 'Edit: multi-reference' if any(x in group for x in ('two_ref','multi')) else 'Edit: bear'
    text=case.get('expected_text')
    if text:
        text=' '.join(as_list(text))
        return 'Chinese text' if any(ord(c)>127 for c in text) else 'English text'
    return 'Ordinary T2I'


class Scorer:
    def __init__(self,skip_lpips=False,allow_download=False,threads=8):
        self.skip_lpips=skip_lpips;self.allow_download=allow_download;self.threads=threads
        self.model=None;self.lpips_error=None;self.details={};self.health_cache={}
    def lpips_init(self):
        if self.model is not None or self.lpips_error or self.skip_lpips:return
        try:
            import torch,lpips
            weight=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
            if not weight.is_file() and not self.allow_download:
                raise FileNotFoundError('AlexNet weights are not cached; supply TORCH_HOME or --allow-weight-download')
            torch.set_num_threads(self.threads)
            try:torch.set_num_interop_threads(1)
            except RuntimeError:pass  # Already initialized by a caller; record the actual value below.
            torch.manual_seed(0);torch.use_deterministic_algorithms(True)
            self.model=lpips.LPIPS(net='alex',version='0.1',lpips=True,pretrained=True,pnet_rand=False,spatial=False,verbose=False).cpu().eval()
            calibration=Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
            self.details={'weights':{p.name:sha(p) for p in (weight,calibration)},'device':'cpu','network':'alex','version':'0.1',
                'torch_intraop_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
                'packages':{p:importlib.metadata.version(p) for p in ('torch','torchvision','lpips','numpy','Pillow','scikit-image')}}
        except Exception as e:self.lpips_error=f'{type(e).__name__}: {e}'
    def score(self,a_path,b_path):
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity
        with Image.open(a_path) as im:a=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
        with Image.open(b_path) as im:b=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
        self.health_cache[str(a_path.resolve())]=array_health(a)
        self.health_cache[str(b_path.resolve())]=array_health(b)
        if a.shape!=b.shape:raise ValueError('paired image shape mismatch')
        mse=float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
        result={'psnr_db':float('inf') if mse==0 else 10*math.log10(255**2/mse),
            'ssim':float(structural_similarity(a,b,data_range=255,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11,channel_axis=-1)),
            'lpips':None,'rgb_equal':mse==0,'mse':mse,'bf16_black':bool(not np.any(a)),'candidate_black':bool(not np.any(b)),
            'bf16_rgb_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'candidate_rgb_sha256':hashlib.sha256(b.tobytes()).hexdigest(),
            'size':[a.shape[1],a.shape[0]],'lpips_error':None}
        self.lpips_init()
        if self.model is not None:
            import torch
            ta=torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/127.5-1
            tb=torch.from_numpy(b).permute(2,0,1).unsqueeze(0).float()/127.5-1
            with torch.inference_mode():value=float(self.model(ta,tb).item())
            if not math.isfinite(value) or value<0:raise ValueError('invalid LPIPS value')
            result['lpips']=value
        else:result['lpips_error']=self.lpips_error or 'skipped by --skip-lpips'
        return result


def array_health(pixels):
    """Cheap spatial checks on already decoded RGB; do not infer semantic quality."""
    import numpy as np
    low=pixels.min(axis=(0,1));high=pixels.max(axis=(0,1))
    return {'decoded_size':[int(pixels.shape[1]),int(pixels.shape[0])],
            'rgb_min':low.tolist(),'rgb_max':high.tolist(),
            'rgb_mean':np.mean(pixels,axis=(0,1)).tolist(),
            'spatially_constant_rgb':bool(np.all(low==high)),
            'exactly_black':bool(np.all(high==0)),'exactly_white':bool(np.all(low==255))}


def image_health(runs,cache):
    output=[]
    for run in runs:
        directory=Path(run['directory']);referenced=set()
        records=[]
        for row in run['rows']:
            filename=(str(row.get('case_id',row.get('id')))+'.png') if row.get('phase')=='quality' else f"{row.get('phase')}_{row.get('index')}.png"
            path=directory/'images'/filename;referenced.add(path)
            records.append((path,row))
        records.extend((path,{}) for path in sorted((directory/'images').glob('*.png')) if path not in referenced)
        for path,row in records:
            item={'directory':str(directory),'path':str(path),'arm':run['arm'],
                  'case_id':row.get('case_id',row.get('id')),'phase':row.get('phase'),
                  'request_error':row.get('error'),'expected_size':[row.get('width'),row.get('height')],
                  'status':'ok','issues':[]}
            if not row:item['issues'].append('image has no associated request row')
            if not path.is_file():item.update(status='missing_image');item['issues'].append('planned output image missing')
            else:
                try:
                    key=str(path.resolve())
                    if key in cache:info=cache[key]
                    else:
                        from PIL import Image,ImageStat
                        with Image.open(path) as im:
                            rgb=im.convert('RGB');extrema=rgb.getextrema()
                            info={'decoded_size':list(rgb.size),'rgb_min':[a for a,b in extrema],
                                  'rgb_max':[b for a,b in extrema],'rgb_mean':ImageStat.Stat(rgb).mean,
                                  'spatially_constant_rgb':all(a==b for a,b in extrema),
                                  'exactly_black':all(b==0 for a,b in extrema),
                                  'exactly_white':all(a==255 for a,b in extrema)}
                        cache[key]=info
                    item.update(info)
                    if row and info['decoded_size']!=item['expected_size']:
                        item['status']='dimension_mismatch';item['issues'].append('decoded dimensions differ from request')
                    item['constant_image_flag']=info['spatially_constant_rgb']
                except Exception as e:item.update(status='decode_error');item['issues'].append(f'{type(e).__name__}: {e}')
            output.append(item)
    return output


def quality_index(runs):
    result=defaultdict(list)
    for run in runs:
        if not Path(run['directory']).name.startswith('quality_'):continue
        for row in run['rows']:
            if row.get('phase')=='quality':result[(run['arm'],row.get('case_id',row.get('id')))].append((run,row))
    return result


def inspect_quality(case,arm,index):
    matches=index.get((arm,case['id']),[])
    if len(matches)!=1:return {'status':'missing_record' if not matches else 'duplicate_record','matches':len(matches)},None,None
    run,row=matches[0];item={'status':'ok','directory':run['directory'],'issues':[],'inference_error':row.get('error')}
    if row.get('error'):item['status']='inference_error'
    for key in CASE_FIELDS:
        if row.get(key)!=case.get(key):item['issues'].append('metadata:'+key)
    expected=case.get('input_images_sha256',case.get('input_image_sha256'))
    if as_list(row.get('input_image_sha256'))!=as_list(expected):item['issues'].append('metadata:input_image_sha256')
    if row.get('arm')!=arm:item['issues'].append('metadata:arm')
    image=Path(run['directory'])/'images'/(case['id']+'.png')
    if not image.is_file():item['issues'].append('image_missing')
    elif sha(image)!=row.get('image_sha256'):item['issues'].append('image_hash_mismatch')
    else:item['png_sha256']=sha(image)
    if item['issues'] and item['status']=='ok':item['status']='invalid_record'
    return item,image,run


def summarize_pairs(pairs,threshold):
    groups=defaultdict(list)
    for p in pairs:groups[('all',p['candidate'])].append(p);groups[(p['category'],p['candidate'])].append(p)
    output=[]
    for (cat,arm),rows in sorted(groups.items()):
        paired=[r for r in rows if r['status']=='paired'];scored=[r for r in paired if isinstance(r.get('lpips'),(float,int))]
        good=[r for r in scored if r['lpips_gate_pass'] is True]
        failed=[r for r in scored if r['lpips_gate_pass'] is False]
        output.append({'category':cat,'candidate':arm,'expected_n':len(rows),'paired_n':len(paired),'lpips_scored_n':len(scored),
            'lpips_pass_n':len(good),'lpips_fail_n':len(failed),'unassessed_n':len(rows)-len(scored),
            'pair_status_counts':dict(Counter(r['status'] for r in rows)),
            'mean_psnr_db':statistics.fmean(r['psnr_db'] for r in paired) if paired else None,
            'mean_ssim':statistics.fmean(r['ssim'] for r in paired) if paired else None,
            'mean_lpips':statistics.fmean(r['lpips'] for r in scored) if scored else None,
            'max_lpips':max((r['lpips'] for r in scored),default=None),
            'worst_case':max(scored,key=lambda r:r['lpips'])['case_id'] if scored else None,
            'failure_ids':[r['case_id'] for r in failed],'lpips_threshold':threshold})
    return output


def quality(manifest_path,runs,scorer,threshold):
    manifest=read_json(manifest_path);cases=manifest['cases']
    if len({c['id'] for c in cases})!=len(cases):raise ValueError('Duplicate manifest case IDs')
    if manifest.get('expected_cases',len(cases))!=len(cases):raise ValueError('Manifest expected case count mismatch')
    arms=[ALIASES.get(a,a) for a in manifest['expected_arms'] if ALIASES.get(a,a)!='bf16']
    index=quality_index(runs);pairs=[]
    for arm in arms:
        for case in cases:
            bi,bp,br=inspect_quality(case,'bf16',index);ci,cp,cr=inspect_quality(case,arm,index)
            row={'suite':manifest_path.stem,'candidate':arm,'case_id':case['id'],'category':category(case),
                'group':case.get('group'),'task':case['task'],'seed':case['seed'],'cfg':case['true_cfg_scale'],
                'expected_text':case.get('expected_text'),'targets':case.get('targets'),
                'bf16':bi,'candidate_record':ci,'status':'unpaired','psnr_db':None,'ssim':None,'lpips':None,'lpips_gate_pass':None,
                'review_status':'unreviewed; no human acceptance recorded'}
            if bi['status']!='ok':row['status']='baseline_'+bi['status']
            elif ci['status']!='ok':row['status']='candidate_'+ci['status']
            else:
                env_issues=environment_comparison(br['env'],cr['env'])
                if env_issues:row.update(status='environment_unverified',environment_issues=env_issues)
                else:
                    try:
                        row.update(scorer.score(bp,cp))
                        if row['size']!=[case['width'],case['height']]:raise ValueError('PNG dimensions differ from manifest')
                        row['status']='paired'
                        if row['lpips'] is not None:row['lpips_gate_pass']=row['lpips']<=threshold
                    except Exception as e:row.update(status='scoring_error',scoring_error=f'{type(e).__name__}: {e}')
            pairs.append(row)
            print(f"{manifest_path.stem} {arm} {case['id']}: {row['status']} LPIPS={row.get('lpips')}",flush=True)
    return {'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),'scope':manifest.get('scope'),
            'gate':{'metric':'LPIPS','max':threshold,'status':'predeclared working screen; maintainer acceptance unconfirmed','text_subject_edit_review':'separate; model assistance is not human review'},
            'cases_per_candidate':len(cases),'pairs':pairs,'summary':summarize_pairs(pairs,threshold)}


def request_marker_errors(data,path,env):
    """Tie observed tensors to an immutable post-init, real-request marker."""
    errors=[];marker=data.get('request_start')
    if not isinstance(marker,dict):return ['real request_start marker missing']
    case=marker.get('case');expected_cases=env.get('cases') or []
    if marker.get('schema_version')!=1:errors.append('request marker schema mismatch')
    if marker.get('engine_initialization_complete') is not True:errors.append('request marker does not follow completed engine initialization')
    if marker.get('arm')!=env.get('arm'):errors.append('request marker arm/env mismatch')
    if len(expected_cases)!=1 or case!=expected_cases[0]:errors.append('request marker case/input fingerprint payload differs from env.cases')
    if not isinstance(case,dict):errors.append('request marker has no case object')
    else:
        canonical=json.dumps(case,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        if hashlib.sha256(canonical).hexdigest()!=marker.get('case_fingerprint_sha256'):
            errors.append('request marker case fingerprint hash mismatch')
        if marker.get('case_id')!=case.get('id'):errors.append('request marker case_id mismatch')
    if marker.get('case_file_sha256')!=env.get('case_file_sha256'):errors.append('request marker case file hash/env mismatch')
    if marker.get('harness_sha256')!=env.get('harness_sha256'):errors.append('request marker harness hash/env mismatch')
    ordinal=marker.get('request_ordinal')
    if not isinstance(ordinal,int) or isinstance(ordinal,bool) or ordinal<1:errors.append('invalid request ordinal')
    if marker.get('request_phase') not in {'warmup','feasibility','measured'}:errors.append('unexpected audit request phase')
    filename=data.get('request_start_file')
    if not isinstance(filename,str) or not re.fullmatch(r'request_start_[0-9]{4,}\.json',filename):
        errors.append('immutable request marker filename missing/invalid')
    else:
        marker_path=path.parent/filename
        try:
            if sha(marker_path)!=data.get('request_start_sha256'):errors.append('immutable request marker byte hash mismatch')
            if read_json(marker_path)!=marker:errors.append('immutable request marker contents mismatch')
        except (OSError,ValueError) as e:errors.append('immutable request marker unreadable: '+str(e))
    try:
        request_rows=[json.loads(line) for line in (path.parent.parent/'runs.jsonl').read_text().splitlines() if line.strip()]
        if not isinstance(ordinal,int) or ordinal<1 or ordinal>len(request_rows):
            errors.append('marked request has no completed runs.jsonl row')
        else:
            row=request_rows[ordinal-1]
            if row.get('error'):errors.append('marked request failed')
            if row.get('phase')!=marker.get('request_phase') or row.get('arm')!=marker.get('arm'):
                errors.append('marked request phase/arm differs from completed row')
            if not isinstance(case,dict) or any(row.get(k)!=v for k,v in case.items()):
                errors.append('marked request case/input differs from completed row')
    except (OSError,ValueError) as e:errors.append('marked request rows unreadable: '+str(e))
    counts=data.get('transformer_call_counts_at_capture') or {}
    if not isinstance(counts.get('first_real_request'),int) or counts['first_real_request']<1:
        errors.append('no transformer call attributed to first real request')
    return errors


def audit_cache(root):
    rows=[]
    for path in sorted(root.rglob('prefix_runtime_*.json')):
        if archived(path,root):continue
        data=read_json(path);directory=path.parent.parent
        env=read_json(directory/'env.json') if (directory/'env.json').is_file() else {}
        case=(env.get('cases') or [{}])[0]
        verification=read_json(path.parent/'verification.json') if (path.parent/'verification.json').is_file() else {}
        for name,obs in data.get('observations',{}).items():
            blocks=obs.get('blocks',[])
            tensors=blocks[0].get('tensors',{}) if blocks else {}
            keyshape=tensors.get('key',{}).get('shape',[])
            issues=request_marker_errors(data,path,env)
            if data.get('schema_version')!=4:issues.append('prefix runtime schema is not 4')
            if data.get('arm')!=env.get('arm'):issues.append('prefix runtime arm/env mismatch')
            if data.get('omni_sha')!=env.get('vllm_omni_sha'):issues.append('prefix runtime source/env mismatch')
            if verification.get('passed') is not True:issues.append('worker audit verification absent or not passed')
            if verification.get('schema_version')!=4:issues.append('worker audit verification schema is not 4')
            if len(blocks)!=32 or len(keyshape)!=4:issues.append('incomplete block/BSHD cache metadata')
            if not isinstance(obs.get('cache_bytes',{}).get('logical_total'),int):issues.append('logical cache byte total missing')
            rows.append({'directory':str(directory),'arm':env.get('arm'),'case_id':case.get('id'),
                'observation':name,'phase':obs.get('phase'),'branch':obs.get('branch'),
                'schema_version':data.get('schema_version'),'audit_passed':verification.get('passed'),
                'request_phase':data.get('request_start',{}).get('request_phase'),
                'request_ordinal':data.get('request_start',{}).get('request_ordinal'),
                'case_fingerprint_sha256':data.get('request_start',{}).get('case_fingerprint_sha256'),
                'request_start_file':data.get('request_start_file'),
                'request_start_sha256':data.get('request_start_sha256'),
                'transformer_call_counts_at_capture':data.get('transformer_call_counts_at_capture'),
                'configured_prefix_dtype':obs.get('configured_prefix_dtype'),**obs.get('cache_bytes',{}),
                'block_count':len(blocks),'prefix_tokens_per_sample':keyshape[1] if len(keyshape)==4 else None,
                'key_dtype':tensors.get('key',{}).get('dtype'),'value_dtype':tensors.get('value',{}).get('dtype'),
                'scale_logical_bytes':sum(v for k,v in obs.get('cache_bytes',{}).get('logical_by_name',{}).items() if k.endswith('_scale')),
                'first_block_tensors':tensors,'source':str(path),'issues':issues,'baseline_comparison':None,
                'delta_logical_total_pct':None,
                '_env':env,'_key':case_key(case)+'|'+str(obs.get('phase'))+'|'+str(obs.get('branch'))})
    add_comparisons(rows,['logical_total'])
    return rows


def markdown(result):
    lines=['# C1 supplement results','','## Latency','','| Case | Configuration | n | Mean ± SD (s) | Δ vs BF16 |','|---|---|---:|---:|---:|']
    for r in result['performance']:
        s=r['wall_s'];v=f"{s['mean']:.3f} ± {s['stdev']:.3f}" if s else '—'
        lines.append(f"| {r['case_id']} | {r['arm']} | {r['successful_n']}/{r['expected_n']} | {v} | {percent(r['delta_wall_pct'])} |")
    lines+=['','## Memory','','| Case | Configuration | Allocated (GiB) | Δ vs BF16 | Reserved (GiB) | Δ vs BF16 |','|---|---|---:|---:|---:|---:|']
    for r in result['memory']:lines.append(f"| {r['case_id']} | {r['arm']} | {number(r['allocated_gib'],2)} | {percent(r['delta_allocated_pct'])} | {number(r['reserved_gib'],2)} | {percent(r['delta_reserved_pct'])} |")
    cache=[r for r in result['cache_observations'] if r['phase']=='decode']
    if cache:
        lines+=['','## Prefix storage','','| Case | Configuration | Branch | Prefix tokens | K / V dtype | KV + scale MiB | Δ vs BF16 |','|---|---|---|---:|---|---:|---:|']
        for r in cache:
            size=r.get('logical_total');size=size/1024**2 if isinstance(size,(int,float)) else None
            lines.append(f"| {r['case_id']} | {r['arm']} | {r['branch']} | {r['prefix_tokens_per_sample']} | {r['key_dtype']} / {r['value_dtype']} | {number(size,2)} | {percent(r['delta_logical_total_pct'])} |")
    for suite in result['quality_suites']:
        lines+=['',f"## Image Quality: {Path(suite['manifest']).stem}",'','| Category | Configuration | PSNR (dB) | SSIM | LPIPS | Pass / planned | Unassessed |','|---|---|---:|---:|---:|---:|---:|']
        for r in suite['summary']:lines.append(f"| {r['category']} | {r['candidate']} | {number(r['mean_psnr_db'],2)} | {number(r['mean_ssim'],4)} | {number(r['mean_lpips'],4)} | {r['lpips_pass_n']}/{r['expected_n']} | {r['unassessed_n']} |")
    lines+=['','LPIPS ≤ 0.15 is the predeclared working screen; maintainer acceptance and requested-text/subject/edit review are recorded separately.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--quality-manifest',type=Path,action='append',default=[])
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--skip-lpips',action='store_true')
    p.add_argument('--allow-weight-download',action='store_true')
    p.add_argument('--torch-home',type=Path)
    p.add_argument('--threads',type=int,default=8)
    args=p.parse_args()
    if args.threads<1:raise SystemExit('--threads must be positive')
    if args.out_dir.exists():raise SystemExit('Use a fresh output directory; preserve prior analysis.')
    if args.torch_home:os.environ['TORCH_HOME']=str(args.torch_home)
    # Set before NumPy/PyTorch imports; avoid oversubscribing the 20-vCPU Pod.
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]=str(args.threads)
    runs=discover(args.results_root)
    if not runs:raise SystemExit('No recognized run directories found')
    scorer=Scorer(args.skip_lpips,args.allow_weight_download,args.threads)
    result={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'results_root':str(args.results_root.resolve()),
        'script_sha256':sha(__file__),'performance':performance(runs),'memory':memory(runs),
        'quality_suites':[quality(path,runs,scorer,.15) for path in args.quality_manifest],
        'cache_observations':audit_cache(args.results_root),'scoring':scorer.details,'lpips_error':scorer.lpips_error,
        'run_errors':[{'directory':r['directory'],'errors':r['errors']} for r in runs if r['errors']]}
    result['image_health']=image_health(runs,scorer.health_cache)
    result['image_health_summary']={'planned_and_extra_images':len(result['image_health']),
        'status_counts':dict(Counter(r['status'] for r in result['image_health'])),
        'constant_rgb_images':sum(r.get('spatially_constant_rgb',False) for r in result['image_health']),
        'exactly_black_images':sum(r.get('exactly_black',False) for r in result['image_health']),
        'exactly_white_images':sum(r.get('exactly_white',False) for r in result['image_health'])}
    result['analysis_status'] = 'incomplete' if (
        result['run_errors'] or any(r['issues'] for r in result['performance']+result['memory']+result['cache_observations'])
        or any(r['unassessed_n'] for s in result['quality_suites'] for r in s['summary'])
        or any(r['status']!='ok' for r in result['image_health'])
    ) else 'complete_for_supplied_manifests_and_discovered_runs'
    args.out_dir.mkdir(parents=True)
    (args.out_dir/'analysis.json').write_text(json.dumps(clean(result),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (args.out_dir/'TABLES.md').write_text(markdown(result))
    for key in ('performance','memory','cache_observations','image_health'):write_csv(args.out_dir/(key+'.csv'),result[key])
    pairs=[r for s in result['quality_suites'] for r in s['pairs']]
    write_csv(args.out_dir/'quality_pairs.csv',pairs)
    write_csv(args.out_dir/'quality_gate.csv',[{'suite':Path(s['manifest']).stem,**r} for s in result['quality_suites'] for r in s['summary']])
    write_csv(args.out_dir/'visual_review.csv',[{k:r.get(k) for k in ('suite','candidate','case_id','category','expected_text','targets','review_status')}|{'reviewer_type':'','requested_text':'','subject_preservation':'','edit_completed':'','baseline_issue':'','candidate_regression':'','notes':''} for r in pairs])
    print(json.dumps({'output':str(args.out_dir),'runs':len(runs),'quality_pairs':len(pairs),
                      'analysis_status':result['analysis_status'],'lpips_error':scorer.lpips_error},indent=2))


if __name__=='__main__':main()
