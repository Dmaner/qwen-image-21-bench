from pathlib import Path
import hashlib,json,shutil
root=Path('/workspace/c1-phase2');out=root/'supplementary/profiler-kernel-source';out.mkdir(parents=True,exist_ok=True)
sources=sorted(p for d in Path('/tmp').glob('torchinductor_*') if d.is_dir() for p in d.rglob('*.py') if p.stat().st_size<10*1024*1024)
import importlib.util
spec=importlib.util.find_spec('vllm.model_executor.layers.quantization.input_quant_fp8')
if spec and spec.origin: sources.append(Path(spec.origin))
rows=[]
for n,p in enumerate(sources):
 dst=out/f'{n:04d}_{p.name}';shutil.copy2(p,dst)
 rows.append({'source_path':str(p),'copied_path':str(dst),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'bytes':dst.stat().st_size})
(out/'manifest.json').write_text(json.dumps({'files':rows,'scope':'this Pod installed source and generated Python; never imported for analysis'},indent=2)+'\n')
print(json.dumps({'files':len(rows),'bytes':sum(x['bytes'] for x in rows)}))
