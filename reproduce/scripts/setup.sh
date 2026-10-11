#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/c1-phase2/results /workspace/c1-phase2/hf
ln -sf /usr/bin/python3 /usr/bin/python
export HF_HOME=/workspace/c1-phase2/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# Pin the same source as A0; no optimization patches.
git init /opt/vllm-omni
git -C /opt/vllm-omni remote add origin https://github.com/vllm-project/vllm-omni.git
git -C /opt/vllm-omni fetch --depth 1 origin 3dc35694b3d458fc1461a8fe71832ac12e8c8cea
git -C /opt/vllm-omni checkout --detach FETCH_HEAD
cd /opt/vllm-omni
uv pip install --python "$(which python3)" --no-cache-dir -c /workspace/c1-phase2/scripts/constraints.txt .

# Remove two unused beta OTEL remnants shipped in the pinned base image.
uv pip uninstall --python "$(which python3)" opentelemetry-exporter-otlp-common opentelemetry-exporter-http-transport
uv pip install --python "$(which python3)" --no-cache-dir ijson==3.4.0.post0 scikit-image==0.25.2
uv pip check --python "$(which python3)" | tee /workspace/c1-phase2/results/pip-check.txt

# Verify the workload dependencies before downloading the fixed model.
python3 - <<'PY'
import importlib.metadata as m, json, subprocess, sys
expected={'vllm':'0.31.0','torch':'2.13.0+cu130','triton':'3.7.1','transformers':'5.14.1','diffusers':'0.40.0'}
actual={p:m.version(p) for p in expected}
assert actual==expected, (actual,expected)
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras.qwen_image_21 import build_image_to_image_prompt
env={'packages':actual,'python':sys.version,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'git_dirty':subprocess.check_output(['git','status','--porcelain'],text=True),'pip_freeze':subprocess.check_output(['uv','pip','freeze','--python',sys.executable],text=True),'nvidia_smi':subprocess.check_output(['nvidia-smi','-q'],text=True),'lscpu':subprocess.check_output(['lscpu'],text=True)}
open('/workspace/c1-phase2/results/setup-env.json','w').write(json.dumps(env,indent=2))
print(json.dumps(actual),flush=True)
PY

# Download the exact public checkpoint from A0.
python3 - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import json,time
t=time.perf_counter()
p=snapshot_download('Qwen/Qwen-Image-2.1',revision='d26bb61231c349cf6b7896fa83353113880e1ba3',max_workers=8)
files=[{'path':str(f.relative_to(p)),'bytes':f.stat().st_size} for f in Path(p).rglob('*') if f.is_file()]
r={'model':'Qwen/Qwen-Image-2.1','revision':'d26bb61231c349cf6b7896fa83353113880e1ba3','snapshot':p,'seconds':time.perf_counter()-t,'files':files,'bytes':sum(x['bytes'] for x in files)}
refs=Path(p).parent.parent/'refs'; refs.mkdir(exist_ok=True); (refs/'main').write_text(r['revision'])
Path('/workspace/c1-phase2/results/model-download.json').write_text(json.dumps(r,indent=2))
print(json.dumps({k:v for k,v in r.items() if k!='files'}),flush=True)
PY
touch /workspace/c1-phase2/setup.ok
