"""Build a self-contained, private, audit-first V6 notebook from local modules."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODULES=['__init__.py','data.py','v6_audit.py','v6_instances.py','v6_forensics.py','v6_detector.py','v6_train.py']


def cell(kind,text):
    result={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
    if kind=='code': result.update(execution_count=None,outputs=[])
    return result


def build():
    sources={f'solar/{name}':(ROOT/'solar'/name).read_text() for name in MODULES}
    hashes={name:hashlib.sha256(code.encode()).hexdigest() for name,code in sources.items()}
    cells=[cell('markdown','''# V6: audit first, bounded detector experiment

V5 Version 5 remains the user-confirmed best official Public PQ **0.32**.
This notebook has no measured solar validation or official score yet.
It embeds repository modules rather than downloading moving GitHub main.
Default execution performs synthetic readiness and artifact inventory, **no training**.
No V5 checkpoint or submission is overwritten. Before training, inspect the current
competition rules and official self-evaluation notebook, archive the latter and
set its path below. It reports public-reference all-pairs PQ and one-to-one local PQ separately; neither is a leaderboard result.
V5 forensics require its original fitting/validation manifests and exported
per-strategy RLE predictions; a new random split is invalid for an old checkpoint.

Repository code MIT; torchvision BSD. V5's Apache reference remains attributed
in the unchanged V5 notebook and third_party/NOTICE.md. No third-party source
notebook or competition data is redistributed here.
'''),cell('code','''import os, sys, json, subprocess
from pathlib import Path
WORK = Path('/kaggle/working/v6') if Path('/kaggle/working').exists() else Path.cwd()/'v6-work'
WORK.mkdir(parents=True, exist_ok=True)
# Keep installed compatible torch/torchvision together; do not replace CUDA wheels blindly.
if Path('/kaggle/working').exists():
    subprocess.run([sys.executable,'-m','pip','install','numpy==1.26.4',
                    'opencv-python-headless==4.10.0.84','scipy==1.13.1','pycocotools==2.0.8'],check=True)
import torch, torchvision
from torchvision.ops import nms
assert nms(torch.tensor([[0.,0.,1.,1.]]),torch.tensor([1.]),.5).numel()==1
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
(WORK/'environment.txt').write_text(freeze)
print('torch',torch.__version__,'torchvision',torchvision.__version__,'GPUs',torch.cuda.device_count())
'''),cell('code','sources = '+repr(sources)+'\nsource_hashes = '+repr(hashes)+'''
PACKAGE = WORK/'package'
for name,code in sources.items():
    path=PACKAGE/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=code:
        raise RuntimeError(f'Refusing to overwrite changed module {path}; use a fresh work directory')
    path.write_text(code)
sys.path.insert(0,str(PACKAGE))
(WORK/'source-hashes.json').write_text(json.dumps(source_hashes,indent=2))
'''),cell('code','''import numpy as np
from solar.v6_audit import measure
from solar.v6_instances import mask_nms
synthetic=np.zeros((2,32,32),bool)
synthetic[0,2:15,2:15]=1; synthetic[1,10:25,10:25]=1
assert measure(synthetic,synthetic)['pq_local']==1
assert mask_nms(np.concatenate([synthetic,synthetic[:1]]),[.9,.8,.7])==[0,1]
# These are software checks, NOT solar validation or official metrics.
print('Synthetic readiness passed; no research accuracy measured.')
INPUT=Path('/kaggle/input')
interesting=[str(p) for p in INPUT.rglob('*') if p.is_file() and p.suffix in {'.pt','.pth','.json','.csv','.ipynb'}] if INPUT.exists() else []
(WORK/'input-inventory.json').write_text(json.dumps(interesting,indent=2))
print('Input artifact count:',len(interesting))
'''),cell('code','''# Optional V5 audit: set all four paths to real, original artifacts.
DATA = '/kaggle/input'
V5_PREDICTIONS = None  # JSON: {strategy: {stem: [{size:[2048,2048],counts:str}]}}
V5_FITTING_STEMS = None  # JSON list, union of detector AND refiner fitting stems
V5_VALIDATION_STEMS = None  # JSON list of original held-out physical stems
if all([V5_PREDICTIONS,V5_FITTING_STEMS,V5_VALIDATION_STEMS]):
    subprocess.run([sys.executable,'-m','solar.v6_forensics','--data',DATA,
                    '--predictions',V5_PREDICTIONS,'--fitting-stems',V5_FITTING_STEMS,
                    '--validation-stems',V5_VALIDATION_STEMS,'--out',str(WORK/'v5-audit')],
                   cwd=PACKAGE,check=True)
else:
    print('V5 forensics pending original predictions and fitting/validation manifests; no failure-mode claim.')
'''),cell('code','''RUN_TRAINING = False
OFFICIAL_EVALUATION = None  # archived, inspected official evaluator notebook; never test labels
if RUN_TRAINING:
    assert OFFICIAL_EVALUATION and Path(OFFICIAL_EVALUATION).is_file(), 'Inspect and archive official evaluation first'
    assert torch.cuda.is_available(), 'Enable GPU before explicit bounded run'
    subprocess.run([sys.executable,'-m','solar.v6_train','--data',DATA,
                    '--out',str(WORK/'detector-pilot'),'--official-evaluation',OFFICIAL_EVALUATION,
                    '--run-training','--epochs','2','--max-steps','100','--resolution','1024'],
                   cwd=PACKAGE,check=True)
else:
    print('No GPU training requested; V5 remains best confirmed Public PQ 0.32.')
''')]
    for c in cells:
        if c['cell_type']=='code': ast.parse(''.join(c['source']))
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(build(),indent=1)+'\n')
    metadata={'id':'minahussain111/solar-filament-research-v6','title':'solar-filament-research-v6',
              'code_file':args.output.name,'language':'python','kernel_type':'notebook',
              'is_private':True,'enable_gpu':False,'enable_internet':True,
              'dataset_sources':[],'competition_sources':['filament-segmentation-2026'],
              'kernel_sources':['minahussain111/solar-filament-research-v5'],'model_sources':[]}
    (args.output.parent/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__': main()
