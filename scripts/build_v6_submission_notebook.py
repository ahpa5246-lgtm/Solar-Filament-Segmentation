"""Build private full-test inference using immutable final V5 checkpoints."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(out):
    names=['__init__.py','v6_audit.py','v6_cached_search.py','v5_replay.py','v6_diagnostics.py','v6_refiner_audit.py','v6_resolution_audit.py','v6_promotion.py','v6_submission.py']
    modules={f'solar/{name}':(ROOT/'solar'/name).read_text() for name in names}
    cache=json.loads((ROOT/'reports/V5_FINAL_CACHE.json').read_text())
    if cache.get('status')!='ready' or cache.get('private_verified') is not True or not cache.get('artifact_sha256'):raise RuntimeError('Fingerprint-verified private final cache is not ready')
    gate=json.loads((ROOT/'reports/V6_KAGGLE_CONFIRMATION_EXECUTION.json').read_text())
    from solar.v6_promotion import promotion_decision
    decision=promotion_decision(gate)
    if not decision['promote']:
        raise RuntimeError('Full confirmation does not support promotion: '+', '.join(decision['reasons']))
    baseline=(ROOT/'notebooks/baselines/v5_training_session_v3.ipynb').read_text()
    code="""import sys,json,subprocess
from pathlib import Path
try:import pycocotools
except ImportError:subprocess.run([sys.executable,'-m','pip','install','pycocotools==2.0.8'],check=True)
"""
    code+='\nSOURCES='+repr(modules)+'\nBASELINE='+repr(baseline)+'\nEXPECTED='+repr(cache['artifact_sha256'])+'\nGATE='+repr(gate)+"""
assert GATE.get('status')=='COMPLETE' and GATE.get('result',{}).get('confirmation_gain_supported') is True, 'Positive unused-date confirmation is required'
assert GATE['result'].get('fixed_nms_confirmation') is True and GATE['result']['confirmation_observations']>=40
assert GATE['result']['paired_macro_delta_ci']['low']>0
WORK=Path('/kaggle/working/v6-final')
PACKAGE=Path('/kaggle/working/v6-final-package')
for name,source in SOURCES.items():
    path=PACKAGE/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
source_path=PACKAGE/'archived_v5_training_v3.ipynb';source_path.write_text(BASELINE)
sys.path.insert(0,str(PACKAGE))
from solar.v6_submission import run
from solar.v6_promotion import promotion_decision
assert promotion_decision(GATE)['promote'], 'Promotion evidence changed or is insufficient'
INPUT=Path('/kaggle/input')
def one(name):
    files=list(INPUT.rglob(name))
    if len(files)!=1:raise RuntimeError(f'Expected one {name}, found {len(files)}')
    return files[0]
test_dirs=[p for p in INPUT.rglob('test_images') if p.is_dir()]
if len(test_dirs)!=1:raise RuntimeError('Expected one official test-images directory')
result=run(test_dirs[0],one('final_detector.pt').parent,EXPECTED,source_path,WORK,
           Path('/kaggle/working/submission.csv'),seconds=1000)
print('Validated full-test CSV created. No training, test labels, or official score used.')
print(json.dumps(result['submission'],indent=2))
"""
    ast.parse(code)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    cells=[{'cell_type':'markdown','metadata':{},'source':['# V6 final-checkpoint test inference\n','Private, no training. Fixed detector mask NMS after unused-date confirmation.\n','Preserved V5 all-training checkpoints and original prediction parity; V5 Public baseline remains 0.32.\n','Apache-2.0 architecture attribution: Nomannic, adapted in the archived V5 source; see repository third_party/NOTICE.md.\n','The CSV still requires official competition evaluation; no higher official score is claimed.\n']},
           {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]
    out.write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells},indent=1)+'\n')
    metadata={'id':'minahussain111/solar-filament-research-v6-submit','title':'solar-filament-research-v6-submit',
              'code_file':out.name,'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,
              'machine_shape':'NvidiaTeslaT4','enable_internet':True,'dataset_sources':[cache['dataset']],
              'competition_sources':['filament-segmentation-2026'],'kernel_sources':[],'model_sources':[]}
    (out.parent/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
