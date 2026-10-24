"""Package one private, inference-only V5 refiner replay for Kaggle T4."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(out,experiment="refiner"):
    if experiment not in {"refiner","resolution","confirmation"}:raise ValueError("Unknown inference experiment")
    modules={f'solar/{name}':(ROOT/'solar'/name).read_text() for name in ['__init__.py','v6_audit.py','v6_cached_search.py','v6_diagnostics.py','v5_replay.py','v6_refiner_audit.py','v6_resolution_audit.py']}
    baseline=(ROOT/'notebooks/baselines/v5_training_session_v3.ipynb').read_text()
    code="""import sys,json,subprocess
from pathlib import Path
WORK=Path('/kaggle/working/v6-refiner-audit');WORK.mkdir(parents=True,exist_ok=True)
try:
    import pycocotools
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','pycocotools==2.0.8'],check=True)
"""
    code+='\nsources='+repr(modules)+'\nbaseline='+repr(baseline)+"""
PACKAGE=WORK/'package'
for name,source in sources.items():
    path=PACKAGE/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
source_path=WORK/'archived_v5_training_v3.ipynb';source_path.write_text(baseline)
sys.path.insert(0,str(PACKAGE))
from solar.v6_refiner_audit import run
INPUT=Path('/kaggle/input')
def one(name):
    files=list(INPUT.rglob(name))
    if len(files)!=1:raise RuntimeError(f'Expected one {name}, found {len(files)}')
    return files[0]
result=run(one('MAGFiLO_1.0_Annotations_kaggle2026_train.json'),one('training_manifest.csv'),
           one('best_detector_predictions_1024.pkl'),one('refined_validation_predictions_2048.pkl'),
           one('best_refiner.pt'),source_path,WORK,seconds=1000)
print('Inference-only local validation complete. No training, test inference, or official submission.')
print(json.dumps({'selected':result['selected'],'confirmation_gain_supported':result['confirmation_gain_supported']},indent=2))
"""
    if experiment in {'resolution','confirmation'}:
        code=code.replace('from solar.v6_refiner_audit import run','from solar.v6_resolution_audit import run')
        code=code.replace("one('best_refiner.pt'),source_path", "one('best_detector.pt'),one('best_refiner.pt'),source_path")
    if experiment=='confirmation':code=code.replace('seconds=1000)','seconds=1000,full_confirmation=True)')
    ast.parse(code)
    cells=[{'cell_type':'markdown','metadata':{},'source':['# V6 bounded refiner replay\n','Private inference experiment. No training or test labels. V5 Public PQ baseline remains 0.32.\n','Reuses the exact archived V5/v3 checkpoint and Apache-2.0 attributed architecture from Nomannic.\n','Compares plain/flip4, four thresholds and three overlap policies, with separate dates for tuning and confirmation.\n']},
           {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]
    if experiment in {'resolution','confirmation'}:
        cells[0]['source']=['# V6 native-resolution detector pilot\n',
                            'Private checkpoint-only inference. No training or test inference.\n',
                            'Exact Apache-attributed V5 architecture, 1024 versus 2048, union versus mask NMS.\n',
                            'Fixed 16 tuning and 16 confirmation observations on separate dates.\n']
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells},indent=1)+'\n')
    meta={'id':'minahussain111/solar-filament-research-v6-refiner','title':'solar-filament-research-v6-refiner',
          'code_file':out.name,'language':'python','kernel_type':'notebook','is_private':True,
          'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':True,
          'dataset_sources':['minahussain111/solar-filament-v5-v3-cache-356bd788'],
          'competition_sources':['filament-segmentation-2026'],'kernel_sources':[],'model_sources':[]}
    if experiment in {'resolution','confirmation'}:
        meta['id']=meta['id'].replace('-refiner','-resolution')
        meta['title']=meta['title'].replace('-refiner','-resolution')
    if experiment=='confirmation':
        meta['id']=meta['id'].replace('-resolution','-confirmation');meta['title']=meta['title'].replace('-resolution','-confirmation')
        cells[0]['source']=['# Fixed NMS confirmation\n','No training. Original pilot-confirmation observations excluded. No new threshold search.\n']
        payload=json.loads(out.read_text());payload['cells']=cells;out.write_text(json.dumps(payload,indent=1)+'\n')
    (out.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
