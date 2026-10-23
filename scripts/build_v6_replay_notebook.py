"""Self-contained CPU-only Kaggle replay; no credentials needed inside notebook."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build(out):
    modules={f'solar/{name}':(ROOT/'solar'/name).read_text() for name in ['__init__.py','v6_audit.py','v6_cached_search.py']}
    modules['solar/v6_replay.py']=(ROOT/'scripts/replay_v5_cached.py').read_text()
    codes=["""import sys,json,subprocess
from pathlib import Path
WORK=Path('/kaggle/working/v6-cpu-audit')
WORK.mkdir(parents=True,exist_ok=True)
# Reuse Kaggle's compatible installed numeric stack. Install only missing COCO support.
try:
    import pycocotools
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','pycocotools==2.0.8'],check=True)
""",'sources='+repr(modules)+"""
PACKAGE=WORK/'package'
for name,source in sources.items():
    path=PACKAGE/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
sys.path.insert(0,str(PACKAGE))
""","""from solar.v6_replay import replay
INPUT=Path('/kaggle/input')
def one(name):
    files=list(INPUT.rglob(name))
    if len(files)!=1:raise RuntimeError(f'Need one original V5 input {name}, found {len(files)}')
    return files[0]
result=replay(one('MAGFiLO_1.0_Annotations_kaggle2026_train.json'),one('training_manifest.csv'),
              one('best_detector_predictions_1024.pkl'),one('refined_validation_predictions_2048.pkl'))
result['execution']='Kaggle CPU replay; no model training or GPU inference'
Path('/kaggle/working/v6_cpu_replay.json').write_text(json.dumps(result,indent=2))
print(json.dumps({name:row['reference_micro_pq'] for name,row in result['all_annotation_results'].items()},indent=2))
print('These are cached LOCAL validation results, not official Kaggle scores.')
"""]
    cells=[{'cell_type':'markdown','metadata':{},'source':['# V6 CPU audit of preserved V5 outputs\n','No GPU, training, submission, or hidden labels. V5 remains the confirmed Public baseline 0.32.\n','Verifies exact V5/v3 cache fingerprints before using official TRAIN annotations.\n']}]
    for code in codes:
        ast.parse(code)
        cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)})
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(nb,indent=1)+'\n')
    meta={'id':'minahussain111/solar-filament-research-v6-audit','title':'solar-filament-research-v6-audit','code_file':out.name,
          'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_internet':True,
          'dataset_sources':['minahussain111/solar-filament-v5-v3-cache-356bd788'],'competition_sources':['filament-segmentation-2026'],
          'kernel_sources':[],'model_sources':[]}
    (out.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
