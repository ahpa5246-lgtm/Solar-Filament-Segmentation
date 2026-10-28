"""Self-contained private V5 warm-start training; original checkpoints remain mounted read-only."""
import argparse, ast, json
from pathlib import Path
from build_v5_score_calibration import compatible_runtime

ROOT=Path(__file__).resolve().parents[1]
KERNEL='minahussain111/solar-v5-finetune-pq50-20261028'


def build(output):
    modules=['v5_finetune.py','v6_audit.py','v6_cached_search.py','v6_diagnostics.py',
             'v5_replay.py','v6_refiner_audit.py','v6_resolution_audit.py']
    sources={'solar/'+name:(ROOT/'solar'/name).read_text() for name in modules}
    sources['solar/__init__.py']=''
    source=(ROOT/'notebooks/baselines/v5_training_session_v3.ipynb').read_text()
    driver='sources='+repr(sources)+'\narchived='+repr(source)+'''
import sys, json
import numpy as np
sys.modules.setdefault('numpy.rec',np.rec)
from pathlib import Path
WORK=Path('/kaggle/working/v5-finetune'); WORK.mkdir(parents=True,exist_ok=True)
PACKAGE=WORK/'package'
for name,text in sources.items():
    path=PACKAGE/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
source_path=WORK/'archived_v5.ipynb';source_path.write_text(archived)
sys.path.insert(0,str(PACKAGE))
def one(name):
    files=list(Path('/kaggle/input').rglob(name))
    if len(files)!=1:raise RuntimeError(f'Expected one {name}; got {len(files)}')
    return files[0]
from solar.v5_finetune import run
run(one('MAGFiLO_1.0_Annotations_kaggle2026_train.json'),one('training_manifest.csv'),
    one('best_detector_predictions_1024.pkl'),one('refined_validation_predictions_2048.pkl'),
    one('best_detector.pt'),one('best_refiner.pt'),source_path,WORK,seconds=6600)
'''
    ast.parse(driver)
    tree=ast.parse(compatible_runtime(driver))
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id=='PACKAGES':
                packages=ast.literal_eval(node.value)+('pandas==2.2.3',)
                node.value=ast.parse(repr(packages),mode='eval').body
            elif node.targets[0].id=='target':
                node.value=ast.parse("Path('/tmp/solar_v5_finetune_packages')",mode='eval').body
    code=ast.unparse(ast.fix_missing_locations(tree))
    nb={'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':[
            '# V5 detector continuation toward official PQ 0.50\n',
            'Original Mask R-CNN + fixed Crop U-Net architecture and archived training augmentations.\n',
            'Warm starts from preserved validation weights; no training-from-scratch substitution.\n',
            'Up to 16 additional epochs. Tune pipeline PQ; reserve confirmation for final selection.\n',
            'Private training only; no test inference or competition submission in this notebook.\n',
            'Original Apache-2.0 source attribution is retained in archived_v5.ipynb.\n']},
                 {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(nb,indent=1)+'\n')
    meta={'id':KERNEL,'title':KERNEL.split('/')[1],'code_file':output.name,
          'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,
          'machine_shape':'NvidiaTeslaT4','enable_internet':True,
          'dataset_sources':['minahussain111/solar-filament-v5-v3-cache-356bd788'],
          'competition_sources':['filament-segmentation-2026'],'kernel_sources':[],'model_sources':[]}
    (output.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    build(parser.parse_args().output)
