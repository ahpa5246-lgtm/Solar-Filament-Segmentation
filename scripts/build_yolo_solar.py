"""Reuse existing V6 notebook for a modern pretrained instance-model comparison."""
import ast,hashlib,json
from pathlib import Path
from build_v5_score_calibration import build as base_build,ROOT

KERNEL='minahussain111/solar-filament-research-v6'


def build(output):
    output=Path(output);meta=base_build(output);nb=json.loads(output.read_text())
    tree=ast.parse(''.join(nb['cells'][1]['source']));values={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    old=ast.literal_eval(values['SOURCE'].value);prefix=ast.parse(old.split('from solar.v5_score_calibration import run',1)[0]);sources=ast.literal_eval(prefix.body[0].value)
    for name in ('disjoint_delivery.py','yolo_solar_data.py','yolo_solar_experiment.py','v6_submission.py','v5_finetune.py','stable_detector_training.py'):
        sources['solar/'+name]=(ROOT/'solar'/name).read_text()
    prefix.body[0].value=ast.parse(repr(sources),mode='eval').body
    prefix.body=[n for n in prefix.body if not isinstance(n,ast.FunctionDef) or n.name!='one']
    driver=ast.unparse(ast.fix_missing_locations(prefix))+'''
import os,subprocess
os.environ['YOLO_CONFIG_DIR']='/tmp/solar_yolo_configuration'
os.environ['WANDB_DISABLED']='true'
os.environ['COMET_MODE']='DISABLED'
subprocess.run([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check','--no-deps','--target','/tmp/solar_yolo_packages','ultralytics==8.3.221','ultralytics-thop==2.0.18'],check=True)
sys.path.insert(0,'/tmp/solar_yolo_packages')
from hashlib import sha256
def one(name):
    paths=list(Path('/kaggle/input').rglob(name))
    if not paths or len({sha256(p.read_bytes()).hexdigest() for p in paths})!=1:
        raise RuntimeError(f'Missing or different-content {name}')
    return sorted(paths)[0]
from solar.yolo_solar_experiment import run
run(one('MAGFiLO_1.0_Annotations_kaggle2026_train.json'),one('training_manifest.csv'),
    one('refined_validation_predictions_2048.pkl'),one('best_refiner.pt'),one('final_refiner.pt'),
    source_path,Path('/kaggle/working/yolo-solar'))
'''
    ast.parse(driver);values['SOURCE'].value=ast.Constant(driver);values['EXPECTED_SHA'].value=ast.Constant(hashlib.sha256(driver.encode()).hexdigest())
    nb['cells'][1]['source']=ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    nb['cells'][0]['source']=['# Anchor-free solar instance segmentation comparison\n',
        'Unmodified Ultralytics 8.3.221 / YOLO11m-seg COCO pretrained model (AGPL-3.0).\n',
        'All fitting annotation variants; conservative fitting-only empty crops; overlap_mask=False.\n',
        '1024 input; pretrained early layers frozen; EMA; full-pipeline PQ selects checkpoints.\n',
        'Tune confidence/refinement before reading confirmation. One final all-training epoch only on positive local gain.\n',
        'No test labels; complete 180-image export; official Public score remains unknown until competition scoring.\n']
    output.write_text(json.dumps(nb,indent=1)+'\n')
    cache=json.loads((ROOT/'reports/V5_FINAL_CACHE.json').read_text())
    meta.update(id=KERNEL,title=KERNEL.split('/')[1]);meta['dataset_sources'].append(cache['dataset'])
    (output.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return meta


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
