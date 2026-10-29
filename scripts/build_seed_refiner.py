"""Reuse the existing refiner notebook for seeded refinement and negative rejection."""
import ast,hashlib,json
from pathlib import Path
from build_v5_score_calibration import build as base_build,ROOT
KERNEL='minahussain111/solar-filament-research-v6-refiner'


def build(output):
    output=Path(output);meta=base_build(output);nb=json.loads(output.read_text())
    tree=ast.parse(''.join(nb['cells'][1]['source']));values={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    old=ast.literal_eval(values['SOURCE'].value);prefix=ast.parse(old.split('from solar.v5_score_calibration import run',1)[0]);sources=ast.literal_eval(prefix.body[0].value)
    for name in ('disjoint_delivery.py','seed_refiner.py','seed_refiner_experiment.py','yolo_solar_experiment.py','yolo_solar_data.py','v6_submission.py','v5_finetune.py','stable_detector_training.py'):
        sources['solar/'+name]=(ROOT/'solar'/name).read_text()
    prefix.body[0].value=ast.parse(repr(sources),mode='eval').body
    prefix.body=[n for n in prefix.body if not isinstance(n,ast.FunctionDef) or n.name!='one']
    evidence=json.loads((ROOT/'reports/YOLO_SOLAR_EXECUTION_20261028.json').read_text())['result']
    if evidence.get('status')!='COMPLETE':raise ValueError('Completed YOLO source model required')
    driver=ast.unparse(ast.fix_missing_locations(prefix))+'\nYOLO_EVIDENCE='+repr(evidence)+'''
import os,subprocess
os.environ['YOLO_CONFIG_DIR']='/tmp/solar_seed_yolo_config'
os.environ['WANDB_DISABLED']='true';os.environ['COMET_MODE']='DISABLED'
subprocess.run([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check','--no-deps','--target','/tmp/solar_seed_yolo_packages','ultralytics==8.3.221','ultralytics-thop==2.0.18'],check=True)
sys.path.insert(0,'/tmp/solar_seed_yolo_packages')
from hashlib import sha256
def one(name):
    paths=list(Path('/kaggle/input').rglob(name))
    if not paths or len({sha256(p.read_bytes()).hexdigest() for p in paths})!=1:raise RuntimeError(f'Missing or ambiguous {name}')
    return sorted(paths)[0]
import csv
manifest=one('training_manifest.csv');rows=list(csv.DictReader(manifest.open()))
fitting={r['observation_id'] for r in rows if r['split']=='train'}
negative={}
for path in Path('/kaggle/input').rglob('*_negative.jpg'):
    stem=path.stem.removesuffix('_negative')
    if stem not in fitting:continue
    if path.name in negative and sha256(path.read_bytes()).hexdigest()!=sha256(negative[path.name].read_bytes()).hexdigest():raise RuntimeError('Different-content negative crop copies')
    negative[path.name]=path
if len(negative)!=YOLO_EVIDENCE['dataset']['negative_crops']:raise RuntimeError('Guarded negative crop inventory mismatch')
from solar.seed_refiner_experiment import run
run(one('MAGFiLO_1.0_Annotations_kaggle2026_train.json'),manifest,
    one('best_detector_predictions_1024.pkl'),one('refined_validation_predictions_2048.pkl'),
    one('best_refiner.pt'),one('final_detector.pt'),source_path,
    one('epoch-%03d.pt'%YOLO_EVIDENCE['selected_epoch']),one('final_yolo.pt'),one('yolo_result.json'),
    list(negative.values()),Path('/kaggle/working/seed-refiner'))
'''
    ast.parse(driver);values['SOURCE'].value=ast.Constant(driver);values['EXPECTED_SHA'].value=ast.Constant(hashlib.sha256(driver.encode()).hexdigest())
    nb['cells'][1]['source']=ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    nb['cells'][0]['source']=['# Box-conditioned refinement and proposal rejection\n',
        'Three channels: grayscale, CLAHE contrast, rectangular proposal-box seed.\n',
        'Fitting-only GT crops and guarded fitting-only negative crops. No full GT mask is supplied as the seed.\n',
        'Eight adaptation epochs; complete-pipeline PQ checkpoint choice; frozen confirmation; one all-training epoch and official assessment.\n',
        'Compare both detector families; empty masks reject proposals rather than restore false positives.\n',
        'Private training and complete test export. Official score is unknown until the competition evaluates the CSV.\n']
    output.write_text(json.dumps(nb,indent=1)+'\n')
    cache=json.loads((ROOT/'reports/V5_FINAL_CACHE.json').read_text())
    pinned=json.loads((ROOT/'reports/YOLO17_PRIVATE_CACHE.json').read_text())
    if pinned.get('status')!='ready' or not pinned.get('private_verified'):raise ValueError('Verified private v17 cache required')
    meta.update(id=KERNEL,title=KERNEL.split('/')[1],kernel_sources=[])
    meta['dataset_sources'].append(pinned['dataset'])
    meta['dataset_sources'].append(cache['dataset'])
    (output.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return meta


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
