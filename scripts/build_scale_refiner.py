"""Reuse the existing research notebook for frozen-model resolution comparison."""
import ast,hashlib,json
from pathlib import Path
from build_seed_refiner import build as seed_build,ROOT
KERNEL='minahussain111/solar-filament-research-v6'
def build(output):
 output=Path(output);meta=seed_build(output);nb=json.loads(output.read_text());tree=ast.parse(''.join(nb['cells'][1]['source']));values={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
 driver=ast.literal_eval(values['SOURCE'].value);prefix=ast.parse(driver.split('from solar.seed_refiner_experiment import run',1)[0]);sources=ast.literal_eval(prefix.body[0].value);sources['solar/scale_refiner_experiment.py']=(ROOT/'solar/scale_refiner_experiment.py').read_text();prefix.body[0].value=ast.parse(repr(sources),mode='eval').body
 seed=json.loads((ROOT/'reports/SEED2_PRIVATE_CACHE.json').read_text());yolo=json.loads((ROOT/'reports/YOLO17_PRIVATE_CACHE.json').read_text())
 if not seed.get('private_verified'):raise ValueError('Private seed checkpoint cache not verified')
 checks={k:v for state in (seed,yolo) for k,v in state['artifact_sha256'].items() if k in ('epoch-015.pt','final_yolo.pt','seed-epoch-08.pt','final_seed_refiner.pt')}
 code=ast.unparse(ast.fix_missing_locations(prefix))+'\nCHECKS='+repr(checks)+'''
for name,digest in CHECKS.items():
    if sha256(one(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen checkpoint fingerprint mismatch: '+name)
from solar.scale_refiner_experiment import run
run(one('MAGFiLO_1.0_Annotations_kaggle2026_train.json'),manifest,source_path,
    one('epoch-015.pt'),one('final_yolo.pt'),one('seed-epoch-08.pt'),one('final_seed_refiner.pt'),Path('/kaggle/working/scale-refiner'))
'''
 ast.parse(code);values['SOURCE'].value=ast.Constant(code);values['EXPECTED_SHA'].value=ast.Constant(hashlib.sha256(code.encode()).hexdigest());nb['cells'][1]['source']=ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
 nb['cells'][0]['source']=['# Frozen model comparison at 1024, 1536, 2048 and combined resolutions\n','Reuse preserved YOLO v17 and officially assessed seed-refiner v2 weights. No training or test-label access.\n','Select resolution/confidence on tuning, inspect confirmation, export all 180 test images; competition score remains unknown until evaluated.\n']
 output.write_text(json.dumps(nb,indent=1)+'\n');meta.update(id=KERNEL,title=KERNEL.split('/')[1]);meta['dataset_sources'].append(seed['dataset']);(output.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return meta
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
