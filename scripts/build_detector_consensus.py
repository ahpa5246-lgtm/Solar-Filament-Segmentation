"""Reuse the existing confirmation notebook for validation and final inference."""
import ast,hashlib,json
from pathlib import Path
from build_v5_score_calibration import build as base_build,ROOT


def build(output):
    output=Path(output);meta=base_build(output);nb=json.loads(output.read_text())
    tree=ast.parse(''.join(nb['cells'][1]['source']))
    assignments={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    old=ast.literal_eval(assignments['SOURCE'].value)
    prefix=ast.parse(old.split('from solar.v5_score_calibration import run',1)[0]);sources=ast.literal_eval(prefix.body[0].value)
    for name in ('detector_consensus.py','detector_consensus_experiment.py','v6_submission.py','v5_finetune.py'):
        sources['solar/'+name]=(ROOT/'solar'/name).read_text()
    prefix.body[0].value=ast.parse(repr(sources),mode='eval').body
    # Shared cache files can appear in both mounted, fingerprint-verified datasets.
    prefix.body=[n for n in prefix.body if not isinstance(n,ast.FunctionDef) or n.name!='one']
    cache=json.loads((ROOT/'reports/V5_FINAL_CACHE.json').read_text())
    code=ast.unparse(ast.fix_missing_locations(prefix))+'\nFINAL_HASHES='+repr(cache['artifact_sha256'])+'''
from hashlib import sha256
def one(name):
    paths=list(Path('/kaggle/input').rglob(name))
    if not paths:raise RuntimeError(f'Missing {name}')
    if len({sha256(p.read_bytes()).hexdigest() for p in paths})!=1:
        raise RuntimeError(f'Ambiguous different-content {name}')
    return sorted(paths)[0]
from solar.detector_consensus_experiment import run
run(one('MAGFiLO_1.0_Annotations_kaggle2026_train.json'),one('training_manifest.csv'),
    one('best_detector_predictions_1024.pkl'),one('refined_validation_predictions_2048.pkl'),
    one('best_detector.pt'),one('best_refiner.pt'),source_path,
    one('final_detector.pt').parent,FINAL_HASHES,Path('/kaggle/working/detector-consensus'),seconds=3300)
'''
    ast.parse(code);assignments['SOURCE'].value=ast.Constant(code)
    assignments['EXPECTED_SHA'].value=ast.Constant(hashlib.sha256(code.encode()).hexdigest())
    nb['cells'][1]['source']=ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    nb['cells'][0]['source']=['# Detector multi-view consensus and official assessment candidate\n',
        'Original attributed V5 architecture and preserved validation/final weights. No retraining.\n',
        'Four correctly inverted detector views; distinct-view support; unchanged crop-refiner pipeline.\n',
        'Tune 65 dates/observations, read 67 confirmation observations, then export all 180 test images only on a micro-PQ gain in both splits.\n',
        'Official PQ is unknown until Kaggle scores the exported prediction file.\n']
    output.write_text(json.dumps(nb,indent=1)+'\n')
    meta['dataset_sources'].append(cache['dataset'])
    (output.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
