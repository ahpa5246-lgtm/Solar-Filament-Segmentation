"""Full-data fit of the measured BN-restored candidate, preserving BN moments."""
import ast,hashlib,json
from pathlib import Path
from build_v5_finetune import build as base_build,ROOT

KERNEL='minahussain111/solar-topology-pilot-20261026'


def build(output):
    output=Path(output);meta=base_build(output);nb=json.loads(output.read_text())
    state=json.loads((ROOT/'reports/V5_BN_RECOVERY_EXECUTION_20261028.json').read_text());result=state['result']
    if result.get('status')!='COMPLETE' or not result.get('official_assessment_allowed'):
        raise ValueError('Measured candidate gain on both local splits required')
    expected=result['candidate_checkpoint_sha256'];final_cache=json.loads((ROOT/'reports/V5_FINAL_CACHE.json').read_text())
    final_refiner_sha=final_cache['artifact_sha256']['final_refiner.pt']
    tree=ast.parse(''.join(nb['cells'][1]['source']));values={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    old=ast.literal_eval(values['SOURCE'].value);prefix=ast.parse(old.split('from solar.v5_finetune import run',1)[0])
    sources=ast.literal_eval(prefix.body[0].value)
    for name in ('v5_finalfit.py','v6_submission.py','detector_consensus.py'):
        sources['solar/'+name]=(ROOT/'solar'/name).read_text()
    prefix.body[0].value=ast.parse(repr(sources),mode='eval').body
    prefix.body=[n for n in prefix.body if not isinstance(n,ast.FunctionDef) or n.name!='one']
    driver=ast.unparse(ast.fix_missing_locations(prefix))+'\nEXPECTED='+repr(expected)+'\nFINAL_REFINER_SHA='+repr(final_refiner_sha)+'\nVALIDATION_EVIDENCE='+repr(result)+'''
from hashlib import sha256
def one(name):
    paths=list(Path('/kaggle/input').rglob(name))
    if not paths or len({sha256(p.read_bytes()).hexdigest() for p in paths})!=1:
        raise RuntimeError(f'Missing or ambiguous different-content {name}')
    return sorted(paths)[0]
from solar.v5_finalfit import run
try:
    run(one('MAGFiLO_1.0_Annotations_kaggle2026_train.json'),one('training_manifest.csv'),
        one('candidate_detector.pt'),one('best_refiner.pt'),source_path,WORK,EXPECTED,seconds=3000,
        stable_finetune=True,final_refiner_checkpoint=one('final_refiner.pt'),
        final_refiner_expected_sha=FINAL_REFINER_SHA)
except Exception as error:
    (WORK/'assessment_failure.json').write_text(json.dumps({'status':'FAILED','error_type':type(error).__name__,'diagnosis':str(error)[:700]}))
    raise
'''
    ast.parse(driver);values['SOURCE'].value=ast.Constant(driver);values['EXPECTED_SHA'].value=ast.Constant(hashlib.sha256(driver.encode()).hexdigest())
    nb['cells'][1]['source']=ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    nb['cells'][0]['source']=['# BN-restored detector: complete training fit and official test predictions\n',
        'Frozen candidate from measured interpolation v2; original early-backbone freezing and fixed BN moments.\n',
        'One declared epoch on all labeled training observations. Full-training original refiner for final test inference.\n',
        'Native test-directory discovery; complete 180-image RLE validation. Original best stays preserved.\n',
        'Confirmation micro gain is exploratory; no Public improvement claimed before official scoring.\n']
    output.write_text(json.dumps(nb,indent=1)+'\n')
    meta.update(id=KERNEL,title=KERNEL.split('/')[1],kernel_sources=['minahussain111/solar-v5-weight-interpolation-pq50'])
    meta['dataset_sources'].append(final_cache['dataset'])
    (output.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return meta


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
