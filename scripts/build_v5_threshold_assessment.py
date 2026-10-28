"""Reuse the completed confirmation notebook for one explicit official assessment.

The .65 refiner candidate improved aggregate validation micro PQ but failed the
paired macro uncertainty gate. This is an exploratory official assessment at the
owner's explicit request, not automatic model promotion. No further fitting.
"""
import ast,hashlib,json
from pathlib import Path
from build_v5_score_calibration import build as base_build,ROOT


def build(output):
    output=Path(output);meta=base_build(output)
    cache=json.loads((ROOT/'reports/V5_FINAL_CACHE.json').read_text())
    if cache.get('status')!='ready' or not cache.get('private_verified'):
        raise RuntimeError('Verified final checkpoint cache required')
    nb=json.loads(output.read_text());tree=ast.parse(''.join(nb['cells'][1]['source']))
    values={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    driver=ast.literal_eval(values['SOURCE'].value)
    prefix=driver.split('from solar.v5_score_calibration import run',1)[0]
    parsed=ast.parse(prefix);sources=ast.literal_eval(parsed.body[0].value)
    sources['solar/v6_submission.py']=(ROOT/'solar/v6_submission.py').read_text()
    parsed.body[0].value=ast.parse(repr(sources),mode='eval').body
    driver=ast.unparse(ast.fix_missing_locations(parsed))+'\nEXPECTED='+repr(cache['artifact_sha256'])+'''
from solar.v6_submission import run
test_dirs=[p for p in Path('/kaggle/input').rglob('test_images') if p.is_dir()]
if len(test_dirs)!=1:raise RuntimeError('Expected one official test directory')
result=run(test_dirs[0],one('final_detector.pt').parent,EXPECTED,source_path,
    Path('/kaggle/working/final-assessment'),Path('/kaggle/working/submission.csv'),
    seconds=1800,candidate_policy='union',refiner_threshold=.65)
print('OFFICIAL_ASSESSMENT_READY',json.dumps(result),flush=True)
'''
    ast.parse(driver);values['SOURCE'].value=ast.Constant(driver)
    values['EXPECTED_SHA'].value=ast.Constant(hashlib.sha256(driver.encode()).hexdigest())
    nb['cells'][1]['source']=ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    nb['cells'][0]['source']=['# Final V5 threshold assessment\n',
        'Frozen all-training V5 weights. Union overlap; flip4 refinement threshold .65.\n',
        'Baseline .85 is reconstructed against the preserved scored CSV before candidate export.\n',
        'Exploratory official assessment requested by the owner; local uncertainty remains.\n',
        'All 180 official test images. No test labels or retraining. Original best preserved.\n']
    output.write_text(json.dumps(nb,indent=1)+'\n')
    meta['dataset_sources']=[cache['dataset']]
    (output.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
