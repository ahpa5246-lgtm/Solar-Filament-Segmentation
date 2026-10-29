"""Recover the already completed v8 validation selection into full-test inference."""
import ast,hashlib,json
from pathlib import Path
from build_v5_threshold_assessment import build as base_build,ROOT


def build(output):
    output=Path(output);meta=base_build(output);nb=json.loads(output.read_text())
    tree=ast.parse(''.join(nb['cells'][1]['source']))
    values={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    driver=ast.literal_eval(values['SOURCE'].value)
    parsed=ast.parse(driver);sources=ast.literal_eval(parsed.body[0].value)
    sources['solar/detector_consensus.py']=(ROOT/'solar/detector_consensus.py').read_text()
    parsed.body[0].value=ast.parse(repr(sources),mode='eval').body
    driver=ast.unparse(ast.fix_missing_locations(parsed))
    before="candidate_policy='union', refiner_threshold=0.65)"
    if before not in driver:raise RuntimeError('Frozen driver structure drift')
    driver=driver.replace(before,"candidate_policy='union', refiner_threshold=0.85, detector_strategy='consensus')")
    state=json.loads((ROOT/'reports/DETECTOR_CONSENSUS_EXECUTION_20261028.json').read_text());r=state['result']
    if r.get('processed_validation')!=132 or not r.get('official_assessment_allowed') or r.get('selected')!='consensus':
        raise ValueError('Completed v8 validation selection is required')
    if r.get('error_type')!='FileNotFoundError':raise ValueError('Unexpected v8 failure; inspect before recovery')
    driver='VALIDATION_EVIDENCE='+repr(r)+'\n'+driver
    ast.parse(driver);values['SOURCE'].value=ast.Constant(driver)
    values['EXPECTED_SHA'].value=ast.Constant(hashlib.sha256(driver.encode()).hexdigest())
    nb['cells'][1]['source']=ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    nb['cells'][0]['source']=['# Frozen detector-consensus official assessment\n',
        'Recover completed v8 tuning/confirmation after its sibling test-directory path error.\n',
        'No repeated validation sweep or retraining. Four-view detector consensus; original .85 flip4 refiner.\n',
        'Preserved final V5 weights; all 180 official test images; baseline parity required.\n',
        'Micro confirmation gain is small and paired CI includes zero; official assessment is exploratory.\n']
    output.write_text(json.dumps(nb,indent=1)+'\n');return meta


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
