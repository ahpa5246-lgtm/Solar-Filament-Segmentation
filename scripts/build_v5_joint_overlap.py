"""Reuse the existing confirmation notebook for the untested joint overlap policy."""
import ast
import hashlib
import json
from pathlib import Path
from build_v5_score_calibration import build as baseline_build


def build(output):
    output=Path(output)
    meta=baseline_build(output)
    nb=json.loads(output.read_text())
    tree=ast.parse(''.join(nb['cells'][1]['source']))
    assignments={n.targets[0].id:n for n in tree.body
                 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    driver=ast.literal_eval(assignments['SOURCE'].value)
    driver=driver.replace('from solar.v5_score_calibration import run',
                          'from solar.v6_resolution_audit import run')
    driver=driver.replace('WORK, seconds=1000)',
                          'WORK, seconds=1000, full_confirmation=True, preserve_refined=True)')
    driver=driver.split("print(json.dumps({'status': result['status']",1)[0]
    driver+="print('JOINT_OVERLAP_RESULT', json.dumps(result), flush=True)\n"
    ast.parse(driver)
    assignments['SOURCE'].value=ast.Constant(driver)
    assignments['EXPECTED_SHA'].value=ast.Constant(hashlib.sha256(driver.encode()).hexdigest())
    nb['cells'][1]['source']=ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    nb['cells'][0]['source']=['# Joint instance preservation experiment\n',
        'Fixed archived V5 weights; baseline union/union versus proposal NMS/refined none.\n',
        'No training or test inference. Historical validation; not official PQ.\n',
        'Fixed candidate; exclude earlier resolution-pilot confirmation dates.\n']
    output.write_text(json.dumps(nb,indent=1)+'\n')
    return meta


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    build(parser.parse_args().output)
