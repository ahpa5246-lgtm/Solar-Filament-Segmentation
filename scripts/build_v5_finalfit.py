"""Package a pinned continuation output for a single full-data fit and inference."""
import ast,hashlib,json
from pathlib import Path
from build_v5_finetune import build as training_build,ROOT
KERNEL='minahussain111/solar-v5-final-assessment-20261028'


def build(output,version,expected_sha):
    if type(version) is not int or version<2:raise ValueError('Require known repaired training version')
    if len(expected_sha)!=64 or any(c not in '0123456789abcdef' for c in expected_sha):raise ValueError('Invalid frozen checkpoint hash')
    output=Path(output);meta=training_build(output)
    nb=json.loads(output.read_text());tree=ast.parse(''.join(nb['cells'][1]['source']))
    values={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    old=ast.literal_eval(values['SOURCE'].value)
    driver_tree=ast.parse(old)
    sources=ast.literal_eval(driver_tree.body[0].value)
    sources['solar/v5_finalfit.py']=(ROOT/'solar/v5_finalfit.py').read_text()
    sources['solar/v6_submission.py']=(ROOT/'solar/v6_submission.py').read_text()
    driver_tree.body[0].value=ast.parse(repr(sources),mode='eval').body
    prefix=ast.unparse(ast.fix_missing_locations(driver_tree)).split('from solar.v5_finetune import run',1)[0]
    driver=prefix+f'''
from hashlib import sha256
from solar.v5_finalfit import run
expected={expected_sha!r}
selected_training_version={version}
checkpoints=[p for p in Path('/kaggle/input').rglob('best_detector.pt') if sha256(p.read_bytes()).hexdigest()==expected]
if len(checkpoints)!=1:raise RuntimeError('Expected exactly one frozen continuation checkpoint')
run(one('MAGFiLO_1.0_Annotations_kaggle2026_train.json'),one('training_manifest.csv'),
    checkpoints[0],one('best_refiner.pt'),source_path,WORK,expected,seconds=6600)
'''
    ast.parse(driver)
    values['SOURCE'].value=ast.Constant(driver)
    values['EXPECTED_SHA'].value=ast.Constant(hashlib.sha256(driver.encode()).hexdigest())
    nb['cells'][1]['source']=ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    nb['cells'][0]['source']=['# Selected V5 continuation: full-data fitting and official test inference\n',
      'One fixed epoch on all labeled TRAIN observations; no test labels or test calibration.\n']
    output.write_text(json.dumps(nb,indent=1)+'\n')
    meta.update(id=KERNEL,title=KERNEL.split('/')[1],kernel_sources=['minahussain111/solar-v5-finetune-pq50-20261028'])
    (output.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta
