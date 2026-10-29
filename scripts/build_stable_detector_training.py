"""Reuse the existing topology notebook for corrected warm-start training."""
import ast,hashlib,json
from pathlib import Path
from build_v5_finetune import build as base_build

KERNEL='minahussain111/solar-topology-pilot-20261026'


def build(output):
    output=Path(output);meta=base_build(output);nb=json.loads(output.read_text())
    tree=ast.parse(''.join(nb['cells'][1]['source']))
    values={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    driver=ast.literal_eval(values['SOURCE'].value)
    original='source_path,WORK,seconds=6600)'
    if original not in driver:raise RuntimeError('Training driver structure drift')
    driver=driver.replace(original,'source_path,WORK,seconds=3000,stable_finetune=True,epochs=4)')
    ast.parse(driver);values['SOURCE'].value=ast.Constant(driver)
    values['EXPECTED_SHA'].value=ast.Constant(hashlib.sha256(driver.encode()).hexdigest())
    nb['cells'][1]['source']=ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
    nb['cells'][0]['source']=['# Corrected detector continuation\n',
        'Restore original three trainable ResNet backbone stages; freeze noisy batch-one BN moments.\n',
        'Four epochs from preserved V5 validation weights; fixed crop refiner and complete-pipeline PQ selection.\n',
        'No test inference or official submission here. Original best remains mounted read-only.\n']
    output.write_text(json.dumps(nb,indent=1)+'\n')
    meta.update(id=KERNEL,title=KERNEL.split('/')[1]);(output.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
