"""Reuse the existing refiner notebook for actual false-proposal supervised rejection."""
import ast,hashlib,json
from pathlib import Path
from build_seed_refiner import build as base_build,ROOT

def build(output):
 output=Path(output);meta=base_build(output);nb=json.loads(output.read_text());tree=ast.parse(''.join(nb['cells'][1]['source']));values={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
 driver=ast.literal_eval(values['SOURCE'].value);parsed=ast.parse(driver);sources=ast.literal_eval(parsed.body[0].value)
 for name in ('proposal_negatives.py','scale_refiner_experiment.py'):sources['solar/'+name]=(ROOT/'solar'/name).read_text()
 parsed.body[0].value=ast.parse(repr(sources),mode='eval').body;driver=ast.unparse(ast.fix_missing_locations(parsed))
 cache=json.loads((ROOT/'reports/SEED2_PRIVATE_CACHE.json').read_text())
 if not cache.get('private_verified'):raise ValueError('Verified original seed v2 checkpoint required')
 sha=cache['artifact_sha256']['seed-epoch-08.pt']
 driver=driver.replace("one('best_refiner.pt'), one('final_detector.pt')","one('seed-epoch-08.pt'), one('final_detector.pt')")
 original="list(negative.values()), Path('/kaggle/working/seed-refiner'))"
 replacement="list(negative.values()), Path('/kaggle/working/hard-refiner'), initial_refiner_sha="+repr(sha)+", epochs=6, learning_rate=3e-5, hard_proposals=True)"
 if original not in driver:raise ValueError('Expected run call not found; refusing ambiguous edit')
 driver=driver.replace(original,replacement);ast.parse(driver);values['SOURCE'].value=ast.Constant(driver);values['EXPECTED_SHA'].value=ast.Constant(hashlib.sha256(driver.encode()).hexdigest());nb['cells'][1]['source']=ast.unparse(ast.fix_missing_locations(tree)).splitlines(keepends=True)
 nb['cells'][0]['source']=['# Train actual false-proposal rejection\n','Initialize the officially assessed seed-refiner v2 validation checkpoint; keep YOLO weights frozen.\n','Mine 1024/1536 proposals on fitting observations only, retaining only crops clear of all known annotations with a 48-pixel margin.\n','Train six adaptation epochs at 3e-5, select complete-pipeline tuning PQ, inspect confirmation once, then one all-training epoch and all test predictions.\n','Compare single and combined proposal scales. Official Public PQ remains unknown until assessment.\n']
 output.write_text(json.dumps(nb,indent=1)+'\n');meta['dataset_sources'].append(cache['dataset']);(output.parent/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return meta
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
