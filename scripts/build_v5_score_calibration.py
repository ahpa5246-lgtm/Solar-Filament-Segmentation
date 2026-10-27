"""Package a fixed-weight V5 comparison into the existing private confirmation notebook."""
import argparse
import ast
import json
from pathlib import Path
from kaggle_isolated_runtime import wrap_final_inference_target

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('numpy==2.2.6', 'scipy==1.15.3',
            'opencv-python-headless==4.11.0.86', 'pycocotools==2.0.11')


def compatible_runtime(source):
    # Change bootstrap assignments only. Never alter the hash-frozen source.
    tree = ast.parse(wrap_final_inference_target(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'PACKAGES':
                node.value = ast.parse(repr(PACKAGES), mode='eval').body
            elif node.targets[0].id == 'preflight':
                node.value = ast.Constant(ast.literal_eval(node.value).replace('1.26.4', '2.2.6'))
        if isinstance(node, ast.List) and any(isinstance(v, ast.Constant) and v.value == '--target' for v in node.elts):
            # Keep sys.executable at argv[0] and pip at argv[2].
            for index, value in enumerate(node.elts):
                if isinstance(value, ast.Constant) and value.value == '--target':
                    node.elts.insert(index, ast.Constant('--only-binary=:all:'))
                    break
    return ast.unparse(ast.fix_missing_locations(tree))


def build(output):
    names = ['__init__.py', 'v6_audit.py', 'v6_cached_search.py', 'v6_diagnostics.py',
             'v5_replay.py', 'v6_refiner_audit.py', 'v6_resolution_audit.py', 'v5_score_calibration.py']
    sources = {'solar/' + name: (ROOT / 'solar' / name).read_text() for name in names}
    archived = (ROOT / 'notebooks/baselines/v5_training_session_v3.ipynb').read_text()
    code = 'sources = ' + repr(sources) + '\narchived = ' + repr(archived) + '''
import sys, json
import numpy as np
sys.modules.setdefault('numpy.rec', np.rec)
from pathlib import Path
WORK = Path('/kaggle/working/v5-score-calibration')
PACKAGE = WORK / 'package'
for name, source in sources.items():
    path = PACKAGE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
source_path = WORK / 'archived_v5.ipynb'
source_path.write_text(archived)
sys.path.insert(0, str(PACKAGE))
def one(name):
    paths = list(Path('/kaggle/input').rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f'Expected one {name}; found {len(paths)}')
    return paths[0]
from solar.v5_score_calibration import run
result = run(one('MAGFiLO_1.0_Annotations_kaggle2026_train.json'), one('training_manifest.csv'),
             one('best_detector_predictions_1024.pkl'), one('refined_validation_predictions_2048.pkl'),
             one('best_detector.pt'), one('best_refiner.pt'), source_path, WORK, seconds=1000)
print(json.dumps({'status': result['status'], 'selected': result['selected'],
                  'local_gain_supported': result['local_gain_supported'], 'promote': result['promote']}))
'''
    ast.parse(code)
    wrapped = compatible_runtime(code)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    nb = {'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
          'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': [
              '# V5 fixed-weight detector score comparison\n',
              'Existing archived V5 weights; Apache attribution retained in the embedded source.\n',
              'No training, test inference or official submission. Historical validation reuse; no automatic promotion.\n']},
                    {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                     'outputs': [], 'source': wrapped.splitlines(keepends=True)}]}
    out.write_text(json.dumps(nb, indent=1) + '\n')
    meta = {'id': 'minahussain111/solar-filament-research-v6-confirmation',
            'title': 'solar-filament-research-v6-confirmation', 'code_file': out.name,
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': True,
            'dataset_sources': ['minahussain111/solar-filament-v5-v3-cache-356bd788'],
            'competition_sources': ['filament-segmentation-2026'], 'kernel_sources': [], 'model_sources': []}
    (out.parent / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    build(parser.parse_args().output)
