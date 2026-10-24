"""Verify wheel delivery and default notebook execution outside the checkout.

Requires the pinned CPU development environment. Does not train or use Kaggle
credentials/data. Random-weight inference proves software plumbing, not accuracy.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def package_version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None

DISTRIBUTION_CHECK=r'''
import csv,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from pycocotools import mask as coco
sys.path.insert(0,sys.argv[1])
import solar
assert Path(solar.__file__).resolve().parent==Path(sys.argv[1]).resolve()/'solar'
from solar.v6_detector import build_detector,save_checkpoint,load_checkpoint,infer
from solar.v6_instances import mask_nms
from solar.v6_audit import export_submission
from solar.v6_train import DetectionDataset
torch.manual_seed(42)
image=np.full((64,64),180,np.uint8);image[28:32,10:54]=60
model=build_detector(False,min_size=64,max_size=64)
config={'min_size':64,'max_size':64,'seed':42,'torch':str(torch.__version__)}
save_checkpoint(model,'roundtrip.pt',config)
restored,loaded=load_checkpoint('roundtrip.pt','cpu')
assert loaded==config
for network in [model,restored]:
    network.roi_heads.detections_per_img=2
    network.roi_heads.score_thresh=0.
    network.rpn._pre_nms_top_n={'training':20,'testing':20}
    network.rpn._post_nms_top_n={'training':10,'testing':10}
a,scores=infer(model,image,score_threshold=0.,mask_threshold=.01)
b,restored_scores=infer(restored,image,score_threshold=0.,mask_threshold=.01)
assert len(a)>0
np.testing.assert_array_equal(a,b)
np.testing.assert_array_equal(scores,restored_scores)
keep=mask_nms(a,scores,.5)
native=np.stack([cv2.resize(mask.astype(np.uint8),(2048,2048),interpolation=cv2.INTER_NEAREST).astype(bool) for mask in a[keep]])
export_submission({'observation_a':native,'observation_empty':native[:0]},['observation_a','observation_empty'],'submission.csv')
rows=list(csv.DictReader(open('submission.csv')))
assert len(rows)==len(native)>0
assert list(rows[0])==['filament_id','segmentation_rle']
assert len({r['filament_id'] for r in rows})==len(rows)
for row,expected in zip(rows,native):
    decoded=coco.decode({'size':[2048,2048],'counts':row['segmentation_rle'].encode()})
    np.testing.assert_array_equal(decoded,expected)
# Missing image inventory must stop export rather than silently omit observations.
try:export_submission({'observation_a':native},['observation_a','missing'],'bad.csv')
except ValueError:pass
else:raise AssertionError('Missing observation was accepted')
del restored
cv2.imwrite('synthetic_train.png',image)
dataset=DetectionDataset([{'path':'synthetic_train.png','stem':'20260101120000Bh','image_id':'software-test',
                          'ann':[{'segmentation':[[10,28,54,28,54,32,10,32]]}]}])
training_image,target=dataset[0]
model.train();optimizer=torch.optim.SGD(model.parameters(),lr=.001)
before=model.roi_heads.box_predictor.cls_score.weight.detach().clone()
losses=model([training_image],[target]);loss=sum(losses.values())
assert len(losses)==5 and torch.isfinite(loss)
optimizer.zero_grad();loss.backward()
gradient=model.roi_heads.box_predictor.cls_score.weight.grad
assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
optimizer.step()
assert not torch.equal(before,model.roi_heads.box_predictor.cls_score.weight.detach())
Path('distribution-result.json').write_text(json.dumps({'status':'passed','imported_built_wheel':True,'checkpoint_predictions_bitwise_equal':True,'native_rle_csv_roundtrip':True,'missing_inventory_rejected':True,'cpu_training_step_updated_head':True,'finite_training_losses':{name:float(value.detach()) for name,value in losses.items()},'rows':len(rows),'model_weights':'random; software test only'}))
'''

NOTEBOOK_CHECK=r'''
import json,sys
from pathlib import Path
notebook=json.loads(Path(sys.argv[1]).read_text());namespace={'__name__':'__main__'};count=0
for cell in notebook['cells']:
    if cell['cell_type']=='code':
        exec(compile(''.join(cell['source']),f'generated-notebook-cell-{count}','exec'),namespace)
        count+=1
import solar
assert Path(solar.__file__).resolve().parent==Path.cwd()/'v6-work/package/solar'
Path('notebook-result.json').write_text(json.dumps({'status':'passed','executed_code_cells':count,'embedded_package_used':True,'training':False}))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    env={**os.environ,'OMP_NUM_THREADS':'2'}
    tracked_sources=[ROOT/'pyproject.toml',*sorted((ROOT/'solar').glob('*.py')),Path(__file__).resolve(),ROOT/'scripts/build_v6_notebook.py']
    report={'status':'running','scope':'CPU delivery and integration; not scientific accuracy or GPU validation',
            'python':sys.version.split()[0],
            'packages':{name:package_version(name) for name in ['torch','torchvision','numpy','scipy','opencv-python-headless','pycocotools','setuptools','wheel']},
            'source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked_sources}}
    try:
        with tempfile.TemporaryDirectory(prefix='solar-verification-') as tmp:
            work=Path(tmp);wheel_dir=work/'wheels';target=work/'installed'
            source=work/'source';source.mkdir()
            shutil.copy2(ROOT/'pyproject.toml',source/'pyproject.toml')
            shutil.copytree(ROOT/'solar',source/'solar',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            if (ROOT/'LICENSE').is_file():shutil.copy2(ROOT/'LICENSE',source/'LICENSE')
            subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',str(source),'--wheel-dir',str(wheel_dir)],check=True,env=env)
            wheels=list(wheel_dir.glob('*.whl'))
            if len(wheels)!=1:raise RuntimeError('Expected one built distribution')
            report['wheel_sha256']=hashlib.sha256(wheels[0].read_bytes()).hexdigest()
            subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(target),str(wheels[0])],check=True,env=env)
            distribution=work/'distribution';distribution.mkdir()
            subprocess.run([sys.executable,'-I','-c',DISTRIBUTION_CHECK,str(target)],cwd=distribution,check=True,env=env)
            report['distribution']=json.loads((distribution/'distribution-result.json').read_text())
            notebook=work/'notebook';notebook.mkdir();path=notebook/'v6.ipynb'
            subprocess.run([sys.executable,str(ROOT/'scripts/build_v6_notebook.py'),'--output',str(path)],check=True,env=env)
            subprocess.run([sys.executable,'-I','-c',NOTEBOOK_CHECK,str(path)],cwd=notebook,check=True,env=env)
            report['notebook']=json.loads((notebook/'notebook-result.json').read_text())
            report['status']='passed'
    except Exception as error:
        report['status']='failed';report['error_type']=type(error).__name__
        raise
    finally:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
