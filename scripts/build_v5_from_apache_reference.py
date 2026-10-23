"""Build an attributed, modified Kaggle notebook from an Apache-2.0 public source.

Reference is downloaded by Kaggle CLI during GitHub Actions, not vendored.
Only official competition TRAIN annotations are used for model fitting.
"""
import argparse, copy, json
from pathlib import Path

UPSTREAM="nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner"
ATTRIBUTION="""# Solar Filament V5 — pretrained Mask R-CNN + crop U-Net

Modified research reproduction of [Nomannic's Apache-2.0 Kaggle notebook](https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner). Original author: **Nomannic**; original title: *[CV] Solar Filament | Mask R-CNN + U-Net Refiner*. Licensed Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0 . See our third_party/NOTICE.md.

Changes: remove EDA-only cells, GPU preflight, four-flip refiner TTA ablation selected on official TRAIN validation, submission RLE gate. The reference's two-T4 detector training, pretrained COCO weights, ResNet18 crop refiner, and validation threshold selection are retained. No test labels or metadata are used for training. Requires Kaggle Internet On + GPU T4 x2. Estimated >1h GPU; score not yet known. Preserve the 0.24 submission until this candidate is verified.
"""
PREFLIGHT="""import torch, cv2, torchvision, importlib.util
assert torch.cuda.is_available(), 'Enable Kaggle GPU.'
assert torch.cuda.device_count() >= 2, 'This two-worker DDP notebook requires Kaggle T4 x2.'
assert importlib.util.find_spec('pycocotools'), 'Install pycocotools.'
from torchvision.ops import nms
assert nms(torch.tensor([[0.,0.,10.,10.]],device='cuda:0'),torch.tensor([1.],device='cuda:0'),0.5).numel()==1
print('Preflight passed: two GPUs, torchvision CUDA ops, COCO RLE.')
"""
TTA_HELPER="""
# Four-view flip TTA, selected on held-out official TRAIN observations.
def predict_refiner_probabilities(model, images, use_tta=False):
    if not use_tta:
        return model(images).sigmoid()[:,0]
    logits=[]
    for flip_y,flip_x in ((False,False),(False,True),(True,False),(True,True)):
        dims=tuple(d for enabled,d in ((flip_y,-2),(flip_x,-1)) if enabled)
        transformed=torch.flip(images,dims) if dims else images
        result=model(transformed).float()
        if dims: result=torch.flip(result,dims)
        logits.append(result)
    return torch.stack(logits).mean(0).sigmoid()[:,0]
"""
GATE="""# Submission integrity; test images remain inference-only.
from pycocotools import mask as mask_utils
assert len(test_files)==180, f'Expected 180 test images, found {len(test_files)}'
assert set(test_predictions)=={p.stem for p in test_files}
assert list(submission.columns)==['filament_id','segmentation_rle']
assert submission.filament_id.is_unique and submission.segmentation_rle.notna().all()
assert len(submission)>0, 'Empty submission rejected'
ids=set(submission.filament_id)
for observation,masks in test_predictions.items():
    for j,rle in enumerate(masks,1):
        assert tuple(rle['size'])==(2048,2048)
        assert int(mask_utils.area(rle))>0
        assert f'{observation}_{j}' in ids
print('Submission gate passed:',len(test_files),'images,',len(submission),'RLE instances.')
print('Official public score is unknown until submission.')
"""

def src(c): return ''.join(c['source'])
def setsrc(c,s): c['source']=s.splitlines(keepends=True)
def md(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)}
def py(s): return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(keepends=True)}

def build_notebook(original):
    cells=original.get('cells',[])
    if len(cells)<34: raise ValueError('Upstream notebook too short')
    signatures={20:'from pycocotools.coco import COCO',21:'class FilamentDataset',
                26:'class CropUNet',27:'refined_unmerged = defaultdict(list)',
                30:'use_refiner = bool(',32:'final_detector = detector.build_model',
                33:'submission_rows = []'}
    for i,needle in signatures.items():
        if needle not in src(cells[i]):
            raise ValueError(f'Upstream changed cell {i}: {needle}')
    result=[md(ATTRIBUTION),py(PREFLIGHT)]
    for i in range(20,34):
        c=copy.deepcopy(cells[i])
        c['metadata']={}
        if c['cell_type']=='code':
            c['execution_count']=None
            c['outputs']=[]
        s=src(c)
        if i==26:
            s+=TTA_HELPER
        if i==27:
            old="refined_unmerged = defaultdict(list)\nrefiner_threshold = 0.85\n"
            new="refined_unmerged = {'plain': defaultdict(list), 'flip4': defaultdict(list)}\nrefiner_threshold = 0.85\n"
            if old not in s: raise ValueError('Upstream changed validation refiner setup')
            s=s.replace(old,new,1)
            start=s.index("        probabilities = refiner(images).sigmoid().cpu().numpy()[:, 0]")
            stop=s.index("\nrefined_predictions_2048 = ",start)
            block="""        probs_by_mode={
            mode: predict_refiner_probabilities(refiner,images,use_tta=(mode=='flip4')).cpu().numpy()
            for mode in ('plain','flip4')
        }
        for task_index,task in enumerate(batch_tasks):
            for mode in ('plain','flip4'):
                probability=probs_by_mode[mode][task_index]
                x0,y0,x1,y1=task['bounds']
                height,width=task['shape']
                crop_mask=cv2.resize(probability,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)>=refiner_threshold
                if np.any(crop_mask):
                    full_mask=np.zeros((height,width),dtype=np.uint8)
                    full_mask[y0:y1,x0:x1]=crop_mask
                    refined_unmerged[mode][task['observation_id']].append(encode_mask(full_mask))
                else:
                    refined_unmerged[mode][task['observation_id']].append(task['fallback'])
"""
            s=s[:start]+block+s[stop:]
            old="""refined_predictions_2048 = {observation_id: merge_overlapping_rles(refined_unmerged.get(observation_id, [])) for observation_id in best_detector_predictions}

comparison = []
for name, predictions in [("Detector", baseline_predictions_2048), ("Detector + refiner", refined_predictions_2048)]:
"""
            new="""refined_by_mode={
    mode:{obs:merge_overlapping_rles(predictions.get(obs,[]))
          for obs in best_detector_predictions}
    for mode,predictions in refined_unmerged.items()
}
refined_predictions_2048=refined_by_mode['plain']
comparison=[]
for name,predictions in [('Detector',baseline_predictions_2048),
                         ('Detector + refiner',refined_by_mode['plain']),
                         ('Detector + refiner + flip4',refined_by_mode['flip4'])]:
"""
            if old not in s: raise ValueError('Upstream changed refiner evaluation')
            s=s.replace(old,new,1)
        if i==30:
            old='use_refiner = bool(comparison.loc["Detector + refiner", "all_dice"] >= comparison.loc["Detector", "all_dice"])'
            new="""best_strategy=str(comparison['all_dice'].idxmax())
use_refiner=best_strategy!='Detector'
use_refiner_tta=best_strategy=='Detector + refiner + flip4'"""
            if old not in s: raise ValueError('Upstream changed refiner selection')
            s=s.replace(old,new,1)
            old='"use_refiner": use_refiner, "refiner_epochs":'
            new='"use_refiner": use_refiner, "refiner_flip4": use_refiner_tta, "refiner_epochs":'
            if old not in s: raise ValueError('Upstream changed final configuration')
            s=s.replace(old,new,1)
        if i==32:
            old='probabilities = final_refiner(crop_tensor).sigmoid().cpu().numpy()[:, 0]'
            new="""probabilities = predict_refiner_probabilities(
                final_refiner,crop_tensor,
                use_tta=final_configuration.get('refiner_flip4',False)
            ).cpu().numpy()"""
            if old not in s: raise ValueError('Upstream changed test refiner inference')
            s=s.replace(old,new,1)
        setsrc(c,s)
        result.append(c)
    result.append(md("## Submission gate\nValidate 180 images and RLE integrity; test labels are never read."))
    result.append(py(GATE))
    return {'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}},'cells':result}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    notebook=build_notebook(json.loads(args.source.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Generated',args.output,'cells:',len(notebook['cells']))

if __name__=='__main__':main()
