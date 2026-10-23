"""V6 tests use independent binary instances, including overlapping masks."""
import csv
import numpy as np
import pytest
from solar import v6_audit as audit


def masks():
    a = np.zeros((16,16), bool); a[2:10,2:10] = True
    b = np.zeros_like(a); b[6:14,6:14] = True
    return np.stack([a,b])


def test_overlapping_perfect_masks_keep_both_instances():
    result = audit.measure(masks(), masks())
    assert result['pq_local'] == result['sq'] == result['rq'] == 1
    assert (result['tp'], result['fp'], result['fn']) == (2,0,0)
    assert result['matched_iou'] == [1.,1.]


def test_split_merge_empty_and_threshold_boundary():
    a = masks()[:1]; split = np.repeat(a,2,axis=0)
    split[0,:,6:] = 0; split[1,:,:6] = 0
    result = audit.measure(a,split)
    assert result['tp'] == 0 and result['likely_splits'] == 1
    result = audit.measure(split,a)
    assert result['likely_merges'] == 1
    assert audit.measure(a,np.zeros((0,16,16),bool))['fn'] == 1
    assert audit.measure(np.zeros((0,16,16),bool),a)['fp'] == 1
    with pytest.raises(ValueError): audit.measure(a,np.zeros((1,8,8),bool))


def test_date_partition_deterministic_and_disjoint():
    records = [{'stem':f'202601{i:02d}120000Bh','image_id':f'{i}-{j}'} for i in range(1,21) for j in range(2)]
    splits = audit.partition(records, seed=42)
    assert splits == audit.partition(records,seed=42)
    days = [{r['stem'][:8] for r in part} for part in splits]
    assert all(days) and not days[0]&days[1] and not days[0]&days[2] and not days[1]&days[2]
    assert sum(map(len,splits)) == len(records)
    with pytest.raises(ValueError): audit.partition(records[:2])


def test_submission_roundtrip_and_inventory(tmp_path):
    # Use true competition dimensions to exercise size and compressed-count checks.
    a = np.zeros((1,2048,2048),bool); a[:,20:25,30:40] = True
    out = tmp_path/'submission.csv'
    audit.export_submission({'obs_with_underscore':a, 'empty':a[:0]}, ['obs_with_underscore','empty'],out)
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0]) == ['filament_id','segmentation_rle']
    from pycocotools import mask as coco
    decoded = coco.decode({'size':[2048,2048],'counts':rows[0]['segmentation_rle'].encode()})
    np.testing.assert_array_equal(decoded,a[0])
    with pytest.raises(ValueError): audit.export_submission({'x':a},['x','missing'],out)
    with pytest.raises(ValueError): audit.export_submission({'x':a},['x','x'],out)
    with pytest.raises(ValueError): audit.export_submission({'x':masks()},['x'],out)


def test_bootstrap_groups_annotation_variants():
    perfect = audit.measure(masks(),masks())
    missing = audit.measure(masks(),masks()[:0])
    interval = audit.bootstrap({'obs-a':[perfect,perfect], 'obs-b':[missing]},seed=3,n=200)
    assert interval['estimate'] == pytest.approx(.5)
    assert 0 <= interval['low'] <= interval['estimate'] <= interval['high'] <= 1
    assert interval == audit.bootstrap({'obs-a':[perfect,perfect], 'obs-b':[missing]},seed=3,n=200)


def test_mask_nms_removes_duplicates_without_union_of_neighbors():
    from solar.v6_instances import mask_nms, resolve_overlap, restore_tile
    a = masks()
    proposals = np.concatenate([a,a[:1]])
    keep = mask_nms(proposals,np.array([.9,.8,.7]),iou_threshold=.5)
    assert keep == [0,1]
    resolved = resolve_overlap(a,np.array([.9,.8]))
    assert not np.any(resolved[0]&resolved[1])
    assert resolved[0].sum() == a[0].sum()
    assert resolved[1].sum() == np.count_nonzero(a[1]&~a[0])
    restored = restore_tile(a,(3,4),(24,24))
    np.testing.assert_array_equal(restored[:,3:19,4:20],a)
    with pytest.raises(ValueError): restore_tile(a,(20,20),(24,24))
    with pytest.raises(ValueError): mask_nms(a,np.array([np.nan,.2]))


def test_overlay_and_artifact_audit(tmp_path):
    import cv2,json
    image = np.full((16,16),100,np.uint8)
    audit.overlay(image,masks(),masks(),tmp_path/'overlay.png')
    assert cv2.imread(str(tmp_path/'overlay.png')).shape == (16,32,3)
    from solar.v6_forensics import run_audit
    from pycocotools import mask as coco
    encoded = [coco.encode(np.asfortranarray(a.astype(np.uint8))) for a in masks()]
    encoded = [{'size':x['size'],'counts':x['counts'].decode()} for x in encoded]
    report = run_audit({'obs':[masks()]},{'detector':{'obs':encoded}},tmp_path/'out')
    assert report['detector']['micro']['pq_local'] == 1
    assert (tmp_path/'out'/'comparison.json').is_file()
    with pytest.raises(ValueError):run_audit({'obs':[masks()]},{'bad':{}},tmp_path/'bad')


def test_detector_checkpoint_and_real_cpu_inference(tmp_path):
    import torch
    from solar.v6_detector import build_detector, save_checkpoint, load_checkpoint, infer
    model = build_detector(pretrained=False,min_size=64,max_size=64)
    save_checkpoint(model,tmp_path/'detector.pt',{'min_size':64,'max_size':64,'seed':42})
    restored,config = load_checkpoint(tmp_path/'detector.pt','cpu')
    assert config['seed'] == 42
    restored.roi_heads.detections_per_img=2
    restored.rpn._pre_nms_top_n={'training':20,'testing':20}
    restored.rpn._post_nms_top_n={'training':10,'testing':10}
    predictions,scores = infer(restored,np.full((64,64),100,np.uint8),score_threshold=0.)
    assert predictions.shape[1:] == (64,64)
    assert len(scores) == len(predictions)
    assert predictions.dtype == bool


@pytest.mark.skipif(__import__('torch').cuda.is_available() is False,reason='CUDA unavailable')
def test_gpu_detector_smoke():
    from solar.v6_detector import build_detector,infer
    model=build_detector(pretrained=False,min_size=64,max_size=64).cuda()
    pred,scores=infer(model,np.zeros((64,64),np.uint8),score_threshold=.9)
    assert pred.shape[1:] == (64,64)


def test_detector_dataset_maps_annotation_and_empty_targets(tmp_path):
    import cv2
    from solar.v6_train import DetectionDataset
    path=tmp_path/'20260101120000Bh.jpg'
    cv2.imwrite(str(path),np.full((64,64),100,np.uint8))
    records=[{'stem':path.stem,'path':str(path),'image_id':'annotator-a',
              'ann':[{'segmentation':[[4,4,20,4,20,20,4,20]]}]}]
    image,target=DetectionDataset(records)[0]
    assert image.shape == (3,64,64) and target['masks'].shape == (1,64,64)
    assert target['boxes'].tolist() == [[4.,4.,20.,20.]]
    records[0]['ann']=[]
    _,target=DetectionDataset(records)[0]
    assert target['boxes'].shape == (0,4) and target['masks'].shape == (0,64,64)


def test_v6_notebook_self_contained_and_private_metadata(tmp_path):
    import ast,json,subprocess,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'scripts/build_v6_notebook.py'),'--output',str(tmp_path/'v6.ipynb')],check=True,cwd=root)
    notebook=json.loads((tmp_path/'v6.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    metadata=json.loads((tmp_path/'kernel-metadata.json').read_text())
    assert metadata['is_private'] is True
    assert metadata['competition_sources']==['filament-segmentation-2026']
    assert metadata['id']=='minahussain111/solar-filament-research-v6'
    assert 'RUN_TRAINING = False' in ''.join(''.join(c['source']) for c in notebook['cells'])


def test_thin_instances_remain_in_training_and_evaluation_targets(tmp_path):
    import cv2
    from solar.v6_train import DetectionDataset
    path=tmp_path/'thin.png';cv2.imwrite(str(path),np.zeros((32,32),np.uint8))
    record={'path':str(path),'stem':'20260101120000Bh','image_id':'thin',
            'ann':[{'segmentation':[[4,4,5,4,5,20,4,20]]}]}
    _,target=DetectionDataset([record])[0]
    assert len(target['masks']) == 1 and target['masks'].sum() == 16


def test_checkpoint_config_normalizes_torch_version_to_safe_primitives(tmp_path):
    import torch
    from solar.v6_detector import save_checkpoint
    model=torch.nn.Linear(2,2)
    path=tmp_path/'safe.pt'
    save_checkpoint(model,path,{'torch':torch.__version__,'min_size':64,'max_size':64,'seed':42})
    checkpoint=torch.load(path,weights_only=True)
    assert type(checkpoint['config']['torch']) is str


def test_public_reference_pq_matches_downloaded_organizer_functions():
    """Execute only cited metric helpers, never data loading or notebook cells."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    text=(Path(__file__).resolve().parents[1]/'reports/OFFICIAL_EVALUATOR_EXCERPTS.md').read_text()
    source=text.split('```python\n',1)[1].rsplit('```',1)[0]
    tree=ast.parse(source)
    names={'get_pq_score','fp_count_hit','fn_count_hit'}
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    namespace={'torch':torch}
    exec('from __future__ import annotations\n'+ast.unparse(ast.Module(body=selected,type_ignores=[])),namespace)
    class Table:
        def __init__(self,gt,pred):
            matrix=audit.pairwise(gt,pred)[3]
            self.row=SimpleNamespace(iou_matrix=torch.tensor(matrix),n_gt=len(gt),n_pred=len(pred))
        def itertuples(self,index=False):return iter([self.row])
    a=masks()[:1]; empty=a[:0]
    for gt,pred in [(a,a),(a,empty),(empty,a),(empty,empty),(a,np.concatenate([a,a]))]:
        expected=namespace['get_pq_score'](Table(gt,pred))
        assert audit.measure(gt,pred)['pq_reference'] == pytest.approx(expected)
    result=audit.measure(a,np.concatenate([a,a]))
    assert result['pq_reference'] == 1 and result['pq_local'] == pytest.approx(2/3)
    assert audit.measure(empty,empty)['pq_reference']==0

    # Aggregation must match the public helper across unequal annotation counts.
    from solar.v6_forensics import run_audit
    from pycocotools import mask as coco
    import tempfile
    encoded=[coco.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in masks()]
    encoded=[{'size':r['size'],'counts':r['counts'].decode()} for r in encoded]
    truth={'x':[masks(),a], 'y':[masks()]}
    predictions={'detector':{'x':encoded,'y':[]}}
    tables=[Table(masks(),masks()).row,Table(a,masks()).row,Table(masks(),empty).row]
    combined=SimpleNamespace(itertuples=lambda index=False: iter(tables))
    with tempfile.TemporaryDirectory() as out:
        report=run_audit(truth,predictions,out)
    assert report['detector']['public_reference_micro']['pq_reference'] == pytest.approx(namespace['get_pq_score'](combined))


def test_v5_any_overlap_merges_distinct_instances_but_v6_nms_keeps_them():
    """Synthetic regression, not evidence of measured solar accuracy gains."""
    import ast,json
    from pathlib import Path
    from pycocotools import mask as coco
    from solar.v6_instances import mask_nms
    source=json.loads((Path(__file__).resolve().parents[1]/'notebooks/baselines/v5_version5.ipynb').read_text())
    code=next(''.join(c['source']) for c in source['cells'] if 'def merge_overlaps(' in ''.join(c['source']))
    if code.startswith('%%writefile'):code=code.split('\n',1)[1]
    nodes=[node for node in ast.parse(code).body if isinstance(node,ast.FunctionDef) and node.name in {'encode_mask','merge_overlaps'}]
    namespace={'np':np,'mask_utils':coco}
    exec(ast.unparse(ast.Module(body=nodes,type_ignores=[])),namespace)
    gt=masks()
    merged=np.stack(namespace['merge_overlaps'](gt))
    assert len(merged)==1
    kept=gt[mask_nms(gt,[.9,.8],iou_threshold=.5)]
    assert len(kept)==2
    assert audit.measure(gt,kept)['pq_reference']==1
    assert audit.measure(gt,merged)['pq_reference']<1


def test_rle_metrics_match_independent_binary_masks():
    from pycocotools import mask as coco
    gt=masks(); pred=gt[:1]
    encoded=lambda stack:[coco.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in stack]
    expected=audit.measure(gt,pred)
    actual=audit.measure_rles(encoded(gt),encoded(pred),shape=(16,16))
    for key in ['pq_reference','pq_local','reference_tp','reference_fp','reference_fn','likely_splits','likely_merges']:
        assert actual[key]==pytest.approx(expected[key])
    assert audit.measure_rles([],[],shape=(16,16))['pq_reference']==0


def test_cached_prediction_pickle_rejects_objects(tmp_path):
    import pickle
    from pathlib import Path
    from scripts.replay_v5_cached import load_predictions
    good=tmp_path/'good.pkl'
    good.write_bytes(pickle.dumps({'obs':[{'size':[2048,2048],'counts':b'PPPP4'}]}))
    assert 'obs' in load_predictions(good)
    bad=tmp_path/'bad.pkl';bad.write_bytes(pickle.dumps(Path('/tmp/example')))
    with pytest.raises(pickle.UnpicklingError):load_predictions(bad)


def test_area_search_excludes_training_dates_and_confirms_on_separate_dates():
    from solar.v6_cached_search import search_area
    from pycocotools import mask as coco
    encode=lambda mask:coco.encode(np.asfortranarray(mask.astype(np.uint8)))
    gt=masks()[:1];noise=np.zeros_like(gt[0]);noise[0,0]=1
    truth={f'202601{day:02d}130000Bh':[[encode(gt[0])]] for day in range(1,9)}
    predictions={stem:[encode(gt[0]),encode(noise)] for stem in truth}
    report=search_area(truth,predictions,{'20260101120000Bh'},thresholds=[0,2,1000],shape=(16,16))
    assert report['excluded_observations_sharing_training_dates']==1
    assert report['selected_min_area']==2
    assert report['tuning_date_overlap_confirmation']==0
    assert report['confirmation']['candidate_pq']==1
    assert report['confirmation']['baseline_pq']==pytest.approx(2/3)
    assert report['confirmation']['paired_macro_delta_ci']['low']>0
    assert report['eligible_for_official_submission'] is False


def test_archived_v5_refiner_helpers_load_without_training_or_downloads():
    import torch
    from pathlib import Path
    from solar.v6_refiner_audit import load_components,postprocess
    from pycocotools import mask as coco
    source=Path(__file__).resolve().parents[1]/'notebooks/baselines/v5_training_session_v3.ipynb'
    helpers=load_components(source)
    model=helpers['CropUNet'](pretrained=False).eval()
    with torch.inference_mode():output=model(torch.zeros(1,3,64,64))
    assert output.shape==(1,1,64,64) and torch.isfinite(output).all()
    rles=[coco.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks()]
    assert len(postprocess(rles,[.9,.8],'union'))==1
    assert len(postprocess(rles,[.9,.8],'nms'))==2
    assert len(postprocess(rles,[.9,.8],'none'))==2


def test_error_profile_distinguishes_missed_thin_objects_and_false_positives():
    from solar.v6_diagnostics import instance_profile
    from pycocotools import mask as coco
    thin=np.zeros((32,32),np.uint8);thin[1,1:21]=1
    compact=np.zeros_like(thin);compact[10:20,10:20]=1
    noise=np.zeros_like(thin);noise[30,30]=1
    enc=lambda m:coco.encode(np.asfortranarray(m))
    profile=instance_profile([enc(thin),enc(compact)],[enc(compact),enc(noise)])
    assert profile['truth']['elongation_ge_5']=={'total':1,'matched':0}
    assert profile['truth']['all']=={'total':2,'matched':1}
    assert profile['prediction']['all']=={'total':2,'matched':1}
    assert profile['unmatched_truth_best_iou']==[0.]
    empty=instance_profile([],[])
    assert empty['truth']['all']=={'total':0,'matched':0}
    assert empty['unmatched_truth_best_iou']==[]
