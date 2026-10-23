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
