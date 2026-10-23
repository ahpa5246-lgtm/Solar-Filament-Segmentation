"""CPU-only replay of known V5 cached RLE predictions. Never loads model pickles.

Only aggregate metrics/configuration are committed; annotations, cached masks,
logs, and signed download URLs remain in temporary Actions storage.
"""
import csv
import hashlib
import io
import json
import pickle
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as coco
from solar.v6_audit import measure_rles,bootstrap


V3_ARTIFACT_SHA256={
    'training_manifest.csv':'356bd788d677819a81922f132b0d65b6a709e773ac0f4f8f2a28a2369e014f21',
    'best_detector_predictions_1024.pkl':'69816e2853e28f4464cf2df3cd89ffa1c4a471d2f7e68b5e91be3d5dbfc10eef',
    'refined_validation_predictions_2048.pkl':'139c6b348eac9295404665ae36872ad3b3251b026580910b6dbb639963670c9c'}


def verify_artifacts(paths,expected):
    if set(paths)!=set(expected):raise ValueError('Artifact fingerprint inventory differs')
    actual={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
    if actual!=expected:raise ValueError('Artifact fingerprint differs from preserved V5 training session v3')
    return actual


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        raise pickle.UnpicklingError('Only primitive RLE containers are accepted')


def load_predictions(path):
    if path.stat().st_size>64*1024*1024: raise ValueError('Prediction cache exceeds 64 MiB budget')
    value=DataOnlyUnpickler(io.BytesIO(path.read_bytes())).load()
    if not isinstance(value,dict):raise ValueError('Expected observation-to-RLE mapping')
    for stem,masks in value.items():
        if not isinstance(stem,str) or not isinstance(masks,list):raise ValueError('Invalid prediction mapping')
        for rle in masks:
            if not isinstance(rle,dict) or set(rle)!={'size','counts'}:raise ValueError('Invalid COCO RLE')
            if not isinstance(rle['counts'],(str,bytes)) or list(rle['size']) not in [[1024,1024],[2048,2048]]:raise ValueError('Invalid cached mask')
    return value


def aggregate(groups):
    rows=[r for variants in groups.values() for r in variants]
    tp,fp,fn=(sum(row[k] for row in rows) for k in ['reference_tp','reference_fp','reference_fn'])
    denominator=tp+.5*(fp+fn)
    local_tp,local_fp,local_fn=(sum(row[k] for row in rows) for k in ['tp','fp','fn'])
    local_denominator=local_tp+.5*(local_fp+local_fn)
    matched=[value for row in rows for value in row['matched_iou']]
    return {'reference_micro_pq':sum(row['reference_sum_iou'] for row in rows)/denominator if denominator else 0.,
            'reference_tp':tp,'reference_fp':fp,'reference_fn':fn,
            'canonical_micro_pq':sum(row['sum_iou'] for row in rows)/local_denominator if local_denominator else 1.,
            'canonical_tp':local_tp,'canonical_fp':local_fp,'canonical_fn':local_fn,
            'canonical_sq':sum(row['sum_iou'] for row in rows)/local_tp if local_tp else 0.,
            'canonical_rq':local_tp/local_denominator if local_denominator else 0.,
            'canonical_precision':local_tp/(local_tp+local_fp) if local_tp+local_fp else 0.,
            'canonical_recall':local_tp/(local_tp+local_fn) if local_tp+local_fn else 0.,
            'likely_splits':sum(row['likely_splits'] for row in rows),
            'likely_merges':sum(row['likely_merges'] for row in rows),
            'matched_iou_quantiles':np.quantile(matched,[0,.25,.5,.75,1]).tolist() if matched else [],
            'reference_macro_observation_ci':bootstrap(groups,metric='pq_reference'),
            'physical_observations':len(groups),'annotation_variants':len(rows)}


def replay(annotation_path,manifest_path,detector_path,refined_path):
    fingerprints=verify_artifacts({'training_manifest.csv':manifest_path,
                                   'best_detector_predictions_1024.pkl':detector_path,
                                   'refined_validation_predictions_2048.pkl':refined_path},V3_ARTIFACT_SHA256)
    payload=json.loads(annotation_path.read_text())
    manifest=list(csv.DictReader(manifest_path.open()))
    train={r['observation_id'] for r in manifest if r['split']=='train'}
    val={r['observation_id'] for r in manifest if r['split']=='val'}
    if not train or not val or train&val:raise ValueError('Invalid original split')
    by_image=defaultdict(list)
    for annotation in payload['annotations']:by_image[str(annotation['image_id'])].append(annotation)
    truth=defaultdict(list)
    for image in payload['images']:
        stem=Path(image['file_name']).stem
        if stem not in val:continue
        if (image['height'],image['width'])!=(2048,2048):raise ValueError('Unexpected annotation dimensions')
        rles=[]
        for annotation in by_image[str(image['id'])]:
            parts=coco.frPyObjects(annotation['segmentation'],2048,2048)
            if parts:
                rle=coco.merge(parts)
                if coco.area(rle)>0:rles.append(rle)
        truth[stem].append(rles)
    detector=load_predictions(detector_path);refined=load_predictions(refined_path)
    if set(detector)!=val or set(refined)!=val or set(truth)!=val:raise ValueError('Cached prediction/holdout inventory differs')
    results={}
    for strategy,predictions in [('detector',detector),('detector_refiner_plain',refined)]:
        groups={}
        for stem,variants in truth.items():
            native=[]
            for rle in predictions[stem]:
                if list(rle['size'])==[1024,1024]:
                    mask=cv2.resize(coco.decode(rle),(2048,2048),interpolation=cv2.INTER_NEAREST)
                    rle=coco.encode(np.asfortranarray(mask))
                native.append(rle)
            groups[stem]=[measure_rles(gt,native) for gt in variants]
        results[strategy]=aggregate(groups)
    from solar.v6_cached_search import search_area
    area_experiment=search_area(truth,refined,train)
    return {'artifact_sha256':fingerprints,'annotation_sha256':hashlib.sha256(annotation_path.read_bytes()).hexdigest(),'area_ablation':area_experiment,'artifact_kernel_version':'minahussain111/solar-filament-research-v5/v3',
            'score_status':'cached LOCAL tuning-validation replay, not an official score; association to Public 0.32 not independently established',
            'matching':'both organizer-public all-pairs and canonical one-to-one',
            'split':{'train_observations':len(train),'validation_observations':len(val),
                     'physical_observation_overlap':len(train&val),
                     'overlapping_acquisition_dates':len({s[:8] for s in train}&{s[:8] for s in val})},
            'all_annotation_results':results,
            'limitations':['Same validation set was used for V5 tuning; not independent confirmation.',
                           'Cached refiner predictions are plain mode only; flip4 not reconstructed.',
                           'Masks have already undergone V5 overlap union; unmerged masks are unavailable.',
                           'No new training, GPU inference, official submission, or score improvement.']}


def main():
    import tempfile
    from kaggle.api.kaggle_api_extended import KaggleApi
    report_path=Path('reports/V5_CACHED_REPLAY.json')
    report={}
    try:
        api=KaggleApi();api.authenticate()
        with tempfile.TemporaryDirectory(prefix='v5-cached-') as tmp:
            root=Path(tmp)
            api.kernels_pull('minahussain111/solar-filament-research-v5/v3',str(root/'source'))
            sources=list((root/'source').glob('*.ipynb'))
            if len(sources)!=1:raise ValueError('Missing training-session source')
            raw=sources[0].read_bytes()
            expected_source='2781c89d49a31a08e4f3b96d39b439e029bc7535fc596bee3d8e074baf9a2bea'
            if hashlib.sha256(raw).hexdigest()!=expected_source:raise ValueError('Training-session source fingerprint changed')
            snapshot=Path('notebooks/baselines/v5_training_session_v3.ipynb')
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            if snapshot.exists() and snapshot.read_bytes()!=raw:raise ValueError('Refusing baseline source overwrite')
            snapshot.write_bytes(raw)
            pattern=r'(training_manifest\.csv|best_detector_predictions_1024\.pkl|refined_validation_predictions_2048\.pkl|detector_thresholds\.json|final_configuration\.json|detector_calibration\.csv)$'
            next_page=None
            for page in range(10):
                _,next_page=api.kernels_output('minahussain111/solar-filament-research-v5/v3',str(root/'output'),file_pattern=pattern,page_token=next_page,page_size=100)
                if not next_page:break
            else:raise RuntimeError('Output inventory pagination budget exceeded')
            desired='MAGFiLO_1.0_Annotations_kaggle2026_train.json'
            annotation_name=None;next_page=None
            for page in range(30):
                listing=api.competition_list_files('filament-segmentation-2026',page_token=next_page,page_size=100)
                matches=[file.name for file in listing.files if Path(file.name).name==desired]
                if len(matches)>1:raise ValueError('Ambiguous official training JSON')
                if matches:annotation_name=matches[0];break
                next_page=getattr(listing,'next_page_token',None) or getattr(listing,'nextPageToken',None)
                if not next_page:break
            if not annotation_name:raise FileNotFoundError('Official training JSON not exposed as an individual competition file')
            api.competition_download_file('filament-segmentation-2026',annotation_name,str(root/'data'),quiet=True,unzip=True)
            def one(name):
                found=list(root.rglob(name))
                if len(found)!=1:raise ValueError(f'Expected one {name}, found {len(found)}')
                return found[0]
            report=replay(one(desired),one('training_manifest.csv'),one('best_detector_predictions_1024.pkl'),one('refined_validation_predictions_2048.pkl'))
            report['selected_configuration']=json.loads(one('final_configuration.json').read_text())
            report['detector_thresholds']=json.loads(one('detector_thresholds.json').read_text())
            try:
                quota=api.quota_view().gpu_quota
                report['gpu_quota']={'total_hours':quota.total_time_allowed.total_seconds()/3600,
                                     'used_hours':quota.time_used.total_seconds()/3600}
            except Exception as error:
                report['gpu_quota']={'status':'unavailable','error_type':type(error).__name__}
    except Exception as error:
        report={'status':'blocked','error_type':type(error).__name__,
                'http_status':getattr(getattr(error,'response',None),'status_code',None)}
        # Controlled invariant errors only; never serialize credential-bearing SDK exceptions.
        if type(error) in (ValueError,FileNotFoundError,RuntimeError,pickle.UnpicklingError):report['diagnosis']=str(error)[:500]
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print('Saved aggregate replay report; raw annotations, predictions and credentials not published.')
    if report.get('status')=='blocked':raise SystemExit(1)


if __name__=='__main__':main()
