"""Cheap cached-mask ablation with a fixed tuning/confirmation date partition.

Post-hoc confirmation is independent of this threshold search, but V5 itself
already used its complete old validation set for selection. No official score.
"""
import numpy as np
from pycocotools import mask as coco
from .v6_audit import measure_rles


def micro(groups):
    rows=[row for variants in groups.values() for row in variants]
    tp,fp,fn=(sum(row[key] for row in rows) for key in ['reference_tp','reference_fp','reference_fn'])
    denominator=tp+.5*(fp+fn)
    return sum(row['reference_sum_iou'] for row in rows)/denominator if denominator else 0.


def evaluate(truth,predictions,stems,area,shape):
    return {stem:[measure_rles(gt,[r for r in predictions[stem] if coco.area(r)>=area],shape) for gt in truth[stem]] for stem in stems}


def paired_ci(baseline,candidate,seed=42,n=2000):
    if set(baseline)!=set(candidate) or not baseline:raise ValueError('Need paired observation results')
    deltas=np.array([np.mean([r['pq_reference'] for r in candidate[stem]])-np.mean([r['pq_reference'] for r in baseline[stem]]) for stem in sorted(baseline)])
    rng=np.random.default_rng(seed)
    samples=[float(rng.choice(deltas,len(deltas),replace=True).mean()) for _ in range(n)]
    low,high=np.quantile(samples,[.025,.975])
    return {'estimate':float(deltas.mean()),'low':float(low),'high':float(high),'unit':'physical observation','replicates':n,'seed':seed}


def search_area(truth,predictions,training_stems,thresholds=(0,32,64,128,256,512,1024,2048),seed=42,shape=(2048,2048)):
    if set(truth)!=set(predictions) or set(truth)&set(training_stems):raise ValueError('Prediction inventory or physical split differs')
    thresholds=sorted(set(thresholds))
    if not thresholds or thresholds[0]!=0 or any(not np.isfinite(x) or x<0 for x in thresholds):raise ValueError('Area grid must include zero and nonnegative finite thresholds')
    train_dates={stem[:8] for stem in training_stems}
    eligible=sorted(stem for stem in truth if stem[:8] not in train_dates)
    dates=sorted({stem[:8] for stem in eligible})
    if len(dates)<4:return {'status':'insufficient disjoint validation dates','eligible_dates':len(dates)}
    np.random.default_rng(seed).shuffle(dates)
    tuning_dates=set(dates[:len(dates)//2]);confirmation_dates=set(dates[len(dates)//2:])
    tuning=sorted(stem for stem in eligible if stem[:8] in tuning_dates)
    confirmation=sorted(stem for stem in eligible if stem[:8] in confirmation_dates)
    trials=[]
    for area in thresholds:
        groups=evaluate(truth,predictions,tuning,area,shape)
        trials.append({'min_area':area,'tuning_reference_micro_pq':micro(groups)})
    # Lower area wins ties: preserve fine structures unless evidence favors filtering.
    selected=max(trials,key=lambda row:(row['tuning_reference_micro_pq'],-row['min_area']))['min_area']
    baseline=evaluate(truth,predictions,confirmation,0,shape)
    candidate=evaluate(truth,predictions,confirmation,selected,shape)
    delta=paired_ci(baseline,candidate,seed)
    base_score=micro(baseline);candidate_score=micro(candidate)
    return {'status':'completed','experiment':'minimum native-mask area; cached plain refiner only',
            'seed':seed,'excluded_observations_sharing_training_dates':len(truth)-len(eligible),
            'tuning_observations':len(tuning),'confirmation_observations':len(confirmation),
            'tuning_dates':len(tuning_dates),'confirmation_dates':len(confirmation_dates),
            'tuning_date_overlap_confirmation':len(tuning_dates&confirmation_dates),
            'tuning_grid':trials,'selected_min_area':selected,
            'confirmation':{'baseline_pq':base_score,'candidate_pq':candidate_score,
                            'micro_delta':candidate_score-base_score,'paired_macro_delta_ci':delta},
            'confirmation_gain_supported':candidate_score>base_score and delta['low']>0,
            'eligible_for_official_submission':False,
            'limitations':['Final saved test predictions used flip4, whereas this cache is plain-refiner mode.',
                           'A matched flip4 validation replay is required before applying to the saved test predictions.',
                           'Original V5 model/strategy selection used all original validation observations.']}
