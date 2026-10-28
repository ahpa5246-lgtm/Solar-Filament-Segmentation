"""Continue the exact archived Apache-attributed V5 detector; refiner stays fixed."""
import math

EPOCHS = 16
LEARNING_RATE = 1e-5


def training_allowed(state):
    return not state.get('push_attempted') and not state.get('version') and state.get('status') != 'COMPLETE'


def better_checkpoint(candidate, incumbent):
    for value in (candidate, incumbent):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('PQ must be finite and in [0,1]')
    return candidate > incumbent


def validate_partitions(train, tuning, confirmation):
    groups = [set(s[:8] for s in part) for part in (train, tuning, confirmation)]
    if not all(groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Training, tuning and confirmation dates must be nonempty and disjoint')


def safe_scaled_step(scaler, optimizer, parameters, max_norm=5.):
    """Never update weights from nonfinite gradients; recover AMP scale safely."""
    import torch
    parameters=list(parameters)
    scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(parameters,max_norm,error_if_nonfinite=False)
    finite=bool(torch.isfinite(norm))
    if finite:
        scaler.step(optimizer)
        scaler.update()
    else:
        # Explicit skip also covers norm overflow with individually finite gradients.
        scaler.update(new_scale=scaler.get_scale()*.5)
    optimizer.zero_grad(set_to_none=True)
    return finite


def run(annotation, manifest, detector_cache, plain_cache, detector_checkpoint,
        refiner_checkpoint, source, out, seconds=6600, hard_negative=False):
    import ast, csv, hashlib, json, random, shutil, time
    from pathlib import Path
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from pycocotools.coco import COCO
    from .v6_resolution_audit import original_detector, select_pilot_dates, detect, refine, DETECTOR_SHA
    from .v6_refiner_audit import load_components, postprocess, references, evaluate, REFINER_SHA, SOURCE_SHA
    from .v5_replay import verify_artifacts, V3_ARTIFACT_SHA256, load_predictions, aggregate
    from .v6_cached_search import micro, paired_ci
    from .v6_audit import measure_rles

    epochs = 4 if hard_negative else EPOCHS
    learning_rate = 2e-6 if hard_negative else LEARNING_RATE
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    report = dict(status='STARTED', training=True, test_inference=False, official_submission=False,
                  target_official_pq=.50, architecture='unchanged V5 Mask R-CNN ResNet50-FPN-v2 + fixed CropUNet ResNet18',
                  epochs_requested=epochs, learning_rate=learning_rate,
                  sampling='fitting-only baseline false-positive weighted sampling' if hard_negative else 'original uniform shuffled fitting',
                  hypothesis='Hard-negative upweighting plus smaller LR may improve V5 proposal precision without 16-epoch regression' if hard_negative else 'original V5 continuation',
                  selection='full detector/refiner tuning micro PQ; baseline included at epoch zero',
                  limitations=['Historically reused V5 validation; not a pristine holdout.',
                               'API-v3 validation weights are attested; exact scored-web-V5 checkpoint identity remains unresolved.',
                               'Validation checkpoints precede final full-data fitting; local PQ is not official PQ.'])
    def save_report():
        pending=out/'finetune_result.pending'
        pending.write_text(json.dumps(report, indent=2)+'\n')
        pending.replace(out/'finetune_result.json')
    save_report()
    started=time.perf_counter()
    random.seed(2028); np.random.seed(2028); torch.manual_seed(2028)
    torch.set_num_threads(2); cv2.setNumThreads(0)
    try:
        fingerprints=verify_artifacts({'training_manifest.csv':Path(manifest),
                          'best_detector_predictions_1024.pkl':Path(detector_cache),
                          'refined_validation_predictions_2048.pkl':Path(plain_cache)},V3_ARTIFACT_SHA256)
        report['input_artifact_sha256']=fingerprints
        for path, expected in [(source,SOURCE_SHA),(detector_checkpoint,DETECTOR_SHA),(refiner_checkpoint,REFINER_SHA)]:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise ValueError('Preserved V5 source/checkpoint fingerprint mismatch')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        rows=list(csv.DictReader(Path(manifest).open()))
        fitting=[r for r in rows if r['split']=='train']
        training=[r['observation_id'] for r in fitting]
        validation={r['observation_id']:r for r in rows if r['split']=='val'}
        tuning, confirmation=select_pilot_dates(training,validation,None)
        validate_partitions(training,tuning,confirmation)
        report.update(split_counts={'train':len(training),'tuning':len(tuning),'confirmation':len(confirmation)},
                      gpu=torch.cuda.get_device_name(0),visible_gpus=torch.cuda.device_count(),used_gpus=1)
        (out/'split.json').write_text(json.dumps({'train':training,'tuning':tuning,'confirmation':confirmation}))
        truth=references(annotation,set(tuning+confirmation))
        images=Path(annotation).parent/'train_images'
        cached=load_predictions(Path(detector_cache))
        model=original_detector(source)
        model.load_state_dict(torch.load(detector_checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
        model=model.cuda()
        helpers=load_components(source)
        refiner=helpers['CropUNet'](pretrained=False)
        refiner.load_state_dict(torch.load(refiner_checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
        refiner=refiner.cuda().eval()
        for p in refiner.parameters(): p.requires_grad_(False)
        def assess(stems, check_parity=False):
            model.eval(); predictions={}; parity=[]
            for stem in stems:
                if time.perf_counter()-started>seconds:
                    raise TimeoutError('Assessment budget exhausted; checkpoints preserved, confirmation incomplete')
                image=cv2.imread(str(images/validation[stem]['file_name']),cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape!=(2048,2048): raise ValueError('Native validation image missing')
                masks,scores=detect(model,image,1024)
                proposals=postprocess(masks,scores,'union')
                if check_parity:
                    parity.append(measure_rles(cached[stem],proposals,shape=(1024,1024))['pq_local'])
                    if parity[-1]<.99: raise RuntimeError('Epoch-zero V5 cache parity failed')
                predictions[stem]=refine(refiner,helpers,image,proposals)
            return evaluate(truth,predictions,stems), min(parity) if parity else None
        baseline_tuning, parity=assess(tuning,True)
        best=micro(baseline_tuning); best_epoch=0
        shutil.copyfile(detector_checkpoint,out/'best_detector.pt')
        report.update(status='TRAINING',baseline_tuning_pq=best,baseline_parity_min=parity,
                      best_tuning_pq=best,best_epoch=0,history=[])
        save_report()
        print('V5 baseline attested; continuing trained detector, tuning PQ:',best,flush=True)
        # Reuse the original open-source dataset/augmentation implementation, not notebook side effects.
        archived=json.loads(Path(source).read_text())
        code=next(''.join(c['source']).split('\n',1)[1] if isinstance(c['source'],list) else c['source'].split('\n',1)[1]
                  for c in archived['cells'] if c['cell_type']=='code' and
                  (''.join(c['source']) if isinstance(c['source'],list) else c['source']).startswith('%%writefile'))
        definitions=[n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in {'masks_to_boxes','FilamentDataset'}]
        if len(definitions)!=2: raise RuntimeError('Archived dataset definitions drifted')
        namespace=dict(np=np,cv2=cv2,torch=torch,COCO=COCO,Dataset=torch.utils.data.Dataset,
                       ANN_PATH=Path(annotation),IMAGE_DIR=images,IMAGE_SIZE=1024)
        exec(compile(ast.Module(body=definitions,type_ignores=[]),'archived_apache_v5_dataset','exec'),namespace)
        dataset=namespace['FilamentDataset'](pd.DataFrame(fitting),augment=True)
        if len(dataset) != len(fitting):
            raise ValueError('Dataset and fitting-row inventories differ')
        sampler = None
        if hard_negative:
            from .v5_hardnegative import sampling_weights, count_unmatched_predictions
            from collections import Counter
            fitting_truth = references(annotation, set(training))
            if set(fitting_truth) != set(training):
                raise ValueError('Hard-negative mining inventory differs from fitting-only observations')
            false_positives = {}
            model.eval()
            for index, row in enumerate(fitting, 1):
                if time.perf_counter()-started > seconds-2500:
                    raise TimeoutError('Hard-negative mining exceeded reserved training budget')
                stem = row['observation_id']
                image = cv2.imread(str(images / row['file_name']), cv2.IMREAD_GRAYSCALE)
                if image is None or image.shape != (2048,2048):
                    raise ValueError('Missing fitting image for hard-negative mining')
                masks, scores = detect(model, image, 1024)
                proposals = postprocess(masks, scores, 'union')
                false_positives[stem] = count_unmatched_predictions(proposals, fitting_truth[stem])
                if index % 100 == 0:
                    print('V5 hard-negative fitting scan', index, '/', len(fitting), flush=True)
            weights = sampling_weights(training, false_positives)
            if len(set(weights)) < 2:
                raise RuntimeError('No variable train-only false-positive weights; refuse redundant training')
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.tensor(weights, dtype=torch.double), num_samples=len(dataset),
                replacement=True, generator=torch.Generator().manual_seed(2031))
            report['hard_negative_mining'] = {
                'source': 'original frozen detector; fitting images and labels only',
                'observations': len(false_positives),
                'with_false_positives': sum(v > 0 for v in false_positives.values()),
                'total_unmatched_proposals': sum(false_positives.values()),
                'weight_min': min(weights), 'weight_max': max(weights),
                'confirmation_or_tuning_used': False}
            save_report()
            del fitting_truth
        loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=sampler is None,
                    sampler=sampler,num_workers=0,
                    collate_fn=lambda batch: tuple(zip(*batch)),generator=torch.Generator().manual_seed(2028))
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=5e-7 if hard_negative else 1e-6)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.); total_steps=0
        successful_updates=0; skipped_updates=0; consecutive_skips=0
        budget_stop=False
        for epoch in range(1,epochs+1):
            if time.perf_counter()-started>seconds-700:
                report['budget_stopped_before_epoch']=epoch; break
            model.train(); losses=[]; optimizer.zero_grad(set_to_none=True)
            for step,(batch,targets,_) in enumerate(loader,1):
                if time.perf_counter()-started>seconds-700:
                    torch.save({'model':model.state_dict(),'epoch':epoch,'partial_epoch':True,'optimizer':optimizer.state_dict()},out/'last_detector.pt')
                    report.update(partial_epoch=epoch,gradient_samples=total_steps)
                    save_report(); budget_stop=True; break
                batch=[im.cuda() for im in batch]
                targets=[{k:v.cuda() for k,v in target.items()} for target in targets]
                with torch.autocast('cuda',dtype=torch.float16):
                    loss=sum(model(batch,targets).values())
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite detector training loss')
                # Average the actual accumulation group, including the final short group.
                group_start=((step-1)//4)*4
                divisor=min(4,len(loader)-group_start)
                scaler.scale(loss/divisor).backward()
                if step%4==0 or step==len(loader):
                    updated=safe_scaled_step(scaler,optimizer,model.parameters())
                    successful_updates+=int(updated); skipped_updates+=int(not updated)
                    consecutive_skips=0 if updated else consecutive_skips+1
                    report.update(optimizer_updates=successful_updates,overflow_skipped_updates=skipped_updates,amp_scale=scaler.get_scale())
                    if consecutive_skips>=20:
                        raise RuntimeError('Persistent gradient overflow after 20 scale reductions; stop safely')
                losses.append(float(loss.detach())); total_steps+=1
                if step%100==0: print('V5 finetune epoch',epoch,'step',step,'/',len(loader),flush=True)
            if budget_stop: break
            if not successful_updates: raise RuntimeError('No finite optimizer updates')
            scheduler.step()
            row={'epoch':epoch,'loss':sum(losses)/len(losses),'learning_rate':optimizer.param_groups[0]['lr']}
            torch.save({'model':model.state_dict(),'epoch':epoch,'optimizer':optimizer.state_dict()},out/'last_detector.pt')
            if hard_negative or epoch%2==0 or epoch==epochs:
                groups,_=assess(tuning)
                row['tuning_pipeline_pq']=micro(groups)
                if better_checkpoint(row['tuning_pipeline_pq'],best):
                    best=row['tuning_pipeline_pq']; best_epoch=epoch
                    torch.save({'model':model.state_dict(),'epoch':epoch},out/'best_detector.pt')
            report['history'].append(row)
            report.update(epochs_completed=epoch,gradient_samples=total_steps,best_tuning_pq=best,best_epoch=best_epoch)
            save_report(); print('V5_TRAINING_EPOCH',json.dumps(row),flush=True)
        # Frozen selection: confirmation is read only now, never for epoch choice.
        model.load_state_dict(torch.load(detector_checkpoint,map_location='cuda',weights_only=True)['model'],strict=True)
        baseline_confirmation,_=assess(confirmation)
        model.load_state_dict(torch.load(out/'best_detector.pt',map_location='cuda',weights_only=True)['model'],strict=True)
        selected_confirmation,_=assess(confirmation)
        delta=paired_ci(baseline_confirmation,selected_confirmation)
        supported=micro(selected_confirmation)>micro(baseline_confirmation) and delta['low']>0 and best_epoch>0
        report.update(status='COMPLETE',confirmation_baseline=aggregate(baseline_confirmation),
                      confirmation_candidate=aggregate(selected_confirmation),paired_macro_delta_ci=delta,
                      local_gain_supported=supported,official_public_pq=None,
                      best_checkpoint_sha256=hashlib.sha256((out/'best_detector.pt').read_bytes()).hexdigest(),
                      elapsed_seconds=time.perf_counter()-started,
                      next_action='Review evidence for full-data fitting and one official assessment' if supported else 'Retain V5; diagnose this completed continuation before next experiment')
        save_report(); print('V5_FINETUNE_RESULT',json.dumps(report),flush=True)
    except Exception as error:
        report.update(status='INCOMPLETE' if isinstance(error,TimeoutError) else 'FAILED',error_type=type(error).__name__,elapsed_seconds=time.perf_counter()-started)
        if type(error) in (RuntimeError,ValueError): report['diagnosis']=str(error)[:500]
        save_report(); raise
    return report
