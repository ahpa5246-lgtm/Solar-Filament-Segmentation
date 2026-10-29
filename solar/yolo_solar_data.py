"""COCO-to-YOLO labels with independent instances and training-only negatives."""
import json,hashlib
from pathlib import Path
import cv2,numpy as np
from pycocotools import mask as coco


def polygon_line(segmentation,width,height):
    if len(segmentation)!=1:raise ValueError('Expected one polygon per original instance; do not silently split identities')
    points=np.asarray(segmentation[0],dtype=float)
    if points.ndim!=1 or len(points)<6 or len(points)%2 or not np.isfinite(points).all():raise ValueError('Invalid instance polygon')
    points=points.reshape(-1,2)/np.array([width,height])
    points=np.clip(points,0,1)
    return '0 '+' '.join(f'{x:.7f}' for x in points.reshape(-1))


def choose_negative_crop(image,forbidden,rng,size=512):
    if image.shape!=forbidden.shape:raise ValueError('Image/annotation shape mismatch')
    h,w=image.shape
    if h<size or w<size:return None
    positions=[(x,y) for y in range(0,h-size+1,size//2) for x in range(0,w-size+1,size//2)]
    rng.shuffle(positions)
    for x,y in positions:
        crop=image[y:y+size,x:x+size]
        if not forbidden[y:y+size,x:x+size].any() and (crop>25).mean()>.95 and crop.std()>2:
            return x,y,x+size,y+size
    return None


def build_dataset(annotation,rows,tuning,out):
    import yaml
    annotation=Path(annotation);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=json.loads(annotation.read_text());images={str(r['id']):r for r in data['images']}
    by_id={};by_stem={}
    for ann in data['annotations']:
        by_id.setdefault(str(ann['image_id']),[]).append(ann)
        stem=Path(images[str(ann['image_id'])]['file_name']).stem
        by_stem.setdefault(stem,[]).append(ann)
    fitting=[r for r in rows if r['split']=='train'];tune=[r for r in rows if r['observation_id'] in set(tuning)]
    if len({r['observation_id'] for r in rows})!=len(rows):raise ValueError('Duplicate physical observations')
    report={'fitting':len(fitting),'tuning':len(tune),'all_training':len(rows),'negative_crops':0,'instance_format':'independent polygons; overlap_mask=False required'}
    variants={}
    for image_id,image in images.items():variants.setdefault(Path(image['file_name']).stem,[]).append(image_id)
    report['fitting_annotation_variants']=sum(len(variants[r['observation_id']]) for r in fitting)
    report['all_annotation_variants']=len(images)
    for name,records in (('fitting',fitting),('tuning',tune),('all_training',rows)):
        if name!='tuning':
            expanded=[]
            for row in records:
                for image_id in variants[row['observation_id']]:
                    item=dict(row);item['id']=image_id;item['_variant']=True;expanded.append(item)
            records=expanded
        image_dir=out/'images'/name;label_dir=out/'labels'/name
        image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
        paths=[]
        for row in records:
            record=images[str(row['id'])];stem=row['observation_id']
            if Path(record['file_name']).stem!=stem:raise ValueError('Annotation variant/physical-image mismatch')
            original=annotation.parent/'train_images'/row['file_name']
            if not original.exists():raise FileNotFoundError('Training image missing')
            suffix=('_ann_'+hashlib.sha256(str(row['id']).encode()).hexdigest()[:12]) if row.get('_variant') else ''
            target=image_dir/(original.stem+suffix+original.suffix)
            if not target.exists():target.symlink_to(original)
            labels=[polygon_line(a['segmentation'],record['width'],record['height']) for a in by_id.get(str(row['id']),[])]
            (label_dir/(target.stem+'.txt')).write_text('\n'.join(labels)+'\n' if labels else '')
            paths.append(str(target))
        (out/(name+'.txt')).write_text('\n'.join(paths)+'\n')
    # No competition test observation can enter this path. Negatives use only
    # fitting images and exclude the union of every known annotator's masks.
    rng=np.random.default_rng(2032);negative_paths=[]
    for index,row in enumerate(fitting):
        if index%4:continue
        image=cv2.imread(str(annotation.parent/'train_images'/row['file_name']),cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape!=(2048,2048):raise ValueError('Bad fitting image')
        pieces=[]
        for ann in by_stem[row['observation_id']]:pieces.extend(coco.frPyObjects(ann['segmentation'],2048,2048))
        forbidden=coco.decode(coco.merge(pieces)) if pieces else np.zeros_like(image)
        forbidden=cv2.dilate(forbidden,np.ones((97,97),np.uint8))
        bounds=choose_negative_crop(image,forbidden,rng)
        if bounds is None:continue
        x0,y0,x1,y1=bounds;name=row['observation_id']+'_negative.jpg'
        for split in ('fitting','all_training'):
            target=out/'images'/split/name
            if not cv2.imwrite(str(target),image[y0:y1,x0:x1]):raise OSError('Negative crop write failed')
            (out/'labels'/split/(target.stem+'.txt')).write_text('')
            with (out/(split+'.txt')).open('a') as handle:handle.write(str(target)+'\n')
        negative_paths.append({'observation':row['observation_id'],'bounds':bounds})
    report['negative_crops']=len(negative_paths)
    (out/'negative_crop_provenance.json').write_text(json.dumps(negative_paths))
    for stage,train in (('select','fitting'),('final','all_training')):
        config={'path':str(out),'train':str(out/(train+'.txt')),'val':str(out/'tuning.txt'),'names':{0:'filament'}}
        (out/(stage+'.yaml')).write_text(yaml.safe_dump(config,sort_keys=False))
    (out/'dataset_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return out/'select.yaml',out/'final.yaml',report
