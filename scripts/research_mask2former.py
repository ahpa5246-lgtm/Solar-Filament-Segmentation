"""Read public model license/config before authorizing alternative weights."""
import hashlib
import json
from pathlib import Path
import urllib.request
MODEL='facebook/mask2former-swin-tiny-coco-instance'


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'solar-filament-research/1.0'}),timeout=45) as response:return response.read()


def main():
    report={'model':MODEL,'library':'transformers==4.49.0','library_license':'Apache-2.0','training':False}
    try:
        metadata=json.loads(get('https://huggingface.co/api/models/'+MODEL))
        revision=metadata['sha'];license=metadata.get('cardData',{}).get('license')
        report['observed_license']=license
        report['card_license_fields']={k:v for k,v in metadata.get('cardData',{}).items() if 'license' in k.lower()}
        card=get(f'https://huggingface.co/{MODEL}/raw/{revision}/README.md')
        report['card_header']=card.decode('utf-8').split('---',2)[1][:3000] if card.startswith(b'---') else ''
        if license=='other':
            url='https://raw.githubusercontent.com/facebookresearch/Mask2Former/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/MODEL_ZOO.md'
            terms=get(url)
            if b'All models available for download through this document are licensed under' not in terms or b'creativecommons.org/licenses/by-nc/4.0/' not in terms:raise RuntimeError('Upstream weights terms could not be verified')
            license='cc-by-nc-4.0';report['weights_terms_url']=url;report['weights_terms_sha256']=hashlib.sha256(terms).hexdigest()
            report['usage_scope']='Noncommercial research; weights retain CC-BY-NC-4.0, distinct from MIT architecture code'
        if license not in ['mit','apache-2.0','cc-by-nc-4.0']:raise RuntimeError('Model license not verified as compatible; observed fields recorded')
        config=json.loads(get(f'https://huggingface.co/{MODEL}/resolve/{revision}/config.json'))
        files=[f['rfilename'] for f in metadata.get('siblings',[])]
        report['use_safetensors']='model.safetensors' in files
        if not report['use_safetensors'] and 'pytorch_model.bin' not in files:raise RuntimeError('No standard safe-loadable PyTorch weights at reviewed revision')
        report.update(status='verified',revision=revision,model_license=license,model_card_sha256=hashlib.sha256(card).hexdigest(),
                      architecture=config.get('architectures'),num_queries=config.get('num_queries'),
                      model_card_url=f'https://huggingface.co/{MODEL}/blob/{revision}/README.md',
                      pretrained_data='COCO instance segmentation; not competition labels')
    except Exception as error:
        report.update(status='blocked',error_type=type(error).__name__,http_status=getattr(error,'code',None))
        if type(error) is RuntimeError:report['diagnosis']=str(error)
    Path('reports/MASK2FORMER_SOURCE.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='verified':raise SystemExit(1)


if __name__=='__main__':main()
