"""Read notebook source only; never execute or print private execution outputs."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def _source(cell):
    source=cell.get('source',[])
    return source if isinstance(source,str) else ''.join(source)


def _redact(source):
    source=re.sub(r'KGAT_[A-Za-z0-9_-]+','[REDACTED]',source)
    return '\n'.join(line for line in source.splitlines()
                     if not re.search(r'(api[_-]?key|api[_-]?token|secret|password|authorization)',line,re.I))


def summarize(v5,official):
    lines=['# V5 Version 5 source audit',
           'Source-only inspection. No execution outputs, scores, checkpoints or solar predictions are printed.',
           'Public official reference: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook',
           '## Official evaluator source excerpts (not executed)']
    for i,cell in enumerate(official.get('cells',[])):
        code=_source(cell)
        if cell.get('cell_type')=='code' and any(word in code.lower() for word in ['def ','panoptic','metric','pq','dice']):
            lines.extend([f'### Official cell {i}','```python',_redact(code)[:24000],'```'])
    lines.append('## V5 source excerpts (not executed)')
    for i,cell in enumerate(v5.get('cells',[])):
        code=_source(cell)
        if cell.get('cell_type')=='code' and any(word in code for word in ['all_dice','merge_overlapping_rles','train_test_split','validation_observation','train_observation','comparison.loc']):
            lines.extend([f'### V5 cell {i}','```python',_redact(code)[:18000],'```'])
    return '\n\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--v5',type=Path,required=True);p.add_argument('--official',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    paths=[]
    for folder in [args.v5,args.official]:
        found=list(folder.glob('*.ipynb'))
        if len(found)!=1:raise ValueError(f'Expected one notebook in {folder}; found {len(found)}')
        paths.append(found[0])
    v5,official=[json.loads(path.read_text()) for path in paths]
    text=summarize(v5,official)
    text+='\n## Downloaded source SHA256\n'+ '\n'.join(f'- {name}: `{hashlib.sha256(path.read_bytes()).hexdigest()}`' for name,path in zip(['V5 Version 5','Official current'],paths))+'\n'
    args.out.write_text(text,encoding='utf-8')


if __name__=='__main__':main()
