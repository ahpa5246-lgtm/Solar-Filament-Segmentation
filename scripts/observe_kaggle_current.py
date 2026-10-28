"""Read-only Kaggle evidence; never launches a kernel or submits predictions."""
import json, re
from datetime import datetime, timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

api=KaggleApi(); api.authenticate()
ref='minahussain111/solar-v5-finetune-pq50-20261028/v1'
response=api.kernels_status(ref)
report={'observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'reference':ref,'kaggle_status':str(response.status),
        'training_launched':False,'submission_created':False}
raw=api.kernels_logs(ref)
try: events=json.loads(raw) if isinstance(raw,str) else raw
except ValueError: events=[]
text='\n'.join(str(e.get('data','')) for e in events if isinstance(e,dict)) if isinstance(events,list) else ''
# Persist controlled progress and exception lines only. Never dump raw logs.
prefixes=('V5 baseline','V5 finetune epoch','V5_TRAINING_EPOCH','V5_FINETUNE_RESULT',
          'RuntimeError:','ValueError:','TypeError:','KeyError:','ImportError:',
          'ModuleNotFoundError:','AssertionError:','TimeoutError:')
report['execution_evidence']=[line[:24000] for line in text.splitlines() if line.startswith(prefixes)][-30:]
rows=api.competition_submissions('filament-segmentation-2026') or []
report['official_submissions']=[{'ref':int(r.ref),'status':str(r.status),
    'public_score':str(r.public_score),'description':str(r.description),'date':str(r.date)}
    for r in rows[:20]]
path=Path('reports/KAGGLE_CURRENT_EVIDENCE.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
