"""Independent open-weight code-model review on the existing private audit notebook."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    notebook=json.loads(next(Path('/workspace/scratch/solar-v7-audit').glob('*.ipynb')).read_text())
    code=''.join(notebook['cells'][1]['source']);tree=ast.parse(code)
    values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SOURCE'}
    driver=values['SOURCE'];sources=ast.literal_eval(ast.parse(driver).body[0].value)
    sections=[]
    names={'original_detector','detect','refine','postprocess','write_rle_submission','run','FilamentDataset','CropDataset','build_model','CropUNet'}
    for path in ('solar/v6_resolution_audit.py','solar/v6_refiner_audit.py','solar/v6_submission.py'):
        sections.append(path+'\n'+'\n'.join(ast.unparse(n) for n in ast.parse(sources[path]).body
                                           if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names))
    sections.append('NEW PROPOSED DETECTOR CONSENSUS:\n'+(ROOT/'solar/detector_consensus.py').read_text())
    source='\n\n'.join(sections)
    prompt='''Audit this solar filament instance-segmentation notebook implementation. Official Public PQ is 0.32. Sixteen extra AdamW epochs lowered training loss but no epoch exceeded the original pipeline tuning PQ. Changing crop-refiner threshold .85 to .65 tied official PQ at .32. Proposal NMS with final no-union, final NMS, and orientation-aware union all reduced local confirmation PQ. Validation is historical and not a pristine holdout. Test labels are never available. New detector consensus is proposed, not yet validated. Identify definite code bugs separately from unproven modeling limitations. Check grayscale scaling, probability versus logits, instance identity, flip inversion, distinct-view consensus, shape/ID/RLE export, overfitting, and proposal recall. Give concrete code references and fixes. Do not promise a score, invent experiments, recommend test labels, or propose merely repeating failed settings. Your output is review advice only and will be checked against code/tests.\n\n'''+source
    code='PROMPT='+repr(prompt)+'\nSOURCE_SHA='+repr(hashlib.sha256(source.encode()).hexdigest())+'''
import gc,json,os,subprocess,sys,time
from pathlib import Path
os.environ['HF_HOME']='/tmp/solar_external_review_cache'
os.environ['HF_HUB_DISABLE_XET']='1'
started=time.perf_counter()
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--quiet','--no-deps','--target','/tmp/solar_external_review_packages','transformers==4.51.3','tokenizers==0.21.4','huggingface-hub==0.30.2','safetensors==0.5.3'],check=True)
sys.path.insert(0,'/tmp/solar_external_review_packages')
import torch
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM,AutoTokenizer
model_id='Qwen/Qwen2.5-Coder-1.5B-Instruct'
revision=model_info(model_id).sha
tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,trust_remote_code=False)
model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,torch_dtype=torch.float16,trust_remote_code=False,use_safetensors=True).cuda().eval()
messages=[{'role':'system','content':'You are an independent ML code reviewer. Distinguish observed code facts from hypotheses.'},{'role':'user','content':PROMPT}]
text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
inputs=tokenizer(text,return_tensors='pt',truncation=True,max_length=12500).to('cuda')
with torch.inference_mode():
 output=model.generate(**inputs,max_new_tokens=1100,do_sample=False,pad_token_id=tokenizer.eos_token_id)
review=tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
result={'status':'COMPLETE','external_model':model_id,'model_revision':revision,'source_sha256':SOURCE_SHA,'input_tokens':int(inputs['input_ids'].shape[1]),'output_tokens':int(output.shape[1]-inputs['input_ids'].shape[1]),'review':review,'elapsed_seconds':time.perf_counter()-started,'limitations':'Independent generated advice; not verified findings or evidence of model accuracy.'}
Path('/kaggle/working/external_ai_review.json').write_text(json.dumps(result,indent=2))
print('EXTERNAL_CODE_REVIEW_COMPLETE',json.dumps(result),flush=True)
'''
    ast.parse(code)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'markdown','metadata':{},'source':['# Independent open-weight code-model audit\n','No competition data, training, test inference or submission. Advice requires verification.\n']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    (folder/'external_review.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
    meta={'id':'minahussain111/solar-filament-research-v6-audit','title':'solar-filament-research-v6-audit','code_file':'external_review.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':True,'dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__=='__main__':build('/workspace/scratch/solar-external-code-review')
