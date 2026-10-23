"""Read-only Kaggle identity probes. Persist allowlisted metadata, never source/outputs."""
import hashlib
import json
from pathlib import Path

FIELDS=('id','ref','title','slug','version_number','current_version_number','version_label',
        'script_version_id','kernel_id','status','last_run_time','creation_date','is_private')


def public_metadata(value):
    result={}
    for name in FIELDS:
        field=getattr(value,name,None)
        if isinstance(field,(str,int,float,bool)):
            result[name]=field
        elif field is not None and name=='status' and hasattr(field,'name'):
            result[name]=field.name
    return result


def source_summary(source):
    result={'sha256':hashlib.sha256(source.encode()).hexdigest()}
    try:
        cells=json.loads(source).get('cells',[])
        result['executed_code_cells']=sum(c.get('cell_type')=='code' and c.get('execution_count') is not None for c in cells)
        result['cells_with_outputs']=sum(bool(c.get('outputs')) for c in cells)
    except (ValueError,AttributeError):
        result['notebook_json']=False
    return result


def error_status(error):
    response=getattr(error,'response',None)
    return {'error_type':type(error).__name__,'http_status':getattr(response,'status_code',None)}


def main():
    # Authentication stays inside the official SDK and its existing Actions binding.
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiGetKernelSessionStatusRequest
    api=KaggleApi();api.authenticate()
    owner='minahussain111';slug='solar-filament-research-v5'
    report={'target':f'https://www.kaggle.com/code/{owner}/{slug}',
            'purpose':'read-only identity/history diagnosis; no model execution or artifact downloads',
            'source_probes':{},'status_probes':{}}
    try:
        report['matching_notebooks']=[public_metadata(item) for item in (api.kernels_list(mine=True,search=slug,page_size=20) or [])]
    except Exception as error:report['matching_notebooks_error']=error_status(error)
    with api.build_kaggle_client() as client:
        for label in [None,'v1','v2','v3','v4','v5','5']:
            key=label or 'latest'
            request=ApiGetKernelRequest();request.user_name=owner;request.kernel_slug=slug
            if label is not None:request.version_label=label
            try:
                response=client.kernels.kernels_api_client.get_kernel(request)
                report['source_probes'][key]={'metadata':public_metadata(response.metadata),
                                              'blob_metadata':public_metadata(response.blob),
                                              'source_summary':source_summary(response.blob.source)}
            except Exception as error:report['source_probes'][key]=error_status(error)
        for label in [None,'v5','5']:
            request=ApiGetKernelSessionStatusRequest();request.user_name=owner;request.kernel_slug=slug
            if label is not None:request.version_label=label
            try:
                response=client.kernels.kernels_api_client.get_kernel_session_status(request)
                report['status_probes'][label or 'latest']=public_metadata(response)
            except Exception as error:report['status_probes'][label or 'latest']=error_status(error)
    Path('reports/V5_IDENTITY.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved allowlisted V5 identity metadata; no notebook source, outputs or credentials printed.')


if __name__=='__main__':main()
