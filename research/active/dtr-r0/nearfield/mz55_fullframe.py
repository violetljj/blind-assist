"""MZ55 owner-side binding adapter for the frozen MZ52 CPU cell derivation."""
import argparse
import importlib.util
import json
from pathlib import Path
import hashlib

CORE_SHA = '5fb19e543e1b6c841a2c61c8a4e4ee4c918985a3883d1c061a2ce673423aeaa3'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run(manifest,specs,capture,dataset,core,output,host):
    assert sha(core)==CORE_SHA
    m=read(manifest);assert m['total_frames']==2560
    ids={};original=None
    captured=read(capture/'source/spec.json')
    inputs={str(p):sha(p) for p in (manifest,core,Path(__file__),capture/'source/spec.json',dataset/'receipt.json')}
    for item in m['shards']:
        path=specs/(item['shard_id']+'.json');assert sha(path)==item['sha256']
        inputs[str(path)]=item['sha256'];spec=read(path)
        if spec['cases'][0]['name']==captured['cases'][0]['name']:original=spec
        for case in spec['cases']:
            assert case['name'] not in ids;ids[case['name']]=len(ids)
    assert len(ids)==2560 and original is not None
    assert {k:v for k,v in captured.items() if k!='map_file'}=={k:v for k,v in original.items() if k!='map_file'}
    receipt=read(dataset/'receipt.json');assert receipt['status']=='PASS'
    result=dataset/'result.json';assert sha(result)==receipt['outputs']['result.json'];inputs[str(result)]=sha(result)
    assert read(result)['source_valid_frames']==len(captured['cases'])
    metadata=dataset/'training-source/evaluator/metadata.json';inputs[str(metadata)]=sha(metadata)
    rows=read(metadata)['records'];assert len(rows)==len(captured['cases'])
    request_rows=[]
    for i,(row,case) in enumerate(zip(rows,captured['cases'])):
        assert row['frame_id']==case['name'] and row['index']==i
        path=capture/f'evaluator/native/{i:04d}.npy'
        assert Path(row['worker_native']).resolve()==path.resolve()
        request_rows.append(dict(global_index=ids[row['frame_id']],frame_id=row['frame_id'],native_path=str(path.resolve()),
            native_sha256=row['native_sha256'],event_counts=row['event_counts'],event_truth=row['event_truth'],
            pair_id=row['pair_id'],site_id=row['site_id'],family=row['family'],role=row['role']))
    output.mkdir(parents=True,exist_ok=False)
    schema=dict(fullframe_event_counts='uint8[N,45,80,4]',valid_counts='uint8[N,45,80]',
        cell_event_presence='DERIVED: fullframe_event_counts>0',cell_known='DERIVED: valid_counts>0',
        authority='MZ55 native training/evaluator only; unchanged45degree predictor packet')
    request=dict(status='FROZEN_DERIVATION_REQUEST',host=host,frames=len(rows),rows=request_rows,
        manifest_sha256=sha(manifest),derivation_code_sha256=CORE_SHA,schema=schema)
    req=output/'request.json';req.write_text(json.dumps(request,indent=2)+'\n',encoding='utf-8')
    module_spec=importlib.util.spec_from_file_location('mz52_bound_core',core)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    module.derive(req,output/'derive-v1')
    result=read(output/'derive-v1/result.json')
    final=dict(status='PASS',inputs=inputs,code_sha256=sha(Path(__file__)),frozen_derivation_sha256=CORE_SHA,
        request_sha256=sha(req),frames=len(rows),global_frame_order='Frozen MZ55 manifest shard order then case order',
        outputs={str(p.relative_to(output)):sha(p) for p in (req,output/'derive-v1/receipt.json',output/'derive-v1/fullframe-cells.npz')},
        result=result,new_captures=0,training_steps=0,model_inference_frames=0)
    (output/'receipt.json').write_text(json.dumps(final,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','specs','capture','dataset','core','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--host',choices=('worker','primary'),required=True)
    a=p.parse_args();run(a.manifest,a.specs,a.capture,a.dataset,a.core,a.output,a.host)
