"""Controlled query-sampling intervention using frozen features and old controls."""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

import run_contact_boundary as old
from contact_boundary_data import queries, camera_boxes, counts, choose_threshold
from query_occupancy_data import read, write, sha, new_stage_directory

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
BASE=old.DEST
DEST=BASE.with_name('ba-contact-sampling-20260923')
PROTOCOL=HERE/'CONTACT_SAMPLING_PROTOCOL_20260923.md'
POS_WEIGHT=50168/12040


def input_entries():
    entries=[('identities',old.PREPARED/'observations/identities.json','observation'),
             ('geometry',old.CAPTURE/'evaluator/geometry.json','evaluator')]
    names=['features.npy','normalization.npz','initial.pt','batch-schedule.npy','cohort.json',
        'source-seal.json','prediction-seal.json','feature-receipt.json','result.json',
        'training-targets.npz','dev-selection-targets.npz','evaluation-targets.npz']
    names += [f'{m}-{s}-predictions.npz' for m in ('direct','geometry') for s in ('train','dev','evaluation')]
    names += [f'{m}-selection.json' for m in ('direct','geometry')]
    configs={'normalization.npz','initial.pt','batch-schedule.npy','cohort.json','source-seal.json',
             'prediction-seal.json','feature-receipt.json','direct-selection.json','geometry-selection.json'}
    for i,name in enumerate(names):
        entries.append((f'base{i}',BASE/name,'observation' if name=='features.npy' else
                        'configuration' if name in configs else 'evaluator'))
    return entries


def make_spec():
    spec=dict(schema='blindassist-asset-run-v1',id='contact-sampling-20260923',route='ue-query-occupancy',
        question='Does boundary-aware continuous sampling improve contact boundaries at the same72-query budget?',
        evaluator='research/active/dtr-r0/nearfield/run_contact_boundary_sampling.py',
        evidence_boundary='Consumed Development; frozen source and baseline; only training query sampling changes',
        reuse=dict(mode='development',query='Consumed contact boundary frozen RGB ToF feature cache and query controls'),
        inputs=[dict(alias=a,path=str(p),role=r,purpose='paired-query-sampling-intervention') for a,p,r in input_entries()],
        outputs=[dict(alias='result',path=str(DEST/'result.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,'-B',str(Path(__file__).resolve()),'run'])
    path=DEST.with_name(DEST.name+'-run.json');write(path,spec);print(path)


def backend(model,features,target,q):
    sys.path.insert(0,str(REPO))
    from tools.research_backend import BackendCandidate,Workload,select_backend,torch_observation
    candidates={}
    for dev in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        m=copy.deepcopy(model).to(dev)
        x=torch.tensor(features[:32],device=dev);y=torch.tensor(target[:32],device=dev,dtype=torch.float32)
        qq=torch.tensor(q[:32],device=dev);pw=torch.tensor(POS_WEIGHT,device=dev)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
        def probe(mm=m,xx=x,yy=y,query=qq,weight=pw,oo=opt):
            oo.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(mm(xx,query),yy,pos_weight=weight)
            loss.backward();oo.step();return loss.detach()
        candidates[dev]=BackendCandidate(dev,dev,probe,lambda output,mm=m:torch_observation(model=mm),
            torch.cuda.synchronize if dev=='cuda' else lambda:None)
    r=select_backend(Workload.BATCH_TENSOR,cpu=candidates['cpu'],gpu=candidates.get('cuda'),
        cpu_reason=None if 'cuda' in candidates else 'ACCELERATOR_UNAVAILABLE',
        record_path=DEST/'backend.json',warmups=1,repeats=3)
    return r['selected_device_type']


def fit(mode,initial,features,indices,q,y,dev_y,schedule,device):
    from contact_boundary_sampling_model import ContactBoundarySamplingModel
    model=ContactBoundarySamplingModel(mode).to(device);model.load_state_dict(initial)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    x=torch.tensor(features[indices['train']],device=device)
    yy=torch.tensor(y,device=device,dtype=torch.float32);qq=torch.tensor(q,device=device)
    dx=torch.tensor(features[indices['dev']],device=device)
    dy=torch.tensor(dev_y,device=device,dtype=torch.float32)
    dq=torch.tensor(queries()['seen'],device=device);pw=torch.tensor(POS_WEIGHT,device=device)
    history=[];best=float('inf');epoch_selected=None;started=time.perf_counter()
    for epoch,order in enumerate(schedule,1):
        model.train();total=0.
        for first in range(0,len(order),32):
            idx=torch.tensor(order[first:first+32],device=device)
            opt.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(model(x[idx],qq[idx]),yy[idx],pos_weight=pw)
            assert torch.isfinite(loss),'Nonfinite loss; preserve failed fit'
            loss.backward();opt.step();total+=float(loss.detach())*len(idx)
        if epoch%10==0:
            model.eval()
            with torch.inference_mode():
                logits=torch.cat([model(dx[i:i+32],dq) for i in range(0,len(dx),32)])
                dl=float(F.binary_cross_entropy_with_logits(logits,dy))
            history.append(dict(epoch=epoch,train_balanced_BCE=total/len(x),dev_BCE=dl))
            print('FIT',mode,epoch,'dev',round(dl,6),flush=True)
            if dl<best:
                best=dl;epoch_selected=epoch
                torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},DEST/(mode+'.pt'))
    duration=time.perf_counter()-started
    model.load_state_dict(torch.load(DEST/(mode+'.pt'),map_location=device,weights_only=True))
    scores=old.predict(model,features[indices['dev']],dict(seen=queries()['seen']),device)['seen']
    cut=choose_threshold(scores,dev_y)
    result=dict(epoch=epoch_selected,dev_BCE=best,threshold=cut,dev=counts(scores,dev_y,cut),
        history=history,training_seconds=duration,parameters=model.parameter_counts(),
        checkpoint_sha256=sha(DEST/(mode+'.pt')),updates=2700)
    write(DEST/(mode+'-selection.json'),result)
    return model,result


def gain(new,base):
    delta={}
    for kind in ('width','horizon'):
        n,b=new['boundaries'][kind],base['boundaries'][kind]
        hit=n['joint_within_5cm']-b['joint_within_5cm']
        false=(n['wrong_crossings_on_right_censored']-b['wrong_crossings_on_right_censored'])/n['right_censored_truth']
        delta[kind]=dict(within5cm_gain=hit,right_censored_false_rate_change=false,
                        improved=hit>=.10 and false<=.05)
    delta['joint_boundary_improvement']=all(delta[k]['improved'] for k in ('width','horizon'))
    delta['query_cost_guard']=(new['both']['FPR']-base['both']['FPR']<=.02 and
                               new['both']['recall']-base['both']['recall']>=-.03)
    return delta


def run():
    from contact_boundary_sampling_model import ContactBoundarySamplingModel
    from contact_boundary_sampling_data import sample_queries
    assert os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL'),'Use governed research-ue'
    new_stage_directory(DEST);old.configure();started=time.perf_counter()
    inputs={str(p):sha(p) for _,p,_ in input_entries()}
    codes=['run_contact_boundary_sampling.py','contact_boundary_sampling_data.py','contact_boundary_sampling_model.py',
           'contact_boundary_model.py','contact_boundary_data.py','run_contact_boundary.py','CONTACT_SAMPLING_PROTOCOL_20260923.md']
    code={n:sha(HERE/n) for n in codes}
    snap=DEST/'source-snapshot';snap.mkdir()
    for n in codes:(snap/n).write_bytes((HERE/n).read_bytes())
    write(DEST/'source-seal.json',dict(inputs=inputs,code=code,positive_weight=POS_WEIGHT))
    priorseal=read(BASE/'prediction-seal.json')
    for n,h in priorseal['files'].items():assert sha(BASE/n)==h
    for m,h in priorseal['selection_hashes'].items():assert sha(BASE/(m+'-selection.json'))==h
    assert sha(BASE/'features.npy')==read(BASE/'feature-receipt.json')['features_sha256']
    inherited=read(BASE/'source-seal.json')
    for n in ('contact_boundary_data.py','contact_boundary_model.py','run_contact_boundary.py'):
        assert sha(HERE/n)==inherited['code'][n],n
    cohort=read(BASE/'cohort.json');indices={k:np.array(v,np.int64) for k,v in cohort['indices'].items()}
    identities=read(old.PREPARED/'observations/identities.json')
    geometry=read(old.CAPTURE/'evaluator/geometry.json')
    qrows=[];ys=[];receipts=[]
    for i in indices['train']:
        assert identities[int(i)]['split']=='train' and geometry[int(i)]['id']==identities[int(i)]['id']
        q,y,receipt=sample_queries(camera_boxes(geometry[int(i)]),202609232+int(i))
        qrows.append(q);ys.append(y);receipts.append(dict(index=int(i),**receipt))
    q=np.array(qrows,np.float32);y=np.array(ys,bool)
    assert q.shape==(864,72,3) and y.shape==(864,72)
    np.savez_compressed(DEST/'training-queries.npz',queries=q,labels=y,indices=indices['train'])
    write(DEST/'sampling-receipts.json',receipts)
    print('SAMPLING',int(y.sum()),'positive of',y.size,flush=True)
    raw=np.load(BASE/'features.npy');norm=np.load(BASE/'normalization.npz')
    features=((raw-norm['mean'])/norm['std']).astype(np.float32)
    initial=torch.load(BASE/'initial.pt',map_location='cpu',weights_only=True)
    schedule=np.load(BASE/'batch-schedule.npy');assert schedule.shape==(100,864)
    dev_y=np.load(BASE/'dev-selection-targets.npz')['seen']
    carrier=ContactBoundarySamplingModel('direct');carrier.load_state_dict(initial)
    device=backend(carrier,features[indices['train']],y,q)
    selection={};files={}
    for mode in ('direct','geometry'):
        model,sel=fit(mode,initial,features,indices,q,y,dev_y,schedule,device);selection[mode]=sel
        for split,rows in indices.items():
            path=DEST/(mode+'-'+split+'-predictions.npz')
            np.savez_compressed(path,**old.predict(model,features[rows],queries(),device))
            files[path.name]=sha(path)
        del model
    write(DEST/'prediction-seal.json',dict(files=files,selection_hashes={m:sha(DEST/(m+'-selection.json')) for m in selection},
        checkpoint_hashes={m:sha(DEST/(m+'.pt')) for m in selection},evaluation_targets_joined=False))
    base=read(BASE/'result.json');reports={}
    for mode in ('direct','geometry'):
        reports[mode]={}
        for split,targetname in [('train','training-targets.npz'),('evaluation','evaluation-targets.npz')]:
            target=dict(np.load(BASE/targetname));boundary={k:target[k+'_boundary'] for k in ('width','horizon')}
            pred=dict(np.load(DEST/(mode+'-'+split+'-predictions.npz')))
            meta=[identities[int(i)] for i in indices[split]]
            summary=old.summary(pred,target,boundary,meta,selection[mode]['threshold'])
            old_summary=base['arms'][mode] if split=='evaluation' else base['arms'][mode]['training_layouts']
            unchanged=old.summary(pred,target,boundary,meta,base['selection'][mode]['threshold'])
            oldpred=dict(np.load(BASE/(mode+'-'+split+'-predictions.npz')))
            paired=old.paired_interval({'direct':oldpred,'geometry':pred},target,meta,
                {'direct':base['selection'][mode],'geometry':selection[mode]})
            paired['direction']='new-sampling-minus-old-fixed-query'
            reports[mode][split]=dict(metrics=summary,change=gain(summary,old_summary),
                at_old_cutoff=unchanged,paired_query_bootstrap=paired)
    for p,h in inputs.items():assert sha(p)==h,p
    for n,h in code.items():assert sha(HERE/n)==h,n
    for n,h in files.items():assert sha(DEST/n)==h,n
    result=dict(status='PASS',arms=reports,selection=selection,actual_device=device,
        source_hashes_unchanged=True,epochs_per_arm=100,unique_queries_per_train_image=72,
        elapsed_s=time.perf_counter()-started,evidence_boundary='Consumed same-generator Development; no new sensing or complete occupancy identification')
    write(DEST/'result.json',result);print('COMPLETE',DEST,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['spec','run']);a=p.parse_args()
    make_spec() if a.command=='spec' else run()
