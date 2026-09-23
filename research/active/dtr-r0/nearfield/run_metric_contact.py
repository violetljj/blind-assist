"""One direct metric current-status fit with immutable sampled-binary controls."""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import run_contact_boundary as old
import run_contact_boundary_sampling as sampling
from contact_boundary_data import queries, counts, choose_threshold
from query_occupancy_data import read, write, sha, new_stage_directory
from metric_contact_model import MetricContactModel
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
BASE=old.DEST
SAMP=BASE.with_name('ba-contact-sampling-20260923')
DEST=BASE.with_name('ba-metric-contact-20260923')
POS_WEIGHT=sampling.POS_WEIGHT

def fit(mode,initial,features,indices,q,y,dev_y,schedule,device):
    from metric_contact_model import MetricContactModel
    model=MetricContactModel().to(device);model.load_state_dict(initial)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    x=torch.tensor(features[indices['train']],device=device)
    yy=torch.tensor(y,device=device,dtype=torch.float32);qq=torch.tensor(q,device=device)
    dx=torch.tensor(features[indices['dev']],device=device)
    dy=torch.tensor(dev_y,device=device,dtype=torch.float32)
    dq=torch.tensor(queries()['seen'],device=device);pw=torch.tensor(POS_WEIGHT,device=device)
    history=[];logit_history=[];best=float('inf');epoch_selected=None;started=time.perf_counter()
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
            logit_history.append(logits.cpu().numpy())
            history.append(dict(epoch=epoch,train_balanced_BCE=total/len(x),dev_BCE=dl))
            print('FIT',mode,epoch,'dev',round(dl,6),flush=True)
            if dl<best:
                best=dl;epoch_selected=epoch
                torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},DEST/(mode+'.pt'))
    duration=time.perf_counter()-started
    np.savez_compressed(DEST/'dev-checkpoint-logits.npz',logits=np.stack(logit_history),epochs=np.arange(10,101,10))
    model.load_state_dict(torch.load(DEST/(mode+'.pt'),map_location=device,weights_only=True))
    scores=old.predict(model,features[indices['dev']],dict(seen=queries()['seen']),device)['seen']
    cut=choose_threshold(scores,dev_y)
    result=dict(epoch=epoch_selected,dev_BCE=best,threshold=cut,dev=counts(scores,dev_y,cut),
        history=history,training_seconds=duration,parameters=model.parameter_counts(),
        checkpoint_sha256=sha(DEST/(mode+'.pt')),updates=2700)
    write(DEST/(mode+'-selection.json'),result)
    return model,result


def entries():
    base=['features.npy','normalization.npz','initial.pt','batch-schedule.npy','cohort.json',
          'feature-receipt.json','source-seal.json','training-targets.npz','dev-selection-targets.npz','evaluation-targets.npz']
    sampled=['training-queries.npz','source-seal.json','prediction-seal.json','result.json',
             'geometry-selection.json','geometry-train-predictions.npz','geometry-evaluation-predictions.npz']
    return [(BASE/n,'evaluator') for n in base]+[(SAMP/n,'evaluator') for n in sampled]+[(old.PREPARED/'observations/identities.json','observation')]


def spec():
    value=dict(schema='blindassist-asset-run-v1',id='metric-contact-20260923',route='ue-query-occupancy',
        question='Does direct censored first-contact parameterization improve sampled geometry boundaries?',
        evaluator='research/active/dtr-r0/nearfield/run_metric_contact.py',
        evidence_boundary='Consumed Development; same sampled binary labels, one new head fit',
        reuse=dict(mode='development',query='contact boundary sampled geometry cached public features same binary labels'),
        inputs=[dict(alias='input'+str(i),path=str(p),role=r,purpose='fixed-sampling-metric-comparison') for i,(p,r) in enumerate(entries())],
        outputs=[dict(alias='result',path=str(DEST/'result.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,'-B',str(Path(__file__).resolve()),'run'])
    path=DEST.with_name(DEST.name+'-run.json');write(path,value);print(path)


def decision(new,base):
    nb,bb=new['boundaries'],base['boundaries']
    terms=dict(horizon_gain=nb['horizon']['joint_within_5cm']-bb['horizon']['joint_within_5cm'],
        horizon_false_crossing_change=nb['horizon']['wrong_crossings_on_right_censored']-bb['horizon']['wrong_crossings_on_right_censored'],
        width_gain=nb['width']['joint_within_5cm']-bb['width']['joint_within_5cm'],
        recall_change=new['both']['recall']-base['both']['recall'],FPR_change=new['both']['FPR']-base['both']['FPR'])
    terms['joint_pass']=bool(terms['horizon_gain']>=.10 and terms['horizon_false_crossing_change']<=0 and terms['width_gain']>=-.03 and terms['recall_change']>=-.03 and terms['FPR_change']<=.02)
    return terms


def tail(pred,truth,kind,cut):
    axis=np.linspace(.2,1.2,51) if kind=='width' else np.linspace(.3,3,55)
    on=pred.reshape(-1,2,len(axis))>=cut
    crossing=np.where(on.any(-1),axis[on.argmax(-1)],np.inf)
    finite=np.isfinite(truth)&(truth>axis[0]+1e-6)&(truth<=axis[-1]+1e-6)
    resolved=finite&np.isfinite(crossing)
    errors=np.abs(crossing[resolved]-truth[resolved])
    return dict(resolved=int(resolved.sum()),missing=int((finite&~resolved).sum()),conditional_p90_m=float(np.quantile(errors,.9)) if len(errors) else None)


def run():
    assert os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL'),'Use governed research-ue'
    new_stage_directory(DEST);old.configure();started=time.perf_counter()
    inputs={str(p):sha(p) for p,_ in entries()}
    names=['run_metric_contact.py','metric_contact_model.py','METRIC_CONTACT_PROTOCOL_20260923.md',
           'run_contact_boundary.py','run_contact_boundary_sampling.py','contact_boundary_data.py','contact_boundary_model.py']
    code={n:sha(HERE/n) for n in names};snap=DEST/'source-snapshot';snap.mkdir()
    for n in names:(snap/n).write_bytes((HERE/n).read_bytes())
    write(DEST/'source-seal.json',dict(inputs=inputs,code=code))
    prior=read(SAMP/'prediction-seal.json')
    for n in ('geometry-train-predictions.npz','geometry-evaluation-predictions.npz'):
        assert sha(SAMP/n)==prior['files'][n]
    assert sha(SAMP/'geometry-selection.json')==prior['selection_hashes']['geometry']
    assert sha(BASE/'features.npy')==read(BASE/'feature-receipt.json')['features_sha256']
    cohort=read(BASE/'cohort.json');indices={k:np.array(v,np.int64) for k,v in cohort['indices'].items()}
    data=np.load(SAMP/'training-queries.npz');q,y=data['queries'],data['labels']
    assert np.array_equal(data['indices'],indices['train']) and q.shape==(864,72,3)
    norm=np.load(BASE/'normalization.npz');features=((np.load(BASE/'features.npy')-norm['mean'])/norm['std']).astype(np.float32)
    carrier=MetricContactModel();initial_old=torch.load(BASE/'initial.pt',map_location='cpu',weights_only=True)
    carrier.trunk.load_state_dict({k[6:]:v for k,v in initial_old.items() if k.startswith('trunk.')})
    initial={k:v.detach().clone() for k,v in carrier.state_dict().items()};torch.save(initial,DEST/'initial.pt')
    schedule=np.load(BASE/'batch-schedule.npy');dev_y=np.load(BASE/'dev-selection-targets.npz')['seen']
    sampling.DEST=DEST;device=sampling.backend(carrier,features[indices['train']],y,q)
    write(DEST/'fit-started.json',dict(device=device,epochs=100,updates=2700,seed=old.SEED))
    model,selection=fit('metric',initial,features,indices,q,y,dev_y,schedule,device)
    files={}
    for split,ix in indices.items():
        path=DEST/('metric-'+split+'-predictions.npz')
        np.savez_compressed(path,**old.predict(model,features[ix],queries(),device));files[path.name]=sha(path)
        continuous=[]
        with torch.inference_mode():
            query=torch.tensor([[.6,3.,0],[.6,3.,1]],device=device)
            for start in range(0,len(ix),32):
                x=torch.tensor(features[ix[start:start+32]],device=device)
                continuous.append(model.continuous_crossing(x,query,selection['threshold']).cpu().numpy())
        path=DEST/('continuous-'+split+'.npy');np.save(path,np.concatenate(continuous));files[path.name]=sha(path)
    write(DEST/'prediction-seal.json',dict(files=files,selection_sha256=sha(DEST/'metric-selection.json'),checkpoint_sha256=sha(DEST/'metric.pt'),evaluation_targets_joined=False))
    # Only now join consumed evaluation targets and old results.
    identities=read(old.PREPARED/'observations/identities.json');base=read(SAMP/'result.json');reports={}
    for split,targetname in [('train','training-targets.npz'),('evaluation','evaluation-targets.npz')]:
        target=dict(np.load(BASE/targetname));boundary={k:target[k+'_boundary'] for k in ('width','horizon')}
        meta=[identities[int(i)] for i in indices[split]];pred=dict(np.load(DEST/('metric-'+split+'-predictions.npz')))
        metrics=old.summary(pred,target,boundary,meta,selection['threshold'])
        control=base['arms']['geometry'][split]['metrics']
        oldpred=dict(np.load(SAMP/('geometry-'+split+'-predictions.npz')))
        bootstrap=old.paired_interval({'direct':oldpred,'geometry':pred},target,meta,{'direct':read(SAMP/'geometry-selection.json'),'geometry':selection})
        bootstrap['direction']='metric-minus-sampled-geometry'
        continuous=np.load(DEST/('continuous-'+split+'.npy'));truth=boundary['horizon']
        finite=np.isfinite(truth)&(truth>.300001)&(truth<=3.000001);resolved=finite&np.isfinite(continuous)
        error=np.abs(continuous[resolved]-truth[resolved])
        reports[split]=dict(metrics=metrics,control=control,decision=decision(metrics,control),paired_query_bootstrap=bootstrap,
            tails={k:dict(metric=tail(pred[k+'_curve'],boundary[k],k,selection['threshold']),control=tail(oldpred[k+'_curve'],boundary[k],k,read(SAMP/'geometry-selection.json')['threshold'])) for k in boundary},
            continuous_horizon=dict(finite_truth=int(finite.sum()),resolved=int(resolved.sum()),within5cm=int((error<=.050001).sum()),conditional_MAE_m=float(error.mean()) if len(error) else None))
    for p,h in inputs.items():assert sha(p)==h,p
    for n,h in code.items():assert sha(HERE/n)==h,n
    write(DEST/'result.json',dict(status='PASS',reports=reports,selection=selection,actual_device=device,elapsed_s=time.perf_counter()-started,inputs_unchanged=True,evidence_boundary='Consumed same-generator simulation Development; no exact-distance labels or App promotion'))
    print('COMPLETE',DEST,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['spec','run']);args=p.parse_args()
    spec() if args.command=='spec' else run()

