"""One direct metric current-status fit with immutable sampled-binary controls."""
import argparse, copy, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import run_contact_boundary as old
import run_contact_boundary_sampling as sampling
from contact_boundary_data import queries, counts, choose_threshold
from query_occupancy_data import read, write, sha, new_stage_directory
from metric_contact_model import MetricContactModel
from exact_contact_loss import supervised_loss
from exact_contact_labels import metric_targets
from contact_boundary_data import camera_boxes
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
BASE=old.DEST
SAMP=BASE.with_name('ba-contact-sampling-20260923')
PRIOR=BASE.with_name('ba-metric-contact-20260923')
DEST=BASE.with_name('ba-exact-contact-20260923')
POS_WEIGHT=sampling.POS_WEIGHT

def fit(mode,initial,features,indices,q,y,z,dev_y,schedule,device):
    from metric_contact_model import MetricContactModel
    model=MetricContactModel().to(device);model.load_state_dict(initial)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    x=torch.tensor(features[indices['train']],device=device)
    yy=torch.tensor(y,device=device,dtype=torch.float32);qq=torch.tensor(q,device=device)
    dx=torch.tensor(features[indices['dev']],device=device)
    dy=torch.tensor(dev_y,device=device,dtype=torch.float32)
    zz=torch.tensor(z,device=device,dtype=torch.float32)
    train_targets=dict(np.load(BASE/'training-targets.npz'))
    dq=torch.tensor(queries()['seen'],device=device);pw=torch.tensor(POS_WEIGHT,device=device)
    history=[];logit_history=[];best=float('inf');epoch_selected=None;started=time.perf_counter()
    for epoch,order in enumerate(schedule,1):
        model.train();total=0.;binary_total=0.;metric_total=0.
        for first in range(0,len(order),32):
            idx=torch.tensor(order[first:first+32],device=device)
            opt.zero_grad(set_to_none=True)
            loss,bl,ml=supervised_loss(model,x[idx],qq[idx],yy[idx],zz[idx],pw)
            assert torch.isfinite(loss),'Nonfinite loss; preserve failed fit'
            loss.backward();opt.step();total+=float(loss.detach())*len(idx);binary_total+=float(bl.detach())*len(idx);metric_total+=float(ml.detach())*len(idx)
        if epoch%10==0:
            model.eval()
            with torch.inference_mode():
                logits=torch.cat([model(dx[i:i+32],dq) for i in range(0,len(dx),32)])
                dl=float(F.binary_cross_entropy_with_logits(logits,dy))
            logit_history.append(logits.cpu().numpy())
            curve=old.predict(model,features[indices['train']],{k:v for k,v in queries().items() if k.endswith('_curve')},device)
            np.savez_compressed(DEST/('train-curves-epoch'+str(epoch)+'.npz'),**curve)
            from contact_boundary_data import boundary_metrics
            diagnostics={k:boundary_metrics(curve[k+'_curve'],.5,train_targets[k+'_boundary'],k) for k in ('width','horizon')}
            history.append(dict(epoch=epoch,train_objective=total/len(x),train_balanced_BCE=binary_total/len(x),train_interval_NLL=metric_total/len(x),dev_BCE=dl,train_curves_at_half=diagnostics))
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
    base=['features.npy','normalization.npz','batch-schedule.npy','cohort.json','feature-receipt.json',
          'training-targets.npz','dev-selection-targets.npz','evaluation-targets.npz','queries.json']
    prior=['initial.pt','source-seal.json','prediction-seal.json','result.json','metric-selection.json',
           'metric-train-predictions.npz','metric-evaluation-predictions.npz']
    sampled=['training-queries.npz','result.json','geometry-selection.json','geometry-train-predictions.npz','geometry-evaluation-predictions.npz']
    return [(BASE/n,'evaluator') for n in base]+[(PRIOR/n,'evaluator') for n in prior]+[(SAMP/n,'evaluator') for n in sampled]+[(old.PREPARED/'observations/identities.json','observation'),(old.CAPTURE/'evaluator/geometry.json','evaluator')]


def spec():
    value=dict(schema='blindassist-asset-run-v1',id='exact-contact-20260923',route='ue-query-occupancy',
        question='Does exact train contact supervision improve the same direct CDF model?',
        evaluator='research/active/dtr-r0/nearfield/run_exact_contact.py',
        evidence_boundary='Consumed Development; extra train-only metric supervision, unchanged model',
        reuse=dict(mode='development',query='contact boundary public feature cache sampled queries geometry exact train first contact'),
        inputs=[dict(alias='input'+str(i),path=str(p),role=r,purpose='matched-supervision-comparison') for i,(p,r) in enumerate(entries())],
        outputs=[dict(alias='result',path=str(DEST/'result.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,'-B',str(Path(__file__).resolve()),'run'])
    path=DEST.with_name(DEST.name+'-run.json');write(path,value);print(path)


def backend(model,features,queries,binary,targets):
    from tools.research_backend import BackendCandidate,Workload,select_backend,torch_observation
    candidates={}
    for dev in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        m=copy.deepcopy(model).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
        tensors=[torch.tensor(a[:32],device=dev,dtype=torch.float32) for a in (features,queries,binary,targets)]
        pw=torch.tensor(POS_WEIGHT,device=dev)
        def probe(mm=m,oo=opt,tt=tensors,weight=pw):
            oo.zero_grad(set_to_none=True);loss=supervised_loss(mm,*tt,weight)[0];loss.backward();oo.step();return loss.detach()
        candidates[dev]=BackendCandidate(dev,dev,probe,lambda output,mm=m:torch_observation(model=mm),torch.cuda.synchronize if dev=='cuda' else lambda:None)
    result=select_backend(Workload.BATCH_TENSOR,cpu=candidates['cpu'],gpu=candidates.get('cuda'),cpu_reason=None if 'cuda' in candidates else 'ACCELERATOR_UNAVAILABLE',record_path=DEST/'backend.json',warmups=1,repeats=3)
    return result['selected_device_type']


def comparisons(metrics,control):
    from run_metric_contact import decision
    d=decision(metrics,control)
    n,b=metrics['boundaries']['horizon'],control['boundaries']['horizon']
    d['horizon_false_rate_change']=(n['wrong_crossings_on_right_censored']-b['wrong_crossings_on_right_censored'])/n['right_censored_truth']
    d['joint_pass']=bool(d['horizon_gain']>=.10 and d['horizon_false_rate_change']<=.02 and d['width_gain']>=-.03 and d['recall_change']>=-.03 and d['FPR_change']<=.02)
    return d


def run():
    assert os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL'),'Use governed research-ue'
    new_stage_directory(DEST);old.configure();started=time.perf_counter()
    inputs={str(p):sha(p) for p,_ in entries()}
    names=['run_exact_contact.py','exact_contact_loss.py','exact_contact_labels.py','EXACT_CONTACT_PROTOCOL_20260923.md',
           'metric_contact_model.py','contact_boundary_model.py','contact_boundary_data.py','run_contact_boundary.py','run_metric_contact.py']
    code={n:sha(HERE/n) for n in names};snap=DEST/'source-snapshot';snap.mkdir()
    for n in names:(snap/n).write_bytes((HERE/n).read_bytes())
    write(DEST/'source-seal.json',dict(inputs=inputs,code=code))
    seal=read(PRIOR/'prediction-seal.json')
    for n in ('metric-train-predictions.npz','metric-evaluation-predictions.npz'):assert sha(PRIOR/n)==seal['files'][n]
    assert sha(PRIOR/'metric-selection.json')==seal['selection_sha256']
    assert sha(BASE/'features.npy')==read(BASE/'feature-receipt.json')['features_sha256']
    indices={k:np.array(v,np.int64) for k,v in read(BASE/'cohort.json')['indices'].items()}
    data=np.load(SAMP/'training-queries.npz');q,y=data['queries'],data['labels']
    assert np.array_equal(data['indices'],indices['train'])
    identities=read(old.PREPARED/'observations/identities.json');geometry=read(old.CAPTURE/'evaluator/geometry.json');zs=[]
    for row,i in enumerate(indices['train']):
        assert identities[int(i)]['split']=='train' and geometry[int(i)]['id']==identities[int(i)]['id']
        zs.append(metric_targets(camera_boxes(geometry[int(i)]),q[row]))
    z=np.array(zs,np.float64);del geometry
    np.savez_compressed(DEST/'exact-training-targets.npz',queries=q,binary=y,z=z,indices=indices['train'])
    write(DEST/'label-receipt.json',dict(rows=int(z.size),finite=int(np.isfinite(z).sum()),right_censored=int(np.isinf(z).sum()),left_censored=int((z<=.300001).sum()),half_bin_m=.025,metric_weight=1.))
    norm=np.load(BASE/'normalization.npz');features=((np.load(BASE/'features.npy')-norm['mean'])/norm['std']).astype(np.float32)
    initial=torch.load(PRIOR/'initial.pt',map_location='cpu',weights_only=True);torch.save(initial,DEST/'initial.pt')
    carrier=MetricContactModel();carrier.load_state_dict(initial)
    device=backend(carrier,features[indices['train']],q,y,z)
    schedule=np.load(BASE/'batch-schedule.npy');dev_y=np.load(BASE/'dev-selection-targets.npz')['seen']
    write(DEST/'fit-started.json',dict(device=device,epochs=100,updates=2700,seed=old.SEED))
    model,selection=fit('metric',initial,features,indices,q,y,z,dev_y,schedule,device)
    files={}
    for split,ix in indices.items():
        path=DEST/('metric-'+split+'-predictions.npz');np.savez_compressed(path,**old.predict(model,features[ix],queries(),device));files[path.name]=sha(path)
        continuous=[]
        with torch.inference_mode():
            query=torch.tensor([[.6,3.,0],[.6,3.,1]],device=device)
            for first in range(0,len(ix),32):
                x=torch.tensor(features[ix[first:first+32]],device=device)
                continuous.append(model.continuous_crossing(x,query,selection['threshold']).cpu().numpy())
        path=DEST/('continuous-'+split+'.npy');np.save(path,np.concatenate(continuous));files[path.name]=sha(path)
    write(DEST/'prediction-seal.json',dict(files=files,selection_sha256=sha(DEST/'metric-selection.json'),checkpoint_sha256=sha(DEST/'metric.pt'),evaluation_targets_joined=False))
    from run_metric_contact import tail
    prior=read(PRIOR/'result.json');sampled=read(SAMP/'result.json');reports={}
    for split,targetname in [('train','training-targets.npz'),('evaluation','evaluation-targets.npz')]:
        target=dict(np.load(BASE/targetname));boundary={k:target[k+'_boundary'] for k in ('width','horizon')}
        meta=[identities[int(i)] for i in indices[split]];pred=dict(np.load(DEST/('metric-'+split+'-predictions.npz')))
        metrics=old.summary(pred,target,boundary,meta,selection['threshold']);control=prior['reports'][split]['metrics']
        oldpred=dict(np.load(PRIOR/('metric-'+split+'-predictions.npz')));oldsel=read(PRIOR/'metric-selection.json')
        bootstrap=old.paired_interval({'direct':oldpred,'geometry':pred},target,meta,{'direct':oldsel,'geometry':selection});bootstrap['direction']='exact-minus-binary-CDF'
        continuous=np.load(DEST/('continuous-'+split+'.npy'));truth=boundary['horizon'];finite=np.isfinite(truth)&(truth>.300001)&(truth<=3.000001);resolved=finite&np.isfinite(continuous);error=np.abs(continuous[resolved]-truth[resolved])
        reports[split]=dict(metrics=metrics,control=control,decision=comparisons(metrics,control),sampled_geometry_comparison=comparisons(metrics,sampled['arms']['geometry'][split]['metrics']),paired_query_bootstrap=bootstrap,
            tails={k:dict(metric=tail(pred[k+'_curve'],boundary[k],k,selection['threshold']),control=tail(oldpred[k+'_curve'],boundary[k],k,oldsel['threshold'])) for k in boundary},
            continuous_horizon=dict(finite_truth=int(finite.sum()),resolved=int(resolved.sum()),within5cm=int((error<=.050001).sum()),conditional_MAE_m=float(error.mean()) if len(error) else None))
    gain=reports['train']['decision']['horizon_gain']>=.10 and reports['evaluation']['decision']['horizon_gain']>=.10
    joint=gain and reports['evaluation']['decision']['joint_pass']
    for p,h in inputs.items():assert sha(p)==h,p
    for n,h in code.items():assert sha(HERE/n)==h,n
    write(DEST/'result.json',dict(status='PASS',reports=reports,selection=selection,actual_device=device,elapsed_s=time.perf_counter()-started,inputs_unchanged=True,supervision_effect_train_and_eval=bool(gain),joint_supervision_pass=bool(joint),evidence_boundary='Consumed simulation Development; extra exact train supervision; no model or App change'))
    print('COMPLETE',DEST,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['spec','run']);args=p.parse_args()
    spec() if args.command=='spec' else run()
