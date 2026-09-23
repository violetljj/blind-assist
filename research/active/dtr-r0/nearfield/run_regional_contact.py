"""Matched regional RGB-ToF alignment contrast; two fixed training runs."""
import argparse, copy, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import run_contact_boundary as old
from contact_boundary_data import queries,counts,choose_threshold
from query_occupancy_data import read,write,sha,new_stage_directory
from query_occupancy_model import sample_region_features
from regional_contact_model import RegionContactModel
from regional_contact_data import derangements,paired_inputs
from exact_contact_loss import supervised_loss
from contact_decomposition import quantile
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
ART=REPO/'artifacts.local/evidence'
BASE=ART/'ba-contact-boundary-20260923'
EXACT=ART/'ba-exact-contact-20260923'
SAMP=ART/'ba-contact-sampling-20260923'
DEST=ART/'ba-regional-contact-20260923'
POS_WEIGHT=50168/12040

def fit(mode,initial,features,indices,q,y,z,dev_y,schedule,device):
    from regional_contact_model import RegionContactModel
    model=RegionContactModel().to(device);model.load_state_dict(initial)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    x=torch.tensor(features[indices['train']],device=device)
    yy=torch.tensor(y,device=device,dtype=torch.float32);qq=torch.tensor(q,device=device)
    dx=torch.tensor(features[indices['dev']],device=device)
    dy=torch.tensor(dev_y,device=device,dtype=torch.float32)
    zz=torch.tensor(z,device=device,dtype=torch.float32)
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
            history.append(dict(epoch=epoch,train_objective=total/len(x),train_balanced_BCE=binary_total/len(x),train_interval_NLL=metric_total/len(x),dev_BCE=dl))
            print('FIT',mode,epoch,'dev',round(dl,6),flush=True)
            if dl<best:
                best=dl;epoch_selected=epoch
                torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},DEST/(mode+'.pt'))
    duration=time.perf_counter()-started
    np.savez_compressed(DEST/(mode+'-dev-checkpoint-logits.npz'),logits=np.stack(logit_history),epochs=np.arange(10,101,10))
    model.load_state_dict(torch.load(DEST/(mode+'.pt'),map_location=device,weights_only=True))
    scores=old.predict(model,features[indices['dev']],dict(seen=queries()['seen']),device)['seen']
    cut=choose_threshold(scores,dev_y)
    result=dict(epoch=epoch_selected,dev_BCE=best,threshold=cut,dev=counts(scores,dev_y,cut),
        history=history,training_seconds=duration,parameters=model.parameter_counts(),
        checkpoint_sha256=sha(DEST/(mode+'.pt')),updates=2700)
    write(DEST/(mode+'-selection.json'),result)
    return model,result


def entries():
    base=['features.npy','feature-receipt.json','batch-schedule.npy','cohort.json','training-targets.npz','dev-selection-targets.npz','evaluation-targets.npz','queries.json']
    return [(BASE/n,'evaluator') for n in base]+[(EXACT/n,'evaluator') for n in ['exact-training-targets.npz','result.json']]+[(old.PREPARED/'observations/identities.json','observation')]


def spec():
    value=dict(schema='blindassist-asset-run-v1',id='regional-contact-20260923',route='ue-query-occupancy',
        question='Does correct regional RGB-ToF pairing improve held-layout contact localization?',
        evaluator='research/active/dtr-r0/nearfield/run_regional_contact.py',
        evidence_boundary='Consumed simulation Development; matched regional pairing contrast',
        reuse=dict(mode='development',query='contact boundary public feature cache exact training contact regional RGB ToF'),
        inputs=[dict(alias='input'+str(i),path=str(p),role=r,purpose='matched-regional-pairing') for i,(p,r) in enumerate(entries())],
        outputs=[dict(alias='result',path=str(DEST/'result.json'),role='result',required=True)],
        result_output='result',command=[sys.executable,'-B',str(Path(__file__).resolve()),'run'])
    path=DEST.with_name(DEST.name+'-run.json');write(path,value);print(path)


def run():
    assert os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL'),'Use governed research-ue'
    new_stage_directory(DEST);old.configure();started=time.perf_counter()
    inputs={str(p):sha(p) for p,_ in entries()}
    names=['run_regional_contact.py','regional_contact_model.py','regional_contact_data.py','REGIONAL_CONTACT_PROTOCOL_20260923.md','exact_contact_loss.py','contact_boundary_model.py','contact_boundary_data.py','run_contact_boundary.py','query_occupancy_model.py','contact_decomposition.py','run_exact_contact.py','run_metric_contact.py']
    code={n:sha(HERE/n) for n in names};snap=DEST/'source-snapshot';snap.mkdir()
    for n in names:(snap/n).write_bytes((HERE/n).read_bytes())
    write(DEST/'source-seal.json',dict(inputs=inputs,code=code))
    assert sha(BASE/'features.npy')==read(BASE/'feature-receipt.json')['features_sha256']
    indices={k:np.array(v,np.int64) for k,v in read(BASE/'cohort.json')['indices'].items()}
    raw=np.load(BASE/'features.npy');sensor=raw[:,4480:].reshape(-1,64,6)
    # GPU-first extraction; tiny indexing/hash/normalization metadata remains CPU.
    pooling_device='cuda' if torch.cuda.is_available() else 'cpu'
    rgb=[]
    with torch.inference_mode():
        for first in range(0,len(raw),32):
            visual=torch.tensor(raw[first:first+32,:4480].reshape(-1,40,8,14),device=pooling_device)
            boxes=torch.tensor(sensor[first:first+32,:,2:],device=pooling_device)
            rgb.append(sample_region_features(visual,boxes).cpu().numpy())
    rgb=np.concatenate(rgb);permutation=derangements(raw)
    aligned,misaligned,norm=paired_inputs(rgb,sensor,indices['train'],permutation)
    np.save(DEST/'regional-rgb.npy',rgb);np.save(DEST/'permutations.npy',permutation)
    np.savez_compressed(DEST/'normalization.npz',**norm)
    features=dict(aligned=aligned,misaligned=misaligned)
    for mode,x in features.items():np.save(DEST/(mode+'-features.npy'),x)
    data=np.load(EXACT/'exact-training-targets.npz');q,y,z=data['queries'],data['binary'],data['z']
    assert np.array_equal(data['indices'],indices['train']) and q.shape==(864,72,3)
    carrier=RegionContactModel();initial=copy.deepcopy(carrier.state_dict());torch.save(initial,DEST/'initial.pt')
    import run_exact_contact as exact_runner
    exact_runner.DEST=DEST
    device=exact_runner.backend(carrier,aligned[indices['train']],q,y,z)
    schedule=np.load(BASE/'batch-schedule.npy');assert schedule.shape==(100,864)
    dev_y=np.load(BASE/'dev-selection-targets.npz')['seen'];selections={};files={}
    for mode,x in features.items():
        write(DEST/(mode+'-fit-started.json'),dict(device=device,epochs=100,updates=2700,seed=old.SEED,initial_sha256=sha(DEST/'initial.pt'),schedule_sha256=sha(BASE/'batch-schedule.npy')))
        model,selection=fit(mode,initial,x,indices,q,y,z,dev_y,schedule,device);selections[mode]=selection
        for split,ix in indices.items():
            path=DEST/(mode+'-'+split+'-predictions.npz');np.savez_compressed(path,**old.predict(model,x[ix],queries(),device));files[path.name]=sha(path)
            parts=[]
            with torch.inference_mode():
                anchor=torch.tensor([[.6,3.,0],[.6,3.,1]],device=device)
                for first in range(0,len(ix),32):
                    parts.append(np.stack([a.cpu().numpy() for a in model.components(torch.tensor(x[ix[first:first+32]],device=device),anchor)],axis=-1))
            parts=np.concatenate(parts);path=DEST/(mode+'-'+split+'-components.npz')
            np.savez_compressed(path,q=parts[...,0],mu=parts[...,1],scale=parts[...,2]);files[path.name]=sha(path)
        for suffix in ['.pt','-selection.json','-dev-checkpoint-logits.npz']:files[mode+suffix]=sha(DEST/(mode+suffix))
    write(DEST/'prediction-seal.json',dict(files=files,evaluation_targets_joined=False))
    identities=read(old.PREPARED/'observations/identities.json');arms={};reports={}
    from contact_decomposition import errors
    from run_exact_contact import comparisons
    for split,targetname in [('train','training-targets.npz'),('evaluation','evaluation-targets.npz')]:
        target=dict(np.load(BASE/targetname));boundary={k:target[k+'_boundary'] for k in ('width','horizon')}
        meta=[identities[int(i)] for i in indices[split]];preds={}
        for mode in features:
            pred=dict(np.load(DEST/(mode+'-'+split+'-predictions.npz')));preds[mode]=pred
            metrics=old.summary(pred,target,boundary,meta,selections[mode]['threshold'])
            parts=np.load(DEST/(mode+'-'+split+'-components.npz'));truth=boundary['horizon']
            median=quantile(parts['mu'],parts['scale'],.5);finite=np.isfinite(truth)&(truth>.300001)&(truth<=3.000001)
            arms.setdefault(mode,{})[split]=dict(metrics=metrics,conditional_median=errors(median,truth,finite))
        a,b=arms['aligned'][split],arms['misaligned'][split]
        decision=comparisons(a['metrics'],b['metrics'])
        decision['conditional_median_gain']=a['conditional_median']['hit_rate']-b['conditional_median']['hit_rate']
        decision['joint_pass']=bool(decision['joint_pass'] and decision['conditional_median_gain']>=.10)
        bootstrap=old.paired_interval({'direct':preds['misaligned'],'geometry':preds['aligned']},target,meta,{'direct':selections['misaligned'],'geometry':selections['aligned']});bootstrap['direction']='aligned-minus-misaligned'
        reports[split]=dict(decision=decision,paired_query_bootstrap=bootstrap)
    for p,h in inputs.items():assert sha(p)==h,p
    for n,h in code.items():assert sha(HERE/n)==h,n
    write(DEST/'result.json',dict(status='PASS',arms=arms,reports=reports,selections=selections,actual_device=device,pooling_device=pooling_device,elapsed_s=time.perf_counter()-started,inputs_unchanged=True,joint_alignment_pass=reports['evaluation']['decision']['joint_pass'],evidence_boundary='Consumed simulation Development; position and range pairing jointly changed; no causal ownership or App claim'))
    print('COMPLETE',DEST,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['spec','run']);args=p.parse_args()
    spec() if args.command=='spec' else run()
