"""Same MZ11 gate/optimizer with expanded original training partitions."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mz5_ensemble_readout import read,write,sha,load_npz
from mz11_selective_addition import gate_features,compose
from mz8_train import metrics


def main(root,features,output):
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    work=root/'artifacts.local/work';run11=work/'mz11-selective-addition-20260910/run-v1';run12=work/'mz12-existing-data-20260910/run-v1'
    receipt=read(features/'receipt.json');assert receipt['status']=='PASS'
    assert sha(features/'features.npz')==receipt['files']['features.npz']
    assert sha(features/'selected.json')==receipt['selected_sha256']
    d=load_npz(features/'features.npz')
    assert len(d['truth'])==7500
    inputs={str(features/'features.npz'):sha(features/'features.npz')}
    for run in [run11,run12]:
        assert sha(run/'predictions.npz')==read(run/'receipt.json')['outputs']['predictions.npz']
        inputs[str(run/'predictions.npz')]=sha(run/'predictions.npz')
    p11=load_npz(run11/'predictions.npz');p12=load_npz(run12/'predictions.npz')
    records=read(work/'mz12-existing-data-20260910/inventory-v1/selected.json')
    cohorts={}
    for key in ['TRAIN','DEV','clean','stress','clean_wrong','stress_wrong']:
        cohorts[key]={k:p11[key+'/'+k] for k in ['features','baseline','source','available','eligible','truth']}
        cohorts[key]['previous']=p11[key+'/candidate']
    for dataset in ['relation','distance']:
        mask=d['dataset']==dataset
        cohorts['train_'+dataset]={k:d[k][mask] for k in ['features','baseline','source','available','eligible','truth']}
    for dataset in ['relation10000','distance5000']:
        mask=p12['dataset']==dataset
        a=dict(baseline=p12['MZ5'][mask],source=p12['SOURCE'][mask],available=p12['available'][mask],truth=p12['truth'][mask],previous=p12['MZ11'][mask])
        a['features']=gate_features(np.where(a['available'],a['source'],0),p12['candidate_count'][mask],p12['candidate_zones'][mask],p12['candidate_returns'][mask])
        a['eligible']=(a['baseline']<0)&(a['source']>=0)&a['available']
        cohorts[dataset]=a
    training=['TRAIN','clean','train_relation','train_distance']
    write(output/'start-receipt.json',dict(seed=111,steps=600,parameters=20,inputs=inputs,
        feature_receipt_sha256=sha(features/'receipt.json'),source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(Path(__file__).with_name('MZ13_TRAINING_COVERAGE_PROTOCOL_20260910.md')),
        training_group_counts={k:dict(frames=len(cohorts[k]['truth']),positive=(cohorts[k]['eligible']&cohorts[k]['truth']).sum(0).tolist(),
            negative=(cohorts[k]['eligible']&~cohorts[k]['truth']).sum(0).tolist()) for k in training}))
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(111)
    weight=torch.nn.Parameter(torch.zeros(4,4,device='cuda'));bias=torch.nn.Parameter(torch.zeros(4,device='cuda'))
    opt=torch.optim.AdamW([weight,bias],lr=.03,weight_decay=.001)
    fit={k:{v:torch.from_numpy(cohorts[k][v]).cuda() for v in ['features','eligible','truth']} for k in training}
    history=[]
    for step in range(600):
        losses=[]
        for a in fit.values():
            z=(a['features']*weight).sum(-1)+bias
            element=F.binary_cross_entropy_with_logits(z,a['truth'].float(),reduction='none')
            for q in range(4):
                for positive in [False,True]:
                    mask=a['eligible'][:,q]&(a['truth'][:,q]==positive)
                    if mask.any():losses.append(element[:,q][mask].mean())
        loss=torch.stack(losses).mean();assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();opt.step()
        if step in [0,199,399,599]:
            history.append(dict(step=step+1,loss=float(loss.detach())));print(history[-1],flush=True)
    torch.save(dict(weight=weight.detach().cpu(),bias=bias.detach().cpu()),output/'gate.pt')
    for a in cohorts.values():
        with torch.inference_mode():a['score']=((torch.from_numpy(a['features']).cuda()*weight).sum(-1)+bias).cpu().numpy()
    dev=cohorts['DEV'];threshold=[]
    for q in range(4):
        e=dev['eligible'][:,q];negative=e&~dev['truth'][:,q]
        threshold.append(np.nextafter(dev['score'][negative,q].max(),np.float32(np.inf)) if negative.any() else (-np.inf if e.any() else np.inf))
    threshold=np.array(threshold,dtype=np.float32);np.save(output/'threshold.npy',threshold)
    result=dict(metrics={},families={},groups={},fit_history=history);arrays={}
    for key,a in cohorts.items():
        z,accepted=compose(a['baseline'],a['source'],a['available'],a['score'],threshold)
        a.update(candidate=z,accepted=accepted)
        t=a['truth'];b=a['baseline']>=0;c=z>=0
        item=dict(candidate=metrics(z,t),baseline=metrics(a['baseline'],t),added_tp=(accepted&t).sum(0).tolist(),
                  added_fp=(accepted&~t).sum(0).tolist(),baseline_positive_lost=int((b&~c).sum()))
        if 'previous' in a:
            prev=a['previous']>=0
            item.update(previous=metrics(a['previous'],t),vs_previous=dict(tp_gained=(c&~prev&t).sum(0).tolist(),tp_lost=(~c&prev&t).sum(0).tolist(),
                fp_added=(c&~prev&~t).sum(0).tolist(),fp_removed=(~c&prev&~t).sum(0).tolist()))
        result['metrics'][key]=item
        if key in ['relation10000','distance5000']:
            rows=[r for r in records if r['dataset']==key]
            result['families'][key]={}
            for family in sorted({r['family'] for r in rows}):
                mask=np.array([r['family']==family for r in rows]);result['families'][key][family]=dict(candidate=metrics(z[mask],t[mask]),
                    previous=metrics(a['previous'][mask],t[mask]),added_fp=(accepted[mask]&~t[mask]).sum(0).tolist())
            result['groups'][key]={}
            for field in ['site','group']:
                units={}
                for i,row in enumerate(rows):units.setdefault(row[field],[]).append(i)
                result['groups'][key][field]=dict(total=len(units),candidate=sum(bool(((z[ids]>=0)==t[ids]).all()) for ids in units.values()),
                    previous=sum(bool(((a['previous'][ids]>=0)==t[ids]).all()) for ids in units.values()))
        arrays.update({key+'/'+k:v for k,v in a.items()})
    thin=p11['clip']=='thin_pole_background_target';result['thin']={}
    for key in ['clean','stress','clean_wrong','stress_wrong']:
        a=cohorts[key];result['thin'][key]=dict(candidate=metrics(a['candidate'][thin],a['truth'][thin]),previous=metrics(a['previous'][thin],a['truth'][thin]))
    m=result['metrics'];sign_fp=sum(sum(result['families'][k]['hanging_sign']['added_fp']) for k in ['relation10000','distance5000'])
    gate=dict(sign_fp_reduced=sign_fp<17,
        transfer_fp_not_increased=all(all(x<=y for x,y in zip(m[k]['candidate']['fp'],m[k]['previous']['fp'])) for k in ['relation10000','distance5000']),
        transfer_far_not_lost=all(sum(m[k]['candidate']['tp'][1::2])>=sum(m[k]['previous']['tp'][1::2]) for k in ['relation10000','distance5000']),
        old_dev_near_not_lost=sum(m['DEV']['added_tp'][::2])>=15,old_dev_far_not_lost=sum(m['DEV']['added_tp'][1::2])>=17,
        thin_clean_not_lost=sum(result['thin']['clean']['candidate']['tp'][1::2])>=46,
        thin_stress_not_lost=sum(result['thin']['stress']['candidate']['tp'][1::2])>=44,
        baseline_positives_preserved=all(v['baseline_positive_lost']==0 for v in m.values()))
    result.update(gate=gate,favorable_challenger=all(gate.values()),sign_added_fp=sign_fp)
    np.savez_compressed(output/'predictions.npz',**arrays)
    write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS',training_steps=600,parameters=20,backend='CUDA',device=torch.cuda.get_device_name(),
        seconds=time.perf_counter()-start,outputs={p.name:sha(p) for p in output.iterdir()}))
    print('RESULT',{k:v for k,v in result.items() if k not in ['families','groups']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','features','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();main(a.root,a.features,a.output)
