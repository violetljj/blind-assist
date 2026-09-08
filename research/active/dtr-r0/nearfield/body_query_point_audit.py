"""Frozen B point decomposition; raw point readouts are not calibrated detectors."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from body_query_model import BodyQueryModel
from body_query_data import QueryRGB,truth,read,write,sha,fresh_output


def softmax(z):
    x=np.exp(z-z.max(-1,keepdims=True));return x/x.sum(-1,keepdims=True)


def near(prob):
    p=prob.reshape(-1,2,6,4);d0=np.ones(p.shape[:2]);d1=np.zeros_like(d0);d2=d1.copy()
    for j in range(6):
        q=p[:,:,j];d0,d1,d2=d0*q[:,:,0],d1*q[:,:,0]+d0*q[:,:,1],d2*q[:,:,0]+d1*q[:,:,1]+d0*q[:,:,2]
    return 1-np.clip(d0+d1+d2,1e-7,1-1e-7)


def counts(pred,y):
    return dict(TP=int((pred&y).sum()),FN=int((~pred&y).sum()),FP=int((pred&~y).sum()),TN=int((~pred&~y).sum()))


def summarize(dump,gt,records,valid,thresholds):
    z=dump['point_logits'].astype(float);cell=dump['cell_logits'].astype(float)
    pp=1-softmax(z)[...,0]
    score=np.logaddexp.reduce(z[...,1:],axis=-1)-z[...,0]
    rank=np.argsort(np.where(valid[None],score,-np.inf),axis=-1)[...,::-1]
    probes={'mean':cell}
    for name,k in (('max',1),('top3',3)):
        picked=np.take_along_axis(z,rank[...,:k,None],axis=2)
        probes[name]=picked.mean(2)
    probs={name:softmax(v) for name,v in probes.items()}
    means=1-probs['mean'][...,0];best=np.max(np.where(valid[None],pp,-np.inf),axis=-1)
    y=gt['counts']>0;miss=y&(means<.5)
    bestid=rank[:,:,0];overlap=np.take_along_axis(dump['point_positive_support'],bestid[...,None],axis=2)[...,0]
    unknown=np.take_along_axis(dump['point_unknown_support'],bestid[...,None],axis=2)[...,0]
    def cells(cols):
        yy=y[:,cols];mm=miss[:,cols];bb=best[:,cols];ov=overlap[:,cols];uu=unknown[:,cols]
        weak=mm&(bb<.5);strong=mm&(bb>=.5)
        return dict(cells=int(yy.size),positive=int(yy.sum()),mean_missed_positive=int(mm.sum()),
            missed_with_point_ge05=int(strong.sum()),missed_with_point_ge09=int((mm&(bb>=.9)).sum()),
            missed_all_points_below05=int(weak.sum()),
            rescued_point_support_any_overlap=int((strong&(ov>0)).sum()),
            rescued_point_support_majority_overlap=int((strong&(ov>=.5)).sum()),
            rescued_point_unknown_majority=int((strong&(uu>=.5)).sum()),
            methods={name:counts(1-p[:,cols,0]>=.5,yy) for name,p in probs.items()})
    result=dict(all=cells(np.arange(12)),heads={},near={},conditions={},missed_cells=[])
    for h,head in enumerate(('BODY','HEAD')):
        result['heads'][head]={'all':cells(np.arange(h*6,h*6+6)),
            'near':cells(np.arange(h*6,h*6+3)),'far':cells(np.arange(h*6+3,h*6+6))}
    for name,p in probs.items():
        n=near(p);alert=n>=thresholds
        result['near'][name]={head:counts(alert[:,h],gt['near'][:,h]==1) for h,head in enumerate(('BODY','HEAD'))}
        result['conditions'][name]={}
        for cond in dict.fromkeys(r['condition'] for r in records):
            ids=[i for i,r in enumerate(records) if r['condition']==cond]
            result['conditions'][name][cond]={head:counts(alert[ids,h],gt['near'][ids,h]==1) for h,head in enumerate(('BODY','HEAD'))}
    for i,c in zip(*np.where(miss)):
        result['missed_cells'].append(dict(sample_index=records[i]['sample_index'],family=records[i]['family'],condition=records[i]['condition'],
            cell=int(c),head='BODY' if c<6 else 'HEAD',range='near' if c%6<3 else 'far',mean_nonempty=float(means[i,c]),
            best_point_nonempty=float(best[i,c]),top3_nonempty=float(1-probs['top3'][i,c,0]),
            best_point=int(bestid[i,c]),positive_support_overlap=float(overlap[i,c]),unknown_support_overlap=float(unknown[i,c])))
    return result


def run(a):
    if not torch.cuda.is_available():raise RuntimeError('GPU inference required')
    out=fresh_output(a.output);start=time.perf_counter()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    selection=read(a.run/'selection.json');checkpoint=a.run/'B-step2000.pt'
    if sha(checkpoint)!=selection['B']['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
    model=BodyQueryModel(a.pretrained,'B').cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    valid=model.query_valid.cpu().numpy();projection=model.query_projection.cpu().numpy()
    if valid.sum(1).min()<3:raise ValueError('Top3 needs at least3 valid points')
    captured={}
    def hook(module,inputs,output):captured.update(point_inputs=inputs[0],point_features=output)
    handle=model.query_point.register_forward_hook(hook)
    hashes={str(checkpoint):sha(checkpoint),str(a.run/'selection.json'):sha(a.run/'selection.json'),str(a.cache/'manifest.json'):sha(a.cache/'manifest.json')}
    result=dict(scope='Frozen point-readout diagnostic; no calibrated point probabilities or causal backbone attribution',
        effective_points=valid.sum(1).tolist(),roles={},checks={})
    try:
        for role in ('train','dev','eval'):
            data=QueryRGB(a.cache,role);collected={k:[] for k in ('point_inputs','point_features','point_logits','cell_logits','near')}
            errors=[]
            with torch.inference_mode():
                for begin in range(0,len(data.ids),32):
                    n,s,c=model(data.tensor(np.arange(begin,min(begin+32,len(data.ids))),'cuda'))
                    point=F.linear(captured['point_features'],model.query_readout.weight,model.query_readout.bias)
                    reconstructed=(point*model.query_valid[None,:,:,None]).sum(2)/model.query_valid.sum(1)[None,:,None]
                    err=float((reconstructed-c).abs().max());errors.append(err)
                    if err>1e-4:raise ValueError('Linear masked-mean logit identity failed')
                    values={**captured,'point_logits':point,'cell_logits':c,'near':n.sigmoid()}
                    for key in collected:collected[key].append(values[key].cpu().numpy())
            dump={k:np.concatenate(v) for k,v in collected.items()}
            saved=a.run/f'B-{role}.npz';old=np.load(saved);hashes[str(saved)]=sha(saved)
            delta=float(np.abs(dump['near']-old['near']).max())
            if delta>1e-5:raise ValueError('Frozen batch near reproduction failed')
            rec,gt=truth(a.cache,role)
            support=gt['support'];positive=[];unknown=[]
            for cellid in range(12):
                w=projection[cellid*27:(cellid+1)*27]
                positive.append((support[:,cellid//6]==1).reshape(len(data.ids),-1)@w.T)
                unknown.append((support[:,cellid//6]<0).reshape(len(data.ids),-1)@w.T)
            dump.update(point_positive_support=np.stack(positive,1),point_unknown_support=np.stack(unknown,1))
            np.savez_compressed(out/f'{role}-points.npz',**dump,valid=valid,query_grid=model.query_grid.cpu().numpy())
            result['roles'][role]=summarize(dump,gt,rec['records'],valid,np.array(selection['B']['thresholds']))
            result['checks'][role]=dict(max_masked_mean_logit_error=max(errors),near_saved_max_error=delta,frames=len(data.ids))
            write(out/'progress.json',dict(role=role,frames=len(data.ids)));print(role,result['roles'][role]['all'],flush=True)
        for path,digest in hashes.items():
            if sha(Path(path))!=digest:raise ValueError('Frozen source changed')
        result['seconds']=time.perf_counter()-start;write(out/'result.json',result)
        write(out/'receipt.json',dict(status='PASS',training_steps=0,inference_frames=320,backend='CUDA',
            device=torch.cuda.get_device_name(),torch=torch.__version__,code_sha256=sha(Path(__file__)),
            input_sha256=hashes,result_sha256=sha(out/'result.json'),arrays={p.name:sha(p) for p in out.glob('*-points.npz')}))
    finally:handle.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache','run','pretrained','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
