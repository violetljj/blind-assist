"""Two matched fixed-budget fits: G13 readout versus native-count bottleneck."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from pathlib import Path
import time
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from body_query_model import BodyQueryModel,CALIBRATION
from body_query_data import QueryRGB,truth,read,write,sha,fresh_output
from city_data import pixel_support_bce
from city_dev_selection import select_threshold,apply_thresholds
from city_pilot_metrics import evaluate

STEPS=2000
SEED=17
BATCH=32
QUERY_WEIGHT=.25
SUPPORT_WEIGHT=.25
INITIAL_SHA='0c14179486102993508dd2385a3b6f9e17d51a6d53e657c0240325e5f78d5c7b'


def digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def uncertainty(scores,thresholds):
    """Fixed half-logit decision margin. A heuristic abstention, not safety."""
    p=np.clip(np.asarray(scores,dtype=float),1e-7,1-1e-7)
    t=np.clip(np.asarray(thresholds,dtype=float),1e-7,1-1e-7)
    return np.abs(np.log(p/(1-p))-np.log(t/(1-t)))<.5


def metrics(pred,target,record,thresholds):
    n,s,c=pred['near'],pred['support'],pred['counts'];y=target['near']
    groups=[r['group_id'] for r in record['records']]
    result=evaluate(n,s,y,target['support'],groups)
    decisions=apply_thresholds(n,thresholds);u=uncertainty(n,thresholds)
    for h,name in enumerate(('BODY','HEAD')):
        pos=y[:,h]==1;neg=y[:,h]==0;p=decisions[:,h]
        tp,fp=int((p&pos).sum()),int((p&neg).sum())
        auc=float(((n[pos,h,None]>n[neg,h]).astype(float)+.5*(n[pos,h,None]==n[neg,h])).mean()) if pos.any() and neg.any() else None
        result['heads'][name]['selected']=dict(TP=tp,FP=fp,FN=int(pos.sum())-tp,TN=int(neg.sum())-fp,
            recall=tp/int(pos.sum()) if pos.any() else None,FPR=fp/int(neg.sum()) if neg.any() else None,
            AUC=auc,threshold=float(thresholds[h]),UNKNOWN=int(u[:,h].sum()),coverage=float((~u[:,h]).mean()),
            retained_correct=int(((p==y[:,h])&~u[:,h]).sum()),abstention='Fixed half-logit margin; primary counts include every frame')
        positive=target['support'][:,h]==1
        hit=(s[:,h]>=.5)&positive
        result['heads'][name]['support'].update(positive_pixels=int(positive.sum()),positive_pixel_hits=int(hit.sum()),
            positive_pixel_recall=float(hit.sum()/positive.sum()) if positive.any() else None)
    allgroups=list(dict.fromkeys(groups));correct=0;rows=[]
    for g in allgroups:
        ids=[i for i,v in enumerate(groups) if v==g]
        good=bool((decisions[ids]==y[ids]).all());correct+=good
        rows.append(dict(group=g,frames=len(ids),correct=good,unknown_frames=int(u[ids].any(1).sum())))
    result['selected_groups']=dict(correct=correct,total=len(allgroups),rows=rows)
    result['query_count_accuracy']=float((c.argmax(-1)==target['counts']).mean())
    qp=1-c[:,:,0];qy=target['counts']>0
    result['query_visible_support']=dict(TP=int(((qp>=.5)&qy).sum()),FP=int(((qp>=.5)&~qy).sum()),
        FN=int(((qp<.5)&qy).sum()),TN=int(((qp<.5)&~qy).sum()))
    result['conditions']={}
    conditions=[r['condition'] for r in record['records']]
    for condition in dict.fromkeys(conditions):
        ids=np.array([i for i,v in enumerate(conditions) if v==condition])
        result['conditions'][condition]=dict(frames=len(ids),truth_positive=y[ids].sum(0).tolist(),
            false_positives=((decisions[ids])&(y[ids]==0)).sum(0).tolist(),
            misses=((~decisions[ids])&(y[ids]==1)).sum(0).tolist(),both_correct=int((decisions[ids]==y[ids]).all(1).sum()))
    return result


@torch.inference_mode()
def predict(model,data):
    model.eval();rows=[[],[],[]]
    for begin in range(0,len(data.ids),BATCH):
        outputs=model(data.tensor(np.arange(begin,min(begin+BATCH,len(data.ids))),'cuda'))
        for j,t in enumerate(outputs):rows[j].append((t.softmax(-1) if j==2 else t.sigmoid()).cpu().numpy())
    return dict(zip(('near','support','counts'),[np.concatenate(v) for v in rows]))


@torch.inference_mode()
def benchmark(model,data):
    x=data.tensor(np.array([0]),'cuda');model.eval()
    for _ in range(5):model(x)
    times=[]
    for _ in range(30):
        torch.cuda.synchronize();start=time.perf_counter();model(x);torch.cuda.synchronize()
        times.append((time.perf_counter()-start)*1000)
    return dict(batch=1,samples=30,p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
        device=torch.cuda.get_device_name(),scope='Warm model forward only, excludes preprocessing and file IO')


def run(a):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for matched training')
    if sha(a.initial)!=INITIAL_SHA:raise ValueError('Original G13 seed17 checkpoint required')
    out=fresh_output(a.output);start=time.perf_counter()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    train=QueryRGB(a.cache,'train');record,target=truth(a.cache,'train',training=True)
    if record['sample_indices']!=train.ids:raise ValueError('TRAIN row mismatch')
    schedule=np.random.default_rng(SEED).integers(0,len(train.ids),(STEPS,BATCH))
    np.save(out/'schedule.npy',schedule,allow_pickle=False)
    protocol=dict(seed=SEED,steps=STEPS,batch=BATCH,optimizer='AdamW',lr=1e-5,weight_decay=1e-4,
        support_weight=SUPPORT_WEIGHT,query_weight=QUERY_WEIGHT,BN='FROZEN_RUNNING_STATISTICS',
        query='Twelve fixed spatial partitions, count classes0/1/2/3+, unweighted mean CE',
        near_B='Probability six independent capped counts sum>=3',calibration=CALIBRATION,
        data_sha256=sha(a.cache/'manifest.json'),initial_sha256=sha(a.initial),schedule_sha256=sha(out/'schedule.npy'),
        training_frames=len(train.ids),sampled_positive=target['near'][schedule].sum((0,1)).tolist(),
        selection='Final2000 only; original DEV FPR<=.10 selector, min_count8 for smaller fresh Development split',
        uncertainty='Same fixed half-logit decision margin, secondary descriptive coverage only',
        source_sha256={p:sha(Path(__file__).with_name(p)) for p in ('body_query_train.py','body_query_model.py','body_query_data.py','body_query_labels.py','city_data.py','city_dev_selection.py','decoupled_model.py','representation_model.py')},
        torch=torch.__version__,device=torch.cuda.get_device_name(),scope='One-seed same-world Development; no protected TEST or default-App promotion')
    write(out/'protocol.json',protocol)
    train_rgb=torch.from_numpy(train.rgb.copy()).cuda().permute(0,3,1,2).float()/255.
    targets={k:torch.from_numpy(v).cuda() for k,v in target.items()}
    first_hash=None;fits={}
    try:
        for arm in ('A','B'):
            torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
            model=BodyQueryModel(a.pretrained,arm).cuda()
            model.initialize_g13(torch.load(a.initial,map_location='cpu',weights_only=True))
            init=digest(model.state_dict())
            if first_hash is None:first_hash=init
            if init!=first_hash:raise ValueError('A/B initial tensors differ')
            before=digest(dict(model.named_buffers()))
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
            model.train()
            for layer in model.modules():
                if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();fit_start=time.perf_counter();history=[]
            for step,indices in enumerate(schedule,1):
                ids=torch.from_numpy(indices).cuda()
                n,s,c=model(train_rgb[ids])
                ln=F.binary_cross_entropy_with_logits(n,targets['near'][ids].float())
                ls=pixel_support_bce(s,targets['support'][ids])
                # Equivalent mean CE; avoid CUDA nll_loss2d's nondeterministic kernel.
                lc=-c.log_softmax(-1).gather(-1,targets['counts'][ids].long().unsqueeze(-1)).mean()
                loss=ln+SUPPORT_WEIGHT*ls+QUERY_WEIGHT*lc
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if step==1:
                    grads={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
                    if arm=='B' and any(name.startswith('near.') for name in grads):raise ValueError('B has near feature bypass')
                    write(out/f'{arm}-first-gradients.json',grads)
                optimizer.step()
                if step%100==0:
                    row=dict(arm=arm,step=step,near=float(ln.detach()),support=float(ls.detach()),query=float(lc.detach()),loss=float(loss.detach()))
                    history.append(row);write(out/'progress.json',row);print(row,flush=True)
            torch.cuda.synchronize()
            if digest(dict(model.named_buffers()))!=before:raise ValueError('Fixed buffers changed')
            torch.save(model.state_dict(),out/f'{arm}-step2000.pt')
            torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),steps=STEPS),out/f'{arm}-resume.pt')
            fits[arm]=dict(steps=STEPS,seconds=time.perf_counter()-fit_start,history=history,initial_digest=init,
                checkpoint_sha256=sha(out/f'{arm}-step2000.pt'),parameters=sum(p.numel() for p in model.parameters()),
                inactive_near_parameters=sum(p.numel() for p in model.near.parameters()) if arm=='B' else 0,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())
            write(out/f'{arm}-fit-complete.json',fits[arm])
            del model,optimizer,n,s,c,loss,ln,ls,lc
            torch.cuda.empty_cache()
        # Both fits finish before any DEV outcomes. EVAL opens only after cutoffs saved.
        del train_rgb,targets
        selection={};result=dict(fits=fits,arms={},scope=protocol['scope'])
        for arm in ('A','B'):
            model=BodyQueryModel(a.pretrained,arm).cuda();model.load_state_dict(torch.load(out/f'{arm}-step2000.pt',weights_only=True,map_location='cuda'))
            data=QueryRGB(a.cache,'dev');devrec,devtruth=truth(a.cache,'dev')
            pred=predict(model,data);np.savez_compressed(out/f'{arm}-dev.npz',**pred)
            heads=[select_threshold(pred['near'][:,h],devtruth['near'][:,h],min_count=8) for h in range(2)]
            selection[arm]=dict(thresholds=[r['threshold'] for r in heads],heads=heads,checkpoint_sha256=fits[arm]['checkpoint_sha256'])
            result['arms'][arm]=dict(dev=metrics(pred,devtruth,devrec,selection[arm]['thresholds']))
            del model
        write(out/'selection.json',selection);selection_sha=sha(out/'selection.json')
        for arm in ('A','B'):
            model=BodyQueryModel(a.pretrained,arm).cuda();model.load_state_dict(torch.load(out/f'{arm}-step2000.pt',weights_only=True,map_location='cuda'))
            for role in ('train','eval'):
                data=QueryRGB(a.cache,role);rec,gt=truth(a.cache,role)
                pred=predict(model,data);np.savez_compressed(out/f'{arm}-{role}.npz',**pred)
                result['arms'][arm][role]=metrics(pred,gt,rec,selection[arm]['thresholds'])
                if role=='eval':result['arms'][arm]['runtime']=benchmark(model,data)
            del model
        if sha(out/'selection.json')!=selection_sha or sha(a.cache/'manifest.json')!=protocol['data_sha256']:
            raise ValueError('Frozen selection/data changed')
        result.update(seconds=time.perf_counter()-start,selection_sha256=selection_sha)
        write(out/'result.json',result)
        write(out/'receipt.json',dict(status='PASS',fits=2,steps_per_fit=STEPS,result_sha256=sha(out/'result.json'),
            selection_sha256=selection_sha,backend='CUDA',device=torch.cuda.get_device_name(),torch=torch.__version__))
        return result
    except Exception as exc:
        write(out/'failure.json',dict(error=repr(exc),completed_fits=list(fits)));raise


def main():
    p=argparse.ArgumentParser()
    for name in ('cache','pretrained','initial','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a)


if __name__=='__main__':main()
