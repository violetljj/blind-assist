"""A separate, preserved Explore fitting check after the negative 12-epoch pilot.

All four arms receive the same additional optimization and final-checkpoint rule.
No frozen T/T2 run is resumed and the first pilot's plan/scores are unchanged.
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

import cnh_delayed_query_dev as D
import cnh_delayed_query_model as M

OUT=D.OUT/'fit64'


def auc(scores,labels):
    positive=labels==1;n=int(positive.sum());m=int((~positive).sum())
    return float((rankdata(scores)[positive].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve fitting check')
    cfg=json.loads((D.OUT/'PLAN.json').read_text(encoding='utf8'))
    previous=json.loads((D.OUT/'run_receipt.json').read_text())['seconds']
    previous+=sum(json.loads(p.read_text())['seconds'] for p in D.OUT.glob('failure_*.json'))
    D.save(OUT/'PLAN.json',dict(task='CNH_DELAYED_QUERY_FIT64_DEV_20261009',lane='EXPLORE fitting diagnostic',
        reason='12epoch all4 negative, online hard BCE still decreasing; distinguish incomplete optimization from geometry mechanism',
        parent_plan_sha256=D.sha(D.OUT/'PLAN.json'),parent_seconds=previous,cumulative_GPU_cap_seconds=1200,
        additional_epochs=52,effective_epochs=64,optimizer_reset='Same new AdamW state in all4 arms',
        adjustable_scope='Focused source repair only for this fitting check; preserve pilot12 and all old frozen T/T2',
        model_selection='Last64epoch checkpoint, no evaluation-based epoch/arm selection',
        initialization='Each arm inherits its exact12epoch checkpoint; original all4 same seed/state and common order retained',
        stop='Additional52epoch complete or shared task GPU1200s reached; final one fit check not automatic baseline promotion',
        source_sha256=D.sha(__file__)))
    start=time.monotonic()
    def check():
        if previous+time.monotonic()-start>=1200:raise TimeoutError('Shared cumulative GPU cap')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(D.SEED)
    data={k:torch.as_tensor(np.array(np.load(D.OLD/'inputs/train'/f'{k}.npy',mmap_mode='r')),device='cuda')
          for k in ('labels','mask','weights','length')}
    y,w,valid,le=data['labels'],data['weights'],data['mask'].bool(),data['length'];n=len(le)
    traces=[];training=[];raws={}
    (OUT/'models').mkdir(exist_ok=True)
    try:
        for mode in ('center','extent'):
            features=torch.as_tensor(np.load(D.OUT/'features'/f'{mode}_train.npy',mmap_mode='r'),device='cuda')
            for supervision in ('bce','pair'):
                arm=f'{mode}_{supervision}';model=M.DelayedQueryReadout().cuda()
                checkpoint=torch.load(D.OUT/'models'/f'{arm}.pt',map_location='cuda',weights_only=True)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
                for epoch in range(12,64):
                    model.train();order=np.random.default_rng(D.SEED+epoch).permutation(n);num=den=0.
                    for offset in range(0,n,cfg['batch']):
                        check();ix=torch.as_tensor(order[offset:offset+cfg['batch']],device='cuda')
                        out=model(features[ix],le[ix]);bce=F.binary_cross_entropy_with_logits(out,y[ix],reduction='none')
                        hard=(bce*w[ix]).sum()/w[ix].sum().clamp_min(1e-9)
                        pp=valid[ix].all(1)&(y[ix,0]!=y[ix,1]);pw=w[ix].mean(1)*pp
                        contrast=(F.softplus(-(out[:,0]-out[:,1])*(y[ix,0]-y[ix,1]))*pw).sum()/pw.sum().clamp_min(1e-9)
                        loss=hard+(cfg['contrast_weight']*contrast if supervision=='pair' else 0)
                        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
                        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                        num+=float((bce.detach()*w[ix]).sum());den+=float(w[ix].sum())
                    row=dict(arm=arm,effective_epoch=epoch+1,online_BCE=num/den,seconds=time.monotonic()-start)
                    traces.append(row)
                    if (epoch+1)%8==0:print('FIT',arm,epoch+1,round(num/den,5),round(time.monotonic()-start,2),flush=True)
                model.eval();out=[]
                with torch.no_grad():
                    for off in range(0,n,cfg['batch']):check();out.append(model(features[off:off+cfg['batch']],le[off:off+cfg['batch']]))
                pred=torch.cat(out);hard=float((F.binary_cross_entropy_with_logits(pred,y,reduction='none')*w).sum()/w.sum())
                mask=valid.cpu().numpy();pn=pred.cpu().numpy();yn=y.cpu().numpy()
                training.append(dict(arm=arm,final_hard_BCE=hard,pooled_valid_train_AUC=auc(pn[mask],yn[mask])))
                torch.save(dict(state_dict=model.cpu().state_dict(),arm=arm,seed=D.SEED,epochs=64,parent_checkpoint_sha256=D.sha(D.OUT/'models'/f'{arm}.pt')),OUT/'models'/f'{arm}.pt')
                del model,optimizer,pred,out
            del features;torch.cuda.empty_cache()
            features=torch.as_tensor(np.load(D.OUT/'features'/f'{mode}_aligned.npy',mmap_mode='r'),device='cuda')
            evalle=torch.as_tensor(np.tile(np.minimum(np.arange(3,16)+1,8),492*4),device='cuda')
            for supervision in ('bce','pair'):
                arm=f'{mode}_{supervision}';model=M.DelayedQueryReadout().cuda().eval()
                model.load_state_dict(torch.load(OUT/'models'/f'{arm}.pt',map_location='cuda',weights_only=True)['state_dict']);pred=[]
                with torch.no_grad():
                    for off in range(0,len(features),cfg['batch']):
                        check();pred.append(model(features[off:off+cfg['batch']],evalle[off:off+cfg['batch']]).cpu().numpy())
                raws[arm]=np.concatenate(pred).reshape(492,4,13,2);del model
            del features;torch.cuda.empty_cache()
        np.savez_compressed(OUT/'scores.npz',arm_names=np.array(D.ARMS),raw=np.stack([raws[a] for a in D.ARMS]))
        D.save(OUT/'loss_curves.json',traces)
        D.save(OUT/'run_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-start,parent_seconds=previous,total_gpu_seconds=previous+time.monotonic()-start,training=training,device=torch.cuda.get_device_name(0),source_sha256=D.sha(__file__)))
    except Exception as e:
        D.save(OUT/f'failure_{time.time_ns()}.json',dict(seconds=time.monotonic()-start,error=repr(e)));D.save(OUT/'loss_curves_partial.json',traces);raise
    finally:torch.cuda.empty_cache()


if __name__=='__main__':run()
