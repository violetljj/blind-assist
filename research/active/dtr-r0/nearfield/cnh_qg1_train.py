"""Frozen QG-1 gates on existing alley Development. Never opens test/City."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import precision_recall_curve
from cnh_rgb_dev_comparison import checked, read_inputs, sha, SEEDS, BATCH, EPOCHS
from cnh_rgb_alley_v2 import train_pos_weight, COLLECTION_SHA256, PARTITION_SHA256
from cnh_learning_diagnostic import select32
from cnh_rgb_visible_depth_metrics import rank_metrics
from cnh_rgb_v2_perturbation import within_layout_cycle
from cnh_qg1_model import QueryGeometryModel, depth_features
from cnh_qg1_geometry import ray_coordinates

COLLECTION=Path('artifacts.local/evidence/cnh-alley-rgb-replay-six-20260925-v2/collection-overlay.json')
PARTITION=Path('artifacts.local/work/cnh-route-comparison-20260924/plan/street-alley-merged-partitions-20260924-v1.json')
PREPARED=Path('artifacts.local/evidence/cnh-visible-depth-audit-20260925-v1')
ARMS=('full_depth','tof_sim','tof_rgb')


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def threshold_from_train(y,s):
    known=y>=0
    p,r,t=precision_recall_curve(y[known],s[known])
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    return float(t[np.flatnonzero(f==f.max())[-1]])


def metrics(y,s,t):
    k=y>=0
    return dict(**rank_metrics(y,s),tp=int(((y==1)&(s>=t)).sum()),fp=int(((y==0)&(s>=t)).sum()),
                fn=int(((y==1)&(s<t)).sum()),tn=int(((y==0)&(s<t)).sum()),known=int(k.sum()))


class Inputs:
    def __init__(self):
        assert sha(COLLECTION)==COLLECTION_SHA256 and sha(PARTITION)==PARTITION_SHA256
        self.data=read_inputs(COLLECTION,PARTITION)
        item=self.data['collection']['layouts'][0]
        overlay=json.loads(checked({'path':item['overlay'],'sha256':item['overlay_sha256']}).read_text(encoding='utf-8-sig'))
        camera=json.loads(checked(overlay['frames'][0]['original_files']['camera.json']).read_text(encoding='utf-8-sig'))
        self.native_k=np.asarray(camera['K'],float)
        self.small_k=self.native_k.copy()
        self.small_k[:2,:2]*=.2
        self.small_k[:2,2]=(self.small_k[:2,2]+.5)*.2-.5
        self.depth=np.load(PREPARED/'visible-depth-f32.npy',mmap_mode='r')
        rec=json.loads((PREPARED/'result.json').read_text())
        assert sha(PREPARED/'visible-depth-f32.npy')==rec['files']['visible-depth-f32.npy']
        self.device=torch.device('cuda')
        self.rays=torch.tensor(ray_coordinates(self.native_k,360,640),device=self.device)
        self.small_rays=torch.tensor(ray_coordinates(self.small_k,72,128),device=self.device)
        self.tensors={name:torch.tensor(self.data[name],device=self.device) for name in ('histogram','ambient','scalar','valid')}
        self.rgb=torch.tensor(self.data['rgb'].astype(np.float32)/255,device=self.device)
        self.labels=torch.tensor(np.maximum(self.data['labels'],0),device=self.device,dtype=torch.float32)
        self.known=torch.tensor(self.data['labels']>=0,device=self.device)
        w,self.balance=train_pos_weight(self.data['labels'],self.data['train'])
        self.weight=torch.tensor(w,device=self.device)

    def model(self,arm,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        k,size=(self.native_k,(640,360)) if arm=='full_depth' else (self.small_k,(128,72))
        return QueryGeometryModel(k,size).to(self.device)

    def forward(self,model,arm,indices,rgb_source=None):
        ix=torch.tensor(indices,device=self.device)
        if arm=='full_depth':
            image=depth_features(torch.tensor(np.asarray(self.depth[indices]).copy(),device=self.device),self.rays)
        elif arm=='tof_rgb':
            source=ix if rgb_source is None else torch.tensor(rgb_source[indices],device=self.device)
            image=torch.cat((self.rgb[source],self.small_rays[None].expand(len(indices),-1,-1,-1)),1)
        else:
            image=None
        return model(image,*(self.tensors[n][ix] for n in ('histogram','ambient','scalar','valid')),arm=arm)

    def predict(self,model,arm,indices,rgb_source=None):
        model.eval()
        with torch.no_grad():
            return np.concatenate([self.forward(model,arm,indices[i:i+BATCH],rgb_source).cpu().numpy()
                                   for i in range(0,len(indices),BATCH)])


def gate0(inputs,out):
    ids=select32(inputs.data)
    original=json.loads(Path('artifacts.local/evidence/cnh-learning-diagnostic-20260925-v2/overfit-selection.json').read_text())
    assert ids.tolist()==original['indices']
    assert [inputs.data['rows'][i]['frame_key'] for i in ids]==original['frame_keys']
    write(out/'gate0-selection.json',dict(indices=ids.tolist(),frame_keys=original['frame_keys']))
    results=[]
    for arm in ARMS:
        model=inputs.model(arm,20260924)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        generator=torch.Generator().manual_seed(20260924)
        started=time.monotonic()
        history=[]
        grad={}
        stop='UPDATE_LIMIT'
        backup=False
        for step in range(1,1201):
            batch=ids[torch.randperm(32,generator=generator)[:BATCH].numpy()]
            model.train()
            opt.zero_grad(set_to_none=True)
            logits=inputs.forward(model,arm,batch)
            loss=F.binary_cross_entropy_with_logits(logits,inputs.labels[batch],pos_weight=inputs.weight,reduction='none')[inputs.known[batch]].mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite Gate0 loss')
            loss.backward()
            if step==1:
                grad={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.grad is not None}
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            if step==1 or step%50==0:
                pred=inputs.predict(model,arm,ids)
                row=dict(step=step,loss=float(loss.detach()),lr=opt.param_groups[0]['lr'],
                         **rank_metrics(inputs.data['labels'][ids],pred),wall_s=time.monotonic()-started)
                history.append(row)
                write(out/'progress.json',dict(stage='gate0',arm=arm,**row))
                print(json.dumps(dict(stage='gate0',arm=arm,**row)),flush=True)
                if row['auprc']>=.99:
                    stop='TRAIN_AP_GE_0.99'
                    break
                if step==600:
                    backup=True
                    for group in opt.param_groups: group['lr']=1e-3
            if time.monotonic()-started>=1200:
                stop='WALL_LIMIT'
                break
        pred=inputs.predict(model,arm,ids)
        torch.save({k:v.cpu() for k,v in model.state_dict().items()},out/f'gate0-{arm}.pt')
        np.savez_compressed(out/f'gate0-{arm}.npz',indices=ids,logits=pred,labels=inputs.data['labels'][ids])
        result=dict(arm=arm,steps=step,stop=stop,backup_lr_used=backup,final=rank_metrics(inputs.data['labels'][ids],pred),
                    history=history,first_grad_norm=grad,wall_s=time.monotonic()-started)
        results.append(result)
        write(out/'gate0.json',dict(status='RUNNING',results=results))
    passed=results[0]['final']['auprc']>=.99
    report=dict(status='PASS' if passed else 'STOP_FULL_DEPTH_FIT_FAILED',results=results,gate1_allowed=passed)
    write(out/'gate0.json',report)
    return report


def gate1(inputs,out):
    g0=json.loads((out/'gate0.json').read_text())
    if not g0['gate1_allowed']:
        raise ValueError('Gate0 failed; Gate1 prohibited')
    train=np.flatnonzero(inputs.data['train'])
    allids=np.arange(len(inputs.data['rows']))
    results=[]
    for seed in SEEDS:
        for arm in ARMS:
            dest=out/f'{arm}-{seed}-result.json'
            if dest.exists():
                raise FileExistsError('No implicit rerun/resume: '+str(dest))
            model=inputs.model(arm,seed)
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
            generator=torch.Generator().manual_seed(seed)
            history=[]
            started=time.monotonic()
            for epoch in range(EPOCHS):
                order=train[torch.randperm(len(train),generator=generator).numpy()]
                losses=[]
                model.train()
                for start in range(0,len(train),BATCH):
                    ids=order[start:start+BATCH]
                    opt.zero_grad(set_to_none=True)
                    logits=inputs.forward(model,arm,ids)
                    loss=F.binary_cross_entropy_with_logits(logits,inputs.labels[ids],pos_weight=inputs.weight,reduction='none')[inputs.known[ids]].mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite Gate1 loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    opt.step()
                    losses.append(float(loss.detach()))
                scheduler.step()
                row=dict(epoch=epoch+1,loss=float(np.mean(losses)),wall_s=time.monotonic()-started)
                history.append(row)
                write(out/'progress.json',dict(stage='gate1',arm=arm,seed=seed,**row))
                print(json.dumps(dict(stage='gate1',arm=arm,seed=seed,**row)),flush=True)
            logits=inputs.predict(model,arm,allids)
            y=inputs.data['labels'];tm=inputs.data['train'];dm=inputs.data['dev']
            threshold=threshold_from_train(y[tm],logits[tm])
            layouts={name:metrics(y[sel],logits[sel],threshold)
                     for name in sorted({r['layout_id'] for r in inputs.data['rows']})
                     for sel in [np.array([r['layout_id']==name for r in inputs.data['rows']])]}
            extra={}
            if arm=='tof_rgb':
                permutation=within_layout_cycle(inputs.data['rows'],dm)
                shuffled=inputs.predict(model,arm,np.flatnonzero(dm),permutation)
                extra['rgb_shuffled_dev']=metrics(y[dm],shuffled,threshold)
                np.savez_compressed(out/f'{arm}-{seed}-shuffle.npz',logits=shuffled,source=permutation)
            modelpath=out/f'{arm}-{seed}.pt'
            predpath=out/f'{arm}-{seed}-predictions.npz'
            torch.save({k:v.cpu() for k,v in model.state_dict().items()},modelpath)
            np.savez_compressed(predpath,logits=logits,frame_key=[r['frame_key'] for r in inputs.data['rows']],train=tm,dev=dm)
            result=dict(arm=arm,seed=seed,train=metrics(y[tm],logits[tm],threshold),dev=metrics(y[dm],logits[dm],threshold),
                        threshold_train_only=threshold,layouts=layouts,history=history,wall_s=time.monotonic()-started,
                        model_sha256=sha(modelpath),predictions_sha256=sha(predpath),**extra)
            write(dest,result);results.append(result)
            write(out/'gate1.json',dict(status='RUNNING',results=results))
    paired={s:{r['arm']:r for r in results if r['seed']==s} for s in SEEDS}
    decision=dict(structure=all(paired[s]['full_depth']['dev']['auprc']>=.80 for s in SEEDS),
                  tof=all(paired[s]['tof_sim']['dev']['auprc']>.2314 and paired[s]['tof_sim']['dev']['auprc']>.2101 for s in SEEDS),
                  rgb=all(paired[s]['tof_rgb']['dev']['auprc']>paired[s]['tof_sim']['dev']['auprc'] and
                          paired[s]['tof_rgb']['rgb_shuffled_dev']['auprc']<paired[s]['tof_rgb']['dev']['auprc'] for s in SEEDS))
    write(out/'gate1.json',dict(status='COMPLETE',results=results,decision=decision,
                              claim_limit='Reused degenerate Development structure diagnostic; no independent RGB/generalization benefit claim.'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('gate0','gate1'),required=True)
    args=p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if args.stage=='gate0':args.output.mkdir(parents=True,exist_ok=False)
    elif not args.output.is_dir():raise FileNotFoundError(args.output)
    try:
        inputs=Inputs()
        source={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('cnh_qg1_model.py'),Path(__file__).with_name('cnh_qg1_geometry.py'),Path(__file__).with_name('CNH_QG1_PROTOCOL_20260925.md')]}
        if args.stage=='gate0':
            write(args.output/'identity.json',dict(source=source,collection_sha256=sha(COLLECTION),partition_sha256=sha(PARTITION),
                                                 balance=inputs.balance,device=torch.cuda.get_device_name(),torch=torch.__version__))
            gate0(inputs,args.output)
        else:
            assert source==json.loads((args.output/'identity.json').read_text())['source']
            gate1(inputs,args.output)
    except BaseException as exc:
        write(args.output/f'{args.stage}-failure.json',dict(status='FAILED',error=repr(exc)))
        raise


if __name__=='__main__':main()
