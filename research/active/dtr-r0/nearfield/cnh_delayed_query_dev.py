"""Explore delayed bin/query interaction; read-only inherited observations.

New architecture and run, not a restart of frozen T/T2. Supervised query contrast
uses only the two valid training queries of the SAME observed row. It is not a
background intervention or a newly generated obstacle-removal counterfactual.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-delayed-query-dev-20261009'
OLD = ROOT/'artifacts.local/work/cnh-temporal-readout-20261004'
ALIGNED = ROOT/'artifacts.local/work/cnh-aligned-shapes-dev-20261008'
SEED = 2026100955
ARMS = ('center_bce', 'extent_bce', 'center_pair', 'extent_pair')


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def plan():
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Existing run plan preserved')
    sources = [Path(__file__), Path(__file__).with_name('cnh_delayed_query_model.py')]
    inputs = [OLD/'inputs/train'/f'{n}.npy' for n in ('histories','transforms','length','labels','mask','weights')]
    inputs += [ALIGNED/'physical.npz', ALIGNED/'geometry.npz', ALIGNED/'evaluated.npz']
    save(OUT/'PLAN.json', dict(task='CNH_DELAYED_QUERY_DEV_20261009', lane='EXPLORE consumed simulated Development',
        authorization='User 推进 the lightweight representation and fair comparison proposal',
        base_commit='41d4baf3beb98dfc556c1841abdf6b0432be2786',
        goal='Test bin-preserving query interaction with angular footprint instead of central-ray-only geometry',
        budgets_seconds=dict(GPU_stage_cumulative_wall=1200, CPU_preparation_analysis=600),
        adjustable_scope='Source repair and useful focused comparisons within budget; preserve old runs and all failed outputs',
        arms=list(ARMS), seed=SEED, epochs=12, batch=512, lr=.001, wd=.0001, contrast_weight=.2,
        geometry='3x3 zone slope midpoint nodes, radial-bin centers; soft-box exp(-L1 outside distance/.05m); mean/min/max. center repeats one ray.',
        mechanism='Keep16 bins, gate per-time bin evidence by query BEFORE temporal mean/current and spatial encoding; no GRU or occupancy reconstruction',
        supervision='Same inherited39936 rows and weights. Pair adds same-observation HEAD/BODY contrast only when both masks valid and labels differ. No background counterfactual claim.',
        fairness='Identical network/initial state/order/optimizer/epochs/calibration. Both representations cached FP16; final checkpoint only. Frozen M3/k5 practical references are unmatched pretrained baselines.',
        primary='492 scenes K4 f3..15; HEAD/BODY688 each; joint clear4576. Whole-score-tie threshold nearest frozen M3 46clear slots; residual reported.',
        decision_check='M3 HEAD517/BODY406 of688; headroom171/282. One event=1/688. Report paired gains/losses and all clear slot/segment/clip/pass costs. Investment decision descriptive, no invented formal promotion gate.',
        limitations='One seed; fixed simulator and finite background; Development workpoint selection. Angular footprint is a query proxy, not recovered subzone object geometry or physical sensor proof.',
        stop='Cumulative GPU1200s or completed scoped comparison; preserve partial evidence and release owning CUDA process',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        deliverables='Runnable model/train/eval, checkpoints, loss curves, full event ledger, focused checks, report and scoped commit/push'))
    print('PLAN SAVED', flush=True)


def prepare_features(histories, transforms, lengths, mode, path, check, batch=64):
    import cnh_delayed_query_model as M
    began = time.monotonic()
    n = len(histories)
    maps = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(n,2,128,8,8))
    with torch.no_grad():
        for start in range(0,n,batch):
            check()
            ix = slice(start,min(start+batch,n))
            h = torch.as_tensor(np.array(histories[ix]), device='cuda')
            t = torch.as_tensor(np.array(transforms[ix]), device='cuda', dtype=torch.float32)
            le = torch.as_tensor(np.array(lengths[ix]), device='cuda')
            g = M.feature_geometry(t,le,mode)
            features = M.build_features(h,g,le)
            if not bool(torch.isfinite(features).all()):
                raise ValueError('Nonfinite observation features')
            maps[ix] = features.cpu().numpy().astype(np.float16)
            if start % (batch*100) == 0:
                print('FEATURES',mode,start,'/',n,round(time.monotonic()-began,2),flush=True)
    maps.flush()
    del maps
    torch.cuda.empty_cache()
    return dict(rows=n, mode=mode, seconds=time.monotonic()-began, dtype='float16', shape=[n,2,128,8,8])


def aligned_observations(query_yaw=0.):
    # Observation and calibration ONLY. Evaluator category is opened by eval script.
    with np.load(ALIGNED/'physical.npz') as z:
        hist, ambient = z['hist'], z['ambient']
    with np.load(ALIGNED/'geometry.npz') as z:
        sensor, query = z['sensor'], z['public_query']
    if query_yaw:
        a=np.deg2rad(query_yaw); c,s=np.cos(a),np.sin(a)
        rot=np.eye(4); rot[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
        query=rot[None]@query
    bias=np.load(ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy').astype(np.float32)
    z=((hist.astype(np.float32)-bias)/np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9))).astype(np.float16)
    n=492*4*13
    histories=np.zeros((n,8,8,8,16),np.float16)
    transforms=np.tile(np.eye(4),(n,8,1,1))
    lengths=np.zeros(n,np.int64)
    offset=0
    for scene in range(492):
        for replica in range(4):
            for f in range(3,16):
                begin=max(0,f-7); le=f-begin+1; pad=8-le
                histories[offset,pad:]=z[scene,replica,begin:f+1]
                transforms[offset,pad:]=query[f]@np.linalg.inv(sensor[f])@sensor[begin:f+1]
                lengths[offset]=le; offset+=1
    return histories, transforms, lengths


def execute():
    import cnh_delayed_query_model as M
    cfg=json.loads((OUT/'PLAN.json').read_text(encoding='utf8'))
    if (OUT/'run_receipt.json').exists():
        raise FileExistsError('Completed run preserved')
    began=time.monotonic()
    previous=sum(json.loads(p.read_text())['seconds'] for p in OUT.glob('failure_*.json'))
    def check():
        if previous+time.monotonic()-began>cfg['budgets_seconds']['GPU_stage_cumulative_wall']:
            raise TimeoutError('Cumulative GPU stage budget reached')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(SEED)
    init=M.DelayedQueryReadout().state_dict()
    data={k:np.load(OLD/'inputs/train'/f'{k}.npy',mmap_mode='r') for k in ('histories','transforms','length','labels','weights','mask')}
    n=len(data['length'])
    if n!=39936:
        raise ValueError('Inherited train identity mismatch')
    (OUT/'features').mkdir(exist_ok=True)
    (OUT/'models').mkdir(exist_ok=True)
    labels=torch.as_tensor(np.array(data['labels']),device='cuda')
    weights=torch.as_tensor(np.array(data['weights']),device='cuda')
    valid=torch.as_tensor(np.array(data['mask']),device='cuda').bool()
    lengths=torch.as_tensor(np.array(data['length']),device='cuda')
    receipts={}
    allraw={}
    trace=[]
    training=[]
    try:
        for mode in ('center','extent'):
            feature_path=OUT/'features'/f'{mode}_train.npy'
            receipts[mode]=prepare_features(data['histories'],data['transforms'],data['length'],mode,feature_path,check)
            features=torch.as_tensor(np.load(feature_path,mmap_mode='r'),device='cuda')
            for supervision in ('bce','pair'):
                arm=f'{mode}_{supervision}'
                model=M.DelayedQueryReadout().cuda()
                model.load_state_dict(init)
                optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
                armstart=time.monotonic()
                for epoch in range(cfg['epochs']):
                    model.train()
                    perm=np.random.default_rng(SEED+epoch).permutation(n)
                    numerator=denominator=pair_sum=0.
                    for start in range(0,n,cfg['batch']):
                        check()
                        ix=torch.as_tensor(perm[start:start+cfg['batch']],device='cuda')
                        out=model(features[ix].float(),lengths[ix])
                        bce=F.binary_cross_entropy_with_logits(out,labels[ix],reduction='none')
                        mass=weights[ix].sum()
                        hard=(bce*weights[ix]).sum()/mass.clamp_min(1e-9)
                        pp=valid[ix].all(1)&(labels[ix,0]!=labels[ix,1])
                        pw=weights[ix].mean(1)*pp
                        difference=(out[:,0]-out[:,1])*(labels[ix,0]-labels[ix,1])
                        contrast=(F.softplus(-difference)*pw).sum()/pw.sum().clamp_min(1e-9)
                        loss=hard+(cfg['contrast_weight']*contrast if supervision=='pair' else 0)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('Nonfinite training loss')
                        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                        numerator+=float((bce.detach()*weights[ix]).sum());denominator+=float(mass)
                        pair_sum+=float(contrast.detach())
                    row=dict(arm=arm,epoch=epoch+1,hard_BCE=numerator/denominator,query_contrast_batchmean=pair_sum/int(np.ceil(n/cfg['batch'])),seconds=time.monotonic()-armstart)
                    trace.append(row);save(OUT/'progress.json',dict(stage='train',**row))
                    print('TRAIN',arm,epoch+1,round(row['hard_BCE'],5),round(row['seconds'],2),flush=True)
                torch.save(dict(state_dict=model.cpu().state_dict(),arm=arm,seed=SEED,epochs=cfg['epochs'],parameters=sum(p.numel() for p in model.parameters()),plan_sha256=sha(OUT/'PLAN.json')),OUT/'models'/f'{arm}.pt')
                training.append(dict(arm=arm,seconds=time.monotonic()-armstart,final_hard_BCE=trace[-1]['hard_BCE'],parameters=sum(p.numel() for p in model.parameters())))
                del model,optimizer
            del features
            torch.cuda.empty_cache()
        hist,t,le=aligned_observations()
        for mode in ('center','extent'):
            fp=OUT/'features'/f'{mode}_aligned.npy'
            receipts[mode+'_aligned']=prepare_features(hist,t,le,mode,fp,check)
            features=torch.as_tensor(np.load(fp,mmap_mode='r'),device='cuda')
            lengths_eval=torch.as_tensor(le,device='cuda')
            for supervision in ('bce','pair'):
                arm=f'{mode}_{supervision}'
                model=M.DelayedQueryReadout().cuda().eval()
                model.load_state_dict(torch.load(OUT/'models'/f'{arm}.pt',map_location='cuda',weights_only=True)['state_dict'])
                result=[]
                with torch.no_grad():
                    for start in range(0,len(le),cfg['batch']):
                        check();ix=slice(start,min(start+cfg['batch'],len(le)))
                        result.append(model(features[ix].float(),lengths_eval[ix]).cpu().numpy())
                allraw[arm]=np.concatenate(result).reshape(492,4,13,2)
                del model
            del features
            torch.cuda.empty_cache()
        np.savez_compressed(OUT/'scores.npz',arm_names=np.array(ARMS),raw=np.stack([allraw[a] for a in ARMS]))
        save(OUT/'loss_curves.json',trace)
        save(OUT/'run_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,previous_failed_seconds=previous,backend='CUDA',device=torch.cuda.get_device_name(0),torch_version=torch.__version__,rows=n,features=receipts,training=training,source_sha256=sha(Path(__file__)),model_source_sha256=sha(Path(M.__file__)),model_initialization='same exact state dict all4',pair_eligible_rows=int((valid.all(1)&(labels[:,0]!=labels[:,1])).sum()),scores_sha256=sha(OUT/'scores.npz')))
    except Exception as e:
        save(OUT/f'failure_{time.time_ns()}.json',dict(status='FAILED',seconds=time.monotonic()-began,error=repr(e)))
        save(OUT/'loss_curves_partial.json',trace)
        raise
    finally:
        # No spawned worker or resident service: owning command exit releases CUDA.
        torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['plan','run'])
    args=parser.parse_args()
    (plan if args.stage=='plan' else execute)()
