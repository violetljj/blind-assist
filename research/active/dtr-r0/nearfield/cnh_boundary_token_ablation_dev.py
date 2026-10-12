"""Matched retraining without the six signed query-face coordinates.

Same architecture/nominal8833 parameters, seed,32epochs, data/order/optimizer.
Zero channels6:12 on GPU copies only; inherited and full-model caches unchanged.
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import cnh_delayed_query_dev as D
import cnh_delayed_query_eval as E
import cnh_delayed_query_supplement as S
import cnh_boundary_token_dev as B
import cnh_boundary_token_model as M

OUT=D.OUT/'boundary_ablation'
ARM='boundary_no_signed_faces'


def load_features(label):
    x=torch.as_tensor(np.load(B.OUT/'features'/f'{label}_center.npy',mmap_mode='r'),device='cuda')
    x[:,:,:,6:12]=0
    return x


def infer(label,lengths,model,check):
    x=load_features(label);le=torch.as_tensor(np.array(lengths),device='cuda');out=[];model.eval()
    with torch.no_grad():
        for off in range(0,len(x),256):check();out.append(model(x[off:off+256],le[off:off+256]).cpu().numpy())
    del x;torch.cuda.empty_cache()
    return np.concatenate(out)


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve ablation')
    previous=json.loads((B.OUT/'run_receipt.json').read_text())['total_gpu_seconds']
    previous+=sum(json.loads(p.read_text(encoding='utf-8-sig'))['seconds'] for p in OUT.glob('startup_failure*.json'))
    D.save(OUT/'PLAN.json',dict(task='CNH_BOUNDARY_SIGNED_FACE_ABLATION_DEV_20261009',lane='EXPLORE matched retraining after full-model result',
        hypothesis='Six signed query-face coordinates add value beyond membership/outside distance/radius and bin-token architecture',
        mask_channels=[6,7,8,9,10,11],retained='All evidence, membership3, outside-distance2, radial radius',
        nominal_parameters=8833,seed=D.SEED,epochs=32,batch=256,lr=.001,wd=.0001,
        initialization='Exact same torch CPU seed and network initialization as full boundary center; same per-epoch row permutations',
        model_selection='Last32 checkpoint only. No evaluation-based checkpoint selection; exploration chose this ablation after previous outcomes.',
        cumulative_GPU_cap=1200,previous_GPU_seconds=previous,stop='One matched32epoch ablation plus aligned/yaw/held check or shared cap',
        source_sha256=D.sha(__file__),parent_PLAN_sha256=D.sha(B.OUT/'PLAN.json')))
    start=time.monotonic()
    def check():
        if previous+time.monotonic()-start>=1200:raise TimeoutError('Shared GPU budget')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(D.SEED)
    model=M.BoundaryTokenReadout().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    data={k:torch.as_tensor(np.array(np.load(D.OLD/'inputs/train'/f'{k}.npy',mmap_mode='r')),device='cuda') for k in ('labels','weights','length')}
    y,w,le=data['labels'],data['weights'],data['length'];n=len(le);x=load_features('train');trace=[]
    try:
        for epoch in range(32):
            model.train();order=np.random.default_rng(D.SEED+epoch).permutation(n);num=den=0.
            for off in range(0,n,256):
                check();ix=torch.as_tensor(order[off:off+256],device='cuda');out=model(x[ix],le[ix])
                hard=F.binary_cross_entropy_with_logits(out,y[ix],reduction='none');loss=(hard*w[ix]).sum()/w[ix].sum().clamp_min(1e-9)
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite ablation loss')
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();num+=float((hard.detach()*w[ix]).sum());den+=float(w[ix].sum())
            trace.append(dict(epoch=epoch+1,online_BCE=num/den,seconds=time.monotonic()-start))
            if (epoch+1)%4==0:print('ABLATION',epoch+1,round(num/den,5),round(time.monotonic()-start,2),flush=True)
        (OUT/'models').mkdir(exist_ok=True)
        torch.save(dict(state_dict=model.cpu().state_dict(),arm=ARM,seed=D.SEED,epochs=32,plan_sha256=D.sha(OUT/'PLAN.json')),OUT/'models'/f'{ARM}.pt')
        model.cuda().eval();del x,opt;torch.cuda.empty_cache()
        evalle=np.tile(np.minimum(np.arange(3,16)+1,8),492*4)
        raw=infer('aligned',evalle,model,check).reshape(492,4,13,2)
        np.savez_compressed(OUT/'scores.npz',arm_names=np.array([ARM]),raw=raw[None])
        ideal=E.evaluate(OUT/'evaluation',OUT/'scores.npz')
        yaw=infer('yaw',evalle,model,check).reshape(492,4,13,2)
        with np.load(D.ALIGNED/'geometry.npz') as z:category=z['category']
        with np.load(S.POSE) as z:base=S.F.smooth(z['m3_raw'])
        bm,bt=S.F.metrics(base>=S.B.THETA,category);threshold=ideal['arms'][ARM]['threshold'];sm=S.F.smooth(yaw);met,tm=S.F.metrics(sm>=threshold,category)
        D.save(OUT/'yaw_metrics.json',dict(baseline=bm,arms={ARM:dict(fixed_ideal_threshold=dict(threshold=threshold,metrics=met,paired_minus_M3=S.F.compare(bt,tm,category)))}))
        np.savez_compressed(OUT/'yaw_scores.npz',arm_names=np.array([ARM]),raw=yaw[None])
        length=np.load(D.OLD/'inputs/fresh_evaluation/length.npy',mmap_mode='r');fresh=infer('held',length,model,check)
        np.savez_compressed(OUT/'fresh_scores.npz',arm_names=np.array([ARM]),raw=fresh[None]);D.save(OUT/'fresh_metrics.json',S.fresh_metrics({ARM:fresh}))
        D.save(OUT/'loss_curves.json',trace)
        D.save(OUT/'run_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-start,previous_seconds=previous,total_gpu_seconds=previous+time.monotonic()-start,source_sha256=D.sha(__file__),ablation_scope='Signed per-face coordinates, not pure direction or physical localization'))
    except Exception as e:
        D.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(e),seconds=time.monotonic()-start,previous_seconds=previous));D.save(OUT/'loss_curves_partial.json',trace);raise
    finally:torch.cuda.empty_cache()


if __name__=='__main__':run()
