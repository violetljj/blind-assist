"""Materially different signed-boundary bin-token Explore comparison.

Two new matched heads learn geometry/evidence BEFORE radial-bin compression.
Original pilot12, fit64, all frozen baselines and negative outputs stay intact.
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import cnh_delayed_query_dev as D
import cnh_delayed_query_eval as E
import cnh_delayed_query_supplement as S
import cnh_boundary_token_model as M

OUT=D.OUT/'boundary_tokens'
ARMS=('boundary_center','boundary_extent')


def features(h,t,le,mode,path,check,batch=64):
    began=time.monotonic();n=len(le)
    cache=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(n,2,16,15,8,8))
    with torch.no_grad():
        for off in range(0,n,batch):
            check();ix=slice(off,min(off+batch,n))
            hh=torch.as_tensor(np.array(h[ix]),device='cuda');tt=torch.as_tensor(np.array(t[ix]),device='cuda',dtype=torch.float32)
            ll=torch.as_tensor(np.array(le[ix]),device='cuda');g=M.feature_geometry(tt,ll,mode)
            f=M.build_features(hh,g,ll)
            if not bool(torch.isfinite(f).all()):raise ValueError('Nonfinite signed-boundary features')
            cache[ix]=f.cpu().numpy().astype(np.float16)
            if off%(batch*100)==0:print('BOUNDARY_FEATURES',mode,off,n,round(time.monotonic()-began,2),flush=True)
    cache.flush();del cache;torch.cuda.empty_cache()
    return dict(mode=mode,rows=n,seconds=time.monotonic()-began,dtype='float16',shape=[n,2,16,15,8,8])


def infer_dataset(h,t,le,label,check):
    result={};receipts={};lengths=torch.as_tensor(np.array(le),device='cuda')
    for mode in ('center','extent'):
        path=OUT/'features'/f'{label}_{mode}.npy'
        receipts[mode]=features(h,t,le,mode,path,check)
        x=torch.as_tensor(np.load(path,mmap_mode='r'),device='cuda');arm=f'boundary_{mode}'
        model=M.BoundaryTokenReadout().cuda().eval()
        model.load_state_dict(torch.load(OUT/'models'/f'{arm}.pt',map_location='cuda',weights_only=True)['state_dict'])
        pred=[]
        with torch.no_grad():
            for off in range(0,len(le),256):
                check();pred.append(model(x[off:off+256],lengths[off:off+256]).cpu().numpy())
        result[arm]=np.concatenate(pred);del x,model;torch.cuda.empty_cache()
    return result,receipts


def yaw_metrics(raw,ideal):
    with np.load(D.ALIGNED/'geometry.npz') as z:category=z['category']
    with np.load(S.POSE) as z:base=S.F.smooth(z['m3_raw'])
    baseline,timely=S.F.metrics(base>=S.B.THETA,category)
    if baseline['counts']!=[467,375] or baseline['clear_slots']!=176:raise ValueError('Yaw baseline parity')
    clear=(category=='clear').all(1);result=dict(baseline=baseline,arms={},query_yaw_degrees=3)
    for arm in ARMS:
        score=S.F.smooth(raw[arm].reshape(492,4,13,2));r=ideal['arms'][arm];threshold=np.inf if r['threshold_is_positive_infinity'] else r['threshold']
        metrics,candidate=S.F.metrics(score>=threshold,category);paired=S.F.compare(timely,candidate,category)
        E.independent_check(score>=threshold,category,metrics,candidate,timely,paired)
        theta,cost=S.F.nearest(score[clear].max(-1),176);mm,mt=S.F.metrics(score>=theta,category)
        result['arms'][arm]=dict(fixed_ideal_threshold=dict(threshold=threshold,metrics=metrics,paired_minus_M3=paired),
            yaw_cost_matched_diagnostic=dict(threshold=theta,metrics=mm,residual=cost-176,paired_minus_M3=S.F.compare(timely,mt,category)))
    return result


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'PLAN.json').exists():raise FileExistsError('Preserve boundary-token run')
    previous=json.loads((D.OUT/'fit64/supplement/run_receipt.json').read_text())['cumulative_seconds']
    D.save(OUT/'PLAN.json',dict(task='CNH_SIGNED_BOUNDARY_TOKEN_DEV_20261009',lane='EXPLORE same consumed simulation',
        reason='Coverage-only features improve held-unit ranking vs central ray but underperform M3; more fitting reduces full-batch retention. Change representation instead of adding epochs.',
        shared_gpu_cap_seconds=1200,previous_gpu_seconds=previous,seed=D.SEED,epochs=32,batch=256,lr=.001,wd=.0001,
        initialization='Both modes use exact same random state, same row order, same BCE and weights; final checkpoint only',
        mechanism='Signed directional query-face margins and observed radial bins enter shared token encoder; only AFTER learned interaction pool radial bins and spatial neighborhoods',
        geometry='Center ray versus angular3x3 quadrature with per-face minimum signed margin; footprints are hypotheses, no true subzone reconstruction',
        supervision='Same39936 inherited training observations, no evaluation labels/boxes in model and no pair loss in these two arms',
        decision_check='688 contact opportunities/height,46 of4576 jointly-clear slots, full paired event losses; final32 checkpoint. One seed Explore not promotion.',
        stop='Complete two matched arms, aligned/yaw/held-unit check or shared GPU cap; retain all partials/failures',
        source_sha256=D.sha(__file__),model_source_sha256=D.sha(M.__file__),parent_plan_sha256=D.sha(D.OUT/'PLAN.json')))
    began=time.monotonic()
    def check():
        if previous+time.monotonic()-began>=1200:raise TimeoutError('Shared task GPU1200s cap')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(D.SEED)
    init=M.BoundaryTokenReadout().state_dict()
    (OUT/'features').mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    data={k:np.load(D.OLD/'inputs/train'/f'{k}.npy',mmap_mode='r') for k in ('histories','transforms','length','labels','weights')}
    y=torch.as_tensor(np.array(data['labels']),device='cuda');w=torch.as_tensor(np.array(data['weights']),device='cuda');le=torch.as_tensor(np.array(data['length']),device='cuda');n=len(le)
    receipt={};traces=[]
    try:
        for mode in ('center','extent'):
            path=OUT/'features'/f'train_{mode}.npy'
            receipt['train_'+mode]=features(data['histories'],data['transforms'],data['length'],mode,path,check)
            x=torch.as_tensor(np.load(path,mmap_mode='r'),device='cuda');model=M.BoundaryTokenReadout().cuda();model.load_state_dict(init)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);arm=f'boundary_{mode}'
            for epoch in range(32):
                model.train();order=np.random.default_rng(D.SEED+epoch).permutation(n);num=den=0.
                for off in range(0,n,256):
                    check();ix=torch.as_tensor(order[off:off+256],device='cuda');out=model(x[ix],le[ix])
                    hard=F.binary_cross_entropy_with_logits(out,y[ix],reduction='none');loss=(hard*w[ix]).sum()/w[ix].sum().clamp_min(1e-9)
                    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite boundary loss')
                    opt.zero_grad(set_to_none=True);loss.backward();opt.step();num+=float((hard.detach()*w[ix]).sum());den+=float(w[ix].sum())
                traces.append(dict(arm=arm,epoch=epoch+1,online_BCE=num/den,seconds=time.monotonic()-began))
                if (epoch+1)%4==0:print('BOUNDARY_TRAIN',arm,epoch+1,round(num/den,5),round(time.monotonic()-began,2),flush=True)
            torch.save(dict(state_dict=model.cpu().state_dict(),arm=arm,seed=D.SEED,epochs=32,plan_sha256=D.sha(OUT/'PLAN.json')),OUT/'models'/f'{arm}.pt')
            del model,opt,x;torch.cuda.empty_cache()
        D.save(OUT/'loss_curves.json',traces)
        h,t,length=D.aligned_observations();raw,receipt['aligned']=infer_dataset(h,t,length,'aligned',check)
        np.savez_compressed(OUT/'scores.npz',arm_names=np.array(ARMS),raw=np.stack([raw[a].reshape(492,4,13,2) for a in ARMS]))
        ideal=E.evaluate(OUT/'evaluation',OUT/'scores.npz');del h,t,length,raw
        h,t,length=D.aligned_observations(3.);raw,receipt['yaw']=infer_dataset(h,t,length,'yaw',check)
        np.savez_compressed(OUT/'yaw_scores.npz',arm_names=np.array(ARMS),raw=np.stack([raw[a].reshape(492,4,13,2) for a in ARMS]))
        D.save(OUT/'yaw_metrics.json',yaw_metrics(raw,ideal));del h,t,length,raw
        folder=D.OLD/'inputs/fresh_evaluation'
        h,t,length=[np.load(folder/f'{name}.npy',mmap_mode='r') for name in ('histories','transforms','length')]
        raw,receipt['held_units']=infer_dataset(h,t,length,'held',check)
        np.savez_compressed(OUT/'fresh_scores.npz',arm_names=np.array(ARMS),raw=np.stack([raw[a] for a in ARMS]))
        D.save(OUT/'fresh_metrics.json',S.fresh_metrics(raw))
        D.save(OUT/'run_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,previous_seconds=previous,total_gpu_seconds=previous+time.monotonic()-began,features=receipt,parameters=sum(p.numel() for p in M.BoundaryTokenReadout().parameters()),device=torch.cuda.get_device_name(0),source_sha256=D.sha(__file__),model_source_sha256=D.sha(M.__file__)))
    except Exception as e:
        D.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(e),seconds=time.monotonic()-began,previous_seconds=previous));D.save(OUT/'loss_curves_partial.json',traces);raise
    finally:torch.cuda.empty_cache()


if __name__=='__main__':run()
