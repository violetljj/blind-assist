"""Frozen M3 conditional score sensitivity, endpoint noise validation.

The gradient belongs to a continuous readout at the quantized mean feature,
not to FP16 rounding. Actual Monte Carlo traverses both FP16 casts.
"""
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S

REP = B.ROOT/'artifacts.local/work/cnh-bar-representation-dev-20261008'
OUT = REP/'readout'
N = 64
ENDS = (12,13)
SCALES = (.25,1.)
NOISE_PREFIX = 2026100843


def imports():
    A,G,R,normal,T=S.imports()
    A.OUT=OUT/'runtime'
    return A,R,normal


def prepare():
    A,R,normal=imports()
    import cnh_temporal_readout_data as D
    fp=json.loads((REP/'fp16/PLAN.json').read_text())
    contacts=fp['contact_scene_ids']; passes={int(k):v for k,v in fp['B_pass_pairs'].items()}
    endpoint_ids=sorted(set(contacts)|set(passes.values()))+[-1]
    with np.load(S.OUT/'geometry.npz') as z: cat=z['category']
    pairs=[]
    for i in contacts:
        q=int(np.flatnonzero(cat[i]=='contact')[0])
        pairs.append(dict(scene=i,reference=-1,contrast='A_absent',height_idx=q))
        if i in passes:pairs.append(dict(scene=i,reference=passes[i],contrast='B_pass',height_idx=q))
    sources={Path(__file__),*[Path(m.__file__).resolve() for n,m in list(sys.modules.items())
               if n.startswith('cnh_') and getattr(m,'__file__',None)]}
    inputs=[S.OUT/'PLAN.json',S.OUT/'physical.npz',S.OUT/'geometry.npz',REP/'fp16/PLAN.json',
            REP/'fp16/reconstruction.npz',REP/'projection_bounds.json',
            *[REP/f'ideal_map_f{f}.npz' for f in ENDS],
            B.ROOT/'artifacts.local/work/cnh-bar-cached-diagnostic-dev-20261008/background_reference.npz',D.D.BIAS]
    plan=dict(task='CNH_BAR_READOUT_DEV_20261008',lane='EXPLORE consumed fixed geometry Development',
        authorization='User continued after representation diagnostics; execute frozen M3 readout sensitivity and actual-noise variance check',
        base_commit='e04b1363',goal='Determine signed A/B score response and whether local Jacobian variance predicts conditional noisy score behavior',
        compute_budgets_wall_seconds=dict(run=900,analysis=180,independent_audit=180),
        adjustable_scope='Implementation repairs, focused validation and same fixed endpoints only; no threshold/layout/model/training change',
        stop='Complete fixed endpoint diagnostics or applicable cumulative cap; preserve failures/partial endpoint files',
        endpoint_ids=endpoint_ids,contacts=contacts,pairs=pairs,ends=list(ENDS),scales=list(SCALES),replicas=N,
        unique_endpoints=37,mean_windows=74,pair_windows=104,ensemble_MC_examples=37*2*2*N,
        M3_models_sha256={B.logical_path(p):B.sha(p) for p in A.M3_MODELS},
        inputs_sha256={B.logical_path(p):B.sha(p) for p in inputs},
        source_sha256={B.logical_path(p):B.sha(p) for p in sorted(sources,key=str)},
        noise='Endpoint independent signed Skellam R.sample(mu,ambient,seed), full16 frames each draw. SeedSequence([2026100843,scene_id+1,k]); shared endpoint reuses draw at both ends/scales. alpha=.25 scales centered noise, not a lower-photon observation law. alpha=1 is original noise law. New conditional noise, not new geometry/confirmation/hardware data.',
        Jacobian='5-net mean logit gradient at actual FP16 mean voxels, prepared input leaf then exact signedlog derivative1/(1+abs(voxel)), including at zero. Casts held fixed, max uses selected subgradient. Pullback tot and last together to hist; V_j=mu_j+16ambient, separately at each endpoint.',
        decision_check='d_J uses signed nominal score delta over sqrt(avg endpoint propagated variance); d_MC uses signed MC mean delta over sqrt(avg actual endpoint variance). Bootstrap intervals and same-draw tracking compare both endpoints and two noise scales. Interval containing1 is UNRESOLVED, not validated linearity. Variance mismatch restricts Jacobian interpretation to sensitivity. None of these statistics is an information-loss chain or timely alarm metric.',
        deliverables='Endpoint and paired readout/variance tables, original cached-score parity, independent chain audit, report/run/current/commit/push')
    B.save(OUT/'PLAN.json',plan);(OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('PREPARED',37,'endpoints',104,'pairs',plan['ensemble_MC_examples'],'MC ensemble examples',flush=True)


class CachedProjection:
    """Precompute original Engine.project geometry; same gather/reduction order."""
    def __init__(self, engine, matrices):
        from cnh_cvr_projection import SUB,SHAPE,EDGE,WIDTH
        torch=engine.torch; p=engine.projector
        t=torch.as_tensor(matrices,dtype=torch.float32,device='cuda')
        pts=p.points.float();vol=p.volumes.float()
        xyz=torch.bmm(pts[None]-t[:,None,:3,3],t[:,:3,:3])
        radius=torch.linalg.vector_norm(xyz,dim=2)
        ij=torch.floor((xyz[:,:,:2]/xyz[:,:,2:3].clamp_min(1e-30)+EDGE)/(2*EDGE)*8).long()
        bins=torch.floor(radius/WIDTH).long()
        valid=(xyz[:,:,2]>0)&(ij>=0).all(2)&(ij<8).all(2)&(bins>=0)&(bins<16)
        self.index=(ij[:,:,1].clamp(0,7)*8+ij[:,:,0].clamp(0,7))*16+bins.clamp(0,15)
        self.weight=valid*p.voxel_volume/(SUB**3)/vol[self.index]
        self.count=valid.reshape(8,-1,SUB**3).float().mean(2).sum(0).reshape(SHAPE)
        self.torch=torch;self.shape=SHAPE;self.sub=SUB

    def __call__(self,z):
        torch=self.torch;c=len(z)
        values=torch.as_tensor(np.ascontiguousarray(z),dtype=torch.float32,device='cuda').reshape(c,8,1024)
        mass=torch.gather(values,2,self.index[None].expand(c,-1,-1))*self.weight[None]
        ev=mass.reshape(c,8,-1,self.sub**3).sum(3).reshape(c,8,*self.shape)
        return torch.stack((ev.sum(1),self.count[None].expand(c,-1,-1,-1),ev[:,-1]),1)


def run():
    started=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text())
    spent=sum(json.loads(f.read_text())['seconds'] for f in OUT.glob('run_failure_*.json'))
    def check():
        if spent+time.monotonic()-started>plan['compute_budgets_wall_seconds']['run']:
            raise TimeoutError('readout cumulative run900s wall cap')
    engine=None;maps=[]
    try:
        for name,digest in {**plan['inputs_sha256'],**plan['source_sha256'],**plan['M3_models_sha256']}.items():
            assert B.sha(B.ROOT/name)==digest,('changed frozen input/source',name)
        A,R,normal=imports()
        import torch
        import cnh_temporal_readout_data as D
        from cnh_temporal_readout_model import prepare_voxels
        engine=A.Engine()
        for net in engine.nets:
            for parameter in net.parameters():parameter.requires_grad_(False)
        ids=plan['endpoint_ids'];scene_ids=ids[:-1]
        with np.load(S.OUT/'physical.npz') as z:
            mu_present=z['expectation'][scene_ids];ambient=z['ambient'];cached_raw=z['raw'][scene_ids]
        with np.load(S.OUT/'geometry.npz') as z:sensor=z['sensor'];query=z['public_query']
        with np.load(B.ROOT/'artifacts.local/work/cnh-bar-cached-diagnostic-dev-20261008/background_reference.npz') as z:
            bg=z['expectation'];np.testing.assert_array_equal(ambient,z['ambient'])
        mu=np.concatenate((mu_present,bg[None]))
        bias=np.load(D.D.BIAS).astype(np.float32)
        den=np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9))
        def normalize32(hist):return (np.asarray(hist,np.float32)-bias)/den
        transforms=[];parities=[]
        for end in ENDS:
            m=(query[end]@np.linalg.inv(sensor[end]))[None]@sensor[end-7:end+1]
            transforms.append(m);maps.append(CachedProjection(engine,m))
        # Replay all144 cached K4 at the two original frames against shape raw logits.
        with np.load(REP/'fp16/reconstruction.npz') as z:
            sample_count=len(z['sample_scene_ids'])*4
            assert z['sample_scene_ids'].tolist()==scene_ids
            original_features=z['feature_fp16'][:sample_count]
        replay=engine.predict(original_features.reshape(-1,3,24,17,33)).reshape(36,4,2,2)
        expected=cached_raw[:,:,[12-3,13-3]]
        np.testing.assert_allclose(replay,expected,rtol=3e-5,atol=3e-5)
        B.save(OUT/'cached_score_parity.json',dict(rows=288,max_abs=float(np.max(np.abs(replay-expected))),
            seconds=time.monotonic()-started,meaning='Both fixed windows of all original144K4; logits only, no new alarm policy'))
        # Preserve exact new conditional draws once; endpoints/scales reuse them.
        draws_path=OUT/'draws.npy';assert not draws_path.exists()
        draws=np.lib.format.open_memmap(draws_path,mode='w+',dtype=np.int32,shape=(37,N,16,8,8,16))
        mean_scores=np.empty((37,2,2),np.float32);jvar=np.empty((37,2,2),np.float64)
        jhist=np.empty((37,2,2,8,8,8,16),np.float64)
        jvox=np.empty((37,2,2,2,24,17,33),np.float32)
        meanvox=np.empty((37,2,3,24,17,33),np.float16)
        gaps=np.empty((37,2,5,2,32),np.float32)
        mc=np.empty((37,2,2,N,2),np.float32);linear=np.empty((37,2,N,2),np.float64)
        sparse_maps=[sparse.load_npz(REP/f'ideal_map_f{end}.npz') for end in ENDS]
        for ei,end in enumerate(ENDS):
            # Meaningful map parity before using precomputed geometry for all new draws.
            first=end-7
            probe=normalize32(mu[:4]).astype(np.float16)[:,first:end+1]
            with torch.no_grad():
                actual=engine.project(probe,np.broadcast_to(transforms[ei],(len(probe),8,4,4)))
                cached=maps[ei](probe)
            err=float((actual-cached).abs().max())
            np.testing.assert_allclose(actual.cpu().numpy(),cached.cpu().numpy(),rtol=0,atol=0)
            parities.append(dict(end=end,maximum_error=err))
            del actual,cached
        for endpoint,scene_id in enumerate(ids):
            check()
            for k in range(N):
                seed=int(np.random.SeedSequence([NOISE_PREFIX,scene_id+1,k]).generate_state(1)[0])
                draws[endpoint,k]=R.sample(mu[endpoint],ambient,seed)[0]
            noise=draws[endpoint].astype(float)-mu[endpoint]
            for ei,end in enumerate(ENDS):
                check();first=end-7
                zmean=normalize32(mu[endpoint]).astype(np.float16)[None,first:end+1]
                with torch.no_grad():voxel=maps[ei](zmean).half()
                meanvox[endpoint,ei]=voxel[0].cpu().numpy()
                leaf=prepare_voxels(voxel,engine.masks).detach().requires_grad_(True)
                fs=[]
                handles=[]
                for net in engine.nets:
                    handles.append(net.body.register_forward_hook(lambda _m,_i,out:fs.append(out.detach())))
                try:
                    score=torch.stack([net(leaf).float() for net in engine.nets]).mean(0)
                finally:
                    for h in handles:h.remove()
                mean_scores[endpoint,ei]=score.detach().cpu().numpy()[0]
                mask=torch.nn.functional.adaptive_avg_pool3d(engine.masks[None],fs[0].shape[2:])[0]
                for ni,f in enumerate(fs):
                    ranked=f[0][None].expand(2,-1,-1,-1,-1).masked_fill(mask[:,None]==0,-1e4).flatten(2).topk(2,dim=2).values
                    gaps[endpoint,ei,ni]=(ranked[:,:,0]-ranked[:,:,1]).cpu().numpy()
                for q in (0,1):
                    grad=torch.autograd.grad(score[0,q],leaf,retain_graph=q==0)[0][0]
                    # Exact continuous derivative even at voxel0; original forward unchanged.
                    gv=grad[[0,2]]/(1+voxel[0,[0,2]].float().abs())
                    jvox[endpoint,ei,q]=gv.detach().cpu().numpy()
                    p=sparse_maps[ei];g=jvox[endpoint,ei,q].reshape(2,-1).astype(float)
                    hist_grad=p.T@g[0]
                    hist_grad[-1024:]+=p[:,-1024:].T@g[1]
                    hist_grad=hist_grad/den[first:end+1].ravel()
                    jhist[endpoint,ei,q]=hist_grad.reshape(8,8,8,16)
                    jvar[endpoint,ei,q]=np.sum(hist_grad**2*(mu[endpoint,first:end+1]+16*ambient[first:end+1,...,None]).ravel())
                    linear[endpoint,ei,:,q]=noise[:,first:end+1].reshape(N,-1)@hist_grad
                del score,grad,gv,leaf,voxel,fs
                for si,alpha in enumerate(SCALES):
                    values=mu[endpoint]+alpha*noise
                    z=normalize32(values).astype(np.float16)[:,first:end+1]
                    features=[]
                    with torch.no_grad():
                        for b in range(0,N,4):features.append(maps[ei](z[b:b+4]).half().cpu().numpy())
                    mc[endpoint,ei,si]=engine.predict(np.concatenate(features))
                check()
            B.save(OUT/f'endpoint_{scene_id}.json',dict(endpoint=scene_id,mean_scores=mean_scores[endpoint].tolist(),
                jacobian_variance=jvar[endpoint].tolist(),seconds=time.monotonic()-started))
            draws.flush()
            print('ENDPOINT',endpoint+1,'/37',scene_id,'wall',round(time.monotonic()-started,1),flush=True)
        arrays=dict(endpoint_ids=ids,ends=ENDS,scales=SCALES,mean_scores=mean_scores,jacobian_variance=jvar,
            jacobian_hist=jhist,jacobian_voxel=jvox,mean_voxels=meanvox,mc_scores=mc,linear_noise=linear,mu=mu,
            ambient=ambient,bias=bias,denominator=den,max_pool_gaps=gaps)
        assert all(np.isfinite(v).all() for k,v in arrays.items() if isinstance(v,np.ndarray))
        assert (jvar>=0).all()
        assert not (OUT/'predictions.npz').exists()
        np.savez_compressed(OUT/'predictions.npz',**arrays)
        draws.flush();del draws
        B.save(OUT/'result.json',dict(status='COMPLETE',seconds=time.monotonic()-started,cumulative_seconds=spent+time.monotonic()-started,
            unique_endpoints=37,mean_windows=74,pair_windows=104,replicas=N,scales=SCALES,
            MC_ensemble_examples=37*2*2*N,nominal_ensemble_examples=74,cached_replay_ensemble_examples=288,
            jacobian_score_backwards=148,backend=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,
                inference='CUDA FP32 five frozen models with original prepare_voxels, two FP16 stages',moment_pullback='CPU SciPy sparse dot: small sparse-vector/metadata TASK_NOT_GPU_SUITABLE'),
            cached_projection_parity=parities,draws_sha256=B.sha(draws_path),prediction_sha256=B.sha(OUT/'predictions.npz'),
            semantics='Continuous local subgradient at quantized mean feature; actual-noise agreement evaluated separately. New conditional noise is not new independent geometry or confirmation.'))
        check()
    except BaseException as exc:
        B.save(OUT/f'run_failure_{time.time_ns()}.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-started));raise
    finally:
        maps.clear()
        if engine is not None:
            engine.projector=None;engine.nets=[]
            for key in ('_pts32','_vol32'):
                if hasattr(engine,key):delattr(engine,key)
            engine.torch.cuda.synchronize();engine.torch.cuda.empty_cache()
            if hasattr(engine,'_dll'):engine._dll.close()
        gc.collect()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run'))
    args=parser.parse_args();globals()[args.stage]()
