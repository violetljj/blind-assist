"""Conditional same-window A/B separation in the frozen M3 projection.

No inference, training or new photon samples. Whitened sparse row-space solves
include shared voxel covariance; cnt has no contrast at fixed pose.
"""
import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S

OUT = B.ROOT/'artifacts.local/work/cnh-bar-representation-dev-20261008'
CACHED = B.ROOT/'artifacts.local/work/cnh-bar-cached-diagnostic-dev-20261008'


def solve_space(p, delta, variance, denominator, tol=1e-9, maxiter=4000):
    """Project standardized mean difference into row(P diag(sqrt(V)/den))."""
    u = np.asarray(delta).ravel()/np.sqrt(np.asarray(variance).ravel())
    w = p.multiply((np.sqrt(variance)/denominator).ravel()[None]).tocsr()
    norm = np.sqrt(np.asarray(w.multiply(w).sum(1)).ravel())
    w = w[norm > 0].multiply((1/norm[norm > 0])[:, None]).tocsr()
    wt = w.T.tocsr()
    answer = lsmr(wt, u, atol=tol, btol=tol, conlim=1e12, maxiter=maxiter)
    q = wt @ answer[0]
    r = u-q
    q2 = float(q@q); ur = float(u@q)
    orth = float(abs(q@r)/max(float(u@u), 1e-30))
    coverage = np.asarray(wt.getnnz(axis=1)).ravel() > 0
    info = dict(d2=q2, input_d2=float(u@u), covered_d2=float(u[coverage]@u[coverage]),
                lower_bound_d2=ur*ur/q2 if q2 else 0.,
                subtraction_d2=float(u@u-r@r), orthogonality_relative=orth,
                istop=int(answer[1]), iterations=int(answer[2]), normal_residual=float(answer[4]),
                condition_estimate=float(answer[6]),
                converged=bool(answer[1] in (0,1,2,4,5) and orth < 2e-7),
                columns=int(w.shape[1]), active_columns=int(coverage.sum()), active_rows=int(w.shape[0]))
    return q, info


def map_window(transforms, projector, torch):
    """Same FP32 indexing/weights as the actual shapes Engine.project path."""
    from cnh_cvr_projection import SHAPE, SUB, EDGE, WIDTH
    n = int(np.prod(SHAPE)); t = torch.as_tensor(transforms, dtype=torch.float32, device='cuda')
    pts = projector.points.float(); vol = projector.volumes.float()
    xyz = torch.bmm(pts[None]-t[:,None,:3,3], t[:,:3,:3])
    radius = torch.linalg.vector_norm(xyz, dim=2)
    ij = torch.floor((xyz[:,:,:2]/xyz[:,:,2:3].clamp_min(1e-30)+EDGE)/(2*EDGE)*8).long()
    bins = torch.floor(radius/WIDTH).long()
    valid = (xyz[:,:,2]>0)&(ij>=0).all(2)&(ij<8).all(2)&(bins>=0)&(bins<16)
    index = (ij[:,:,1].clamp(0,7)*8+ij[:,:,0].clamp(0,7))*16+bins.clamp(0,15)
    weight = valid*projector.voxel_volume/(SUB**3)/vol[index]
    inds = index.cpu().numpy(); weights = weight.cpu().numpy()
    row = np.repeat(np.arange(n), SUB**3)
    blocks = [sparse.coo_matrix((v[v!=0].astype(float), (row[v!=0], i[v!=0])), shape=(n,1024)).tocsr()
              for i,v in zip(inds,weights)]
    p = sparse.hstack(blocks,format='csr')
    return p, blocks, inds, weights


def fixture():
    rng = np.random.default_rng(734)
    p = rng.normal(size=(4,9)); p[-1] = p[0]
    v = rng.uniform(.4,3,size=9); d = rng.normal(size=9); den = rng.uniform(.7,2,size=9)
    _, a = solve_space(sparse.csr_matrix(p),d,v,den,tol=1e-12)
    pp = p/den
    covariance = (pp*v)@pp.T
    direct = float((pp@d)@np.linalg.pinv(covariance)@(pp@d))
    np.testing.assert_allclose(a['d2'],direct,rtol=1e-10,atol=1e-10)
    return dict(sparse_d2=a['d2'],dense_pinv_d2=direct)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    cap = 900
    def check():
        if time.monotonic()-started > cap: raise TimeoutError('projection diagnostic cumulative wall cap900s')
    for key in ('TEMP','TMP','CUPY_CACHE_DIR'):
        path = OUT/'runtime'/key.lower(); path.mkdir(parents=True,exist_ok=True); os.environ[key]=str(path)
    A,G,R,normal,T = S.imports()
    import torch
    import cnh_temporal_readout_data as D
    from cnh_cvr_projection import Projector
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    plan0=json.loads((S.OUT/'PLAN.json').read_text()); rows=plan0['scene_rows']
    ids=json.loads((CACHED/'PLAN.json').read_text())['dark4cm_scene_ids']
    with np.load(S.OUT/'evaluated.npz') as data: cat=data['category']
    ids=[i for i in ids if (cat[i]=='contact').any()]
    pairs=[]
    for i in ids:
        row=rows[i]
        ref=None if row['placement']=='center' else next(j for j,b in enumerate(rows)
            if b['family']==row['family'] and b['placement']=='pass' and
            all(b[k]==row[k] for k in ('side','group','variant','rho')))
        pairs.append((i,ref))
    inputs=[S.OUT/'PLAN.json',S.OUT/'physical.npz',S.OUT/'evaluated.npz',CACHED/'background_reference.npz',D.D.BIAS]
    plan=dict(task='CNH_BAR_REPRESENTATION_DEV_20261008',lane='EXPLORE consumed Development',
        goal='Same-past8 A/B conditional linear projection separation and reconstruction of two FP16 rounding stages',
        authorization='User explicitly requested the first two diagnostics; no inference/training/sampling or pitch/layout changes',
        compute_budgets_wall_seconds=dict(projection=900,fp16=300,focused_audit=180),
        scene_pairs=[dict(scene=i,pass_scene=j) for i,j in pairs],geometries=28,A_pairs=28,B_pairs=24,frames=[12,13],
        windows=dict(f12=list(range(5,13)),f13=list(range(6,14))),
        projection='FP32 geometric indices/weights of actual shape Engine.project, ideal real linear arithmetic. Input z normalization included. Joint tot/lst covariance via equivalent disjoint past7-sum/last blocks. cnt fixed.',
        solver='SciPy sparse row-whitened LSMR tolerance1e-9/maxiter4000; normal residual, orthogonality, termination and lower bound recorded. No independent voxel variance approximation.',
        decision_check='28 A and24 B geometry pairs, 104 paired windows; geometry not K4 inflated. Separate unrepresented input-bin d2 from mixing residual. Loss localizes ideal projection only; no alarm-gain or model-failure claim. Pos/neg modes include cross terms. No diagnostic pass-rate threshold or promotion.',
        stop='Complete fixed diagnostics within separate new-run budgets, retain partial failures; no new M3 response or new photon samples',
        adjustable_scope='Implementation repair and focused numerical verification only',
        deliverables='Projection per-pair/aggregate, FP16 reconstruction, scoped report/run record/commit/push',
        inputs_sha256={B.logical_path(p):B.sha(p) for p in inputs},
        source_sha256={B.logical_path(Path(__file__)):B.sha(__file__)})
    B.save(OUT/'PLAN.json',plan); (OUT/'projection_source.py').write_bytes(Path(__file__).read_bytes())
    audits=dict(fixture=fixture(),backend=dict(projection='CUDA FP32 map/reconstruction',solver='CPU SciPy sparse LSMR',
        cpu_reason='GPU_BACKEND_UNAVAILABLE: CuPy sparse linalg import failed on missing cublas DLL in verified runtime',device=torch.cuda.get_device_name()))
    with np.load(S.OUT/'physical.npz') as data:
        mu=data['expectation']; ambient=data['ambient']; hist=data['hist'][ids[:1]]
    with np.load(CACHED/'background_reference.npz') as data: bg=data['expectation']
    bias=np.load(D.D.BIAS).astype(np.float32)
    den=np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9))
    sensor,query=B.poses(-10.)
    records=[]; numeric=[]; projector=Projector()
    try:
        for end in (12,13):
            check(); start=end-7
            transforms=(query[end]@np.linalg.inv(sensor[end]))[None]@sensor[start:end+1]
            before=time.monotonic(); p,blocks,inds,weights=map_window(transforms,projector,torch)
            sparse.save_npz(OUT/f'ideal_map_f{end}.npz',p)
            # Actual original Engine.project, without constructing/loading nets.
            engine=SimpleNamespace(torch=torch,projector=projector)
            z=normal(hist,ambient[None])[0,0,start:end+1]
            runtime=A.Engine.project(engine,z[None],transforms[None]).cpu().numpy()[0]
            actual=np.r_[runtime[0].ravel(),runtime[2].ravel()]
            ideal=np.r_[p@z.ravel(),blocks[-1]@z[-1].ravel()]
            err=float(np.max(np.abs(actual-ideal)))
            np.testing.assert_allclose(actual,ideal,rtol=3e-5,atol=1e-5)
            audits[f'map_f{end}']=dict(nnz=int(p.nnz),seconds=time.monotonic()-before,fp32_vs_ideal64_max_abs=err)
            for scene,pass_scene in pairs:
                check()
                height=('HEAD','BODY')[int(np.flatnonzero(cat[scene]=='contact')[0])]
                for contrast,reference in (('A_absent',bg),('B_pass',mu[pass_scene] if pass_scene is not None else None)):
                    if reference is None:continue
                    delta=(mu[scene]-reference)[start:end+1]
                    var=(.5*(mu[scene]+reference)+16*ambient[...,None])[start:end+1]
                    denominator=den[start:end+1]
                    t0=time.monotonic()
                    components=[]; qs=[]
                    for name,pp,dd,vv,nn in (('past7_sum',p[:,:7*1024],delta[:-1],var[:-1],denominator[:-1]),
                                              ('last',blocks[-1],delta[-1],var[-1],denominator[-1])):
                        qp,ip=solve_space(pp,np.maximum(dd,0),vv,nn)
                        qn,inn=solve_space(pp,np.minimum(dd,0),vv,nn)
                        q=qp+qn; u=(dd/np.sqrt(vv)).ravel(); q2=float(q@q)
                        components.append(dict(component=name,positive=ip,negative=inn,
                            positive_d2=float(qp@qp),negative_d2=float(qn@qn),cross_d2=float(2*qp@qn),
                            d2=q2,input_d2=float(u@u),covered_d2=ip['covered_d2']+inn['covered_d2'],
                            orthogonality_relative=float(abs(q@(u-q))/max(float(u@u),1e-30)),
                            lower_bound_d2=float(u@q)**2/q2 if q2 else 0.))
                        qs.append(q)
                    _,itot=solve_space(p,delta,var,denominator)
                    joint=sum(c['d2'] for c in components); obs=float(np.sum(delta**2/var))
                    covered=sum(c['covered_d2'] for c in components)
                    ok=all(c['positive']['converged'] and c['negative']['converged'] and c['orthogonality_relative']<2e-7 for c in components)
                    rec=dict(scene=scene,reference_scene=pass_scene if contrast=='B_pass' else None,height=height,
                        variant=rows[scene]['variant'],placement=rows[scene]['placement'],side=rows[scene]['side'],contrast=contrast,end=end,
                        observation_d=np.sqrt(obs),covered_d=np.sqrt(covered),joint_d=np.sqrt(joint),tot_d=np.sqrt(itot['d2']),last_d=np.sqrt(components[-1]['d2']),
                        retained_d2_fraction=joint/obs,covered_d2_fraction=covered/obs,
                        mixing_loss_d2=covered-joint,unrepresented_d2=obs-covered,
                        positive_joint_d2=sum(c['positive_d2'] for c in components),negative_joint_d2=sum(c['negative_d2'] for c in components),
                        cross_joint_d2=sum(c['cross_d2'] for c in components),seconds=time.monotonic()-t0,converged=ok and itot['converged'])
                    if rec['converged']:
                        assert itot['d2'] <= joint+max(1e-6,obs*1e-6)
                        assert joint <= covered+max(1e-6,obs*1e-6) <= obs+max(1e-6,obs*1e-6)
                    records.append(rec); numeric.append(dict(scene=scene,contrast=contrast,end=end,components=components,tot=itot))
                    # Each pair is durable, and can be audited if execution stops.
                    B.save(OUT/f'pair_f{end}_{scene}_{contrast}.json',dict(metrics=rec,numerics=numeric[-1]))
                    print('PAIR',len(records),'/104',scene,contrast,end,'d',round(rec['observation_d'],3),round(rec['joint_d'],3),'ok',rec['converged'],'wall',round(time.monotonic()-started,1),flush=True)
            del p,blocks,inds,weights
        summaries=[]
        keys=['observation_d','covered_d','joint_d','tot_d','last_d','retained_d2_fraction','covered_d2_fraction','mixing_loss_d2','unrepresented_d2','positive_joint_d2','negative_joint_d2','cross_joint_d2']
        for end in (12,13):
            for height in ('HEAD','BODY'):
                for contrast in ('A_absent','B_pass'):
                    rr=[r for r in records if r['end']==end and r['height']==height and r['contrast']==contrast]
                    summaries.append(dict(end=end,height=height,contrast=contrast,n=len(rr),converged=sum(r['converged'] for r in rr),
                        **{k:dict(min=float(min(r[k] for r in rr)),median=float(np.median([r[k] for r in rr])),max=float(max(r[k] for r in rr))) for k in keys}))
        with (OUT/'projection_pairs.csv').open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        B.save(OUT/'projection_result.json',dict(status='COMPLETE',pairs=len(records),converged=sum(r['converged'] for r in records),summaries=summaries,
            numerics=numeric,audits=audits,seconds=time.monotonic()-started,interpretation='Conditional ideal linear diagnostic only. Rounding is a separate empirical reconstruction; no monotone information-loss chain.'))
        check()
    except BaseException as e:
        B.save(OUT/f'projection_failure_{time.time_ns()}.json',dict(error=repr(e),completed=len(records),seconds=time.monotonic()-started));raise
    finally:
        del projector;torch.cuda.empty_cache()


if __name__=='__main__':run()
