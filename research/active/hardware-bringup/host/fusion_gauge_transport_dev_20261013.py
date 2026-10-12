"""Consumed Development: motion-corresponded affine gauge transport, CPU only.

Optical flow aligns scene points, but does not recover camera translation or
changing metric depth. Short-lived transport therefore remains a hypothesis.
Failures retain the complete original readout, never convert missing to FREE.
"""
import copy
import json
import time
from collections import Counter
import cv2
import numpy as np
from atom_camera import AtomCamera, ROOT
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_query_feasibility_dev_20261012 as F
from fusion_refinement_cache_dev_20261013 import OUT as CACHE
from fusion_refinement_readout_dev_20261013 import project
from fusion_refinement_score_dev_20261013 import metrics
from fusion_temporal_candidates_dev_20261013 import run_candidates

OUT = ROOT / 'artifacts.local/work/fusion-refinement-v2-dev-20261013'
POLICY = dict(max_pair_s=.35, max_fresh_age_s=1., max_fb_px=1.,
              max_photo=20., min_points=40, min_cells=6,
              max_normalized_residual=.06, min_inlier_fraction=.70,
              flow_width=320, min_slope=.5, max_slope=2.)


def affine(x, y):
    """Deterministic RANSAC in disparity; validate on all correspondences."""
    rng = np.random.default_rng(20261013)
    span = float(np.percentile(y,90)-np.percentile(y,10))
    xspan = float(np.percentile(x,90)-np.percentile(x,10))
    if min(span,xspan) < 1e-5:
        return None, dict(reason='depth_span')
    ij = rng.integers(len(x),size=(160,2))
    dx = x[ij[:,1]]-x[ij[:,0]]
    good = np.abs(dx) > .1*xspan
    ij,dx = ij[good],dx[good]
    aa = (y[ij[:,1]]-y[ij[:,0]])/dx
    bb = y[ij[:,0]]-aa*x[ij[:,0]]
    good = (aa >= POLICY['min_slope']) & (aa <= POLICY['max_slope'])
    aa,bb = aa[good],bb[good]
    if not len(aa): return None,dict(reason='slope')
    err=np.abs(y[None,:]-aa[:,None]*x[None,:]-bb[:,None])/span
    best=np.lexsort((np.median(err,axis=1),-(err<.10).sum(1)))[0]
    keep=err[best]<.10
    for _ in range(3):
        if keep.sum()<POLICY['min_points']: return None,dict(reason='consensus')
        a,b=np.linalg.lstsq(np.c_[x[keep],np.ones(keep.sum())],y[keep],rcond=None)[0]
        residual=np.abs(y-a*x-b)/span
        keep=residual<.10
    diag=dict(A=float(a),B=float(b),points=len(x),inliers=int(keep.sum()),
              normalized_residual=float(np.median(residual)),inlier_fraction=float(keep.mean()))
    ok=(POLICY['min_slope']<=a<=POLICY['max_slope'] and
        diag['normalized_residual']<=POLICY['max_normalized_residual'] and
        diag['inlier_fraction']>=POLICY['min_inlier_fraction'])
    diag['reason']='accepted' if ok else 'fit_quality'
    return (float(a),float(b)) if ok else None,diag


def sample(img,p):
    return cv2.remap(img,p[:,0].astype(np.float32).reshape(-1,1),
                     p[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).ravel()


def motion_fit(prev,cur):
    g0,d0,v0=prev; g1,d1,v1=cur
    pts=cv2.goodFeaturesToTrack(g0,maxCorners=600,qualityLevel=.015,minDistance=7,
                                mask=(v0.astype(np.uint8)*255))
    if pts is None or len(pts)<POLICY['min_points']:return None,dict(reason='few_corners')
    p1,status,_=cv2.calcOpticalFlowPyrLK(g0,g1,pts,None,winSize=(21,21),maxLevel=3)
    back,status_back,_=cv2.calcOpticalFlowPyrLK(g1,g0,p1,None,winSize=(21,21),maxLevel=3)
    p0=pts[:,0];p1=p1[:,0];back=back[:,0]
    h,w=g0.shape
    good=(status.ravel()>0)&(status_back.ravel()>0)&(np.linalg.norm(back-p0,axis=1)<=POLICY['max_fb_px'])
    good&=(p1[:,0]>=2)&(p1[:,0]<w-2)&(p1[:,1]>=2)&(p1[:,1]<h-2)
    good&=np.abs(sample(g0.astype(np.float32),p0)-sample(g1.astype(np.float32),p1))<=POLICY['max_photo']
    good&=sample(v1.astype(np.float32),p1)>.99
    p0,p1=p0[good],p1[good]
    if len(p0)<POLICY['min_points']:return None,dict(reason='flow_quality',points=len(p0))
    cells=np.unique((p1[:,0]*4/w).astype(int)+4*(p1[:,1]*4/h).astype(int))
    if len(cells)<POLICY['min_cells']:return None,dict(reason='flow_coverage',cells=len(cells))
    fit,diag=affine(sample(d0,p0),sample(d1,p1))
    diag.update(cells=len(cells),median_flow_px=float(np.median(np.linalg.norm(p1-p0,axis=1))))
    return fit,diag


def numerical_checks():
    x=np.linspace(1,10,120);y=1.2*x+.7;y[::6]+=20
    fit,diag=affine(x,y)
    assert np.allclose(fit,(1.2,.7))
    a,b=3.,2.;A,B=fit;z=np.linspace(.3,4,100)
    assert np.allclose((A*(a/z+b)+B-(A*b+B))/(A*a),1/z)
    assert affine(x,-x)[0] is None
    return dict(outlier_fit=fit,affine_transport_identity=True,negative_slope_rejected=True)


def main():
    start=time.perf_counter();cv2.setNumThreads(2);OUT.mkdir(parents=True,exist_ok=True)
    source=json.loads((CACHE/'robust_readout.json').read_text())
    cache=json.loads((CACHE/'cache_manifest.json').read_text())
    result=dict(policy=POLICY,checks=numerical_checks(),sessions={})
    summary=dict(sessions={})
    cam=AtomCamera();K=cam.K_upright;w,h=cam.upright_size
    vv,uu=np.mgrid[:h,:w]
    dirs=np.stack(((uu-K[0,2])/K[0,0],(vv-K[1,2])/K[1,1],np.ones_like(uu)),-1).reshape(-1,3)
    Rct=np.array(json.loads((X.OUT/'camera_tof_extrinsic.json').read_text())['rotation'])
    for key,s in source['sessions'].items():
        rows=cache['sessions'][key]['frames'];folder=F.RAW/s['session']
        rgb=list(F.lines(folder/'rgb.jsonl')); memo={}
        def load(j):
            if j not in memo:
                c=rows[j]
                with np.load(CACHE/c['cache']) as z:disp,valid=z['disp'],z['valid']
                raw=cv2.imread(str(folder/rgb[c['rgb_index']]['file']))
                upright=cam.undistort_upright(raw)
                size=(320,round(h*320/w))
                gray=cv2.resize(cv2.cvtColor(upright,cv2.COLOR_BGR2GRAY),size)
                small=(gray,cv2.resize(disp,size),cv2.resize(valid.astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0)
                memo[j]=(small,disp,valid)
            return memo[j]
        arms={k:copy.deepcopy(s['arms'][k]) for k in ('baseline','robust_replace')}
        arms.update({k+'_transport':copy.deepcopy(arms[k]) for k in tuple(arms)})
        diagnostics=[];fresh={k:None for k in ('baseline','robust_replace')};chain={k:False for k in fresh}
        for j,c in enumerate(rows):
            if time.perf_counter()-start>565:raise TimeoutError('transport 600s budget reserve')
            diag=dict(i=c['i'],reason='not_reused',
                      pair_s=(c['t_ns']-rows[j-1]['t_ns'])/1e9 if j else None,
                      rgb_pair_s=(c['rgb_t_ns']-rows[j-1]['rgb_t_ns'])/1e9 if j else None);fit=None
            eligible=[]
            for base in fresh:
                orig=arms[base][j]
                if orig['anchor'] in ('tof','floor','robust_tof'):
                    fresh[base]=c['t_ns'];chain[base]=True
                elif orig['anchor']=='reused':eligible.append(base)
                else:chain[base]=False
            if eligible:
                if j==0 or 'cache' not in c or 'cache' not in rows[j-1]:diag['reason']='missing_adjacent'
                elif not (0<(c['t_ns']-rows[j-1]['t_ns'])/1e9<=POLICY['max_pair_s']) or c['rgb_t_ns']<=rows[j-1]['rgb_t_ns'] or c['seq']<=rows[j-1]['seq'] or c['device_ms']<=rows[j-1]['device_ms']:diag['reason']='time_break'
                elif not any(chain[b] and fresh[b] is not None and (c['t_ns']-fresh[b])/1e9<=POLICY['max_fresh_age_s'] for b in eligible):diag['reason']='stale_or_broken_chain'
                else:
                    fit,info=motion_fit(load(j-1)[0],load(j)[0]);diag.update(info)
            for base in eligible:
                target=arms[base+'_transport'][j]
                age=(c['t_ns']-fresh[base])/1e9 if fresh[base] is not None else None
                apply=fit is not None and chain[base] and age is not None and age<=POLICY['max_fresh_age_s']
                if apply:
                    a,b=arms[base+'_transport'][j-1]['fit'];A,B=fit
                    newfit=(A*a,A*b+B)
                    _,disp,valid=load(j)
                    corridor=dirs@(F.rot(c['pitch'],c['roll'])@Rct.T).T
                    target.update(project(disp,valid,newfit,corridor),fit=newfit,anchor='flow_transport')
                else:chain[base]=False
                target['transport']=dict(diag,applied=bool(apply),fresh_age_s=age)
            diagnostics.append(diag)
            memo={k:v for k,v in memo.items() if k>=j-1}
        predictions={k:run_candidates(fr)['baseline_2of3'] for k,fr in arms.items()}
        ms={k:metrics(fr,predictions[k]) for k,fr in arms.items()}
        for base in fresh:
            name=base+'_transport';p,q=predictions[base],predictions[name]
            ms[name].update(added_query_frames=int((q&~p).sum()),removed_query_frames=int((p&~q).sum()),
                            added_any_frames=int((q.any(1)&~p.any(1)).sum()),removed_any_frames=int((p.any(1)&~q.any(1)).sum()),
                            transported_frames=sum(r['anchor']=='flow_transport' for r in arms[name]))
            assert all(a['available']==b['available'] for a,b in zip(arms[base],arms[name]))
        result['sessions'][key]=dict(session=s['session'],label=s['label'],arms=arms,diagnostics=diagnostics)
        summary['sessions'][key]=dict(label=s['label'],metrics=ms,reasons=dict(Counter(d['reason'] for d in diagnostics)))
        print(key,summary['sessions'][key]['reasons'],{k:ms[k] for k in ms if 'transport' in k},flush=True)
    result['wall_s']=summary['wall_s']=time.perf_counter()-start
    summary['checks']=result['checks'];summary['policy']=POLICY
    (OUT/'transport_readout.json').write_text(json.dumps(result),encoding='utf8')
    (OUT/'transport_summary.json').write_text(json.dumps(summary,indent=1),encoding='utf8')
    print('wall_s',summary['wall_s'],flush=True)


if __name__=='__main__':main()
