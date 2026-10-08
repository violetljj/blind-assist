"""Cached raw trigger and privileged A/B/radial pooling diagnostics, no inference."""
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import cnh_aligned_boundary_dev as B
import cnh_aligned_shapes_dev as S

OUT=B.ROOT/'artifacts.local/work/cnh-bar-cached-diagnostic-dev-20261008'
OLD=S.OUT


def save(name,value):B.save(OUT/name,value)


def matched_threshold(scores,cat,target):
    v=scores[(cat=='clear').all(1)].max(-1).reshape(-1);ordered=np.sort(v)
    candidates=np.r_[np.inf,np.unique(v)];cost=len(v)-np.searchsorted(ordered,candidates,'left')
    ix=np.lexsort((-candidates,abs(cost-target)))[0];theta=candidates[ix]
    return np.nextafter(float(v.max()),np.inf) if not np.isfinite(theta) else float(theta)


def stats(x):
    x=np.asarray(x,float)
    return dict(n=int(x.size),min=float(x.min()),median=float(np.median(x)),max=float(x.max())) if x.size else None


def shift_matrix(shift,bins=16):
    """Conservative fractional translation of equal-width radial bins."""
    matrix=np.zeros((bins,bins),float)
    for j in range(bins):
        x=j+shift;i=int(np.floor(x));alpha=x-i
        if 0<=i<bins:matrix[i,j]+=1-alpha
        if 0<=i+1<bins:matrix[i+1,j]+=alpha
    return matrix


def pooled(mu0,mu1,hist,ambient,ranges,end,start,aligned,width):
    """Whole-FOV radial pooling, covariance from shared interpolation weights."""
    delta=np.zeros(16);cov=np.zeros((16,16));obs=np.zeros((len(hist),16));base=np.zeros(16)
    for f in range(start,end+1):
        matrix=shift_matrix((ranges[end]-ranges[f])/width) if aligned else np.eye(16)
        d=(mu1[f]-mu0[f]).sum((0,1));v=(.5*(mu0[f]+mu1[f])+16*ambient[f,...,None]).sum((0,1))
        delta+=matrix@d;cov+=(matrix*v[None,:])@matrix.T
        obs+=hist[:,f].sum((1,2))@matrix.T;base+=matrix@mu0[f].sum((0,1))
    weight=np.linalg.solve(cov,delta);d2=float(delta@weight)
    z=(obs-base)@weight/np.sqrt(d2) if d2>0 else np.zeros(len(hist))
    return dict(d2=max(d2,0.),observed_standardized_score=z.tolist())


def run():
    began=time.monotonic();A,G,R,normal,T=S.imports()
    plan0=json.loads((OLD/'PLAN.json').read_text());rows=plan0['scene_rows']
    ids=[i for i,r in enumerate(rows) if r['family']=='horizontal' and r['rho']==.25 and 'thick0.04' in r['variant']]
    sources=[Path(__file__),Path(B.__file__),Path(S.__file__),*[
        Path(m.__file__).resolve() for n,m in list(sys.modules.items()) if n.startswith('cnh_') and getattr(m,'__file__',None)]]
    inputs=[OLD/f for f in ('PLAN.json','run_result.json','physical.npz','evaluated.npz')]
    plan=dict(task='CNH_BAR_CACHED_DIAGNOSTIC_DEV_20261008',lane='EXPLORE consumed shape caches',
        authorization='Continue scoped diagnostics from user corrected roundtable; no heading/device/layout/training work',
        shared_CPU_wall_cap_seconds=180,adjustable_scope='Cache-only; fixed inputs, thresholds and diagnostic formulas. No new sampled observations or M3 inference.',
        dark4cm_scene_ids=ids,replicas=4,contact_events=112,HEAD_BODY_each=56,
        raw='Raw float64 comparisons; original theta and whole-score-tie nearest integer all13 jointly-clear cost matching baseline46/4576. Report residual, per-height paired losses/gains, late, clear runs/clips and pass, no automatic promotion.',
        A='Target present vs target removed, frozen no-target background expectation16frames, conditional known geometry/pose/noise diagnostic only.',
        B='In_1/4/12cm vs existing same side/height/rho/variant pass inner|x|=.35; gaps6/9/17cm. Center not paired. Shared pass reused, no independent pair inflation.',
        separation='d2_frame=sum((mu1-mu0)^2/(.5*(mu0+mu1)+16ambient)); positive and negative mean terms retained; full prefix and past8 at f12/f13. Gaussian variance-standardized proxy, not exact discrimination limit.',
        pooling='Whole-FOV radial sum; static target inner-front anchor known from geometry, r_f=norm(anchor-sensor_position_f). Shift by (r_end-r_f)/coarsewidth using linear bin overlaps, retain covariance. Compare aligned vs unaligned past8 and prefixes at f12/f13. Ideal hypothesis only, angular identity collapsed; no learned/query-safe detector or alert threshold.',
        decision_check='Baseline and dark-bar deficits already consumed; one paired noise event minimum1. Raw>=theta should contain any smoothed timely event at fixed theta. Seven suppressed events are a lead to recompute, not a promised gain. Finite matched-cost raw differences and retention losses change its priority only; no inference that other representations, negative background, side information or layout options are exhausted.',
        stop='Complete cached diagnostics within shared180s; preserve denominators/failures, no retuning, no new model, no pitch/layout change',
        deliverables='Raw-trigger paired comparisons/costs; A/B prefix/sign-bin table; radial pooling comparison and observed conditional scores; independent review, scoped report/commit/push',
        source_sha256={B.logical_path(p):B.sha(p) for p in sorted(set(sources),key=str)},
        input_sha256={B.logical_path(p):B.sha(p) for p in inputs})
    save('PLAN.json',plan);(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    def check():
        if time.monotonic()-began>120:raise TimeoutError('main120s within shared180')
    with np.load(OLD/'physical.npz') as z:
        raw=np.asarray(z['raw'],np.float64);mu=z['expectation'];ambient=z['ambient'];hist=z['hist'][ids]
    with np.load(OLD/'evaluated.npz') as z:cat=z['category'];smooth=z['scores']
    original=B.metrics(smooth,cat,B.THETA);fixed=B.metrics(raw,cat,B.THETA)
    theta=matched_threshold(raw,cat,original['clear_slots']);matched=B.metrics(raw,cat,theta)
    assert np.all(~original['timely']|fixed['timely'])
    darkmask=np.zeros(len(rows),bool);darkmask[ids]=True
    def compare(m):
        return dict(metrics=B.serial_metrics(m),paired=B.compare(original,m,cat),
            dark4cm_paired=B.compare(B.metrics(smooth[darkmask],cat[darkmask],B.THETA),
                                   B.metrics(raw[darkmask],cat[darkmask],m['threshold']),cat[darkmask]))
    suppressed=[]
    for i in ids:
        for q in (0,1):
            if cat[i,q]!='contact':continue
            for k in range(4):
                if fixed['timely'][i,k,q] and not original['timely'][i,k,q]:
                    frames=B.FRAMES[:11][raw[i,k,:11,q]>=B.THETA]
                    suppressed.append(dict(scene=i,replica=k,height=('HEAD','BODY')[q],raw_crossing_frames=frames.tolist()))
    sensor,_=B.poses(-10.)
    bg=R.expected(dict(poses=sensor,boxes=S.BASE_BG))['expectation']
    details=[];prefix=[];pooling=[];bins=[]
    width=8*R.SENSOR.RAW_BIN_M if hasattr(R,'SENSOR') else 8*sys.modules[R.S.synthesize_response.__module__].RAW_BIN_M
    mu_selected=mu[ids]
    for local,i in enumerate(ids):
        check();r=rows[i]
        if cat[i].tolist().count('contact')==0:continue
        q=int(np.flatnonzero(cat[i]=='contact')[0]);pair=None
        if r['placement']!='center':
            pair=next(j for j,b in enumerate(rows) if b['family']==r['family'] and b['placement']=='pass' and
                b['side']==r['side'] and b['group']==r['group'] and b['variant']==r['variant'] and b['rho']==r['rho'])
        for label,reference in (('A_absent',bg),('B_pass',mu[pair] if pair is not None else None)):
            if reference is None:continue
            delta=mu[i]-reference;variance=.5*(mu[i]+reference)+16*ambient[...,None]
            info=delta**2/variance;d2=info.sum((1,2,3));pos=np.where(delta>0,info,0).sum((1,2,3));neg=np.where(delta<0,info,0).sum((1,2,3))
            for end in (12,13):
                for window,start in (('prefix',0),('past8',max(0,end-7))):
                    prefix.append(dict(scene=i,reference_scene=pair if label=='B_pass' else None,height=('HEAD','BODY')[q],variant=r['variant'],placement=r['placement'],side=r['side'],contrast=label,end=end,window=window,
                        d=float(np.sqrt(d2[start:end+1].sum())),positive_d2=float(pos[start:end+1].sum()),negative_d2=float(neg[start:end+1].sum())))
            for b in range(16):
                bins.append(dict(scene=i,contrast=label,bin=b,range_low=b*width,range_high=(b+1)*width,
                    positive_d2=float(np.where(delta[:14,...,b]>0,info[:14,...,b],0).sum()),
                    negative_d2=float(np.where(delta[:14,...,b]<0,info[:14,...,b],0).sum())))
            details.append(dict(scene=i,contrast=label,d2_frame=d2.tolist()))
        # Known inner-front point is a hypothesis anchor, not an estimator input.
        x=0. if r['side']==0 else r['lo'][0] if r['side']>0 else r['hi'][0]
        anchor=np.array([x,(r['lo'][1]+r['hi'][1])/2,r['lo'][2]])
        radial=np.linalg.norm(anchor[None]-sensor[:,:3,3],axis=1)
        for end in (12,13):
            for window,start in (('prefix',0),('past8',max(0,end-7))):
                a=pooled(bg,mu[i],hist[local],ambient,radial,end,start,True,width)
                u=pooled(bg,mu[i],hist[local],ambient,radial,end,start,False,width)
                pooling.append(dict(scene=i,height=('HEAD','BODY')[q],variant=r['variant'],placement=r['placement'],end=end,window=window,
                    aligned_d=float(np.sqrt(a['d2'])),unaligned_d=float(np.sqrt(u['d2'])),
                    aligned_observed_scores=a['observed_standardized_score'],unaligned_observed_scores=u['observed_standardized_score'],anchor=anchor.tolist()))
    summaries=[]
    for q in ('HEAD','BODY'):
        for label in ('A_absent','B_pass'):
            for end in (12,13):
                for window in ('prefix','past8'):
                    rs=[p for p in prefix if p['height']==q and p['contrast']==label and p['end']==end and p['window']==window]
                    summaries.append(dict(height=q,contrast=label,end=end,window=window,d=stats([p['d'] for p in rs]),positive_d2=stats([p['positive_d2'] for p in rs]),negative_d2=stats([p['negative_d2'] for p in rs])))
    pool_summary=[]
    for q in ('HEAD','BODY'):
        for end in (12,13):
            for window in ('prefix','past8'):
                rs=[p for p in pooling if p['height']==q and p['end']==end and p['window']==window]
                pool_summary.append(dict(height=q,end=end,window=window,
                    aligned=stats([p['aligned_d'] for p in rs]),unaligned=stats([p['unaligned_d'] for p in rs]),
                    aligned_larger_scenes=sum(p['aligned_d']>p['unaligned_d'] for p in rs),scenes=len(rs)))
    result=dict(status='COMPLETE',baseline=B.serial_metrics(original),raw_fixed=compare(fixed),raw_cost_matched=compare(matched),
        matched_cost_residual=matched['clear_slots']-original['clear_slots'],suppressed_events=suppressed,
        event_count=len(suppressed),suppressed_slots=int(((raw[darkmask,:,:11]>=B.THETA)&(smooth[darkmask,:,:11]<B.THETA)&(cat[darkmask]=='contact')[:,None,None,:]).sum()),
        prefix_summaries=summaries,radial_pool_summaries=pool_summary,
        decision='RETAIN_ORIGINAL_M3_POLICY; raw and ideal pooling are diagnostic candidates only, other mechanisms remain open',
        width_m=width,seconds=time.monotonic()-began)
    for name,records in (('prefix.csv',prefix),('sign_bins.csv',bins),('radial_pooling.csv',pooling)):
        with (OUT/name).open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    np.savez_compressed(OUT/'background_reference.npz',expectation=bg,ambient=ambient)
    save('frame_information.json',details);save('result.json',result);check()
    print(json.dumps({k:v for k,v in result.items() if k not in ('prefix_summaries','radial_pool_summaries','suppressed_events')},ensure_ascii=False),flush=True)


if __name__=='__main__':run()
