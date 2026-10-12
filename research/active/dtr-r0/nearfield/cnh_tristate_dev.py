"""CPU replay of consumed CNH Development; no inference or certificates.

Online score decisions are saved before evaluator-only box labels are joined.
Missing tracking validity stays unavailable; mathematical SE3 checks do not
invent a tracking flag. The inherited 2.1m range is retained without retuning.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
OUT = WORK / 'cnh-tristate-dev-20261006'
MARGIN = WORK / 'cnh-margin-confirm-20261002'
FUSION = WORK / 'cnh-dual-gated-fusion-20261005'
AUG = WORK / 'cnh-extrinsic-aug-20261006'
CONT = AUG / 'continuation-r1'
THRESHOLD = .8557642486787612
FRAMES = np.arange(3, 16)
HEIGHTS = ((-.20, .42), (.42, .90))
EDGE = float(np.tan(np.pi/8))
EPS = 1e-10


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def deadline():
    if time.time() >= read(OUT/'PLAN.json')['deadline_unix']:
        raise TimeoutError('75 minute wall limit; preserve completed stages')


def smooth(raw):
    raw = np.asarray(raw, float)
    if raw.shape[-2:] != (13, 2) or not np.isfinite(raw).all():
        raise ValueError('13 finite raw frames and two queries required')
    result = np.empty_like(raw)
    for t in range(13):
        begin = max(0, t-4); w = 2.**np.arange(t-begin+1)
        result[..., t, :] = np.sum(raw[..., begin:t+1, :]*w[:, None], axis=-2)/w.sum()
    return result


def rotation(degrees, axis='y'):
    a = np.deg2rad(degrees); c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]]) if axis=='y' else np.array([[1,0,0],[0,c,-s],[0,s,c]])


def extrinsic(angle):
    e = np.eye(4); e[:3,:3] = rotation(10,'x')@rotation(angle)@rotation(-10,'x')
    return e


def original_motion(unit, config):
    # Frozen public schedule, same original noise generator; no scene input.
    source = WORK/'cnh-track-a-v5-20260928/data/source'
    if str(source) not in sys.path: sys.path.insert(0,str(source))
    from cnh_track_a_readout import noisy_poses
    mode = unit%3
    yaw = np.linspace(-20.,0.,16) if mode==2 else np.zeros(16)
    position = np.zeros((16,3))
    for i in range(1,16):
        position[i] = position[i-1]+.16*rotation((yaw[i-1]+yaw[i])/2)@np.array([0.,0.,1.])
    position -= position[-1]
    sy = 20*np.sin(np.linspace(-np.pi/2,np.pi/2,16)) if mode==1 else np.full(16,15. if mode==0 else 0.)
    sensor = np.repeat(np.eye(4)[None],16,axis=0); travel = sensor.copy()
    for i in range(16):
        travel[i,:3,:3]=rotation(yaw[i]); travel[i,:3,3]=position[i]
        sensor[i,:3,:3]=rotation(yaw[i]+sy[i])@rotation(-10,'x'); sensor[i,:3,3]=position[i]
    return sensor, travel, noisy_poses(sensor,2026092900+unit*1000+config*10+7,dt=.2)


def fov(points, pose, range_limit=2.1):
    local = (np.asarray(points)-pose[:3,3])@pose[:3,:3]
    z = local[:,2]
    angular = (z>0)&(np.abs(local[:,0])<=EDGE*z+EPS)&(np.abs(local[:,1])<=EDGE*z+EPS)
    return angular if range_limit is None else angular&(np.linalg.norm(local,axis=1)<=range_limit+EPS)


def geometric_gate(noisy, angles):
    """Exact far-boundary rejection witness; full lattice only if needed.

    A stale required sentinel suffices to reject an all-points rule. This avoids
    calculating millions of interior points when the far boundary is stale.
    """
    passes=np.zeros(13,bool); unavailable=np.zeros(13,bool); far_stale=np.zeros(13,bool)
    min_far=np.full(13,np.nan); counts=np.zeros(13,np.int32)
    finite = np.isfinite(noisy).all() and np.allclose(noisy[:,3,:],[0,0,0,1])
    if not finite:
        unavailable[:]=True; return passes,unavailable,far_stale,min_far,counts
    for ti,f in enumerate(FRAMES):
        if f<5: unavailable[ti]=True; continue
        d=(noisy[f,:3,3]-noisy[f-5,:3,3])[[0,2]]; norm=np.linalg.norm(d)
        if norm<.05: unavailable[ti]=True; continue
        forward=d/norm; right=np.array([forward[1],-forward[0]])
        origin=noisy[f,:3,3]; yy=noisy[0,1,3]+np.array([v for pair in HEIGHTS for v in pair])
        far=np.array([[*(origin[[0,2]]+a*right+2.5*forward),y] for a in (-.29,.29) for y in yy])[:,[0,2,1]]
        poses=[noisy[h]@extrinsic(a) for h in range(f-3,f+1) for a in angles]
        fresh=np.any([fov(far,p) for p in poses],axis=0)
        min_far[ti]=min(float(np.linalg.norm(far-p[:3,3],axis=1).min()) for p in poses)
        if not fresh.all(): far_stale[ti]=True; counts[ti]=int((~fresh).sum()); continue
        # Full 5cm world XZ lattice at each query-height grid slice.
        initial=noisy[0,[0,2],2]; initial/=np.linalg.norm(initial)
        basis=np.stack(([initial[1],-initial[0]],initial),axis=1)
        anchor=noisy[0,[0,2],3]; current=origin[[0,2]]
        corners=np.array([current+a*right+b*forward for a in (-.29,.29) for b in (.9,2.5)])
        ij=(corners-anchor)@basis/.05; low=np.floor(ij.min(0)).astype(int)-1; high=np.ceil(ij.max(0)).astype(int)+1
        x,z=np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1))
        xz=np.stack((x.ravel(),z.ravel()),axis=1)*.05@basis.T+anchor
        rel=xz-current; keep=(np.abs(rel@right)<=.29+EPS)&(rel@forward>=.9-EPS)&(rel@forward<=2.5+EPS)
        sent=np.array([current+a*right+b*forward for a in (-.29,.29) for b in np.arange(.9,2.5001,.05)])
        xz=np.concatenate((xz[keep],sent))
        ys=np.unique(np.concatenate([np.arange(lo,hi+EPS,.05).tolist()+[hi] for lo,hi in HEIGHTS]))+noisy[0,1,3]
        points=np.array([[p[0],y,p[1]] for p in xz for y in ys])
        full=np.any([fov(points,p) for p in poses],axis=0)
        passes[ti]=full.all(); counts[ti]=int((~full).sum())
    return passes,unavailable,far_stale,min_far,counts


def clip_polygon(poly, axis, bound, greater):
    if not len(poly): return poly
    result=[]; prev=poly[-1]; pin=(prev[axis]>=bound-EPS) if greater else (prev[axis]<=bound+EPS)
    for cur in poly:
        cin=(cur[axis]>=bound-EPS) if greater else (cur[axis]<=bound+EPS)
        if pin!=cin:
            t=(bound-prev[axis])/(cur[axis]-prev[axis]); result.append(prev+t*(cur-prev))
        if cin: result.append(cur)
        prev,pin=cur,cin
    return np.asarray(result).reshape(-1,2)


def contact_boxes(boxes, travel_pose, width=.30, zmax=2.5):
    pose=np.asarray(travel_pose,float); result=[]
    for index,box in enumerate(boxes):
        lo,hi=np.asarray(box['lo'],float),np.asarray(box['hi'],float)
        queries=[q for q,(yl,yh) in enumerate(HEIGHTS) if hi[1]>=pose[1,3]+yl-EPS and lo[1]<=pose[1,3]+yh+EPS]
        if not queries: continue
        corners=np.array([[lo[0],0,lo[2]],[hi[0],0,lo[2]],[hi[0],0,hi[2]],[lo[0],0,hi[2]]])
        local=(corners-pose[:3,3])@pose[:3,:3]; poly=local[:,[0,2]]
        for ax,b,gt in ((0,-width,True),(0,width,False),(1,0.,True),(1,zmax,False)):
            poly=clip_polygon(poly,ax,b,gt)
        if not len(poly): continue
        xmin,xmax=poly[:,0].min(),poly[:,0].max()
        depth=max(0.,min(width-xmin,xmax+width))
        yc=[max(lo[1]-pose[1,3],HEIGHTS[q][0]) for q in queries]
        yh=[min(hi[1]-pose[1,3],HEIGHTS[q][1]) for q in queries]
        pts=np.array([[p[0],y,p[1]] for p in np.concatenate((poly,poly.mean(0)[None])) for y in (min(yc),max(yh),(min(yc)+max(yh))/2)])
        result.append(dict(box_index=index,queries=queries,depth_m=float(depth),
            boundary=bool(depth<=.01+EPS or min(zmax-poly[:,1].min(),poly[:,1].max())<=.01+EPS
                or min(HEIGHTS[-1][1]-(lo[1]-pose[1,3]),(hi[1]-pose[1,3])-HEIGHTS[0][0])<=.01+EPS),
            witness_world=(pts@pose[:3,:3].T+pose[:3,3]).tolist()))
    return result


def labels(boxes, travel_pose):
    core=contact_boxes(boxes,travel_pose)
    return dict(contact=bool(core),graze=bool(not core and contact_boxes(boxes,travel_pose,width=.40)),
        boundary=bool(core and any(r['boundary'] for r in core)))


def score_sources():
    return {95000:FUSION/'natural95000/scores',96000:WORK/'cnh-dual-sensor-envelope-20261005/natural/scores',
        97000:FUSION/'natural97000/scores',98000:CONT/'scores/calibration/ensemble3',99000:CONT/'scores/evaluation/ensemble3'}


def freeze(started):
    OUT.mkdir(parents=True,exist_ok=True)
    counts={str(b):len(list(p.glob('unit*.npz'))) for b,p in score_sources().items()}
    save(OUT/'PLAN.json',dict(task='CNH_TRISTATE_DEV_20261006',phase='EXPLORE / consumed synthetic Development / descriptive, not guarantee',
        started_unix=started,deadline_unix=started+75*60,maximum_wall_seconds=4500,CPU_only=True,no_inference=True,no_render=True,no_train=True,no_binary_verdict=True,
        units_inventoried=counts,frames=FRAMES.tolist(),configs='Use actual retained configs; do not impute absent ones',
        rule=dict(alarm_threshold=THRESHOLD,alarm_operator='>=',clear_operator='<=',arms=['single M3','dual +/-15deg M3'],
            smoothing='Causal last5 logits weights1,2,4,8,16 per branch/query then max',
            coverage='97e4a6f4 recency0.6s,5cm world grid, exact lateral sentinels; extend query0.9..2.5m and full two height slices',
            range_m=2.1,range_conflict='Retained inherited radial limit; far2.5m can fail. No silent relaxation',
            direction='last1s estimated displacement, >=.05m; frames0..4 unavailable',
            validity='Missing explicit pose/tracking/gravity validity NOT_AVAILABLE and gate fails; finiteSE3 only a separate geometry diagnostic',
            heights=HEIGHTS,grid_width_m=.29,dual='causal FOV union, not sum',
            thresholds='20 pooled label-blind quantiles q0..1 of both arms nonalarm scores capped at original alarm threshold; common grid; keep ties'),
        truth=dict(front_window_m=[0.,2.5],width_m=.30,heights=HEIGHTS,contact='Closed-volume exact polygon clip, any object/query',
            graze='Outer width.40 minus primary contact, same finite z/y window',boundary='Primary intersection within1cm of lateral/vertical/forward boundary',
            person_front='Current saved travel origin proxy; distinct anatomical front offset NOT_AVAILABLE; no fabricated offset',
            no_clear_all=True,no_curved_future=True),
        bootstrap=dict(unit='Whole scene unit; configs/frames/heights/arms stay together',draws=1000,seed=2026100611,stratified_by='batch, mode, turn',
            interval='percentile95%; undefined denominators remain null'),
        source_split=dict(actual_errors='At each20threshold, by arm/mode/turn/batch',
            blocked_candidates='Separate nongated counterfactual diagnostics, never actual outputs or certificate',
            fov='Contact-intersection witness angularFOV proxy only; does not prove unoccluded return; unproven absence remains unresolved'),
        stage2='Recommendation only; alpha2%/5%, delta.025 per arm; one config+frame per new independent unit, zero-error CP planning',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'research/active/dtr-r0/nearfield/cnh_coverage_recency.py',
            WORK/'cnh-track-a-v5-20260928/data/source/cnh_track_a_readout.py',ROOT/'research/active/dtr-r0/nearfield/cnh_cvr_pilot.py']}))
    (OUT/'PLAN.sha256').write_text(sha(OUT/'PLAN.json')+'\n')


def online():
    start=time.monotonic(); rows=[]; ledger={k:[] for k in ('score','geometric_gate','direction_unavailable','far_stale','min_far_range','stale_sentinel_count','aug_score')}; hashes={}; fields={}
    early=np.load(MARGIN/'frame_scores_M3_early.npz'); late=np.load(MARGIN/'frame_scores_M3.npz')
    for p in (MARGIN/'frame_scores_M3_early.npz',MARGIN/'frame_scores_M3.npz'): hashes[str(p.relative_to(ROOT))]=sha(p)
    for batch,folder in score_sources().items():
        paths=sorted(folder.glob('unit*.npz'))
        for path in paths:
            deadline(); hashes[str(path.relative_to(ROOT))]=sha(path)
            with np.load(path) as z:
                u=int(z['unit']); configs=z['configs'].tolist(); np.testing.assert_array_equal(z['frames'],FRAMES)
                raw=z['reference'] if batch>=98000 else z['raw']; sm=smooth(raw)
                if batch in (95000,96000):
                    baseline=smooth(np.concatenate((early[str(u)],late[str(u)]),axis=1))[configs]
                    score=np.stack((baseline.max(-1),sm.max((0,3))),axis=-1)
                else: score=np.stack((sm[0].max(-1),sm[1:].max((0,3))),axis=-1)
                aug=np.stack((smooth(z['augmented_mean'])[0].max(-1),smooth(z['augmented_mean'])[1:].max((0,3))),axis=-1) if batch>=98000 else np.full_like(score,np.nan)
            if batch>=98000:
                obs=(AUG if batch==98000 else CONT)/f'observations/{"calibration" if batch==98000 else "evaluation"}/unit{u}.npz'
                hashes[str(obs.relative_to(ROOT))]=sha(obs)
                with np.load(obs) as z:
                    fields[str(batch)]=z.files; noises=z['noisy_center']; oc=z['configs'].tolist()
            for ci,c in enumerate(configs):
                est=noises[oc.index(c)] if batch>=98000 else original_motion(u,c)[2]
                values=[geometric_gate(est,angles) for angles in ((0,),(-15,15))]
                ledger['score'].append(score[ci]); ledger['aug_score'].append(aug[ci])
                for name,vi in [('geometric_gate',0),('direction_unavailable',1),('far_stale',2),('min_far_range',3),('stale_sentinel_count',4)]:
                    ledger[name].append(np.stack([v[vi] for v in values],axis=-1))
                rows.append(dict(batch=batch,unit=u,config=c,mode=u%3,turn='none' if u%3!=2 else ('right' if batch<98000 or not bool((u//3)%2) else 'left')))
            if len(rows)%480==0: print('online',batch,u,len(rows),'sequences',round(time.monotonic()-start,1),'s',flush=True)
    early.close(); late.close()
    data={k:np.asarray(v) for k,v in ledger.items()}
    data['gate']=np.zeros_like(data['geometric_gate']) # explicit tracking validity absent, never invented
    candidates=data['score'][data['score']<THRESHOLD]
    thresholds=np.quantile(candidates,np.linspace(0,1,20)).clip(max=THRESHOLD)
    data['thresholds']=thresholds
    np.savez_compressed(OUT/'online.npz',**data); save(OUT/'rows.json',rows)
    save(OUT/'online_receipt.json',dict(rows=len(rows),frames=len(rows)*13,tracking_validity='NOT_AVAILABLE',
        observed_pose_fields=fields,old_pose='Deterministic frozen schedule replay, not saved estimator output',
        source_sha256=hashes,online_sha256=sha(OUT/'online.npz'),rows_sha256=sha(OUT/'rows.json'),plan_sha256=sha(OUT/'PLAN.json'),seconds=time.monotonic()-start))


def evaluate():
    start=time.monotonic(); receipt=read(OUT/'online_receipt.json'); assert receipt['online_sha256']==sha(OUT/'online.npz')
    assert receipt['rows_sha256']==sha(OUT/'rows.json')
    rows=read(OUT/'rows.json'); manifest={}
    for path in (MARGIN/'scene_manifest.json',FUSION/'natural97000/scene_manifest.json'):
        manifest.update({(int(s['unit']),int(s['config'])):s for s in read(path)})
    contact=[]; graze=[]; boundary=[]; fov_in=[]; fov_unresolved=[]; hashes={}; current_unit=None
    for i,row in enumerate(rows):
        deadline(); u,c,batch=row['unit'],row['config'],row['batch']
        if batch>=98000:
            if current_unit!=u:
                path=(AUG if batch==98000 else CONT)/f'truth/{"calibration" if batch==98000 else "evaluation"}/unit{u}.json'
                truth=read(path); hashes[str(path.relative_to(ROOT))]=sha(path); travel=np.asarray(truth['travel']); sensor=np.asarray(truth['sensor_center'])
                scenes={s['config']:s for s in truth['scenes']}; current_unit=u
            boxes=scenes[c]['boxes']
        else:
            sensor,travel,_=original_motion(u,c); boxes=manifest[u,c]['boxes']
        cc=[]; gg=[]; bb=[]; ff=[]; unresolved=[]
        for f in FRAMES:
            core=contact_boxes(boxes,travel[f]); cc.append(bool(core))
            gg.append(bool(not core and contact_boxes(boxes,travel[f],width=.40)))
            bb.append(bool(core and any(r['boundary'] for r in core)))
            witnesses=np.concatenate([np.asarray(r['witness_world']) for r in core]) if core else np.zeros((0,3))
            # Witness IN proves angular intersection; no witness is not proof OUT.
            hits=[any(fov(witnesses,sensor[f]@extrinsic(a),None).any() for a in angles) for angles in ((0,),(-15,15))]
            ff.append(hits); unresolved.append([bool(core and not h) for h in hits])
        contact.append(cc); graze.append(gg); boundary.append(bb); fov_in.append(ff); fov_unresolved.append(unresolved)
        if (i+1)%480==0: print('truth',u,i+1,'sequences',round(time.monotonic()-start,1),'s',flush=True)
    np.savez_compressed(OUT/'truth.npz',contact=contact,graze=graze,boundary=boundary,fov_in=fov_in,fov_unresolved=fov_unresolved)
    for path in (MARGIN/'scene_manifest.json',FUSION/'natural97000/scene_manifest.json'): hashes[str(path.relative_to(ROOT))]=sha(path)
    save(OUT/'truth_receipt.json',dict(source_sha256=hashes,truth_sha256=sha(OUT/'truth.npz'),seconds=time.monotonic()-start,
        inputs='Evaluator-only boxes and true current travel; predictions already saved',person_front_offset='NOT_AVAILABLE; saved travel origin proxy'))


def summarize():
    deadline(); start=time.monotonic(); rows=read(OUT/'rows.json')
    with np.load(OUT/'online.npz') as z: d={k:z[k] for k in z.files}
    with np.load(OUT/'truth.npz') as z: truth={k:z[k] for k in z.files}
    contact,graze=truth['contact'],truth['graze']; safe=~contact&~graze
    units=np.asarray([r['unit'] for r in rows]); modes=np.asarray([r['mode'] for r in rows]); batches=np.asarray([r['batch'] for r in rows]); turns=np.asarray([r['turn'] for r in rows])
    groups={'all':np.ones(len(rows),bool)}
    groups.update({f'mode{m}':modes==m for m in range(3)})
    groups.update({f'turn_{t}':turns==t for t in ('left','right','none')})
    groups.update({f'batch{b}':batches==b for b in score_sources()})
    for b in score_sources():
        for m in range(3): groups[f'batch{b}/mode{m}']=(batches==b)&(modes==m)
    results={}; rng=np.random.default_rng(2026100611)
    for group,selection in groups.items():
        if not selection.any(): continue
        uu,inv=np.unique(units[selection],return_inverse=True); clusters=len(uu)
        strata=np.array([f'{u//1000}/{u%3}/{"left" if u>=98000 and (u//3)%2 else "right"}' for u in uu])
        weights=np.zeros((1000,clusters),np.int16)
        for key in np.unique(strata):
            ids=np.flatnonzero(strata==key); draw=rng.integers(0,len(ids),size=(1000,len(ids)))
            for j in range(1000): weights[j,ids]=np.bincount(draw[j],minlength=len(ids))
        def ratio(num,den):
            num=np.asarray(num)[selection]; den=np.asarray(den)[selection]
            ns=np.bincount(inv,weights=num.sum(1),minlength=clusters); ds=np.bincount(inv,weights=den.sum(1),minlength=clusters)
            n=int(ns.sum()); total=int(ds.sum()); bn=weights@ns; bd=weights@ds; valid=bd>0
            ci=np.quantile(bn[valid]/bd[valid],[.025,.975]).tolist() if valid.any() else [None,None]
            return dict(numerator=n,denominator=total,value=n/total if total else None,ci95=ci,bootstrap_defined=int(valid.sum()))
        whole=np.ones_like(contact,bool)
        cell={}
        for ai,arm in enumerate(('single','dual')):
            score=d['score'][...,ai]; alarm=score>=THRESHOLD; curve=[]
            for ti,tau in enumerate(d['thresholds']):
                candidate=~alarm&(score<=tau); clear=candidate&d['gate'][...,ai]; unknown=~alarm&~clear
                error=clear&contact; blocked=candidate&contact&~d['gate'][...,ai]
                curve.append(dict(index=ti,tau=float(tau),risk=ratio(error,clear),unknown=ratio(unknown,whole),clear=ratio(clear,whole),
                    obstacle=ratio(alarm,whole),graze_clear=ratio(clear&graze,clear),availability=ratio(clear&safe,safe),
                    true_clear_obstacle=ratio(alarm&safe,safe),true_clear_unknown=ratio(unknown&safe,safe),
                    source_actual=dict(coverage_gap=int((error&~d['geometric_gate'][...,ai])[selection].sum()),
                        fov_in_readout_proxy=int((error&truth['fov_in'][...,ai])[selection].sum()),boundary=int((error&truth['boundary'])[selection].sum()),
                        fov_unresolved=int((error&truth['fov_unresolved'][...,ai])[selection].sum())),
                    blocked_candidate=dict(primary=ratio(blocked,candidate),gate_blocks=int(blocked[selection].sum()),
                        geometric_gap=int((blocked&~d['geometric_gate'][...,ai])[selection].sum()),
                        fov_in_readout_proxy=int((blocked&truth['fov_in'][...,ai])[selection].sum()),
                        boundary=int((blocked&truth['boundary'])[selection].sum()),fov_unresolved=int((blocked&truth['fov_unresolved'][...,ai])[selection].sum()))))
            cell[arm]=curve
        results[group]=dict(sequences=int(selection.sum()),frames=int(selection.sum()*13),units=clusters,curves=cell)
    aug={}
    aug_threshold=read(CONT/'evaluator/final_calibration.json')['threshold']
    for b in (98000,99000):
        mask=batches==b; aug[str(b)]={}
        for ai,arm in enumerate(('single','dual')):
            alarm=d['aug_score'][...,ai]>=aug_threshold
            aug[str(b)][arm]=dict(threshold=aug_threshold,frames=int(mask.sum()*13),obstacle=int(alarm[mask].sum()),
                clear=0,unknown=int((~alarm)[mask].sum()),risk=None,scope='Separate descriptive augmented ensemble3, same unavailable gate; no new threshold calibration')
    nzero={str(alpha):math.ceil(math.log(.025)/math.log(1-alpha)) for alpha in (.02,.05)}
    result=dict(status='DESCRIPTIVE_COMPLETE',guarantee=False,binary_verdict='NOT_APPLICABLE',groups=results,augmented=aug,
        tracking_validity='NOT_AVAILABLE',geometric_gate_pass_frames=d['geometric_gate'].sum((0,1)).tolist(),
        direction_unavailable_frames=d['direction_unavailable'].sum((0,1)).tolist(),far_stale_frames=d['far_stale'].sum((0,1)).tolist(),
        min_far_range_m=float(np.nanmin(d['min_far_range'])),thresholds=d['thresholds'].tolist(),
        labels=dict(contact=int(contact.sum()),graze_only=int(graze.sum()),strict_clear=int(safe.sum()),no_contact=int((~contact).sum())),
        stage2=dict(alpha_recommendation=.05,reason='5% needs fewer independent clear outputs; neither2% nor5% is established reachable because actual clear yield is zero',
            zero_error_clear_outputs=nzero,per_arm_delta=.025,observed_clear_yield=0.,certification_units='NOT_ESTIMABLE / no finite size at zero yield',
            units_at_hypothetical_yield={str(p):{a:math.ceil(n/p) for a,n in nzero.items()} for p in (.1,.2,.3,.5)},
            threshold_grid='Development grid archived but not frozen for certification until the range/validity contract is resolved',
            sampling='One config and one frame per independent unit;400units x2configs means at most400 independent cert points; audit separately'),
        provenance=dict(plan_sha256=sha(OUT/'PLAN.json'),online_sha256=sha(OUT/'online.npz'),truth_sha256=sha(OUT/'truth.npz'),
            source_sha256=sha(__file__)),summary_seconds=time.monotonic()-start,
        limits=['Consumed synthetic Development only; no hardware, independent certificate, or real-user safety evidence',
            'Strict rule produces zero clear;0/0 risk and its bootstrap are undefined, not zero-risk',
            'No explicit tracking/gravity validity; replayed/saved SE3 cannot supply these flags',
            'Retained2.1m radial gate fails far2.5m sentinel; no adjustment after seeing outcome',
            'True-clear availability excludes graze; no-contact denominator also reported',
            'TargetFOV witnesses prove angular intersection only, not unoccluded return or causal readout fault',
            'FOV/readout and boundary diagnostics overlap; not an exclusive causal partition',
            'Front label uses saved travel origin; anatomical front offset is unavailable',
            'Batch/mode/left-right distributions differ; left-turn data only98000/99000, not isolated yaw effect'])
    save(OUT/'result.json',result)
    with (OUT/'curves.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.writer(f); writer.writerow(['group','arm','tau','errors','clear_outputs','risk','unknown','unknown_lo','unknown_hi','availability'])
        for g,v in results.items():
            for arm,curve in v['curves'].items():
                for p in curve: writer.writerow([g,arm,p['tau'],p['risk']['numerator'],p['risk']['denominator'],p['risk']['value'],p['unknown']['value'],*p['unknown']['ci95'],p['availability']['value']])
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','augmented','limits')},ensure_ascii=False,indent=2),flush=True)


def plot():
    os.environ['MPLCONFIGDIR']=str(OUT/'mpl-cache')
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=read(OUT/'result.json'); fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    colors={'single':'#2764b8','dual':'#cd6b24'}
    for arm,color in colors.items():
        curve=r['groups']['all']['curves'][arm]; point=curve[-1]
        axes[0].errorbar(100*point['unknown']['value'],0.,xerr=np.array([[100*(point['unknown']['value']-point['unknown']['ci95'][0])],[100*(point['unknown']['ci95'][1]-point['unknown']['value'])]]),fmt='x',color=color,label=arm+' M3: risk undefined (0/0)')
        pct=lambda value: np.nan if value is None else 100*value
        y=[pct(p['blocked_candidate']['primary']['value']) for p in curve]
        axes[1].plot(range(1,21),y,'o-',color=color,label=arm+' blocked candidates')
        axes[1].fill_between(range(1,21),[pct(p['blocked_candidate']['primary']['ci95'][0]) for p in curve],[pct(p['blocked_candidate']['primary']['ci95'][1]) for p in curve],color=color,alpha=.12)
    axes[0].set(xlabel='Unable-to-judge outputs (%)',ylabel='Empirical false-clear risk (%)',title='Actual outputs: all 20 thresholds coincide')
    axes[0].set_yticks([0],['Undefined']); axes[0].text(.5,.8,'No clear outputs in either arm\nDo not interpret marker height as zero risk',ha='center',transform=axes[0].transAxes)
    axes[1].set(xlabel='Shared label-blind quantile grid, strict -> loose',ylabel='Contact among ungated low-score candidates (%)',title='Diagnostic only: all candidates blocked by gate')
    for ax in axes: ax.grid(alpha=.2); ax.legend(loc='lower right',fontsize=8)
    fig.suptitle('Consumed synthetic Development — descriptive, not a guarantee',fontsize=13)
    fig.savefig(OUT/'tristate_curves.png',dpi=170); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['freeze','online','evaluate','summarize','plot']); p.add_argument('--started-unix',type=float,default=time.time()); a=p.parse_args()
    if a.stage=='freeze': freeze(a.started_unix)
    elif a.stage=='online': online()
    elif a.stage=='evaluate': evaluate()
    elif a.stage=='summarize': summarize()
    else: plot()
