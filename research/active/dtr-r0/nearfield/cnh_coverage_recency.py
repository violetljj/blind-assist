"""Causal coverage-age marker and consumed Development replay; no model calls.

The online marker accepts estimated SE3 only. Evaluator geometry is joined after
all marker decisions are saved. The horizontal footprint is not occupied-volume
truth, and UNKNOWN is never an alarm veto.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cnh_coverage_policy_unknown as OLD

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
OUT = WORK / 'cnh-coverage-recency-20261005'
PILOT = WORK / 'cnh-readout-pilot2-20261005'
NATURAL = WORK / 'cnh-margin-confirm-20261002'
TS = (.6, 1.)
DT = .2
EDGE = np.tan(np.deg2rad(22.5))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(started_unix):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'PLAN.json'
    if path.exists():
        raise FileExistsError('Existing plan: resume its stage, do not reset budget')
    units = json.loads((WORK/'cnh-coverage-policy-20261004/result.json').read_text())['units']
    plan = dict(task='CNH_COVERAGE_RECENCY_20261005', phase='EXPLORE consumed synthetic Development',
        created_utc=datetime.now(timezone.utc).isoformat(), started_unix_s=started_unix,
        deadline_unix_s=started_unix+5400, maximum_wall_seconds=5400,
        controlled_units=units, natural_units=dict(calib=list(range(95000,95048)), evaluation=list(range(96000,96096))),
        K=4, natural_configs=40, frames=list(range(16)), T_seconds=TS, dt_seconds=DT,
        threshold=.8557642486787612, M3_weights=[1,2,4,8,16], no_train=True, no_photon_render=True,
        no_images=True, no_protected_test=True,
        rule=dict(direction='last 1s estimated net displacement projected to assumed gravity-horizontal XZ',
            min_displacement_m=.05, direction_warmup_frames=5,
            grid='5cm fixed world XZ lattice anchored at first estimated sensor position and initial horizontal forward basis; sensor-height horizontal plane',
            corridor_width_m=[-.29,.29], forward_query_m=[.9,1.4],
            boundary_sentinels='exact +/- .29m boundaries at forward .9:.05:1.4m, projected into world; history queried at same world point',
            observed='point inside square 45deg FOV and sensor radial distance <=2.1m; current frame inclusive',
            installation='saved SE3 is sensor pose and already contains nominal -10deg pitch; do not apply pitch a second time',
            recency='any query point without an observation at age <= T triggers UNKNOWN; never observed starts stale',
            unavailable='frames0..4 WARMUP_UNKNOWN; insufficient net translation DIRECTION_UNKNOWN; separate from postwarmup flags',
            permitted_inputs='estimated sensor SE3, declared dt/gravity and mounting geometry only'),
        truth=dict(direction='true travel forward at current frame', footprint='same lattice and boundary sentinels at true initial sensor height',
            coverage='true sensor SE3 + nearest 128x128 tangent subray; first-hit radial range >= point radial distance-.025m; recent history includes current frame',
            point_tolerance_m=.025, target='target box nearest-front point at corridor edge when intruding, target mid-height; evaluation only',
            sealed_ids='original controlled target IDs retained as independent exposure crosscheck',
            natural='user approved evaluator-only ray geometry replay from existing boxes and poses because full sealed rays absent'),
        metrics='both T retained, no outcome-based selection; compare all/mode0/1/2 after prediction; flag frames and segment starts / all and postwarmup proxy minutes; gap sensitivity and fresh-trigger fractions',
        reaction=dict(reaction_s=[.5,1.], turn_s=.4, grid_m=[2.5,2.1,1.7,1.3,1.],
            formula='first trigger true front range minus .8 * delay; separate turn-start delay=reaction and turn-complete delay=reaction+.4; classify intervals only, no efficacy interpolation'),
        limits=['horizontal footprint is not full HEAD/BODY volume coverage', 'estimated translation and gravity are assumptions; head IMU alone gives no travel direction',
            '3.05m controlled start is artificial; no late onset of head bias', 'no detection delay or real-user motion data',
            'natural mode is evaluator annotation only; K and configs not independent scenes'],
        input_sha256={str(p):sha(p) for p in [WORK/'cnh-coverage-policy-20261004/evaluation/ledger.npz', NATURAL/'scene_manifest.json',
            Path(OLD.__file__), WORK/'cnh-track-a-v5-20260928/data/source/cnh_track_a_readout.py']})
    save(path, plan)
    print('PLAN SEALED', sha(path), flush=True)


def direction(poses, frame):
    if frame < 5:
        return None
    d = (poses[frame,:3,3]-poses[frame-5,:3,3])[[0,2]]
    norm = np.linalg.norm(d)
    return None if norm < .05 else d/norm


def query_points(origin_pose, current_pose, forward):
    """Fixed world cell centers plus explicit corridor boundary points."""
    origin = origin_pose[:3,3]
    initial = origin_pose[[0,2],2]
    initial = initial/np.linalg.norm(initial)
    right0 = np.array([initial[1],-initial[0]])
    basis = np.stack((right0,initial),axis=1)
    current = current_pose[[0,2],3]
    right = np.array([forward[1],-forward[0]])
    corners = np.array([current+a*right+b*forward for a in [-.29,.29] for b in [.9,1.4]])
    local = (corners-origin[[0,2]])@basis
    low = np.floor(local.min(0)/.05).astype(int)-1
    high = np.ceil(local.max(0)/.05).astype(int)+1
    xx,zz = np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1))
    xz = np.stack((xx.ravel(),zz.ravel()),axis=-1)*.05@basis.T+origin[[0,2]]
    rel = xz-current
    mask = (np.abs(rel@right)<=.29+1e-9)&(rel@forward>=.9-1e-9)&(rel@forward<=1.4+1e-9)
    sentinel = np.array([current+a*right+b*forward for a in [-.29,.29] for b in np.arange(.9,1.4001,.05)])
    xz = np.concatenate((xz[mask],sentinel))
    points = np.zeros((len(xz),3)); points[:,[0,2]]=xz; points[:,1]=origin[1]
    return points


def fov(points, pose):
    local = (points-pose[:3,3])@pose[:3,:3]
    z = local[:,2]
    return ((z>0)&(np.abs(local[:,0])<=EDGE*z)&(np.abs(local[:,1])<=EDGE*z)
            &(np.linalg.norm(local,axis=1)<=2.1+1e-9))


def marker(estimated):
    estimated = np.asarray(estimated,float)
    if estimated.shape!=(16,4,4) or not np.isfinite(estimated).all():
        raise ValueError('finite16 sensor SE3 required')
    flags=np.zeros((2,16),bool); stale=np.full((2,16),-1,np.int16)
    counts=np.zeros(16,np.int16); state=np.zeros(16,np.int8)
    for f in range(5,16):
        d=direction(estimated,f)
        if d is None:
            state[f]=1; flags[:,f]=True; continue
        state[f]=2
        pts=query_points(estimated[0],estimated[f],d); counts[f]=len(pts)
        for j,T in enumerate(TS):
            fresh=np.zeros(len(pts),bool)
            for h in range(max(0,f-int(round(T/DT))),f+1):
                fresh |= fov(pts,estimated[h])
            stale[j,f]=int((~fresh).sum()); flags[j,f]=not fresh.all()
    return dict(flag=flags,stale_count=stale,point_count=counts,state=state)


def online():
    import cnh_cvr_pilot as CP
    start=time.monotonic(); plan=json.loads((OUT/'PLAN.json').read_text())
    if (OUT/'online.npz').exists():
        raise FileExistsError('Online ledger already saved')
    rows=[]; decisions=[]; old=[]; pose_hash=[]
    def add(split,u,c,k,est):
        if time.time()>=plan['deadline_unix_s']: raise TimeoutError('wall budget')
        decisions.append(marker(est)); old.append(OLD.marker(est)['state'])
        pose_hash.append(hashlib.sha256(np.asarray(est).tobytes()).hexdigest())
        rows.append(dict(split=split,unit=u,config=c,replica=k))
    for u in plan['controlled_units']:
        with np.load(PILOT/'observations/evaluation'/f'unit{u}.npz',allow_pickle=False) as z:
            for k,est in enumerate(z['noisy']): add('controlled',u,-1,k,est)
    for split,units in plan['natural_units'].items():
        for u in units:
            for c in range(40): add(split,u,c,0,CP.motion_metadata(u,c)[2])
        print('ONLINE',split,len(rows),'seconds',round(time.monotonic()-start,2),flush=True)
    np.savez_compressed(OUT/'online.npz', **{key:np.asarray([d[key] for d in decisions]) for key in decisions[0]},
        old_state=np.asarray(old),pose_sha256=np.asarray(pose_hash))
    save(OUT/'online_rows.json',rows)
    save(OUT/'online_receipt.json',dict(plan_sha256=sha(OUT/'PLAN.json'),ledger_sha256=sha(OUT/'online.npz'),
        rows_sha256=sha(OUT/'online_rows.json'),source_sha256=sha(Path(__file__)),elapsed_s=time.monotonic()-start,
        backend='TASK_NOT_GPU_SUITABLE; small causal geometry/metadata arrays on NumPy CPU'))
    print('ONLINE COMPLETE',len(rows),flush=True)


def truth_sequence(sensor, travel, boxes, target=None):
    import cnh_coverage_recency_truth as E
    gaps=np.zeros((2,16),bool); miss_count=np.full((2,16),-1,np.int16)
    target_fresh=np.zeros((2,16),bool); target_distance=np.full(16,np.nan)
    for f in range(5,16):
        d=travel[f,[0,2],2]; d=d/np.linalg.norm(d)
        pts=query_points(sensor[0],sensor[f],d)
        if target is not None:
            target_distance[f]=float((target-travel[f,:3,3])@travel[f,:3,2])
        history=list(range(max(0,f-5),f+1))
        observed=[]; target_observed=[]
        for h in history:
            observed.append(E.observed_points(pts,sensor[h],boxes=boxes))
            target_observed.append(False if target is None else bool(E.observed_points(target[None],sensor[h],boxes=boxes)[0]))
        for j,T in enumerate(TS):
            length=int(round(T/DT))+1
            fresh=np.any(observed[-length:],axis=0); tf=any(target_observed[-length:])
            miss_count[j,f]=int((~fresh).sum()); gaps[j,f]=not fresh.all(); target_fresh[j,f]=tf
    return gaps,miss_count,target_fresh,target_distance


def summarize(flags,gaps,selection):
    flag=flags[selection,5:]; gap=gaps[selection,5:]
    starts=flag&~np.pad(flag[:,:-1],((0,0),(1,0)),constant_values=False)
    n=flag.size; exposure=n*DT/60
    return dict(sequences=int(flag.shape[0]), evaluated_frames=int(n), flagged_frames=int(flag.sum()),
        flag_fraction=float(flag.mean()) if n else None, flagged_sequences=int(flag.any(1).sum()),
        proxy_minutes=float(exposure),segments=int(starts.sum()),segments_per_proxy_minute=float(starts.sum()/exposure) if exposure else None,
        gap_frames=int(gap.sum()),gap_sequences=int(gap.any(1).sum()),
        flag_and_gap_frames=int((flag&gap).sum()),flag_on_fresh_frames=int((flag&~gap).sum()),
        unflagged_gap_frames=int((~flag&gap).sum()),fresh_frames=int((~gap).sum()),
        gap_sensitivity=float(flag[gap].mean()) if gap.any() else None,
        fresh_flag_fraction=float(flag[~gap].mean()) if (~gap).any() else None,
        segments_started_on_gap=int((starts&gap).sum()),segments_started_on_fresh=int((starts&~gap).sum()))


def range_bin(r):
    edges=[2.5,2.1,1.7,1.3,1.]
    if r>=edges[0]: return '>=2.5m'
    for hi,lo in zip(edges[:-1],edges[1:]):
        if r>=lo: return f'[{lo:g},{hi:g})m'
    return '<1.0m'


def evaluate():
    import cnh_coverage_recency_truth as E
    if (OUT/'result.json').exists():
        raise FileExistsError('Completed evaluation is immutable; preserve its ledger')
    start=time.monotonic();plan=json.loads((OUT/'PLAN.json').read_text());rows=json.loads((OUT/'online_rows.json').read_text())
    receipt=json.loads((OUT/'online_receipt.json').read_text())
    assert sha(OUT/'online.npz')==receipt['ledger_sha256'] and sha(OUT/'online_rows.json')==receipt['rows_sha256']
    with np.load(OUT/'online.npz',allow_pickle=False) as z: online_data={k:z[k] for k in z.files}
    manifest=E.load_natural_manifest(); cache={}; source_hashes=dict(plan['input_sha256'])
    gaps=[];targets=[];distances=[];true_counts=[]; controlled_targets=[];sealed_checks=[]
    for index,row in enumerate(rows):
        if time.time()>=plan['deadline_unix_s']:raise TimeoutError('wall budget')
        split,u,c,k=row['split'],row['unit'],row['config'],row['replica']
        key=(split,u,c)
        if key not in cache:
            if split=='controlled':
                tp=PILOT/'truth/evaluation'/f'unit{u}.json';op=PILOT/'observations/evaluation'/f'unit{u}.npz'
                truth=json.loads(tp.read_text());source_hashes[str(tp)]=sha(tp);source_hashes[str(op)]=sha(op)
                with np.load(op,allow_pickle=False) as z:sensor=z['sensor'];travel=z['travel']
                boxes=truth['boxes'][0]
                cache[key]=truth_sequence(sensor,travel,boxes)
                # Targets differ across the two shallow branches; keep their geometry separate.
                branch=[]
                for b in range(2):
                    box=truth['boxes'][b][0];pt=E.intrusion_point(box)
                    branch.append(truth_sequence(sensor,travel,truth['boxes'][b],pt)[2:])
                controlled_targets.append(dict(unit=u,branches=[dict(fresh=x[0].tolist(),distance=x[1][5:].tolist()) for x in branch]))
                with np.load(PILOT/'templates/evaluation'/f'unit{u}.npz',allow_pickle=False) as z:
                    sealed=z['object_id'][:2]
                # Independent first-hit-ID replay, all16 frames, both shallow branches.
                sealed_checks.append(E.check_sealed_ids(sensor,truth['boxes'][:2],sealed))
            else:
                item=manifest[(split,u,c)];sensor,travel,est=E.natural_poses(u,c)
                assert hashlib.sha256(np.asarray(est).tobytes()).hexdigest()==online_data['pose_sha256'][index]
                box=item['boxes'][0];pt=E.intrusion_point(box)
                cache[key]=truth_sequence(sensor,travel,item['boxes'],pt)
        gap,count,tf,td=cache[key]
        gaps.append(gap);true_counts.append(count);targets.append(tf);distances.append(td)
        if index%400==0:print('TRUTH',index,len(rows),'seconds',round(time.monotonic()-start,2),flush=True)
    gaps=np.asarray(gaps);target_fresh=np.asarray(targets);target_distance=np.asarray(distances)
    np.savez_compressed(OUT/'truth_ledger.npz',gap=gaps,stale_count=np.asarray(true_counts),target_fresh=target_fresh,target_distance=target_distance)
    summaries={}; splits=np.asarray([r['split'] for r in rows]); modes=np.asarray([r['unit']%3 for r in rows])
    old_flag=online_data['old_state']>=3
    for split in ('controlled','calib','evaluation'):
        for mode in ('all',0,1,2):
            selection=(splits==split)&(True if mode=='all' else modes==mode)
            if not selection.any():continue
            arms={}
            for j,T in enumerate(TS):
                arms[f'recency_{T:g}s']=summarize(online_data['flag'][:,j],gaps[:,j],selection)
                arms[f'8deg_truth_T{T:g}s']=summarize(old_flag,gaps[:,j],selection)
                relevant=(target_distance>=.9)&(target_distance<=1.4)&(np.arange(16)[None]>=5)
                relevant &=selection[:,None]
                target_gap=relevant&~target_fresh[:,j]
                for name,flags in [(f'recency_{T:g}s',online_data['flag'][:,j]),(f'8deg_truth_T{T:g}s',old_flag)]:
                    arms[name]['target_gap_frames']=int(target_gap.sum())
                    arms[name]['target_gap_flagged_frames']=int((target_gap&flags).sum())
                    arms[name]['target_gap_sequences']=int(target_gap.any(1).sum())
            summaries[f'{split}/mode{mode}']=arms
    with np.load(WORK/'cnh-coverage-policy-20261004/evaluation/ledger.npz',allow_pickle=False) as z:
        ledger={k:z[k] for k in z.files}
    cohort=json.loads((WORK/'cnh-coverage-policy-20261004/PLAN.json').read_text())['cohorts']['FOV_OUT']
    controlled=[]; reaction={}
    for name,flags in [('recency_.6s',online_data['flag'][:,0]),('recency_1s',online_data['flag'][:,1]),('8deg',old_flag)]:
        hits=0;records=[]
        for index,row in enumerate(rows):
            if row['split']!='controlled':continue
            u,k=row['unit'],row['replica'];i=ledger['units'].tolist().index(u)
            truth=json.loads((PILOT/'truth/evaluation'/f'unit{u}.json').read_text())
            ranges=np.asarray(truth['front_range_m']);ix=np.flatnonzero(flags[index]);first=int(ix[0]) if len(ix) else None
            timely=bool((flags[index]&(ranges>=.9)).any())
            if u in cohort:
                hits+=int((ledger['baseline_timely'][i,:2,k]|timely).sum())
            records.append(dict(unit=u,replica=k,first_frame=first,first_time_s=None if first is None else first*DT,
                first_range_m=None if first is None else float(ranges[first]),unknown_timely=timely))
        controlled.append(dict(arm=name,first_triggers=records,opposite_shallow_alarm_or_unknown=hits,denominator=88))
        for delay in (.5,1.):
            counts=dict(turn_start={},turn_complete={})
            for r in records:
                if r['first_range_m'] is None:continue
                for action,latency in [('turn_start',delay),('turn_complete',delay+.4)]:
                    b=range_bin(r['first_range_m']-.8*latency); counts[action][b]=counts[action].get(b,0)+1
            reaction[f'{name}/reaction{delay:g}s']=counts
    result=dict(status='COMPLETE',summaries=summaries,controlled=controlled,reaction=reaction,
        elapsed_s=time.monotonic()-start,online_elapsed_s=receipt['elapsed_s'],
        provenance=dict(plan_sha256=sha(OUT/'PLAN.json'),online_sha256=sha(OUT/'online.npz'),truth_sha256=sha(OUT/'truth_ledger.npz'),
            sources=source_hashes,rule_source_sha256=sha(Path(__file__)),truth_source_sha256=sha(Path(E.__file__))),
        sealed_checks=sealed_checks,limits=plan['limits'],warmup_frames_per_sequence=5)
    save(OUT/'controlled_target_coverage.json',controlled_targets);save(OUT/'result.json',result)
    print('EVALUATION COMPLETE',round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['plan','online','evaluate'],required=True)
    p.add_argument('--started-unix',type=float,default=time.time());a=p.parse_args()
    if a.stage=='plan':freeze(a.started_unix)
    elif a.stage=='online':online()
    else:evaluate()
