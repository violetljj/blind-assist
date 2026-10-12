"""Paired consumed-Development yaw intervention; frozen M3/VD, no training.

Physical rendering may read sealed boxes; prediction receives observations only.
The nominal commanded mount is known, as in the inherited pilot input contract.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import time

import numpy as np
import cnh_proposal_attribution_scenes as S
import cnh_displacement_ceiling_render as R
import cnh_temporal_readout_data as DATA
import cnh_location_reference_evaluate as L
import cnh_readout_pilot2_evaluate as P2
import cnh_three_level_sequence as SE
from cnh_fov_failure_split import read, save, sha

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
REF = ROOT/'artifacts.local/work/cnh-location-reference-20261004'
OUT = ROOT/'artifacts.local/work/cnh-head-recenter-dose-20261005'
DOSES = {'none': None, '2.5': 2.5, '2.1': 2.1, '1.7': 1.7,
         '1.3': 1.3, '1.0': 1., 'zero': 0.}
FRAMES = np.arange(3, 16)
START_UNIX = datetime(2026, 10, 4, 16, 23, 33, tzinfo=timezone.utc).timestamp()


def model_paths():
    import cnh_margin_confirm as MC
    return {'M3': MC.model_paths('M3'),
            'VD': [OLD/f'models/VD/seed{s}/model.pt' for s in range(3)]}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Use existing frozen plan')
    ref = read(REF/'result.json')
    selected = [s for s in ref['scenes'] if s['unit'] % 3 == 0]
    opposite = [s['unit'] for s in selected if not s['fov_in']]
    same = [s['unit'] for s in selected if s['fov_in']]
    assert (len(opposite), len(same)) == (11, 13)
    inputs = [REF/'result.json', REF/'episode_ledger.npz', DATA.D.BIAS]
    for u in opposite+same:
        tp = OLD/'truth/evaluation'/f'unit{u}.json'
        t = read(tp)
        assert t['mode'] == 0
        side = np.sign(np.mean([t['boxes'][0][0]['lo'][0], t['boxes'][0][0]['hi'][0]]))
        assert side == (-1 if u in opposite else 1)
        inputs += [tp, OLD/'observations/evaluation'/f'unit{u}.npz',
                   OLD/'templates/evaluation'/f'unit{u}.npz']
    for paths in model_paths().values():
        inputs += paths
    source = [Path(__file__), Path(DATA.__file__), Path(L.__file__), Path(P2.__file__),
              Path(SE.__file__), ROOT/'research/active/dtr-r0/nearfield/cnh_cvr_pilot.py',
              ROOT/'research/active/dtr-r0/nearfield/cnh_displacement_ceiling.py',
              ROOT/'research/active/dtr-r0/nearfield/cnh_temporal_readout_model.py']
    save(OUT/'PLAN.json', dict(run='CNH_HEAD_RECENTER_DOSE_20261005',
        created_utc=datetime.now(timezone.utc).isoformat(), source_commit='279de37a',
        scope='EXPLORE consumed simulated Development; same worlds, no training/readout search',
        opposite_units=opposite, same_units=same, doses=DOSES, K=4,
        deltas_cm=[1., 2., 5., -5., -10., -15., -20.], frames=FRAMES.tolist(),
        recenter='Keep yaw15 until crossing rc. First sampled frame with front_range<=rc has yaw0; preceding sampled frame stays15. Entire transition completes within one .2s frame interval; no interpolated exposure. zero means yaw0 at every frame. Pitch -10, position/travel/boxes unchanged.',
        sampled_rc_m={k: None if v is None else 3.05 if k=='zero' else float((3.05-.16*np.arange(16))[np.where(3.05-.16*np.arange(16)<=v)[0][0]]) for k,v in DOSES.items()},
        inputs='Inherited public commanded mount and noisy relative motion; ideal schedule supplied to input builder. True rc is controlled experiment intervention, not target-aware deployable trigger.',
        noise='Same photon and pose seeds as original for paired doses; K varies BOTH photon and estimated-pose noise, not isolated pose assay',
        photon_seed_prefix=2026100406, pose_seed_prefix=2026100407,
        thresholds={a:ref['thresholds'][a] for a in ('M3','VD')},
        primary='Opposite shallow first alarm >=.9m;88 sequences=22 trajectories/11 scene clusters. Both frozen arms reported; VD is decision arm, M3 separately.',
        secondary='Same shallow104; all outside5/10/15/20cm first stops. Clear outside15/20 exact original categories separately; never relabel pass0-10 as false stops.',
        geometry='Same angular_rays16 first-hit code; all16 and predeadline>=.9 summaries, last visible range, visible frames, total/mean first-hit target rays divided16384. K shares geometry.',
        bootstrap=dict(n=1000, seed=2026100507, unit='whole paired scene cluster, all branches/K/doses together'),
        r_star='For each arm: minimum NUMERIC rc with absolute timely count difference from all-zero <=5/88; no monotonic assumption. none reported but excluded from rc. No qualifying dose -> NOT_REACHED.',
        interpretation='VD r*<=1.3: late ideal recenter sufficient; propose estimated-pose uncovered-side UNKNOWN hint. r*>=2.1: early ideal recenter needed; mount/dual/wider-FOV options require user decision. Between -> INTERMEDIATE. Same-side decline>5/104 vs none separately flagged. Software/hardware feasibility remains untested.',
        budget_seconds=3600, started_unix=START_UNIX, deadline_unix=START_UNIX+3600,
        budget_priority='All opposite doses before same group; preserve per-unit checkpoints. Do not expand budget.',
        no_training=True, source_sha256={str(p):sha(p) for p in source}|R.source_sha256(),
        input_sha256={str(p):sha(p) for p in inputs}))
    print('PLAN FROZEN', sha(OUT/'PLAN.json'), opposite, same, flush=True)


def yaw_schedule(ranges, dose):
    if dose == 'none':
        return np.full(16, 15.)
    if dose == 'zero':
        return np.zeros(16)
    return np.where(np.arange(16) >= np.where(ranges <= DOSES[dose])[0][0], 0., 15.)


def render_one(job):
    unit, dose, deadline = job
    if time.time() >= deadline:
        return None
    target = OUT/'observations'/dose/f'unit{unit}.npz'
    if target.exists():
        return str(target)
    tick = time.monotonic()
    truth = read(OLD/'truth/evaluation'/f'unit{unit}.json')
    with np.load(OLD/'observations/evaluation'/f'unit{unit}.npz') as z:
        original = {k:z[k].copy() for k in z.files}
    ranges = np.asarray(truth['front_range_m'])
    yaw = yaw_schedule(ranges, dose)
    sensor = original['sensor'].copy()
    sensor[:,:3,:3] = np.stack([S.ry(y)@S.rx(-10) for y in yaw])
    if dose == 'none':
        np.testing.assert_array_equal(sensor, original['sensor'])
        observation = original
        with np.load(OLD/'templates/evaluation'/f'unit{unit}.npz') as z:
            ids = z['object_id'].copy()
    else:
        rendered = [R.expected(dict(poses=sensor, boxes=boxes)) for boxes in truth['boxes']]
        ambient = rendered[0]['ambient']
        assert all(np.array_equal(ambient,r['ambient']) for r in rendered)
        hist = np.stack([np.stack([R.sample(r['expectation'], ambient,
            int(np.random.SeedSequence([2026100406,unit,v,k]).generate_state(1)[0]))[0]
            for k in range(4)]) for v,r in enumerate(rendered)])
        from cnh_track_a_readout import noisy_poses
        noisy = np.stack([noisy_poses(sensor, int(np.random.SeedSequence([2026100407,unit,k]).generate_state(1)[0]),dt=.2) for k in range(4)])
        observation = dict(hist=hist, ambient=ambient, sensor=sensor, travel=original['travel'], noisy=noisy)
        ids = np.stack([r['object_id'] for r in rendered]).astype(np.int8)
        # Unchanged physical prefix must retain the same sampled observation.
        prefix = np.flatnonzero(yaw == 15.)
        np.testing.assert_array_equal(hist[:,:,prefix], original['hist'][:,:,prefix])
    counts = (ids == 0).sum(axis=(2,3,4))
    pre = ranges >= .9
    geometry = dict(unit=unit,dose=dose,yaw_deg=yaw.tolist(),ranges_m=ranges.tolist(),
        counts=counts.tolist(),last_visible_range_m=[None if not np.any(c>0) else float(ranges[np.where(c>0)[0][-1]]) for c in counts],
        last_visible_predeadline_m=[None if not np.any((c>0)&pre) else float(ranges[np.where((c>0)&pre)[0][-1]]) for c in counts],
        visible_frames_predeadline=((counts>0)&pre).sum(1).tolist(),
        exposure_ray_frames_predeadline=counts[:,pre].sum(1).tolist(),
        mean_ray_fraction_predeadline=(counts[:,pre].mean(1)/16384).tolist(),
        seconds=time.monotonic()-tick)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**observation)
    save(target.with_suffix('.geometry.json'), geometry)
    return str(target)


class Predictor:
    def __init__(self):
        import torch
        import cnh_temporal_readout_model as M
        from cnh_cvr_v2_materialize import BatchedProjector
        self.torch, self.M = torch, M
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available()
        self.projector = BatchedProjector()
        self.nets = {}
        for arm, paths in model_paths().items():
            self.nets[arm] = []
            for path in paths:
                net = M.CVR().cuda().eval()
                net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
                self.nets[arm].append(net)

    def predict(self, path):
        import cnh_cvr_pilot as CP
        import cnh_displacement_ceiling as D
        torch = self.torch
        tick = time.monotonic()
        with np.load(path) as obs:
            z = DATA.normalized_z(obs['hist'],obs['ambient'])
            sensor, travel, noisy = (obs[k].copy() for k in ('sensor','travel','noisy'))
        raw = {a:np.empty((7,4,13,2), np.float32) for a in self.nets}
        with torch.inference_mode():
            for k in range(4):
                voxels=[]
                for f in FRAMES:
                    transforms=CP.relative_transforms(sensor,travel,noisy[k],int(f))
                    v=D.project_many(self.projector,z[:,k,max(0,f-7):f+1],transforms)
                    voxels.append(v.half())
                x=torch.stack(voxels,1).reshape(7*13,3,24,17,33)
                for arm,nets in self.nets.items():
                    parts=[]
                    for j in range(0,len(x),64):
                        xb=self.M.prepare_voxels(x[j:j+64])
                        parts.append(torch.stack([net(xb) for net in nets]).mean(0).cpu().numpy())
                    raw[arm][:,k]=np.concatenate(parts).reshape(7,13,2)
        assert all(np.isfinite(v).all() for v in raw.values())
        return raw,time.monotonic()-tick

    def close(self):
        self.nets.clear()
        self.projector=None
        self.torch.cuda.empty_cache()


def run(workers=3):
    plan=read(OUT/'PLAN.json')
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    assert all(sha(p)==h for p,h in plan['input_sha256'].items())
    save(OUT/'request.json',dict(pid=os.getpid(),plan_sha256=sha(OUT/'PLAN.json'),deadline_unix=plan['deadline_unix']))
    predictor=Predictor()
    timings=[]
    try:
        for group in ('opposite','same'):
            jobs=[(u,d,plan['deadline_unix']) for u in plan[group+'_units'] for d in DOSES]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for path in pool.map(render_one,jobs,chunksize=1):
                    if path is None:
                        continue
                    path=Path(path);dose=path.parent.name;unit=int(path.stem[4:])
                    dest=OUT/'scores'/dose/path.name
                    if dest.exists():
                        continue
                    if time.time()>=plan['deadline_unix']:
                        continue
                    raw,seconds=predictor.predict(path)
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest,**raw)
                    timings.append(dict(unit=unit,dose=dose,prediction_s=seconds))
                    save(dest.with_suffix('.json'),dict(status='COMPLETE',observation_sha256=sha(path),score_sha256=sha(dest),seconds=seconds,plan_sha256=sha(OUT/'PLAN.json')))
                    save(OUT/'progress.json',dict(stage=group,completed=len(list((OUT/'scores').glob('*/*.npz'))),total=168,last_unit=unit,last_dose=dose,last_activity_utc=datetime.now(timezone.utc).isoformat()))
                    print('DONE',group,unit,dose,'prediction_s',round(seconds,2),'remaining_s',round(plan['deadline_unix']-time.time()),flush=True)
            if time.time()>=plan['deadline_unix']:
                break
    finally:
        predictor.close()
        save(OUT/'timing.json',dict(records=timings,elapsed_from_authorization_s=time.time()-plan['started_unix'],backend='CUDA projection+M3/VD, CPU analytic renderer',device='RTX5060 Laptop',cpu_reason='GPU_BACKEND_UNAVAILABLE for inherited numpy expectation renderer',render_workers=workers))
    evaluate()


def smoothed(raw):
    return SE.smooth(np.asarray(raw,np.float64).reshape(28,1,13,2))[:,0].reshape(7,4,13,2)


def evaluate():
    plan=read(OUT/'PLAN.json')
    baseline, scenes, _, _, _, covered, parity = L.load_baselines()
    index={s['unit']:i for i,s in enumerate(scenes)}
    cells={};records=[];checks=[];geometries=[]
    boot_rng=np.random.default_rng(plan['bootstrap']['seed'])
    for group in ('opposite','same'):
        units=plan[group+'_units'];boots=boot_rng.integers(len(units),size=(1000,len(units)))
        cells[group]={}
        for dose in DOSES:
            values={a:[] for a in ('M3','VD')}
            for unit in units:
                path=OUT/'scores'/dose/f'unit{unit}.npz'
                if not path.exists():
                    continue
                truth=read(OLD/'truth/evaluation'/f'unit{unit}.json');i=index[unit]
                ranges=np.asarray(truth['front_range_m'])[FRAMES]
                geom=read(OUT/'observations'/dose/f'unit{unit}.geometry.json');geometries.append(dict(group=group,**geom))
                with np.load(path) as z:
                    scores={a:smoothed(z[a])[...,truth['group']] for a in values}
                for arm,score in scores.items():
                    if dose=='none':
                        error=float(np.max(np.abs(score-baseline[arm][i])))
                        assert error<1e-5,('Baseline score parity',unit,arm,error)
                        checks.append(dict(unit=unit,arm=arm,max_abs=error))
                    ev=L.events(score[None],plan['thresholds'][arm],ranges[None],covered[i:i+1])
                    timely=ev['timely'][0];stopped=ev['stopped'][0]
                    outside_clear=np.all(np.asarray(truth['categories'])[[5,6],:,truth['group']]=='clear',axis=-1)
                    values[arm].append(dict(unit=unit,timely=int(timely[:2].sum()),outside=stopped[[3,4,5,6]].sum(1).tolist(),clear15_20=int(stopped[[5,6]][outside_clear].sum()),clear_n=int(outside_clear.sum()*4)))
                    for v in range(7):
                        for k in range(4):
                            f=int(ev['first_index'][0,v,k])+3 if stopped[v,k] else None
                            records.append(dict(group=group,unit=unit,dose=dose,arm=arm,variant=v,intrusion_cm=truth['intrusion_cm'][v],replica=k,stopped=bool(stopped[v,k]),timely=bool(timely[v,k]),first_frame=f,first_range_m=None if f is None else float(truth['front_range_m'][f]),visible_at_first=None if f is None else bool(geom['counts'][v][f]>0)))
            complete=len(values['VD'])==len(units)
            cell=dict(status='COMPLETE' if complete else 'PARTIAL_OR_NOT_RUN',completed_scenes=len(values['VD']),expected_scenes=len(units),expected_shallow_n=len(units)*8,arms={})
            for arm,rows in values.items():
                num=np.array([r['timely'] for r in rows])
                ci=None if not complete else np.quantile(num[boots].sum(1),[.025,.975]).tolist()
                cell['arms'][arm]=dict(timely=int(num.sum()),shallow_n=len(rows)*8,scene_cluster_count_ci95=ci,outside_stops_5_10_15_20=np.asarray([r['outside'] for r in rows],int).sum(0).tolist() if rows else [0]*4,outside_each_n=len(rows)*4,clear15_20_stops=sum(r['clear15_20'] for r in rows),clear15_20_n=sum(r['clear_n'] for r in rows),per_scene=rows)
            cells[group][dose]=cell
    decisions={}
    for arm in ('M3','VD'):
        c=cells['opposite'];upper=c['zero']['arms'][arm]['timely']
        eligible=[float(d) for d in DOSES if d not in ('none','zero') and c[d]['status']=='COMPLETE' and c['zero']['status']=='COMPLETE' and abs(c[d]['arms'][arm]['timely']-upper)<=5]
        rc=min(eligible) if eligible else None
        branch='NOT_REACHED' if rc is None else 'LATE_IDEAL_RECENTER_SUFFICIENT' if rc<=1.3 else 'EARLY_IDEAL_RECENTER_REQUIRED' if rc>=2.1 else 'INTERMEDIATE'
        declines={d:cells['same']['none']['arms'][arm]['timely']-v['arms'][arm]['timely'] for d,v in cells['same'].items() if v['status']=='COMPLETE' and cells['same']['none']['status']=='COMPLETE'}
        decisions[arm]=dict(r_star_m=rc,branch=branch,zero_timely=upper,same_side_declines=declines,same_decline_over5=[d for d,n in declines.items() if n>5])
        # Paired cluster CIs of dose minus zero/none counts, using identical scene resamples.
        for group in cells:
            units=plan[group+'_units'];rng=np.random.default_rng(plan['bootstrap']['seed']+(group=='same'))
            ix=rng.integers(len(units),size=(1000,len(units)))
            for dose,cell in cells[group].items():
                for ref in ('none','zero'):
                    if cell['status']==cells[group][ref]['status']=='COMPLETE':
                        delta=np.array([r['timely'] for r in cell['arms'][arm]['per_scene']])-np.array([r['timely'] for r in cells[group][ref]['arms'][arm]['per_scene']])
                        cell['arms'][arm]['paired_minus_'+ref]=dict(count=int(delta.sum()),scene_cluster_count_ci95=np.quantile(delta[ix].sum(1),[.025,.975]).tolist())
    save(OUT/'sequences.json',records);save(OUT/'geometry.json',geometries)
    result=dict(status='COMPLETE' if all(c['status']=='COMPLETE' for g in cells.values() for c in g.values()) else 'PARTIAL',cells=cells,decision_arm='VD',decisions=decisions,baseline_score_parity=checks,full48_baseline_parity=parity['status'],plan_sha256=sha(OUT/'PLAN.json'),source_inputs_unchanged=all(sha(p)==h for p,h in plan['input_sha256'].items()),budget=read(OUT/'timing.json') if (OUT/'timing.json').exists() else None)
    save(OUT/'result.json',result)
    print('RESULT',result['status'],decisions,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('plan','run','evaluate'));parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    {'plan':freeze,'run':lambda:run(args.workers),'evaluate':evaluate}[args.stage]()
