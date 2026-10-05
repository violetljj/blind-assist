"""Paired dual-sensor EXPLORE: frozen M3, fixed poses, no training.

Only this run's ignored directory is written. Rendering truth is not exported
to inference observations; the two sensors inherit one head-pose estimate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-dual-sensor-alarm-20261005'
OLD = ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
PROFILE = ROOT/'artifacts.local/work/cnh-location-reference-20261004/performance/cohort_profiles.json'
PHOTON_SEED = 2026100515
SPLIT_SEED = 2026100517
FRAMES = np.arange(3, 16)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    for key, name in [('TEMP','tmp'),('TMP','tmp'),('CUPY_CACHE_DIR','cupy-cache'),('MPLCONFIGDIR','mpl-cache')]:
        path = OUT/name; path.mkdir(exist_ok=True); os.environ[key] = str(path)


def deadline():
    if time.time() >= read(OUT/'PLAN.json')['deadline_unix']:
        raise TimeoutError('Authorized two-hour wall clock reached; retain partial evidence')


def freeze():
    setup()
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('Existing frozen PLAN: resume by stage, never overwrite')
    profiles = read(PROFILE)['profiles']
    units = read(OLD/'PLAN.json')['eval_units']
    assert [p['unit'] for p in profiles] == units and len(units) == 48
    rng = np.random.default_rng(SPLIT_SEED)
    strata, calibration, evaluation = [], [], []
    for i, key in enumerate((g,m,c) for g in (0,1) for m in (0,1) for c in ('none','panel')):
        rows = [p for p in profiles if (p['group'],p['mode'],p['context']) == key]
        assert len(rows) == 6
        outside = rng.permutation([p['unit'] for p in rows if not p['fov_in']]).tolist()
        inside = rng.permutation([p['unit'] for p in rows if p['fov_in']]).tolist()
        nout = (len(outside)+1)//2 if i%2 == 0 else len(outside)//2
        ev = sorted(outside[:nout]+inside[:3-nout])
        ca = sorted(set(p['unit'] for p in rows)-set(ev))
        evaluation += ev; calibration += ca
        strata.append(dict(group=key[0], mode=key[1], context=key[2], calibration=ca, evaluation=ev))
    assert len(evaluation)==len(calibration)==24 and not set(evaluation)&set(calibration)
    inputs = [OLD/'PLAN.json', PROFILE]
    for unit in units:
        inputs += [OLD/'observations/evaluation'/f'unit{unit}.npz', OLD/'truth/evaluation'/f'unit{unit}.json']
    import cnh_margin_confirm as MC
    import cnh_displacement_ceiling_render as R
    started = time.time()
    plan = dict(run='CNH_DUAL_SENSOR_ALARM_20261005',lane='EXPLORE / consumed synthetic Development',
        created_utc=datetime.now(timezone.utc).isoformat(), started_unix=started, deadline_unix=started+7200,
        units=units, calibration_units=sorted(calibration), evaluation_units=sorted(evaluation), strata=strata,
        frames=FRAMES.tolist(), K=4, intrusion_cm=[1,2,5,-5,-10,-15,-20], modes=[0,1], mode2='NO_SAMPLES',
        threshold=.8557642486787612, operator='>=', splay_deg=[-15,15], pitch_deg=-10,
        sensor_names=['LEFT','RIGHT'], photon_seed_prefix=PHOTON_SEED,
        pose_seed_prefix=read(OLD/'PLAN.json').get('noise_seed_prefix',2026100407),
        pose_seed_rule='Reuse sealed single-center noisy poses, not redraw. One shared head error trajectory transformed through both known extrinsics; improves pairing and supersedes proposed new pose RNG.',
        extrinsics='H = original pitched center sensor @ Rx(+10); left/right = H @ Ry(-/+15) @ Rx(-10). No lateral translation. Pitch stays -10. Same nominal query frame is the original travel frame.',
        inference='Frozen five M3 models; each sensor independently projects observations to the same public travel-frame HEAD/BODY queries. Original <=8 observations / <=1.4s history and [1,2,4,8,16] causal logit smoothing; fuse max after smoothing.',
        alternating='Left samples even physical frames, right odd, .2s system ticks. Rebuild raw-observation voxels over same <=1.4s window (at most4 samples per sensor); smooth latest5 acquired logits from retained frames>=3 only. Hold other sensor newest smoothed score for <=.4s; no duplicated observations, no future input. Extra inference, no extra render.',
        matching='Calibrate only one dual-OR threshold on 24 calibration scenes: greatest attainable joint actual-clear first-stop count <=single; among tied solutions take lowest >= cutoff. Evaluate unchanged on other24. Joint clear = both queries clear throughout13 frames on outside15/20 branches; physical episode counted once.',
        interpretation=dict(supported='Evaluation FOV_OUT shallow delta>=.30 and FOV_IN delta>=-.03 and joint-clear extra stops<=1 -> DUAL_ALARM_SUPPORTED_SIM',
            unsupported='Evaluation FOV_OUT shallow delta<.10 OR FOV_IN delta<-.03 -> DUAL_ALARM_NOT_SUPPORTED_SIM',
            otherwise='MIXED; missing required denominator -> NOT_EVALUABLE',clear_tolerance_extra_stops=1,
            fov='Freeze original single FOV labels, never reclassify dual',bootstrap='2000 scene paired resamples; K4 and displacements not independent scenes',
            secondary='2.5Hz frozen-threshold arm reported independently; no primary gate',scope='Synthetic alarm evidence only; no real inter-sensor crosstalk/SNR/power or human/safety claims'),
        priority='Full +/-15 main arm and alternating inference first; Step2 and extra splay render optional only within original deadline. No downloads, hardware or training.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        model_sha256={str(p):sha(p) for p in MC.model_paths('M3')},renderer_source_sha256=R.source_sha256(),
        source_sha256={Path(__file__).name:sha(__file__)})
    save(OUT/'PLAN.json',plan)
    (OUT/'PLAN.sha256').write_text(sha(OUT/'PLAN.json')+'\n',encoding='utf8')
    save(OUT/'request.json',dict(pid=os.getpid(),started_unix=started,budget_seconds=7200,plan_sha256=sha(OUT/'PLAN.json')))
    print('FROZEN',sha(OUT/'PLAN.json'),'calibration',len(calibration),'evaluation',len(evaluation),flush=True)


def extrinsic(angle):
    import cnh_displacement_ceiling_render as R
    e = np.eye(4); e[:3,:3] = R.S.rx(10)@R.S.ry(angle)@R.S.rx(-10)
    return e


def render_one(unit):
    deadline(); started = time.monotonic()
    import cnh_displacement_ceiling_render as R
    import cnh_location_reference_gpu as G
    from cnh_coverage_policy_render import target_ray_counts
    op = OLD/'observations/evaluation'/f'unit{unit}.npz'
    tp = OLD/'truth/evaluation'/f'unit{unit}.json'
    plan = read(OUT/'PLAN.json')
    for p in (op,tp):
        assert sha(p)==plan['input_sha256'][str(p.relative_to(ROOT))]
    truth = read(tp)
    with np.load(op,allow_pickle=False) as d:
        sensor, travel, base_noisy = d['sensor'].copy(),d['travel'].copy(),d['noisy'].copy()
        old_ambient = d['ambient'].copy()
    ex = np.stack([extrinsic(a) for a in (-15,15)])
    physical = sensor[None]@ex[:,None]
    noisy = base_noisy[None]@ex[:,None,None]
    query = np.linalg.inv(travel)[None]@physical
    assert physical.shape==(2,16,4,4) and noisy.shape==(2,4,16,4,4)
    # Both noisy sensors undo to exactly the same original head estimate.
    np.testing.assert_allclose(noisy[0]@np.linalg.inv(ex[0]),noisy[1]@np.linalg.inv(ex[1]),atol=1e-12,rtol=0)
    targets = [b[0] for b in truth['boxes']]; background = truth['boxes'][0][1:]
    assert all(b[1:]==background for b in truth['boxes'])
    engine = G.ExpectedRenderer(physical.reshape(-1,4,4),background)
    try:
        endpoints = engine.render(targets,candidate_batch=4,pose_batch=32,deadline_check=deadline)
        rhos = np.asarray([b['rho'] for b in targets])
        means = endpoints[:,0]+rhos[:,None,None,None,None]/G.ENDPOINT_RHO*(endpoints[:,1]-endpoints[:,0])
        means = means.reshape(7,2,16,8,8,16).transpose(1,0,2,3,4,5)
        visible = target_ray_counts(engine,targets).reshape(7,2,16).transpose(1,0,2)
        ambient = engine.ambient.reshape(2,16,8,8)
        np.testing.assert_array_equal(ambient,np.repeat(old_ambient[None],2,0))
        meta = engine.metadata
    finally:
        engine.close()
    hist = np.empty((2,7,4,16,8,8,16),np.int32)
    for s in range(2):
        for v in range(7):
            for k in range(4):
                seed=int(np.random.SeedSequence([PHOTON_SEED,unit,s,v,k]).generate_state(1)[0])
                hist[s,v,k]=R.sample(means[s,v],ambient[s],seed)[0]
    path=OUT/'observations'/f'unit{unit}.npz'; path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,hist=hist,ambient=ambient,noisy=noisy,query=query,unit=unit,sensors=['LEFT','RIGHT'])
    private=OUT/'templates'/f'unit{unit}.npz'; private.parent.mkdir(exist_ok=True)
    np.savez_compressed(private,expected=means,target_ray_counts=visible,physical=physical)
    result=dict(status='COMPLETE',unit=unit,seconds=time.monotonic()-started,plan_sha256=sha(OUT/'PLAN.json'),
                observation_sha256=sha(path),template_sha256=sha(private),renderer=meta,
                shared_head_pose=True,source_sha256=sha(__file__))
    save(OUT/'render_receipts'/f'unit{unit}.json',result)
    return result


def render():
    setup(); started=time.monotonic(); receipts=[]
    for i,u in enumerate(read(OUT/'PLAN.json')['units']):
        rp=OUT/'render_receipts'/f'unit{u}.json'
        if rp.exists():
            r=read(rp); assert r['plan_sha256']==sha(OUT/'PLAN.json')
            assert sha(OUT/'observations'/f'unit{u}.npz')==r['observation_sha256']
        else:
            r=render_one(u)
        receipts.append(r)
        save(OUT/'progress.json',dict(stage='render',completed=i+1,total=48,unit=u,elapsed_s=time.monotonic()-started))
        print('render',i+1,'/48',u,round(time.monotonic()-started,2),flush=True)
    save(OUT/'render_result.json',dict(status='COMPLETE',elapsed_s=time.monotonic()-started,units=[r['unit'] for r in receipts]))


def infer():
    setup(); deadline()
    import torch
    import cnh_displacement_ceiling as D
    import cnh_margin_confirm as MC
    import cnh_cvr_pilot as CP
    from cnh_temporal_readout_data import normalized_z
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import query_masks
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    started=time.monotonic(); nets=[]; projector=None
    def network(x):
        x=x.float().clone(); x[:,0]=x[:,0].sign()*x[:,0].abs().log1p()
        x[:,2]=x[:,2].sign()*x[:,2].abs().log1p(); x[:,1]/=8
        x=torch.cat((x,masks[None].expand(len(x),-1,-1,-1,-1)),1)
        return torch.stack([n(x) for n in nets]).mean(0).cpu().numpy()
    def scores(z,query,noisy,alternating=False,sensor=0):
        raw=np.full((7,4,13,2),np.nan,np.float32)
        for k in range(4):
            vox=[]; frames=[]
            for f in FRAMES:
                if alternating and f%2!=sensor: continue
                deadline()
                idx=np.arange(max(0,f-7),f+1)
                if alternating: idx=idx[idx%2==sensor]
                matrices=query[f]@np.linalg.inv(noisy[k,f])@noisy[k,idx]
                vox.append(D.project_many(projector,z[:,k,idx],matrices).half()); frames.append(f)
            x=torch.stack(vox,1).reshape(7*len(frames),3,24,17,33)
            logits=[]
            with torch.inference_mode():
                for begin in range(0,len(x),64): logits.append(network(x[begin:begin+64]))
            logits=np.concatenate(logits).reshape(7,len(frames),2)
            raw[:,k,np.asarray(frames)-3]=logits
        return raw
    try:
        for path in MC.model_paths('M3'):
            assert sha(path)==read(OUT/'PLAN.json')['model_sha256'][str(path)]
            net=CP.CVR().cuda().eval(); net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)); nets.append(net)
        projector=BatchedProjector(); masks=torch.as_tensor(query_masks(),device='cuda')
        # Identity engineering control: stored observations and no extra splay
        # reproduce original raw M3 before new sensor inference is accepted.
        import cnh_temporal_readout_evaluate as T
        rows,_=T.rows_for(OLD,'fresh_evaluation'); original,_=T.prediction(OLD,'fresh_evaluation','M3')
        u=read(OUT/'PLAN.json')['units'][0]
        with np.load(OLD/'observations/evaluation'/f'unit{u}.npz') as d:
            z=normalized_z(d['hist'],d['ambient'][None,None]); q=np.linalg.inv(d['travel'])@d['sensor']; np_=d['noisy'].copy()
        control=scores(z,q,np_)
        ref=original[rows['unit']==u].reshape(7,4,13,2)
        error=float(np.abs(control-ref).max())
        assert error<1e-5, f'Original single raw inference parity failed: {error}'
        save(OUT/'engineering/inference_identity.json',dict(status='PASS',unit=u,raw_logit_max_abs_error=error))
        del z,control
        for i,u in enumerate(read(OUT/'PLAN.json')['units']):
            deadline(); path=OUT/'scores'/f'unit{u}.npz'; rp=path.with_suffix('.json')
            if rp.exists():
                r=read(rp); assert r['plan_sha256']==sha(OUT/'PLAN.json') and r['score_sha256']==sha(path)
            else:
                op=OUT/'observations'/f'unit{u}.npz'; rr=read(OUT/'render_receipts'/f'unit{u}.json')
                assert sha(op)==rr['observation_sha256']
                with np.load(op,allow_pickle=False) as d:
                    z=normalized_z(d['hist'],d['ambient'][:,None,None]); q=d['query'].copy(); np_=d['noisy'].copy()
                full=np.stack([scores(z[s],q[s],np_[s]) for s in range(2)])
                alt=np.stack([scores(z[s],q[s],np_[s],True,s) for s in range(2)])
                assert np.isfinite(full).all()
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path,raw_full=full,raw_alternating=alt,frames=FRAMES,unit=u,sensors=['LEFT','RIGHT'])
                save(rp,dict(status='COMPLETE',plan_sha256=sha(OUT/'PLAN.json'),observation_sha256=sha(op),score_sha256=sha(path)))
                del z,full,alt
            save(OUT/'progress.json',dict(stage='inference',completed=i+1,total=48,unit=u,elapsed_s=time.monotonic()-started))
            print('inference',i+1,'/48',u,round(time.monotonic()-started,2),flush=True)
        save(OUT/'inference_result.json',dict(status='COMPLETE',elapsed_s=time.monotonic()-started,
            runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),tf32=False),
            source_sha256=sha(__file__),models=read(OUT/'PLAN.json')['model_sha256']))
    finally:
        nets.clear(); net=projector=masks=None; gc.collect(); torch.cuda.empty_cache()
        save(OUT/'inference_release.json',dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()))


def engineering():
    setup(); deadline()
    import cnh_displacement_ceiling_render as R
    import cnh_location_reference_gpu as G
    u=read(OUT/'PLAN.json')['units'][0]; truth=read(OLD/'truth/evaluation'/f'unit{u}.json')
    with np.load(OLD/'observations/evaluation'/f'unit{u}.npz') as z:
        poses=z['sensor'][[0,13]]@extrinsic(-15)
    boxes=truth['boxes'][0]; cpu=R.expected(dict(poses=poses,boxes=boxes))
    engine=G.ExpectedRenderer(poses,boxes[1:])
    try:
        endpoints=engine.render([boxes[0]],deadline_check=deadline)
        gpu=endpoints[0,0]+boxes[0]['rho']/G.ENDPOINT_RHO*(endpoints[0,1]-endpoints[0,0])
        error=float(np.abs(cpu['expectation']-gpu).max())
        assert error<1e-9
        np.testing.assert_array_equal(cpu['ambient'],engine.ambient)
        metadata=engine.metadata
    finally: engine.close()
    save(OUT/'engineering/render_identity.json',dict(status='PASS',unit=u,expectation_max_abs_error=error,renderer=metadata))
    print('render engineering PASS',error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['freeze','engineering','render','infer']); args=p.parse_args()
    try:
        {'freeze':freeze,'engineering':engineering,'render':render,'infer':infer}[args.stage]()
    except Exception as exc:
        save(OUT/'failures'/f'{args.stage}-{time.time_ns()}.json',dict(stage=args.stage,error=repr(exc),utc=datetime.now(timezone.utc).isoformat()))
        raise
