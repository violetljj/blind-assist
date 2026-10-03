"""Known-scene lateral-displacement diagnostic; frozen M3, no training.

Truth/templates have separate files. M3 inference opens observations only.
New straight-motion units belong to the consumed synthetic Development family.
"""
import argparse
import copy
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import cnh_proposal_attribution_scenes as S
import cnh_near_range as NR
import cnh_structure_space as SS

OUT = SS.WORK / 'cnh-displacement-ceiling-20261003'
RUN = 'CNH_DISPLACEMENT_CEILING_20261003'
UNITS = [u for u in range(110001, 110073) if u % 3 in (0, 1)][:48]
SMOKE = 110998
INTRUSION_CM = np.array([1., 2., 5., -5., -10., -15., -20.])
FRAMES = np.arange(3, 16)
K = 4
THRESHOLD = .8557642486787612
BIAS = SS.WORK / 'cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy'


def save(path, value):
    SS.save(path, value)


def plan_sha():
    p = json.loads((OUT / 'PLAN.json').read_text(encoding='utf8'))
    assert p['run'] == RUN and p['units'] == UNITS and p['K'] == K
    assert p['intrusion_cm'] == INTRUSION_CM.tolist() and p['frames'] == FRAMES.tolist()
    return SS.sha(OUT / 'PLAN.json')


def scenes(unit, index):
    """Draw one fixed world; all seven variants change only target x."""
    from cnh_expected_separability import target_at
    rng = np.random.default_rng([2026100305, unit])
    ref = S.make_scenes(unit)[0]
    assert ref['mode'] in (0, 1)
    group, condition = (index // 2) % 2, 'none' if (index // 4) % 2 == 0 else 'panel'
    target = NR._target(rng, -.01, group)
    dz = .65 - target['z']
    target['box']['lo'][2] += dz
    target['box']['hi'][2] += dz
    target['z'] = .65
    if condition == 'none':
        base = NR._none_scene(S, unit, 0, ref, target)
    else:
        # Predesigned gap avoids target/panel interpenetration at all seven x.
        shape = dict(SS.draw_shape(rng), same_side=1, gap=float(rng.uniform(.34, .40)))
        base = SS.panel_scene(unit, 0, ref, target, shape, float(rng.uniform(SS.EDGES[0], SS.EDGES[5])))
    result = []
    for penetration in INTRUSION_CM:
        scene = copy.deepcopy(base)
        scene['boxes'][0] = target_at(target, -penetration / 100)
        result.append(scene)
    for sc in result:
        assert sc['boxes'][1:] == base['boxes'][1:]
        assert np.array_equal(sc['poses'], base['poses']) and np.array_equal(sc['travel'], base['travel'])
        for b in sc['boxes'][1:]:
            overlap = np.minimum(sc['boxes'][0]['hi'], b['hi']) - np.maximum(sc['boxes'][0]['lo'], b['lo'])
            assert not (overlap > 0).all(), 'Target/context interpenetration'
    return result, group, condition


def plan():
    import cnh_margin_confirm as MC
    import cnh_displacement_ceiling_render as R
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'PLAN.json').exists():
        raise FileExistsError('Inspect/resume existing PLAN')
    # Exact candidate filenames and numeric literal reservations, not substring IDs.
    reservations = []
    import re
    for root in SS.WORK.glob('cnh*'):
        if root == OUT or not root.is_dir():
            continue
        for p in root.rglob('unit110*.npz'):
            m = re.fullmatch(r'unit(\d+)', p.stem)
            if m and int(m[1]) in (*UNITS, SMOKE):
                reservations.append(str(p))
        for name in ('PLAN.json', 'plan.json', 'request.json'):
            for p in root.glob(name):
                text = p.read_text(encoding='utf8')
                if any(re.search(r'(?<!\d)' + str(u) + r'(?!\d)', text) for u in (*UNITS, SMOKE)):
                    reservations.append(str(p))
    assert not reservations, reservations
    params, _ = S.nominal_parameters()
    assert params.noise_scale == 1 and params.output_gain == 1
    from dataclasses import asdict
    record = dict(run=RUN, units=UNITS, excluded_smoke_unit=SMOKE, K=K,
        intrusion_cm=INTRUSION_CM.tolist(), frames=FRAMES.tolist(),
        bins=[[.6,.9],[.9,1.2],[1.2,1.6],[1.6,2.1],[2.1,2.6]],
        threshold=THRESHOLD, template_window=8, M3_smoothing_weights=[1,2,4,8,16],
        nominal_sensor_params=asdict(params), role='EXPLORE consumed synthetic Development measurement only; no training',
        scene_design='48 balanced HEAD/BODY x none/panel x straight mode0/1, six per cell; final target front .65m; only target lateral faces moved',
        panel_change='Fixed same-side gap .34-.40m avoids interpenetration for outside20cm target; wider than prior near-panel cohort',
        replicas='K4 independent photon seeds per delta; independent noisy ego estimates across K shared across delta; true sensor/travel fixed',
        oracle='Exact signed-CNH Skellam class-mixture likelihood ratio; clean expected templates all 2 positive and 3 clear candidate deltas with equal class priors; known scene/true pose privileged',
        likelihood='coarse signed=P(E+8ambient)-P(8ambient); no latent counts given to oracle; 8 causal exposures',
        primary='Per-scene AUC nominal inside1/2cm vs outside10/15/20cm, frames with front1.2-2.1m, then equal-scene macro mean; M3 five-logit causal smoothing',
        descriptive_branches=dict(observation_limited='oracle macro AUC<=.75: only this known-scene/noise experiment limited; recommend pause tested readout/loss recipes, no proof all methods impossible',
            unused_information='oracle macro AUC>=.9 and M3 macro AUC<=.75: information/readout gap; propose different training units, measurement batch stays evaluation',
            otherwise='Report distribution, no automatic new training'),
        delta_crossing='For each inside1/2/5cm candidate vs outside10/15/20cm, farthest sampled distance bin with scene-specific AUC>=.9; report not-reached and visibility, .8m/s relative to .5m and timely .9m',
        slope_intercept='Per-scene/bin OLS smoothed raw logit vs signed intrusion; threshold crossing and fit residuals; counterfactual pooled slope/intercept descriptive only, not causal resolution/bias attribution',
        sensitivity='First8 balanced design scenes additional32-ray expectation; fixed-template Gaussian-distance sensitivity separate from exact main likelihood, never select primary results',
        budget_seconds=3600, audit=dict(candidate_ids_found=0, range='110001..110071 mode0/1, smoke110998', scope='RUNS/source literal search plus prior CNH top-level plans/requests and exact unit filenames'),
        models_sha256={str(p): SS.sha(p) for p in MC.model_paths('M3')}, bias_sha256=SS.sha(BIAS),
        source_sha256={p.name:SS.sha(p) for p in (Path(__file__), Path(R.__file__), Path(NR.__file__), Path(SS.__file__), Path(S.__file__), S.SOURCE/'cnh_route_sensor.py')})
    save(OUT / 'PLAN.json', record)
    print('PLAN', len(UNITS), 'scenes', 'SHA', plan_sha(), flush=True)


def unit_receipt(path, unit, paths, extra=None):
    save(path, dict(status='COMPLETE', unit=unit, plan_sha256=plan_sha(),
        output_sha256={str(p.relative_to(OUT)).replace('\\','/'):SS.sha(p) for p in paths}, **(extra or {})))


def generate_one(unit, index, folder=OUT, smoke=False):
    import cnh_displacement_ceiling_render as R
    from cnh_track_a_readout import noisy_poses
    from cnh_corridor_labels import labels_for_all
    import cnh_sequence_observed_geometry as G
    for name in ('observations','templates','truth'):
        (folder/name).mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    variants, group, condition = scenes(unit, index)
    ex = [R.expected(sc) for sc in variants]
    expectation = np.stack([item['expectation'] for item in ex])
    ambient = ex[0]['ambient']
    assert all(np.array_equal(ambient, item['ambient']) for item in ex)
    obs = np.empty((7,K,*expectation.shape[1:]),np.int32)
    counts, estimates = np.empty_like(obs), np.empty_like(obs)
    for d in range(7):
        for k in range(K):
            seed = int(np.random.SeedSequence([2026100305,unit,d,k]).generate_state(1)[0])
            obs[d,k],counts[d,k],estimates[d,k] = R.sample(expectation[d],ambient,seed)
    sensor, travel = variants[0]['poses'], variants[0]['travel']
    noisy = np.stack([noisy_poses(sensor, int(np.random.SeedSequence([2026100306,unit,k]).generate_state(1)[0]), dt=.2) for k in range(K)])
    truth = dict(unit=unit, group=group, mode=unit%3, context=condition,
        front_range_m=(.65-travel[:,2,3]).tolist(), intrusion_cm=INTRUSION_CM.tolist(),
        boxes=[sc['boxes'] for sc in variants], labels=[], categories=[],
        template_ray_subdivision=16, panel_meta=variants[0].get('meta', {}))
    for sc in variants:
        truth['labels'].append(labels_for_all(sc['boxes'],travel).tolist())
        triangles = np.concatenate([S.box_mesh(b['lo'],b['hi']) for b in sc['boxes']])
        truth['categories'].append([[G.surface_category((triangles-p[:3,3])@p[:3,:3],q) for q in (0,1)] for p in travel])
    obs_path = folder/'observations'/f'unit{unit}.npz'
    np.savez_compressed(obs_path,hist=obs,counts=counts,ambient_estimate=estimates,
        ambient=ambient,sensor=sensor,travel=travel,noisy=noisy)
    template_path = folder/'templates'/f'unit{unit}.npz'
    templates = dict(expected=expectation,object_id=np.stack([item['object_id'] for item in ex]).astype(np.int8))
    if index < 8:
        templates['expected32'] = np.stack([R.expected(sc,sub=32)['expectation'] for sc in variants])
    np.savez_compressed(template_path,**templates)
    truth_path = folder/'truth'/f'unit{unit}.json'
    save(truth_path, truth)
    elapsed = time.monotonic()-start
    if not smoke:
        unit_receipt(obs_path.with_suffix('.json'),unit,[obs_path],dict(elapsed_s=elapsed,noise_scale=1,output_gain=1))
        unit_receipt(template_path.with_suffix('.json'),unit,[template_path])
        unit_receipt(truth_path.with_name(f'unit{unit}.receipt.json'),unit,[truth_path])
    return elapsed


def generate():
    digest = plan_sha()
    started=time.monotonic()
    save(OUT/'request.json',dict(run=RUN,plan_sha256=digest,started_unix=time.time(),pid=os.getpid(),budget_seconds=3600))
    for i,u in enumerate(UNITS):
        path=OUT/'observations'/f'unit{u}.json'
        if path.exists():
            r=json.loads(path.read_text());assert r['plan_sha256']==digest
            assert all(SS.sha(OUT/p)==h for p,h in r['output_sha256'].items())
            continue
        if time.monotonic()-started>3500:
            raise TimeoutError('Rendering budget reached; retain partial cohort')
        elapsed=generate_one(u,i)
        print('generated',i+1,'/48',u,'scene seconds',round(elapsed,2),'total',round(time.monotonic()-started,1),flush=True)
    save(OUT/'generation_receipt.json',dict(status='COMPLETE',units=UNITS,plan_sha256=digest,
        elapsed_s=time.monotonic()-started,source_sha256=SS.sha(__file__)))


def project_many(projector, z, transforms):
    """Reuse identical quadrature geometry for seven independent photon arrays."""
    import torch
    from cnh_cvr_projection import EDGE, WIDTH, SUB, SHAPE
    t=torch.as_tensor(transforms,dtype=torch.float64,device='cuda')
    p=torch.bmm(projector.points[None]-t[:,None,:3,3],t[:,:3,:3])
    radius=torch.linalg.vector_norm(p,dim=2)
    xy=p[:,:,:2]/p[:,:,2:3].clamp_min(1e-30)
    ij=torch.floor((xy+EDGE)/(2*EDGE)*8).long();bins=torch.floor(radius/WIDTH).long()
    valid=(p[:,:,2]>0)&(ij>=0).all(2)&(ij<8).all(2)&(bins>=0)&(bins<16)
    idx=(ij[:,:,1].clamp(0,7)*8+ij[:,:,0].clamp(0,7))*16+bins.clamp(0,15)
    weight=valid*projector.voxel_volume/(SUB**3)/projector.volumes[idx]
    values=torch.as_tensor(z,dtype=torch.float64,device='cuda').reshape(len(z),len(t),-1)
    mass=torch.gather(values,2,idx[None].expand(len(z),-1,-1))*weight
    evidence=mass.reshape(len(z),len(t),-1,SUB**3).sum(3).reshape(len(z),len(t),*SHAPE).float()
    coverage=valid.reshape(len(t),-1,SUB**3).double().mean(2).reshape(len(t),*SHAPE).float()
    total=torch.zeros_like(evidence[:,0]);count=torch.zeros_like(coverage[0])
    for frame in range(len(t)):
        total+=evidence[:,frame];count+=coverage[frame]
    return torch.stack([total,count[None].expand(len(z),*SHAPE),evidence[:,-1]],1)


def infer(units=None, folder=OUT, check=False):
    import torch
    import cnh_margin_confirm as MC
    import cnh_cvr_pilot as CP
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import query_masks
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    nets=[];started=time.monotonic();projection_s=0.;network_s=0.
    paths=MC.model_paths('M3')
    digest=plan_sha()
    (folder/'scores').mkdir(parents=True,exist_ok=True)
    try:
        for path in paths:
            net=CP.CVR().cuda().eval();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));nets.append(net)
        projector=BatchedProjector();masks=torch.as_tensor(query_masks(),device='cuda')
        bias=torch.as_tensor(np.load(BIAS),dtype=torch.float32,device='cuda')
        for u in (UNITS if units is None else units):
            obs_path=folder/'observations'/f'unit{u}.npz'
            with np.load(obs_path) as d:
                hist=torch.as_tensor(d['hist'],dtype=torch.float32,device='cuda')
                ambient=torch.as_tensor(d['ambient'],dtype=torch.float32,device='cuda')
                z=((hist-bias)/(16*ambient[...,None]+bias.clamp_min(0)).clamp_min(1e-9).sqrt()).half().float()
                sensor,travel,noisy=d['sensor'],d['travel'],d['noisy']
            raw=np.empty((7,K,13,2),np.float32)
            for k in range(K):
                vox=[];torch.cuda.synchronize();tick=time.monotonic()
                for f in FRAMES:
                    transforms=CP.relative_transforms(sensor,travel,noisy[k],int(f))
                    v=project_many(projector,z[:,k,max(0,f-7):f+1],transforms)
                    if check and k==0 and f in (3,15):
                        reference=projector.sequence(z[0,k,max(0,f-7):f+1],transforms)
                        assert torch.equal(reference,v[0]),float((reference-v[0]).abs().max())
                    vox.append(v.half())
                torch.cuda.synchronize();projection_s+=time.monotonic()-tick
                x=torch.stack(vox,1).reshape(7*13,3,24,17,33)
                torch.cuda.synchronize();tick=time.monotonic()
                with torch.inference_mode():
                    logits=[]
                    for j in range(0,len(x),64):
                        xb=x[j:j+64].float().clone()
                        xb[:,0]=xb[:,0].sign()*xb[:,0].abs().log1p();xb[:,2]=xb[:,2].sign()*xb[:,2].abs().log1p();xb[:,1]/=8
                        xb=torch.cat((xb,masks[None].expand(len(xb),-1,-1,-1,-1)),1)
                        logits.append(torch.stack([n(xb) for n in nets]).mean(0).cpu().numpy())
                    raw[:,k]=np.concatenate(logits).reshape(7,13,2)
                torch.cuda.synchronize();network_s+=time.monotonic()-tick
            assert np.isfinite(raw).all()
            path=folder/'scores'/f'unit{u}.npz';np.savez_compressed(path,raw=raw)
            if not check:
                unit_receipt(path.with_suffix('.json'),u,[path],dict(observation_sha256=SS.sha(obs_path),models_sha256={str(p):SS.sha(p) for p in paths}))
            print('M3',u,'elapsed',round(time.monotonic()-started,1),flush=True)
        if not check:
            save(OUT/'scores_receipt.json',dict(status='COMPLETE',units=UNITS,plan_sha256=digest,
                elapsed_s=time.monotonic()-started,projection_s=projection_s,network_s=network_s,
                source_sha256=SS.sha(__file__),models_sha256={str(p):SS.sha(p) for p in paths},bias_sha256=SS.sha(BIAS),
                outputs={str((folder/'scores'/f'unit{u}.npz').relative_to(OUT)).replace('\\','/'):SS.sha(folder/'scores'/f'unit{u}.npz') for u in UNITS},
                runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),tf32=False)))
    finally:
        nets.clear();net=projector=bias=masks=hist=z=vox=x=xb=v=None
        gc.collect();torch.cuda.empty_cache()


def smoke():
    import cnh_displacement_ceiling_render as R
    started=time.monotonic()
    variants,_,_=scenes(SMOKE,0)
    expected=R.expected(variants[0])
    old=S.render(variants[0],123,noise_scale=0)
    error=float(np.max(abs(expected['expectation']-old['hist'])))
    assert error<=1e-9,error
    folder=OUT/'smoke';elapsed=generate_one(SMOKE,0,folder,smoke=True)
    infer([SMOKE],folder,check=True)
    save(folder/'check.json',dict(status='PASS',excluded_unit=SMOKE,noiseless_renderer_max_error=error,
        generation_s=elapsed,total_s=time.monotonic()-started,projector_exact=True,only_lateral_target_geometry=True))
    print('SMOKE PASS',round(time.monotonic()-started,1),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('plan','smoke','generate','infer'))
    a=p.parse_args();{'plan':plan,'smoke':smoke,'generate':generate,'infer':infer}[a.stage]()
