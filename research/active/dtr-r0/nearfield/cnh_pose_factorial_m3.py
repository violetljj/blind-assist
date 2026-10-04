"""Frozen M3 inference with true or coupled quarter-scale pose estimates.

No model fitting, photon sampling, scene rendering or evaluator-truth access.
Import the displacement module first to install the frozen sensor snapshot.
The true-pose arm is a test-time intervention on noisy-pose-trained models.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import cnh_displacement_ceiling as D
import cnh_track_a_readout as READOUT
from scipy.spatial.transform import Rotation

OUT = D.OUT.parent/'cnh-pose-factorial-20261004'
M3 = OUT/'m3'
RUN = 'CNH_POSE_FACTORIAL_20261004'
POSE_LAMBDA = .25
DT = .2
POSE_SEED_PREFIX = 2026100306


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def pose_seed(unit, replica):
    return int(np.random.SeedSequence([POSE_SEED_PREFIX, int(unit), int(replica)]).generate_state(1)[0])


def scaled_noisy_poses(poses, seed, noise_lambda=POSE_LAMBDA, dt=DT):
    """Shrink noise draws, preserving frozen RNG seed/order and causal integration.

    Return world_from_tof poses plus raw draws. Both step translation noise and
    rotation noise are drawn at the original standard deviations, then scaled.
    Lambda1 is bitwise checked against the unmodified frozen noisy_poses.
    """
    poses = READOUT._poses(poses, len(poses))
    if dt <= 0 or not np.isfinite(dt) or not np.isfinite(noise_lambda) or not 0 <= noise_lambda <= 1:
        raise ValueError('Finite dt>0 and noise lambda in [0,1] required')
    rng = np.random.default_rng(int(seed))
    scale = rng.uniform(-.2, .2)
    bias = rng.choice([-1., 1.], 3)*np.deg2rad(1.)
    out = poses.copy()
    translation_noise, rotation_noise = [], []
    for frame in range(1, len(poses)):
        delta = np.linalg.inv(poses[frame-1])@poses[frame]
        translation_step = rng.normal(0, .02)
        delta[:3, 3] *= 1+scale*noise_lambda+translation_step*noise_lambda
        rotation_step = rng.normal(0, np.deg2rad(.1), 3)
        delta[:3, :3] = delta[:3, :3]@Rotation.from_rotvec(bias*noise_lambda*dt+rotation_step*noise_lambda).as_matrix()
        out[frame] = out[frame-1]@delta
        translation_noise.append(float(translation_step))
        rotation_noise.append(rotation_step.tolist())
    if noise_lambda == 0:
        out = poses.copy()  # True arm uses sensor directly, avoiding integration roundoff.
    return out, dict(seed=int(seed), noise_lambda=float(noise_lambda), dt=float(dt),
        scale=float(scale), gyro_bias_rad_per_s=bias.tolist(),
        translation_step_noise=translation_noise, rotation_step_noise_rad=rotation_noise)


def source_hashes():
    import cnh_cvr_pilot as CP
    import cnh_cvr_v2_materialize as MATERIALIZE
    import cnh_cvr_projection as PROJECTION
    modules = (D, READOUT, CP, MATERIALIZE, PROJECTION)
    return {str(Path(module.__file__).resolve()): D.SS.sha(module.__file__) for module in modules} | {
        str(Path(__file__).resolve()): D.SS.sha(__file__)}


def quarter_poses(unit, sensor, stored_noisy):
    if sensor.shape != (16,4,4) or stored_noisy.shape != (4,16,4,4):
        raise ValueError('Frozen sensor/noisy pose axes differ')
    quarter, components = [], []
    for replica in range(D.K):
        seed = pose_seed(unit, replica)
        reconstructed, _ = scaled_noisy_poses(sensor, seed, 1.)
        reference = READOUT.noisy_poses(sensor, seed, dt=DT)
        if not np.array_equal(reconstructed, reference) or not np.array_equal(reference, stored_noisy[replica]):
            raise ValueError(f'Frozen noise/RNG bitwise reconstruction failed: {unit}/{replica}')
        q, draws = scaled_noisy_poses(sensor, seed)
        quarter.append(q)
        components.append(draws)
    return np.stack(quarter), components


def noise_check():
    """CPU-only observation-metadata replay; no scientific model inference."""
    M3.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    observations = {}
    for unit in D.UNITS:
        path = D.OUT/'observations'/f'unit{unit}.npz'
        with np.load(path, allow_pickle=False) as source:
            sensor, noisy = source['sensor'], source['noisy']
        quarter, _ = quarter_poses(unit, sensor, noisy)
        READOUT._poses(quarter.reshape(64,4,4),64)
        observations[str(path)] = D.SS.sha(path)
    report = dict(status='PASS_BITWISE_NOISE_REPLAY', run=RUN, units=D.UNITS,
        sequences=len(D.UNITS)*D.K, pose_shape=[4,16,4,4], all_lambda1_bitwise_equal=True,
        coupled_lambda=.25, dt=DT, seed_prefix=POSE_SEED_PREFIX,
        formula='sequence scale U(-.2,.2), signed axis gyro bias 1deg/s; step translation N(0,.02), step rotation N(0,.1deg); all four components multiplied by lambda before causal integration',
        sources=source_hashes(), observation_sha256=observations,
        elapsed_s=time.monotonic()-started, scientific_inference=False)
    path = M3/'noise_check.json'
    if path.exists():
        old = read(path)
        if old['sources'] != report['sources'] or old['observation_sha256'] != observations:
            raise ValueError('Prior noise check inputs differ; preserve prior evidence')
    else:
        save_new(path, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','observation_sha256')}, indent=2), flush=True)
    return report


def frozen_plan():
    path = OUT/'PLAN.json'
    plan = read(path)
    if plan['run'] != RUN or plan['units'] != D.UNITS:
        raise ValueError('Frozen factorial plan run/units differ')
    settings = plan['m3']
    required = dict(pose_lambda=POSE_LAMBDA, dt=DT, pose_seed_prefix=POSE_SEED_PREFIX,
                    K=D.K, history=8, frames=D.FRAMES.tolist(), five_seed_mean=True,
                    threshold=D.THRESHOLD)
    if any(settings.get(key) != value for key,value in required.items()):
        raise ValueError('Frozen factorial M3 pose/history/settings differ')
    return plan, D.SS.sha(path)


def _raw_for_poses(torch, CP, nets, projector, masks, z, sensor, travel, estimates):
    raw = np.empty((7,D.K,13,2),np.float32)
    projection_s = network_s = 0.
    for replica in range(D.K):
        voxels=[]
        torch.cuda.synchronize(); tick=time.monotonic()
        for frame in D.FRAMES:
            transforms=CP.relative_transforms(sensor,travel,estimates[replica],int(frame))
            values=D.project_many(projector,z[:,replica,max(0,frame-7):frame+1],transforms)
            voxels.append(values.half())
        torch.cuda.synchronize(); projection_s+=time.monotonic()-tick
        features=torch.stack(voxels,1).reshape(7*13,3,24,17,33)
        torch.cuda.synchronize(); tick=time.monotonic()
        with torch.inference_mode():
            logits=[]
            for begin in range(0,len(features),64):
                batch=features[begin:begin+64].float().clone()
                batch[:,0]=batch[:,0].sign()*batch[:,0].abs().log1p()
                batch[:,2]=batch[:,2].sign()*batch[:,2].abs().log1p()
                batch[:,1]/=8
                batch=torch.cat((batch,masks[None].expand(len(batch),-1,-1,-1,-1)),1)
                logits.append(torch.stack([net(batch) for net in nets]).mean(0).cpu().numpy())
            raw[:,replica]=np.concatenate(logits).reshape(7,13,2)
        torch.cuda.synchronize(); network_s+=time.monotonic()-tick
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite frozen M3 output')
    return raw, dict(projection_s=projection_s,network_s=network_s)


def infer():
    plan, digest = frozen_plan()  # Before CUDA allocation/model loading.
    import torch
    import cnh_margin_confirm as MC
    import cnh_cvr_pilot as CP
    from cnh_cvr_v2_materialize import BatchedProjector
    from cnh_cvr_projection import query_masks
    M3.mkdir(parents=True,exist_ok=True)
    (M3/'noisy_quarter').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA frozen M3 inference required')
    source_sha=source_hashes()
    model_paths=MC.model_paths('M3')
    model_sha={str(p):D.SS.sha(p) for p in model_paths}
    old_plan=read(D.OUT/'PLAN.json')
    if model_sha!=old_plan['models_sha256'] or D.SS.sha(D.BIAS)!=old_plan['bias_sha256']:
        raise ValueError('Frozen M3 model/bias changed')
    started=time.monotonic();nets=[]; outputs={}; observation_sha={}; timings=[]
    try:
        for path in model_paths:
            network=CP.CVR().cuda().eval()
            network.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            nets.append(network)
        projector=BatchedProjector()
        masks=torch.as_tensor(query_masks(),device='cuda')
        bias=torch.as_tensor(np.load(D.BIAS),dtype=torch.float32,device='cuda')
        baseline_parity=None
        for index,unit in enumerate(D.UNITS):
            unit_start=time.monotonic()
            input_path=D.OUT/'observations'/f'unit{unit}.npz'
            observation_sha[str(input_path)]=D.SS.sha(input_path)
            output=M3/f'unit{unit}.npz'; receipt=output.with_suffix('.json')
            if receipt.exists():
                old=read(receipt)
                if (old['plan_sha256']!=digest or old['source_sha256']!=source_sha
                        or old['observation_sha256']!=observation_sha[str(input_path)]):
                    raise ValueError('Existing per-unit M3 evidence differs')
                for file,sha in old['output_sha256'].items():
                    if D.SS.sha(OUT/file)!=sha:
                        raise ValueError('Prior M3 output changed')
                outputs.update(old['output_sha256']);timings.append(old['timing'])
                if index==0: baseline_parity=old['baseline_parity']
                continue
            if output.exists():
                raise FileExistsError('Unreceipted prior output requires inspection')
            with np.load(input_path,allow_pickle=False) as source:
                hist=torch.as_tensor(source['hist'],dtype=torch.float32,device='cuda')
                ambient=torch.as_tensor(source['ambient'],dtype=torch.float32,device='cuda')
                sensor,travel,noisy=source['sensor'],source['travel'],source['noisy']
            quarter,components=quarter_poses(unit,sensor,noisy)
            z=((hist-bias)/(16*ambient[...,None]+bias.clamp_min(0)).clamp_min(1e-9).sqrt()).half().float()
            if index==0:
                reconstructed,_=_raw_for_poses(torch,CP,nets,projector,masks,z,sensor,travel,noisy)
                with np.load(D.OUT/'scores'/f'unit{unit}.npz',allow_pickle=False) as stored:
                    original=stored['raw']
                if not np.array_equal(reconstructed,original):
                    raise ValueError(f'Frozen first-unit M3 reconstruction differs: {np.max(abs(reconstructed-original))}')
                baseline_parity=dict(unit=unit,bitwise=True,max_abs_error=0.,
                    original_scores_sha256=D.SS.sha(D.OUT/'scores'/f'unit{unit}.npz'))
            true_estimates=np.broadcast_to(sensor,(D.K,*sensor.shape))
            raw_true,true_timing=_raw_for_poses(torch,CP,nets,projector,masks,z,sensor,travel,true_estimates)
            raw_quarter,quarter_timing=_raw_for_poses(torch,CP,nets,projector,masks,z,sensor,travel,quarter)
            pose_path=M3/'noisy_quarter'/f'unit{unit}.npz'
            if pose_path.exists():
                raise FileExistsError('Existing quarter poses require inspected resume')
            np.savez_compressed(pose_path,sensor=sensor,travel=travel,noisy_quarter=quarter,beliefroot=sensor[0])
            np.savez_compressed(output,raw_true=raw_true,raw_quarter=raw_quarter,noisy_quarter=quarter)
            hashes={str(p.relative_to(OUT)).replace('\\','/'):D.SS.sha(p) for p in (output,pose_path)}
            timing=dict(true=true_timing,quarter=quarter_timing,elapsed_s=time.monotonic()-unit_start)
            save_new(receipt,dict(status='COMPLETE',run=RUN,unit=unit,plan_sha256=digest,
                source_sha256=source_sha,models_sha256=model_sha,bias_sha256=D.SS.sha(D.BIAS),
                observation_sha256=observation_sha[str(input_path)],output_sha256=hashes,
                timing=timing,noise_draws=components,baseline_parity=baseline_parity if index==0 else None,
                evidence='True poses replace test-time noisy estimates; original training used noisy poses'))
            outputs.update(hashes);timings.append(timing)
            print('M3 true/quarter',index+1,'/48',unit,'elapsed_s',round(time.monotonic()-started,2),flush=True)
        # Verify immutable scientific inputs again after the run.
        for path,sha in {**source_sha,**model_sha,**observation_sha,str(D.BIAS):old_plan['bias_sha256']}.items():
            if D.SS.sha(path)!=sha:
                raise ValueError('M3 source/model/observation changed during inference: '+path)
        result=dict(status='COMPLETE',run=RUN,units=D.UNITS,plan_sha256=digest,
            source_sha256=source_sha,models_sha256=model_sha,bias_sha256=D.SS.sha(D.BIAS),
            observation_sha256=observation_sha,output_sha256=outputs,baseline_parity=baseline_parity,
            elapsed_s=time.monotonic()-started,timings=timings,
            arms=['M3_true','M3_quarter'],raw_shape=[7,4,13,2],pose_lambda=POSE_LAMBDA,
            runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),tf32=False),
            interpretation='Frozen noisy-pose-trained model with test-time true/quarter poses; no training or calibration')
        save_new(M3/'inference_receipt.json',result)
        print('COMPLETE M3 true/quarter elapsed_s',round(result['elapsed_s'],2),flush=True)
        return result
    finally:
        nets.clear()
        network=projector=bias=masks=hist=ambient=z=None
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=('noise-check','infer'))
    args=parser.parse_args()
    {'noise-check':noise_check,'infer':infer}[args.stage]()
