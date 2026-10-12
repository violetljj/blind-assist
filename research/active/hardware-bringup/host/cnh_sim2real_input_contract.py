"""Existing real CNH descriptive input-contract comparison; no device access.

No protected data, threshold selection, model modification or physical truth.
Prepare records estimation and interpretation rules before model inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
NEAR = ROOT/'research/active/dtr-r0/nearfield'
sys.path.insert(0,str(NEAR))
from cnh_h3_recording_diagnostics import load_recording, distribution

OUT = ROOT/'artifacts.local/work/sim2real-input-contract-dev-20261011'
CAP = ROOT/'artifacts.local/hardware-bringup/captures'
SIMPLE = CAP/'simple-20260927T164234Z-cc4d0e'
SPECS = [
    (f'h3-{name[:2]}',SIMPLE/name/'tof/raw.bin',desc,'nominal stationary -10deg; unmeasured','cabinet approx 63-64cm; measurement origin unknown')
    for name,desc in [('01-background-attempt1','cabinet/background'),('02-object-attempt2','object introduced before cabinet'),
                      ('03-movement-attempt3','object movement; no per-frame positions'),('04-restored-attempt4','restored cabinet/background')]
] + [(f'h3-{tag}',CAP/f'cnh-h3-{tag}-20260927'/'raw.bin',desc,'unknown','unknown')
     for tag,desc in [('1mhz','I2C 1MHz static engineering recording'),('static','I2C 400kHz static engineering recording'),
                      ('initdiag','initialization diagnostic short recording')]] + [
    ('live-demo02',ROOT/'artifacts.local/work/m3-live-demo-20261010/live-session-02/serial.bin','stationary bench; actions not timestamp-labelled','nominal -10deg; unmeasured','unknown'),
    ('live-opt01',ROOT/'artifacts.local/work/m3-live-opt-20261010/live-session-01/serial.bin','stationary bench; actions not timestamp-labelled','nominal -10deg; unmeasured','unknown')] + [
    (sid,CAP/'live-small-1790532816778'/folder/'tof/raw.bin',scene,'unknown; nominal stationary replay','cabinet approx 64cm; weak recollection')
    for sid,folder,scene in [('small-01','01-background','cabinet background'),('small-02','02-thin-center35','prompted pen but prior RGB inspection showed background; not positive truth'),('small-04','04-restored-background','restored background; no introduced object per prior RGB inspection')]
] + [('preview',ROOT/'artifacts.local/hardware-bringup/previews/preview-b46be2db/data/tof/raw.bin','PREVIEW_ONLY_NOT_GUIDED_COLLECTION: pen/hand/cylinder mixed; no frame truth','unknown','unknown')]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Preserve the prior plan')
    save('PLAN.json',dict(task='SIM2REAL_INPUT_CONTRACT_DEV_20261011',lane='EXPLORE',
        budgets=dict(cpu_command_wall_seconds=1800,gpu_command_wall_seconds=300),
        authority='Existing recordings only; no acquisition/flashing/model/threshold change; no protected480/test',
        decision='Does preserving existing surface evidence change frozen replay response, and which electronics mismatch should be modelled next?',
        baseline='Frozen M3/original5/S955956957 ensemble; same stationary nominal -10deg transform and FP32 engineering features on both paths; thresholds remain simulation references',
        estimators=dict(bias='Only valid distance>max(1.5m,(bin+2.5)*0.3002784m), >=8 eligible frames/cell, median; unsupported cells zero with NOT_EVALUABLE mask. No scene-mean subtraction in new adapter.',
            noise='Adjacent-frame difference squared/2, contiguous sequence, stable distance and ambient, exclude firmware peak +/-2 bins; nonnegative regression intercept+ambient+positive_bias. Descriptive coefficients, correlation and residuals; not a Poisson proof.',
            pulse='Valid stable single-return plane-like zones; firmware distance is weak centre; aggregate16 bins cannot identify native128 optical kernel; estimate effective width only.',
            gain='Nominal electronics weak-distance surface response, assumed rho; effective amplitude ratio confounded with reflectance/occupancy; physical photon calibration NOT_EVALUABLE',
            fixed='Per zone/bin robust median waveform and reference-shift/std, with split-block uncertainty; surface/bias ambiguity retained'),
        weak_support=dict(valid='distance_mm>0 AND status==5 AND nb_target>0',
            geometry='4x4 midpoint unit rays per zone, nominal -10deg X rotation, radial firmware distance. A zone supports a query if >=25% rays inside its box.',
            boxes=dict(HEAD=[[-.3,-.2,.3],[.3,.42,3.]],BODY=[[-.3,.42,.3],[.3,.9,3.]]),
            classes='Among triggered full-history frames: fixed-pattern-sensitive first if diagnostic removal of preidentified row4col2 bin1/2 cancels trigger; otherwise observed surface support; otherwise no observed surface support. Invalid firmware ranges remain UNKNOWN, never known absence.',
            fixed_counterfactual='Zero only two preidentified feature cells for attribution; never an adapter or proposed fix. Sensitivity does not identify optical bias vs real surface.',
            denominator='All successfully decoded frames; full8 history and original reference epochs/gaps for paired comparisons. Report warmup/UNKNOWN and per-segment counts; no true accuracy rates'),
        selection='Exploratory pooled real recordings reused for estimation and description; parameters never chosen by trigger rate; no heldout transfer claim',
        stop='Stop before CPU1800 or GPU300 cumulative command wall; preserve partial outputs and failures',
        next='Draft simulator augmentation and future labelled capture only; no retraining'))
    inventory=[]
    for sid,path,scene,pose,geometry in SPECS:
        d=load_recording(path)
        np.savez_compressed(OUT/f'{sid}.npz',**{k:d[k] for k in ('H','A','D','S','N','seq','ms')})
        good=(d['D']>0)&(d['S']==5)&(d['N']>0)
        first=next(json.loads(line) for line in path.read_bytes().splitlines() if b'"cnh_frame"' in line and line.strip().endswith(b'}'))
        inventory.append(dict(id=sid,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),scene=scene,pose=pose,
            weak_geometry=geometry,frames=len(d['H']),quality=d['quality'],fields=list(first),
            independent_signal_field=False,ambient_units='signed raw/scaler decoded units, no photon count calibration',
            distance_valid_zone_frames=int(good.sum()),zone_frame_denominator=int(good.size),
            valid_distance_mm=distribution(d['D'][good]),ambient=distribution(d['A']),
            duration_sensor_ms=int(d['ms'][-1]-d['ms'][0]),
            structural_zero_fraction_by_bin=(d['H']==0).mean((0,1)).tolist()))
    noframes=CAP/'cnh-h3-framerate-20260927/raw.bin'
    inventory.append(dict(id='h3-framerate',path=noframes.relative_to(ROOT).as_posix(),sha256=sha(noframes),frames=0,status='NOT_EVALUABLE: error records only'))
    noframes=CAP/'live-small-1790532816778/03-positioning-observation/tof/raw.bin'
    inventory.append(dict(id='small-03',path=noframes.relative_to(ROOT).as_posix(),sha256=sha(noframes),frames=0,status='NOT_EVALUABLE: concurrent serial connection rejected'))
    save('inventory.json',inventory)
    save('prepare_receipt.json',dict(seconds=time.perf_counter()-START,source_sha256=sha(__file__),plan_sha256=sha(OUT/'PLAN.json'),raw_inputs={r['path']:r['sha256'] for r in inventory}))
    print('PREPARED',[(r['id'],r['frames']) for r in inventory],flush=True)

def recordings():
    return {sid:{k:v.copy() for k,v in np.load(OUT/f'{sid}.npz').items()} for sid,*_ in SPECS}

def adapt(hist,ambient,bias,noise):
    """Subtract instrument template only; preserve signed measured surfaces.

    Coefficients are in decoded recording units. This is a fitted training-floor
    analogue, not a uniquely identified pseudo-count calibration.
    """
    variance=noise[0]+noise[1]*np.maximum(ambient,0)[...,None]+noise[2]*np.maximum(bias,0)
    return (hist-bias)/np.sqrt(np.maximum(variance,1e-12))

def support(d):
    from cnh_route_sensor import angular_rays
    from cnh_cvr_pilot import rotation
    rays,_=angular_rays(4)
    p=d['D'][:,:,None,None]/1000*(rays.reshape(64,16,3)@rotation(-10,'x').T)[None]
    good=(d['D']>0)&(d['S']==5)&(d['N']>0)
    result=[]
    for lo,hi in [([-.3,-.2,.3],[.3,.42,3.]),([-.3,.42,.3],[.3,.9,3.])]:
        within=((p>=lo)&(p<=hi)).all(-1).mean(-1)>=.25
        result.append((within&good).any(-1))
    return np.stack(result,-1),good.sum(-1)

def old_input(sid,d,all_d):
    """Actual stored live reference epochs; explicit scene self-replay otherwise."""
    n=len(d['H']); epoch=np.zeros(n,int)
    if sid.startswith('live'):
        folder=next(p.parent for name,p,*_ in SPECS if name==sid)
        refs={int(p.stem.split('-')[1]):np.load(p)['histograms'].reshape(-1,64,16) for p in folder.glob('reference-*.npz')}
        versions={json.loads(line)['seq']:json.loads(line).get('reference_version',0) for line in (folder/'inference.jsonl').read_text().splitlines()}
        current=0
        for i,seq in enumerate(d['seq']):
            current=versions.get(int(seq),current);epoch[i]=current
        # Frames before first infer are retained as reference/warmup descriptive input.
        z=np.empty_like(d['H'])
        for key,ref in refs.items():
            std=ref.std(0,ddof=1);pos=std[std>0];floor=max(np.median(pos)*1e-3 if len(pos) else 0,1e-9)
            z[epoch==key]=(d['H'][epoch==key]-ref.mean(0))/np.maximum(std,floor)
        receipt=dict(reference_kind='saved actual reference; seq->inference reference_version',reference_versions=sorted(refs),reference_hashes={p.name:sha(p) for p in folder.glob('reference-*.npz')})
    else:
        ref=all_d['h3-01']['H'] if sid.startswith('h3-0') else all_d['small-01']['H'] if sid.startswith('small') else d['H']
        std=ref.std(0,ddof=1);pos=std[std>0];floor=max(np.median(pos)*1e-3 if len(pos) else 0,1e-9)
        z=(d['H']-ref.mean(0))/np.maximum(std,floor)
        receipt=dict(reference_kind='01 scene reference for simple01-04 and small01/02/04; self-scene for engineering/preview',reference_frames=len(ref))
    return z,epoch,receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','estimate','infer']);a=p.parse_args();START=time.perf_counter()
    if a.stage=='prepare':prepare()
    elif a.stage=='estimate':
        from cnh_sim2real_instrument import analyze
        analyze(recordings(),OUT)
    else:
        from cnh_sim2real_run import run
        run()
