"""Histogram-only wall calibration and continuous expected-CNH forward.

Wall 2 identifies effective instrument parameters; wall 3 may fit its own
plane geometry but cannot change them. Profile sets are descriptive sensitivity
sets, not confidence intervals. No model score or protected dataset is read.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import ndtr
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'research/active/dtr-r0/nearfield'))
import cnh_proposal_attribution_scenes as S
from cnh_route_sensor import angular_rays
import real_pilot_calibration as P

OUT=ROOT/'artifacts.local/work/real-twin-joint-calibration-dev-20261011'
WIDTH=.3002784
NAMES=('offset_m','sigma_m','tail_fraction','tail_scale_m','gain','wall_distance_m','normal_pitch_deg','normal_yaw_deg')
LOW=np.array([-.30,.02,0,.03,.10,.45,-15,-15])
HIGH=np.array([.30,.30,.40,.50,2.,.85,15,15])

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

@lru_cache(maxsize=1)
def frozen_parameters():return S.nominal_parameters()[0]

def predict(distance,rho,cosine,weights,parameters,bias=None,raw_orientation=True):
    """Return expected decoded [...,8,8,16] without ambient mean.

    Kernel is an integrated Gaussian plus delayed exponential mixture. Offset
    shifts observed radial range; the physical inverse-square energy still uses
    unshifted range. Escaped kernel energy is lost. Bias is in returned/raw
    orientation. Input rays are canonical simulator orientation.
    """
    d=np.asarray(distance,float);valid=np.isfinite(d)&(d>0)
    safe=np.where(valid,d,1.)
    w=np.broadcast_to(weights,d.shape).astype(float);w=w/w.sum(-1,keepdims=True)
    energy=frozen_parameters().signal_counts*np.broadcast_to(rho,d.shape)*np.broadcast_to(cosine,d.shape)*w/np.maximum(safe,.05)**2
    energy=np.where(valid,energy,0.)*parameters['gain']
    edges=np.arange(17)*WIDTH
    delta=edges-safe[...,None]-parameters['offset_m']
    gaussian=ndtr(delta/parameters['sigma_m'])
    # The delayed exponential is a separate mixture component (not an
    # exponentially modified Gaussian); tail shape remains a weak proxy.
    exponential=-np.expm1(-np.maximum(delta,0)/parameters['tail_scale_m'])
    cdf=(1-parameters['tail_fraction'])*gaussian+parameters['tail_fraction']*exponential
    out=(energy[...,None]*np.diff(cdf,axis=-1)).sum(-2)
    leak=frozen_parameters().neighbour_leak/4;old=out.copy()
    out[...,1:,:,:]+=leak*(old[...,:-1,:,:]-old[...,1:,:,:])
    out[...,:-1,:,:]+=leak*(old[...,1:,:,:]-old[...,:-1,:,:])
    out[...,:,1:,:]+=leak*(old[...,:,:-1,:]-old[...,:,1:,:])
    out[...,:,:-1,:]+=leak*(old[...,:,1:,:]-old[...,:,:-1,:])
    out*=frozen_parameters().output_gain
    if raw_orientation:out=out[...,::-1,::-1,:]
    if bias is not None:out=out+np.asarray(bias).reshape(8,8,16)
    return out

@lru_cache(maxsize=2)
def plane_rays(samples):return angular_rays(samples)

def plane_inputs(distance,pitch,yaw,samples=4):
    rays,weights=plane_rays(samples)
    p,y=np.deg2rad([pitch,yaw]);normal=np.array([np.sin(y)*np.cos(p),np.sin(p),np.cos(y)*np.cos(p)])
    cosine=np.einsum('...j,j->...',rays,normal)
    radial=np.divide(distance*normal[2],cosine,out=np.full_like(cosine,np.inf),where=cosine>0)
    return radial,.5,cosine,weights

def plane_prediction(x,samples=4):
    return predict(*plane_inputs(x[5],x[6],x[7],samples),dict(zip(NAMES,x))).reshape(64,16)

def stable_wall(sid):
    # Load only the requested already-decoded recording, never other scenes.
    with np.load(P.OUT/(sid+'.npz')) as d:
        frames=d['H'][d['stable']].astype(float);ambient=d['A'][d['stable']].astype(float)
    mean=frames.mean(0);var=frames.var(0,ddof=1)
    # Mean precision plus a modest 5% response discrepancy term prevents tiny
    # quantized variance cells from dominating; no zone/bin is excluded.
    scale=np.sqrt(var/len(frames)+(.05*np.maximum(abs(mean),1))**2)
    return dict(frames=frames,ambient=ambient,mean=mean,var=var,scale=scale,n=len(frames))

def nuisance(mean,pred):
    bias=np.zeros((64,16));bias[:,0]=np.maximum(mean[:,0]-pred[:,0],0)
    return bias

def metrics(obs,pred,scale):
    res=pred-obs
    return dict(relative_l2=float(np.linalg.norm(res)/max(np.linalg.norm(obs),1e-12)),
                weighted_rmse=float(np.sqrt(np.mean((res/scale)**2))),rmse_decoded=float(np.sqrt(np.mean(res**2))),
                signed_residual_mean=float(res.mean()),by_bin_relative_l2=[float(np.linalg.norm(res[:,b])/max(np.linalg.norm(obs[:,b]),1e-12)) for b in range(16)],
                by_bin_rmse=[float(np.sqrt(np.mean(res[:,b]**2))) for b in range(16)])

def fit():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'instrument_fit.json').exists():raise FileExistsError('Preserve existing wall fit')
    w2=stable_wall('2');calls=0
    def residual(x):
        nonlocal calls
        calls+=1
        if time.monotonic()-started>390:raise TimeoutError('Wall fit CPU internal390s cap')
        pred=plane_prediction(x);bg=nuisance(w2['mean'],pred)
        data=((pred+bg-w2['mean'])/w2['scale']).ravel()
        prior=np.array([(x[5]-.623)/.08,x[6]/10,x[7]/10])
        return np.r_[data,prior]
    initial=np.array([0,.12,.10,.15,.75,.623,0,0])
    starts=[initial,initial+np.array([.10,.08,.1,.05,-.2,-.08,8,-8]),initial+np.array([-.10,-.07,-.08,.15,.2,.08,-8,8])]
    solutions=[]
    for start in starts:
        opt=least_squares(residual,np.clip(start,LOW+1e-7,HIGH-1e-7),bounds=(LOW,HIGH),max_nfev=180,xtol=1e-7,ftol=1e-7,gtol=1e-7)
        solutions.append(opt)
    opt=min(solutions,key=lambda q:q.fun@q.fun);x=opt.x;objective=float(opt.fun@opt.fun)
    profiles={};threshold=1.05*objective
    for j,name in enumerate(NAMES):
        grid=np.unique(np.r_[np.linspace(LOW[j],HIGH[j],7),x[j]])
        rows=[];free=np.arange(8)!=j
        for value in grid:
            trial=x.copy();trial[j]=value
            def reduced(v):
                full=trial.copy();full[free]=v;return residual(full)
            prof=least_squares(reduced,trial[free],bounds=(LOW[free],HIGH[free]),max_nfev=45,xtol=2e-5,ftol=2e-5,gtol=2e-5)
            row=dict(value=float(value),objective=float(prof.fun@prof.fun),parameters=dict(zip(np.array(NAMES)[free],map(float,prof.x))),converged=bool(prof.success))
            rows.append(row)
        admitted=[r['value'] for r in rows if r['objective']<=threshold]
        profiles[name]=dict(grid=rows,descriptive_5percent_set=[min(admitted),max(admitted)] if admitted else None,
                            bound_included=[LOW[j] in admitted,HIGH[j] in admitted],interval_kind='Grid profile objective sensitivity, not confidence interval')
    pred4=plane_prediction(x,4);bg=nuisance(w2['mean'],pred4)
    pred16=plane_prediction(x,16);fixed=x[:5].copy()
    save('instrument_parameters_frozen.json',dict(instrument_parameters=dict(zip(NAMES[:5],map(float,fixed))),
         wall2_geometry=dict(zip(NAMES[5:],map(float,x[5:]))),wall2_bias=bg.tolist(),
         only_input_sha256=sha(P.OUT/'2.npz'),source_sha256=sha(__file__),
         freeze_before_wall3=True,cpu_seconds_at_freeze=time.monotonic()-started))
    w3=stable_wall('3')
    def residual3(geometry):
        xx=np.r_[fixed,geometry];pred=plane_prediction(xx)+bg
        return np.r_[((pred-w3['mean'])/w3['scale']).ravel(),(geometry[0]-1.452)/.10,geometry[1:]/10]
    check=least_squares(residual3,[1.452,0,0],bounds=([1.1,-15,-15],[1.8,15,15]),max_nfev=180)
    x3=np.r_[fixed,check.x];pred3=plane_prediction(x3,16)+bg
    j=opt.jac;singular=np.linalg.svd(j,compute_uv=False);cov=np.linalg.pinv(j.T@j,rcond=1e-9);sd=np.sqrt(np.maximum(np.diag(cov),1e-30));corr=cov/np.outer(sd,sd)
    boundary=[NAMES[k] for k in range(8) if min(abs(x[k]-LOW[k]),abs(HIGH[k]-x[k]))<1e-3*(HIGH[k]-LOW[k])]
    rawparams=dict(zip(NAMES,map(float,x)))
    result=dict(status='DESCRIPTIVE_EFFECTIVE_JOINT_WALL_FIT',fit_condition='2',instrument_parameters={k:rawparams[k] for k in NAMES[:5]},
                wall2_geometry={k:rawparams[k] for k in NAMES[5:]},wall3_check_geometry=dict(zip(NAMES[5:],map(float,check.x))),
                rho_assumed=.5,parameter_bounds={n:[float(a),float(b)] for n,a,b in zip(NAMES,LOW,HIGH)},boundary_parameters=boundary,
                objective=objective,profile_threshold=threshold,profiles=profiles,jacobian_singular_values=singular.tolist(),
                jacobian_parameter_correlation=corr.tolist(),parameter_order=list(NAMES),
                wall2=metrics(w2['mean'],pred16+bg,w2['scale']),wall3_check=metrics(w3['mean'],pred3,w3['scale']),
                quadrature_4_to_16_relative_l2=float(np.linalg.norm(pred16-pred4)/max(np.linalg.norm(pred16),1e-12)),
                fit_attempts=[dict(objective=float(o.fun@o.fun),nfev=o.nfev,converged=bool(o.success),parameters=dict(zip(NAMES,map(float,o.x)))) for o in solutions],
                background=dict(source='Wall2-only profiled nonnegative bin0 per-zone',cells=64,all_other_cells=0,
                                limit='Bin0 residual cannot distinguish instrument bias, early pulse mass, or wall2-only physical return; carried unchanged to wall3 and other scenes'),
                limits=['Range offset, wall distance, effective gain and reflectance remain confounded.',
                        'Fixed background has no ambient slope or full-space crosstalk calibration.',
                        'Wall3 geometry-only check is not a full independent calibration; instrument frozen before opening wall3.',
                        '5percent profile sets include discrepancy/prior choices and are not statistical confidence intervals.'],
                runtime=dict(cpu_command_wall_internal_s=time.monotonic()-started,gpu_s=0,residual_calls=calls),
                frozen_sensor_parameters=asdict(frozen_parameters()),source_sha256=sha(__file__),
                input_sha256={s:sha(P.OUT/(s+'.npz')) for s in ('2','3')})
    np.savez_compressed(OUT/'instrument_fit_arrays.npz',bias=bg,wall2_mean=w2['mean'],wall3_mean=w3['mean'],wall2_prediction=pred16+bg,wall3_prediction=pred3,
                        wall2_variance=w2['var'],wall3_variance=w3['var'],wall2_scale=w2['scale'],wall3_scale=w3['scale'],jacobian=j,parameters=x,wall3_parameters=x3)
    save('instrument_fit.json',result)
    print(json.dumps({k:result[k] for k in ('instrument_parameters','wall2_geometry','wall3_check_geometry','boundary_parameters','wall2','wall3_check','runtime')},ensure_ascii=False),flush=True)

def refine_profiles():
    """Increase profile resolution without changing the already frozen fit."""
    started=time.monotonic()
    if (OUT/'instrument_profile_refined.json').exists():raise FileExistsError('Preserve existing refined profiles')
    result=json.loads((OUT/'instrument_fit.json').read_text(encoding='utf8'))
    with np.load(OUT/'instrument_fit_arrays.npz') as a:x=a['parameters']
    w2=stable_wall('2')
    def residual(full):
        pred=plane_prediction(full);bg=nuisance(w2['mean'],pred)
        return np.r_[((pred+bg-w2['mean'])/w2['scale']).ravel(),(full[5]-.623)/.08,full[6:]/10]
    threshold=1.05*result['objective'];profiles={}
    for j,name in enumerate(NAMES):
        span=.05*(HIGH[j]-LOW[j]);grid=np.unique(np.clip(x[j]+np.linspace(-span,span,9),LOW[j],HIGH[j]));free=np.arange(8)!=j
        rows=list(result['profiles'][name]['grid'])
        for value in grid:
            trial=x.copy();trial[j]=value
            def reduced(v):
                full=trial.copy();full[free]=v;return residual(full)
            o=least_squares(reduced,trial[free],bounds=(LOW[free],HIGH[free]),max_nfev=60,xtol=2e-5,ftol=2e-5,gtol=2e-5)
            rows.append(dict(value=float(value),objective=float(o.fun@o.fun),parameters=dict(zip(np.array(NAMES)[free],map(float,o.x))),converged=bool(o.success)))
        rows.sort(key=lambda r:r['value']);admitted=[r['value'] for r in rows if r['objective']<=threshold]
        profiles[name]=dict(grid=rows,descriptive_5percent_set=[min(admitted),max(admitted)],
                            bound_included=[LOW[j] in admitted,HIGH[j] in admitted],interval_kind='Coarse plus local grid profile sensitivity, not confidence interval')
    report=dict(profiles=profiles,profile_threshold=threshold,parameter_fit_sha256=sha(OUT/'instrument_fit.json'),
                fixed_parameters_unchanged=True,runtime_cpu_s=time.monotonic()-started,source_sha256=sha(__file__))
    save('instrument_profile_refined.json',report)
    print(json.dumps({k:v['descriptive_5percent_set'] for k,v in profiles.items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','refine']);a=p.parse_args()
    {'fit':fit,'refine':refine_profiles}[a.stage]()
