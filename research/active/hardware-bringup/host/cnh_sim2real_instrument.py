"""Existing-recording CNH instrument diagnostics, with a surface-preserving bias.

No serial/device API, model, threshold or protected dataset is used here. Decoded
CNH units are not identified physical photon counts. Firmware's strongest range
is only a weak reference; it cannot exclude a second, undetected near surface.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

WIDTH = .3002784
MIN_BIAS_FRAMES = 8
MIN_FAR_M = 1.5
DISTANCE_STABILITY_MM = 30.
AMBIENT_RELATIVE_STABILITY = .20
PAIR_BLOCK = 32


def distribution(x):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    return dict(n=int(len(x)), mean=float(x.mean()) if len(x) else None,
                quantiles={str(p):float(np.percentile(x,p)) for p in (0,5,25,50,75,95,100)} if len(x) else {})


def nnls_small(x, y):
    """Exhaustive active-set NNLS for three or fewer columns."""
    best = np.zeros(x.shape[1]); loss = float(y@y)
    for k in range(1, x.shape[1]+1):
        for active in itertools.combinations(range(x.shape[1]), k):
            sub = x[:, active]
            c = np.linalg.lstsq(sub, y, rcond=None)[0]
            if np.any(c < 0):
                continue
            candidate = np.zeros(x.shape[1]); candidate[list(active)] = c
            e = float(np.sum((x@candidate-y)**2))
            if e < loss:
                best, loss = candidate, e
    return best


def _sensor():
    source = Path(__file__).resolve().parents[4]/'artifacts.local/work/cnh-track-a-v5-20260928/data/source/cnh_route_sensor.py'
    name = '_sim2real_frozen_route_sensor'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name], source


def _nominal(f):
    """Recompute frozen tier6 gain analytically; no random photons are drawn."""
    p = replace(f.SensorParameters(), signal_counts=1., noise_scale=0.)
    matrix = f._pulse_matrix(p)
    raw = np.zeros(128); raw[int(np.floor(2/f.RAW_BIN_M))] = .5/4
    signal = raw@matrix
    # A lone native zone exchanges 2% energy with its four neighbours.
    signal *= 1-p.neighbour_leak
    xtalk = p.crosstalk_fraction*matrix[int(np.floor(p.crosstalk_range_m/f.RAW_BIN_M))]
    window = np.arange(int(np.floor(2/f.RAW_BIN_M))-3, int(np.floor(2/f.RAW_BIN_M))+4)
    increment, target = float(signal[window].sum()), float((signal+xtalk)[window].sum())
    lo, hi = 1e-3, 1e7
    for _ in range(40):
        g = (lo+hi)/2
        if g*increment/np.sqrt(g*target+2*p.ambient_counts*len(window)) < 6:
            lo = g
        else:
            hi = g
    return replace(f.SensorParameters(), signal_counts=(lo+hi)/2)


def _expected_plane(f, p, axial_m, rho=.3):
    rays, weights = f.angular_rays(16)
    radial = axial_m/rays[...,2]
    normalized = weights/weights.sum(-1, keepdims=True)
    energy = p.signal_counts*rho*rays[...,2]*normalized/np.maximum(radial,.05)**2
    raw, _, _ = f._histogram(radial, energy, np.ones(radial.shape,bool),128,f.RAW_BIN_M,0.,'numpy',None)
    signal = raw@f._pulse_matrix(p)
    old = signal.copy(); leak = p.neighbour_leak/4
    signal[1:] += leak*(old[:-1]-old[1:]); signal[:-1] += leak*(old[1:]-old[:-1])
    signal[:,1:] += leak*(old[:,:-1]-old[:,1:]); signal[:,:-1] += leak*(old[:,1:]-old[:,:-1])
    xtalk = p.signal_counts*p.crosstalk_fraction*f._pulse_matrix(p)[int(np.floor(p.crosstalk_range_m/f.RAW_BIN_M))]
    return signal.reshape(64,16,8).sum(-1), xtalk.reshape(16,8).sum(-1)


def _pulse_summary(profile, center):
    bins = np.arange(16)
    local = np.where(abs(bins-center)<=2, np.maximum(profile,0),0.)
    total = float(local.sum())
    if total <= 0:
        return None
    probability = local/total
    mean = float(probability@bins)
    sigma = float(np.sqrt(probability@((bins-mean)**2)))
    half = np.flatnonzero(local >= local.max()/2)
    return dict(center_bin=mean,sigma_aggregate_bins=sigma,
                fwhm_aggregate_bins=int(half[-1]-half[0]+1),
                mass_outside_peak_bin=float(1-probability[int(np.argmax(local))]),
                relative_kernel=[float(sum(probability[(bins-center)==j])) for j in range(-2,3)])


def analyze(recordings, out):
    """Analyze dict[id: {H,A,D,S,N,seq,ms}], returning the adapter parameters.

    D is in mm; S=5 and N>0 are the pre-existing conservative firmware-valid
    rule. Optional `known_plane` controls eligibility for effective amplitude
    mapping; absent/False still permits explicitly named weak-plane proxies.
    Coverage failures become zero subtraction, never scene-mean fallback.
    """
    started = time.perf_counter(); out = Path(out); out.mkdir(parents=True,exist_ok=True)
    f, source = _sensor(); p = _nominal(f)
    arrays = {}; records = {}; masked = []; per_record = {}; fixed = {}
    for name, r in recordings.items():
        h,a,d,s,n = (np.asarray(r[k]) for k in ('H','A','D','S','N'))
        if not len(h) or h.shape != (len(h),64,16) or any(x.shape != (len(h),64) for x in (a,d,s,n)):
            raise ValueError(f'{name}: expected H[N,64,16], A/D/S/N[N,64]')
        if not np.isfinite(h).all() or not np.isfinite(a).all() or np.any(a<0):
            raise ValueError(f'{name}: nonfinite histogram or invalid ambient')
        good = (s==5)&(n>0)&(d>0)
        distance = d/1000
        safe = good[...,None]&(distance[...,None]>np.maximum(MIN_FAR_M,(np.arange(16)+2.5)*WIDTH))
        masked.append(np.where(safe,h,np.nan))
        records[name] = (r,h,a,d,good,safe)
        per_record[name] = dict(frames=len(h),valid_range_zone_frames=int(good.sum()),
                               ambient=distribution(a),bias_candidate_cells=int(safe.sum()),
                               candidate_by_bin=safe.sum((0,1)).tolist())
    pooled = np.concatenate(masked,axis=0)
    coverage = np.isfinite(pooled).sum(0)
    bias = np.zeros((64,16)); bias_uncertainty = np.full((64,16),np.nan)
    for z,b in np.ndindex(64,16):
        x = pooled[:,z,b]; x = x[np.isfinite(x)]
        if len(x)>=MIN_BIAS_FRAMES:
            bias[z,b] = np.median(x)
            # Temporal dependence prevents interpreting this descriptive width as CI.
            bias_uncertainty[z,b] = 1.4826*np.median(abs(x-np.median(x)))
    available = coverage>=MIN_BIAS_FRAMES
    arrays.update(bias=bias,bias_coverage=coverage,bias_available=available,bias_mad_width=bias_uncertainty)
    # Noise blocks pool stable consecutive pairs; differencing removes static scene
    # means but does not establish independence or eliminate moving secondary echoes.
    block_x=[]; block_y=[]; block_names=[]; mean_var=[]; far_pairs=0
    first_median = None
    for name,(r,h,a,d,good,safe) in records.items():
        med = np.median(h,axis=0); var = np.var(h,axis=0,ddof=1) if len(h)>1 else np.zeros((64,16))
        if first_median is None:first_median=med
        seq,ms = np.asarray(r['seq']),np.asarray(r['ms'])
        dt=np.diff(ms); valid_dt=dt[(np.diff(seq)==1)&(dt>0)]
        period=float(np.median(valid_dt)) if len(valid_dt) else 0.
        consecutive=(np.diff(seq)==1)&(dt>0)&(dt<=1.5*period)
        stable=good[1:]&good[:-1]&(abs(np.diff(d,axis=0))<=DISTANCE_STABILITY_MM)
        amid=(a[1:]+a[:-1])/2
        stable &= abs(np.diff(a,axis=0))<=AMBIENT_RELATIVE_STABILITY*np.maximum(amid,1e-12)
        far=stable&((d[1:]>1500)&(d[:-1]>1500))&consecutive[:,None]
        far_pairs+=int(far.sum())
        peak=np.floor((d[1:]+d[:-1])/2000/WIDTH).astype(int)
        off_peak=(abs(np.arange(16)[None,None,:]-peak[...,None])>2)
        pairs=far[...,None]&off_peak
        diff=np.diff(h,axis=0); pair_var=diff**2/2
        local_noise=np.full((64,16),np.nan)
        for z,b in np.ndindex(64,16):
            ids=np.flatnonzero(pairs[:,z,b])
            if len(ids)>=8:
                local_noise[z,b]=float(pair_var[ids,z,b].mean())
                mean_var.append([float((h[:-1,z,b][ids]+h[1:,z,b][ids]).mean()/2),local_noise[z,b]])
            if not available[z,b]:continue
            for start in range(0,len(ids),PAIR_BLOCK):
                ix=ids[start:start+PAIR_BLOCK]
                if len(ix)<8:continue
                block_x.append([1.,float(amid[ix,z].mean()),max(float(bias[z,b]),0.)])
                block_y.append(float(pair_var[ix,z,b].mean()))
                block_names.append(name)
        arrays[f'{name}_median']=med; arrays[f'{name}_variance']=var; arrays[f'{name}_pair_variance']=local_noise
        four_medians=np.asarray([np.median(h[ix],axis=0) for ix in np.array_split(np.arange(len(h)),4) if len(ix)])
        half=int(len(h)//2)
        half_shift=np.median(h[half:],axis=0)-np.median(h[:half],axis=0) if half else np.full((64,16),np.nan)
        block_range=four_medians.max(0)-four_medians.min(0)
        arrays[f'{name}_four_block_medians']=four_medians
        arrays[f'{name}_half_median_shift']=half_shift
        arrays[f'{name}_four_block_median_range']=block_range
        shift=med-first_median
        arrays[f'{name}_median_shift']=shift
        fixed[name]=dict(raw_median=distribution(med),temporal_variance=distribution(var),
                         median_shift_from_first=distribution(shift),
                         second_minus_first_half=distribution(half_shift),four_block_median_range=distribution(block_range),
                         row4_col2_bins12=dict(median=med[34,1:3].tolist(),variance=var[34,1:3].tolist(),
                             shift_from_first=shift[34,1:3].tolist(),bias=bias[34,1:3].tolist(),coverage=coverage[34,1:3].tolist(),
                             four_block_median_range=block_range[34,1:3].tolist(),half_median_shift=half_shift[34,1:3].tolist() if half else None),
                         limits='Scene medians and inter-recording shifts are diagnostics only; cannot separate fixed instrument pattern from a persistent surface.')
    x,y=np.asarray(block_x,float).reshape(-1,3),np.asarray(block_y,float)
    noise=dict(status='NOT_EVALUABLE',intercept=0.,ambient_coefficient=16.,positive_bias_coefficient=1.,
               fallback='Frozen coefficient16 and positive-bias coefficient1 retained; physical noise-unit mapping unestablished',blocks=len(y),far_stable_zone_pairs=far_pairs)
    if len(y)>=12 and np.linalg.matrix_rank(x)>=2:
        coefficient=nnls_small(x,y); prediction=x@coefficient
        # A positive empirical floor is a pooled moment estimate, not chosen by alerts.
        if np.any(coefficient>0):
            noise.update(status='EXPLORATORY_FIT',intercept=float(coefficient[0]),ambient_coefficient=float(coefficient[1]),
                         positive_bias_coefficient=float(coefficient[2]),design_rank=int(np.linalg.matrix_rank(x)),
                         relative_rmse=float(np.linalg.norm(prediction-y)/max(np.linalg.norm(y),1e-12)),
                         r_squared=float(1-np.sum((prediction-y)**2)/max(np.sum((y-y.mean())**2),1e-12)),
                         ambient_span=distribution(x[:,1]),ambient_coefficient_over_16=float(coefficient[1]/16),fallback=None)
    noise['method']='Nonnegative LS of stable-pair diff²/2 block means on 1, decoded ambient, positive estimated bias; no alert-based fit. Ambient limited range and missing bias cells limit identifiability.'
    noise['poisson_skellam_status']='NOT_IDENTIFIED: decoded unit/scaler, strongest-range ambiguity, drift, serial correlation and firmware processing prevent identifying photon-law from empirical mean/variance.'
    noise['leave_one_recording_out']={}
    for name in records:
        keep=np.asarray([v!=name for v in block_names],bool)
        if keep.sum()>=12 and np.linalg.matrix_rank(x[keep])>=2:
            c=nnls_small(x[keep],y[keep])
            noise['leave_one_recording_out'][name]=dict(status='EXPLORATORY_FIT',blocks=int(keep.sum()),design_rank=int(np.linalg.matrix_rank(x[keep])),intercept=float(c[0]),ambient_coefficient=float(c[1]),positive_bias_coefficient=float(c[2]))
        else:
            noise['leave_one_recording_out'][name]=dict(status='NOT_EVALUABLE',reason='Insufficient covered stable-pair blocks or rank')
    noise['floor_units']='Decoded CNH units squared; empirical coefficient is not physical photon variance. NOT_EVALUABLE fallback 16*A+positive_bias uses synthetic-contract numbers with unidentified real-unit correspondence.'
    arrays['noise_fit_x']=x;arrays['noise_fit_y']=y;arrays['mean_variance_pairs']=np.asarray(mean_var).reshape(-1,2)
    mapping={}; pulse={}; ray,_=f.angular_rays(1); zcos=ray.reshape(64,3)[:,2]
    _,nominal_bias=_expected_plane(f,p,.64)
    for name,(r,h,a,d,good,safe) in records.items():
        med_d=np.zeros(64); stable_zone=np.zeros(64,bool)
        for z in range(64):
            values=d[good[:,z],z]
            if len(values)>=8:
                med_d[z]=np.median(values)/1000
                stable_zone[z]=(np.std(values)<=DISTANCE_STABILITY_MM)&(.30<=med_d[z]<=2.)
        if stable_zone.sum()<4:
            mapping[name]=dict(status='NOT_EVALUABLE',reason='Fewer than four stable valid zones with firmware range .3..2m')
            continue
        axial=float(np.median(med_d[stable_zone]*zcos[stable_zone]))
        expected,xtalk=_expected_plane(f,p,axial)
        observed=np.median(h,axis=0)-bias
        target=stable_zone[:,None]&(abs(np.arange(16)[None,:]-np.floor(med_d[:,None]/WIDTH))<=2)
        xx,yy=expected[target],observed[target]
        gain=max(0.,float(xx@yy/max(xx@xx,1e-12)))
        residual=yy-gain*xx
        block_gains=[]
        for ix in np.array_split(np.arange(len(h)),4):
            if len(ix)<4:continue
            block_observed=np.median(h[ix],axis=0)-bias
            block_gains.append(max(0.,float(xx@block_observed[target]/max(xx@xx,1e-12))))
        arrays[f'{name}_nominal_expectation']=expected+xtalk[None,:]
        arrays[f'{name}_mapping_residual']=np.where(target,observed-gain*expected,np.nan)
        mapping[name]=dict(status='EFFECTIVE_GAIN_WEAK_PLANE_PROXY',known_plane=bool(r.get('known_plane',False)),
            axial_m_firmware_proxy=axial,stable_zones=int(stable_zone.sum()),rho_assumed=.3,effective_gain=gain,
            relative_l2_residual=float(np.linalg.norm(residual)/max(np.linalg.norm(yy),1e-12)),
            residual=distribution(residual),ambient_to_nominal_ratio=float(np.median(a)/p.ambient_counts),
            four_block_gain=distribution(block_gains),gain_range_is='Descriptive contiguous-block range, not confidence interval or held-out calibration',
            limits='Firmware range is weak geometry; approximate front-parallel plane, unknown NIR rho/material/pose and zero offset confound gain. Effective gain is not a physical count conversion.')
        profiles=[];sim_profiles=[];kernels=[]
        for z in np.flatnonzero(stable_zone):
            center=int(np.floor(med_d[z]/WIDTH))
            real=_pulse_summary(observed[z],center); sim=_pulse_summary(expected[z],center)
            if real and sim:
                profiles.append(real);sim_profiles.append(sim);kernels.append(real['relative_kernel'])
        kernel=np.median(kernels,axis=0) if kernels else None
        if kernel is not None:kernel=kernel/max(float(kernel.sum()),1e-12)
        pulse[name]=dict(status='AGGREGATE_BIN_WEAK_SINGLE_RETURN_PROXY' if profiles else 'NOT_EVALUABLE',zones=len(profiles),
            sigma_real=distribution([q['sigma_aggregate_bins'] for q in profiles]),sigma_nominal=distribution([q['sigma_aggregate_bins'] for q in sim_profiles]),
            fwhm_real=distribution([q['fwhm_aggregate_bins'] for q in profiles]),
            spread_real=distribution([q['mass_outside_peak_bin'] for q in profiles]),
            fitted_kernel_offsets=[-2,-1,0,1,2],fitted_kernel=kernel.tolist() if kernel is not None else None,
            limits='Positive-part local shape within firmware peak±2 aggregate bins; unknown multiple echoes and incompletely estimated bias confound pulse. No raw-bin kernel/range zero identification.')
    arrays['nominal_bias']=np.broadcast_to(nominal_bias,(64,16)).copy()
    bias_comparison=dict(near_bins=[0,1,2,3],real_zone_median=np.median(bias,axis=0).tolist(),nominal=nominal_bias.tolist(),
                         limits='Only covered cells represent estimated residual templates; uncovered zero is a subtraction fallback, not measured zero.')
    report=dict(schema='cnh.sim2real.instrument.v1',scope='Existing real replay description; no detection/false-alarm ground truth',
        methods=dict(bias='Median of valid status5/N>0 zone-frames with D>max(1.5m,(bin+2.5)*WIDTH); at least8 frames/cell. No whole-scene subtraction; strongest range cannot rule out hidden near returns.',
                     noise='Stable consecutive-pair difference moments in far zones and outside strongest firmware peak±2 bins; fit V=c0+cA*A+cB*positive_bias.',
                     fixed_pattern='All per-cell scene medians/variance, far-template candidates, and shifts from first recording retained; stable scene echo is not identified instrument bias.',
                     pulse='Stable firmware single-return proxy zones, peak±2 bins, positive-part mass; coarse16-bin shape only.'),
        constants=dict(width_m=WIDTH,min_bias_frames=MIN_BIAS_FRAMES,min_far_m=MIN_FAR_M,distance_stability_mm=DISTANCE_STABILITY_MM,ambient_relative_stability=AMBIENT_RELATIVE_STABILITY,pair_block=PAIR_BLOCK),
        bias=dict(covered_cells=int(available.sum()),total_cells=1024,coverage_by_bin=available.sum(0).tolist(),uncovered_policy='Subtract0 and mark NOT_EVALUABLE; no scene-mean fallback',comparison=bias_comparison),
        recordings=per_record,fixed_pattern=fixed,noise_parameters=noise,mapping=mapping,pulse=pulse,
        nominal_sensor_parameters=asdict(p),sources={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},elapsed_s=time.perf_counter()-started)
    np.savez_compressed(out/'instrument_arrays.npz',**arrays)
    (out/'instrument_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return dict(bias=bias,bias_coverage=coverage,bias_available=available,noise_parameters=noise,report=report)
