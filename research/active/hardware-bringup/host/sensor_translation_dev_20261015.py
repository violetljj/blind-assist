"""Consumed Development: public geometry -> H3 proxy -> weak real readout.

No hardware, downloads, protected data or dense-network training. Firmware D/S/N
are evaluation-only except explicitly declared calibration-wall shape fitting.
No real recording is mean-subtracted for inference. Empirical short-window
centering creates nuisance draws only, not a device background correction.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '4')

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import joblib
import numpy as np
from scipy.ndimage import convolve1d
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[4]
NEAR = ROOT / 'research/active/dtr-r0/nearfield'
sys.path.insert(0, str(NEAR))
from cnh_rgb_zone_sensor import quadrature
import cnh_route_sensor as C
from cnh_zone_orientation import check_mounting, to_model
from cnh_h3_recording_diagnostics import load_recording

WORK = ROOT / 'artifacts.local/work'
PILOT = WORK / 'real-pilot-calibration-replay-dev-20261011/relaxed-pilot'
PUBLIC = WORK / 'data-potential-dev-20261013/round4/native_support'
DATA = ROOT / 'artifacts.local/datasets/hypersim-ba-nfo'
PHONE = WORK / 'phone-new-capture-20261012'
OUT = WORK / 'sensor-translation-dev-20261015'
WIDTH = C.RAW_BIN_M * C.H3.sub_sample
SCALES = (.45, .60, .80, 1., 1.25, 1.50)
ARMS = ('geometry', 'assumed', 'empirical')
SEED = 20261015


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def good(d):
    return (d['S'] == 5) & (d['N'] > 0) & (d['D'] >= WIDTH * 1000) & (d['D'] < 4500)


def model_order(d):
    return {k: to_model(v, zone_axis=1) if k in ('H', 'A', 'D', 'S', 'N') else v
            for k, v in d.items()}


def pilot_data():
    check_mounting()
    return {sid: model_order(dict(np.load(PILOT / (sid + '.npz'), allow_pickle=False)))
            for sid in ('1', '2', '3', '4', '4b', '5', '6', '7', '8', '9')}


def fit_nuisance(data):
    """Fit only 2/4/4b. Everything is in decoded units, not photon counts."""
    wall = data['2']
    offsets = np.arange(-2, 3)
    shapes, amp_at_1m = [], []
    valid = good(wall) & wall['stable'][:, None]
    for t, z in zip(*np.where(valid)):
        r = wall['D'][t, z] / 1000
        if not .4 <= r <= 1.1:
            continue
        k = int(r / WIDTH)
        h = wall['H'][t, z]
        floor = float(np.median(h[8:]))
        ii = k + offsets
        # Bin0 cannot be identified as a target pulse from this recording.
        keep = (ii > 0) & (ii < 16)
        v = np.zeros(5)
        v[keep] = np.maximum(h[ii[keep]] - floor, 0)
        if v.sum() > 1:
            shapes.append(v / v.sum())
            amp_at_1m.append(v.sum() * r*r)
    if len(shapes) < 20:
        raise ValueError('Insufficient calibration-wall shape support')
    kernel = np.median(shapes, axis=0)
    kernel /= kernel.sum()
    residuals, ambients, check_residuals, check_ambients, blocks = [], [], [], [], []
    for sid in ('4', '4b'):
        d = data[sid]
        for start in range(0, len(d['H']) - 7, 8):
            ix = np.arange(start, start + 8)
            if not (d['stable'][ix].all() and np.all(np.diff(d['seq'][ix]) == 1)
                    and np.all((np.diff(d['ms'][ix]) > 0) & (np.diff(d['ms'][ix]) <= 500))):
                continue
            h = d['H'][ix]
            # Contains motion/scene changes too: deliberately called nuisance.
            is_check = (start//8) % 2 == 1
            bank = check_residuals if is_check else residuals
            ambient_bank = check_ambients if is_check else ambients
            bank.extend(h - np.median(h, axis=0, keepdims=True))
            ambient_bank.extend(d['A'][ix])
            blocks.append(dict(segment=sid, start=int(start), frames=8, role='check' if is_check else 'train'))
    if not residuals:
        raise ValueError('No contiguous calibration nuisance blocks')
    residuals = np.asarray(residuals)
    ambients = np.asarray(ambients)
    # Only bin0 carrier; no full-scene template, no estimated bin1 deletion.
    b0 = np.quantile(wall['H'][wall['stable'], :, 0], .25, axis=0)
    amplitudes = np.asarray(amp_at_1m)
    return dict(kernel=kernel, residuals=residuals, ambients=ambients,
                check_residuals=np.asarray(check_residuals), check_ambients=np.asarray(check_ambients),
                b0=b0, amplitude_at_1m=amplitudes), dict(
                    fit_segments=['2', '4', '4b'], kernel=kernel.tolist(),
                    wall_shape_zone_frames=len(shapes), amplitude_at_1m_quantiles=
                    np.quantile(amplitudes, [.1, .5, .9]).tolist(),
                    nuisance_frames=len(residuals), check_nuisance_frames=len(check_residuals), nuisance_blocks=blocks,
                    residual_sd_quantiles=np.quantile(residuals.std(0), [.1,.5,.9]).tolist(),
                    caveat='Effective decoded-unit nuisance; includes movement; wall D is a weak centre; no physical IRF/gain identification')


def geometry():
    """8 fixed consumed families, original radial depth, no legacy resizing."""
    manifest = read(PUBLIC / 'manifest.json')
    rows = manifest['rows']
    if len(rows) != 8 or any(r['source'] != 'hypersim' or r['split'] != 'train' for r in rows):
        raise ValueError('Need the declared eight consumed Hypersim train frames')
    out, receipts = [], []
    for n, r in enumerate(rows):
        p = DATA / r['depth']
        side = PUBLIC / (r['id'] + '.npz')
        with h5py.File(p, 'r') as f:
            radial = f['dataset'][:]
        with np.load(side, allow_pickle=False) as f:
            mat = f['M_cam_from_uv']
        q = quadrature(radial, mat)
        weights = q['solid_angle_weights'].reshape(64, -1)
        weights /= weights.sum(-1, keepdims=True)
        rays = q['radial_m'].reshape(64, -1)
        for scale in SCALES:
            rr = rays * scale
            hist = np.zeros((64, 16))
            ys = np.full(64, np.nan)
            coverage = np.sum(np.isfinite(rr) * weights, axis=-1)
            spread = np.zeros(64)
            for z in range(64):
                ok = np.isfinite(rr[z]) & (rr[z] >= WIDTH) & (rr[z] < WIDTH * 16)
                if coverage[z] < .8 or ok.sum() < 4:
                    continue
                bins = np.floor(rr[z, ok] / WIDTH).astype(int)
                # One shared diffuse inverse-square target distribution for all arms.
                ww = weights[z, ok] / np.maximum(rr[z, ok]**2, .01)
                hist[z] = np.bincount(bins, ww, minlength=16)
                k = int(hist[z].argmax())
                sel = bins == k
                ys[z] = np.average(rr[z, ok][sel], weights=ww[sel])
                spread[z] = float(np.quantile(rr[z, ok], .9) - np.quantile(rr[z, ok], .1))
            near_mass = np.sum(weights * ((rr >= WIDTH) & (rr < 2.)), axis=-1)
            far_mass = np.sum(weights * (rr >= 2.), axis=-1)
            too_close = np.sum(weights * (rr < WIDTH), axis=-1)
            # Partial/unknown rays or sub-resolution near support are not FREE.
            task_valid = (coverage > 1-1e-9) & (too_close == 0) & (hist.sum(-1) > .005)
            task_valid &= (near_mass == 0) | (near_mass >= .02)
            near_target = near_mass >= .02
            weak_near = task_valid & near_target & (near_mass <= .2) & (far_mass >= .5)
            response = C.synthesize_response(rr.reshape(8,8,-1), .5, 1., weights.reshape(8,8,-1),
                params=replace(C.SensorParameters(), noise_scale=0.), seed=SEED+n)
            assumed = C.derive_readout(response, C.H3)
            out.append(dict(source_id=r['id'], family=r['family'], role='train' if n < 6 else 'check',
                            scale=scale, hist=hist, y=ys, coverage=coverage, spread=spread,
                            task_valid=task_valid, near_target=near_target, weak_near=weak_near,
                            assumed_mean=assumed['histogram'].reshape(64,16),
                            assumed_ambient=assumed['ambient'].reshape(64)))
        receipts.append(dict(**r, depth_sha256=sha(p), camera_sidecar_sha256=sha(side),
                             radial_shape=list(radial.shape)))
    return out, receipts


def render(row, arm, nuisance, rng, *, check_noise=False):
    """All variants use the same underlying public geometry and target."""
    base = row['hist']
    if arm == 'geometry':
        h = base * 2000
        a = np.zeros(64)
    elif arm == 'assumed':
        # Sum of independent Poisson draws equals a Poisson of the summed means.
        # Mean is produced by the unchanged native128 simulator above.
        mean = row['assumed_mean']
        a = row['assumed_ambient'].copy()
        bg = a[:,None] * 8
        h = rng.poisson(np.maximum(mean + bg, 0)) - rng.poisson(np.broadcast_to(bg, mean.shape))
    elif arm == 'empirical':
        amp = float(rng.choice(nuisance['amplitude_at_1m']))
        h = convolve1d(base, nuisance['kernel'], axis=-1, mode='constant') * amp
        bank = nuisance['check_residuals'] if check_noise else nuisance['residuals']
        ambient_bank = nuisance['check_ambients'] if check_noise else nuisance['ambients']
        i = int(rng.integers(len(bank)))
        # Preserve the whole-frame spatial/bin residual; do not iid shuffle it.
        h += bank[i] * rng.uniform(.7, 1.4)
        h[:,0] += nuisance['b0'] * rng.uniform(.9, 1.1)
        a = ambient_bank[i].copy()
    else:
        raise ValueError(arm)
    return np.asarray(h, dtype=float), np.asarray(a, dtype=float)


def features(h, ambient):
    """Observation only. No range/status, query truth or source role enters."""
    h = np.asarray(h, float).copy()
    if h.shape[-2:] != (64,16):
        raise ValueError('Expected [...,64,16]')
    h[...,0] = 0  # fixed declared contaminated interval, not a tuned deletion
    floor = np.quantile(h[...,1:], .25, axis=-1, keepdims=True)
    centered = h - floor
    pos = np.maximum(centered, 0)
    pos[...,0] = 0
    total = pos.sum(-1, keepdims=True)
    norm = pos / np.maximum(total, 1e-8)
    signed = np.arcsinh(centered) / 8
    signed[...,0] = 0
    z = np.broadcast_to(np.arange(64) / 63, total.shape[:-1])[...,None]
    f = np.concatenate((norm, signed, np.log1p(total) / 10,
                        np.log1p(np.maximum(ambient,0))[...,None] / 10, z), axis=-1)
    if not np.isfinite(f).all():
        raise ValueError('Nonfinite features')
    return f.reshape(-1,f.shape[-1]).astype(np.float32)


def metrics(pred, target):
    e = np.asarray(pred) - np.asarray(target)
    n = len(e)
    if not n:
        return dict(n=0, mae_m=None, median_abs_m=None, p90_abs_m=None, within_one_bin=None,
                    bias_m=None, false_near=None, missed_near=None)
    near = target < 2
    return dict(n=n, mae_m=float(np.mean(abs(e))), median_abs_m=float(np.median(abs(e))),
                p90_abs_m=float(np.quantile(abs(e),.9)), within_one_bin=float(np.mean(abs(e)<=WIDTH)),
                bias_m=float(e.mean()), false_near=dict(numerator=int(np.sum((pred<2)&~near)), denominator=int((~near).sum())),
                missed_near=dict(numerator=int(np.sum((pred>=2)&near)), denominator=int(near.sum())))


def task_metrics(pred, target, weak):
    pos = target.astype(bool)
    return dict(n=len(pos), positive=int(pos.sum()), negative=int((~pos).sum()),
                hits=int(np.sum(pred & pos)), false_alarms=int(np.sum(pred & ~pos)),
                misses=int(np.sum(~pred & pos)), weak_positive=int(weak.sum()),
                weak_hits=int(np.sum(pred & weak)))


def public_checks(models, classifiers, rows, nuisance):
    """All methods see the SAME check frames per target domain and seed."""
    regression, tasks, details = {}, {}, []
    for domain in ARMS:
        pr={arm:[] for arm in (*ARMS,'raw_peak')}
        cp={arm:[] for arm in (*ARMS,'raw_peak')}
        ys=[]; labels=[]; weak=[]
        rng=np.random.default_rng(SEED+1)
        for row in rows:
            if row['role']!='check':
                continue
            for repeat in range(4):
                h,a=render(row,domain,nuisance,rng,check_noise=True)
                x=features(h,a)
                valid=np.isfinite(row['y']); mask=row['task_valid']
                hh=h.copy();hh[:,0]=-np.inf
                peak=(hh.argmax(-1)+.5)*WIDTH
                ys.extend(row['y'][valid]);labels.extend(row['near_target'][mask]);weak.extend(row['weak_near'][mask])
                pr['raw_peak'].extend(peak[valid]);cp['raw_peak'].extend((peak<2)[mask])
                for arm in ARMS:
                    pp=models[arm].predict(x); pred=classifiers[arm].predict_proba(x)[:,1] >= .5
                    pr[arm].extend(pp[valid]);cp[arm].extend(pred[mask])
                    if domain=='empirical':
                        tm=task_metrics(pred[mask],row['near_target'][mask],row['weak_near'][mask])
                        details.append(dict(source_id=row['source_id'],scale=row['scale'],repeat=repeat,arm=arm,**tm))
        regression[domain]={arm:metrics(np.array(v),np.array(ys)) for arm,v in pr.items()}
        tasks[domain]={arm:task_metrics(np.asarray(v,bool),np.asarray(labels,bool),np.asarray(weak,bool)) for arm,v in cp.items()}
    return regression,tasks,details


def evaluate(models, d):
    mask = good(d)
    x = features(d['H'], d['A'])
    target = d['D'][mask] / 1000
    preds = {arm: m.predict(x).reshape(d['D'].shape) for arm,m in models.items()}
    h = d['H'].copy(); h[...,0] = -np.inf
    preds['raw_peak'] = (h.argmax(-1)+.5)*WIDTH
    result = {arm:metrics(v[mask], target) for arm,v in preds.items()}
    return dict(frames=len(d['H']), all_zone_frames=int(mask.size), weak_valid_zone_frames=int(mask.sum()),
                weak_valid_fraction=float(mask.mean()), by_arm=result), preds


def run(out, budget_s):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists():
        raise FileExistsError('Preserve prior results; choose a new --output')
    if not out.resolve().is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Output must be inside the canonical artifact tree')
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    def check():
        if time.perf_counter()-start > budget_s:
            raise TimeoutError('Recorded execution budget reached')
    data = pilot_data()
    nuisance, fit_receipt = fit_nuisance(data)
    source, public_receipts = geometry()
    save(out/'inputs.json', dict(public=public_receipts, real_inventory_sha256=sha(PILOT/'inventory.json'),
         real=[dict(id=sid,sha256=sha(PILOT/(sid+'.npz')),role='fit' if sid in ('2','4','4b') else 'check') for sid in data],
         public_selection_sha256=sha(PUBLIC/'manifest.json')))
    save(out/'nuisance.json', fit_receipt)
    np.savez_compressed(out/'nuisance.npz', **nuisance)
    models, classifiers, learned = {}, {}, {}
    for arm in ARMS:
        check()
        rng = np.random.default_rng(SEED)
        xx, yy, cx, cy = [], [], [], []
        for row in source:
            if row['role'] != 'train':
                continue
            mask = np.isfinite(row['y'])
            for repeat in range(8):
                h,a = render(row, arm, nuisance, rng)
                xx.append(features(h,a)[mask]); yy.append(row['y'][mask])
                cx.append(features(h,a)[row['task_valid']]);cy.append(row['near_target'][row['task_valid']])
        x, y = np.concatenate(xx), np.concatenate(yy)
        model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                    min_samples_leaf=30, l2_regularization=1., early_stopping=False, random_state=SEED)
        model.fit(x,y)
        models[arm] = model
        joblib.dump(model,out/(arm+'.joblib'))
        class_x,class_y=np.concatenate(cx),np.concatenate(cy)
        classifier=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=30,
                    l2_regularization=1.,early_stopping=False,random_state=SEED,class_weight='balanced')
        classifier.fit(class_x,class_y)
        classifiers[arm]=classifier
        joblib.dump(classifier,out/(arm+'-near.joblib'))
        learned[arm] = dict(samples=len(y), feature_count=x.shape[1], iterations=model.n_iter_,
                           near_task_samples=len(class_y),near_task_positive=int(class_y.sum()),
                           target_min_m=float(y.min()),target_max_m=float(y.max()))
        print(arm, 'train', len(y), 'near',len(class_y),flush=True)
    public_result, public_tasks, task_rows=public_checks(models,classifiers,source,nuisance)
    save(out/'public-task-rows.json',task_rows)
    pilot={}
    for sid,d in data.items():
        check()
        pilot[sid],pred=evaluate(models,d)
        pilot[sid]['role']='fit' if sid in ('2','4','4b') else 'check'
        np.savez_compressed(out/('pilot-'+sid+'.npz'),**pred,weak_valid=good(d),weak_reference_m=d['D']/1000)
    phone={}
    # Explicit Development registry, not directory-wide arbitrary/test discovery.
    sessions=read(PHONE/'prepared.json')
    if isinstance(sessions,dict):
        sessions=sessions.get('sessions',sessions.get('rows',sessions))
    # Source roster schema is inspected in the caller; list paths from prepared.
    for entry in sessions:
        sid=entry if isinstance(entry,str) else entry['session']
        raw=PHONE/'raw'/sid/'tof.raw.jsonl'
        if not raw.exists():
            raise FileNotFoundError(raw)
        check()
        d=model_order(load_recording(raw))
        phone[sid],pred=evaluate(models,d)
        np.savez_compressed(out/('phone-'+sid+'.npz'),**pred,weak_valid=good(d),weak_reference_m=d['D']/1000)
    def macro(rows,ids):
        return {arm:dict(segment_count=sum(rows[s]['by_arm'][arm]['n']>0 for s in ids),
            mae_m=float(np.mean([rows[s]['by_arm'][arm]['mae_m'] for s in ids if rows[s]['by_arm'][arm]['n']>0])))
            for arm in (*ARMS,'raw_peak')}
    result=dict(status='COMPLETE',schema='sensor-translation-dev.v1',seed=SEED,
        training=learned,public_check=public_result,public_near_task=public_tasks,pilot=pilot,phone=phone,
        pilot_check_macro=macro(pilot,[s for s in pilot if pilot[s]['role']=='check']),
        phone_macro=macro(phone,list(phone)),bin_width_m=WIDTH,
        seconds=time.perf_counter()-start,source_sha256=sha(__file__),
        evidence='Consumed Development. Firmware distance agreement, not independent range/obstacle truth. No model selection on real check. No deployment.',
        resources=dict(new_hardware_cny=0,download_bytes=0,gpu_seconds=0,budget_wall_s=budget_s))
    save(out/'result.json',result)
    print(json.dumps(dict(status='COMPLETE',seconds=result['seconds'],pilot=result['pilot_check_macro'],phone=result['phone_macro']),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--budget-s',type=float,default=900)
    a=p.parse_args()
    try:
        run(a.output,a.budget_s)
    except Exception as exc:
        a.output.mkdir(parents=True,exist_ok=True)
        if not (a.output/'result.json').exists():
            save(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
