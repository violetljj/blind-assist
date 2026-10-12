"""Candidate distance diagnostics on consumed Development; no annotations or refits.

ToF shares the affine fit's anchors, so agreement is internal consistency only.
A thin line may not occupy a zone's dominant return: disagreement is not a
negative object label, and fewer than three valid zones stays UNSUPPORTED.
"""
from __future__ import annotations
import hashlib
import json
import time
from collections import Counter
import numpy as np
from atom_camera import AtomCamera
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_small_model_dev_20261012 as S
import real_rgb_query_feasibility_dev_20261012 as F

CACHE = F.ROOT / 'artifacts.local/work/fusion-refinement-dev-20261013'
SOURCE = F.ROOT / 'artifacts.local/work/thin-local-geometry-dev-20261013'
OUT = F.ROOT / 'artifacts.local/work/thin-confidence-dev-20261013'
ARMS = ('range_only', 'fit_gate', 'tof_agreement', 'local_consistency')


def strip_sample(c, disp, valid, fit, K):
    p = np.array([[c['x1'], c['y1']], [c['x2'], c['y2']]], float)
    delta = p[1] - p[0]
    normal = np.array([-delta[1], delta[0]]) / max(np.linalg.norm(delta), 1e-12)
    center = p[0] + np.linspace(0, 1, 40)[:, None] * delta
    profiles = {}; raw = None
    for name, offset in [('center', 0), ('left', -6), ('right', 6)]:
        xy = np.rint(center + offset * normal).astype(int)
        bounded = (xy[:, 0] >= 0) & (xy[:, 0] < disp.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < disp.shape[0])
        safe = np.clip(xy, [0, 0], [disp.shape[1]-1, disp.shape[0]-1])
        inv = (disp[safe[:, 1], safe[:, 0]] - fit[1]) / fit[0]
        good = bounded & valid[safe[:, 1], safe[:, 0]] & np.isfinite(inv) & (inv > 1e-3)
        z = np.full(40, np.nan); z[good] = 1 / inv[good]
        pair = good[1:] & good[:-1]
        q = np.quantile(z[good], [.1, .5, .9]) if good.any() else None
        continuity = np.abs(np.diff(z))[pair] <= np.maximum(.2, .2 * np.maximum(z[1:], z[:-1])[pair])
        profiles[name] = dict(valid_points=int(good.sum()), z_p10_p50_p90_m=q.tolist() if q is not None else None,
                              relative_span=float((q[2]-q[0])/q[1]) if q is not None else None,
                              neighbor_pairs=int(pair.sum()), neighbor_continuity=float(continuity.mean()) if pair.any() else None)
        if name == 'center':
            dirs = np.stack(((xy[:, 0]-K[0, 2])/K[0, 0], (xy[:, 1]-K[1, 2])/K[1, 1], np.ones(40)), axis=1)
            raw = dirs, z, good
    return profiles, raw


def tof_support(raw, dist, Rct, zone_pred):
    dirs, z, good = raw
    td = dirs @ Rct
    slopes = td[:, :2] / td[:, 2, None]
    cells = np.floor((slopes + X.EDGE) / (2*X.EDGE/8)).astype(int)
    inside = (td[:, 2] > 0) & (cells >= 0).all(1) & (cells < 8).all(1)
    zone = cells[:, 1]*8 + cells[:, 0]
    ranges = z * np.linalg.norm(dirs, axis=1)
    rows = []
    for j in sorted(set(zone[inside].tolist())):
        keep = inside & good & (zone == j)
        if not keep.any(): continue
        pred = float(np.median(ranges[keep])); observed = float(dist[j])
        usable = np.isfinite(observed) and .3 < observed < 4
        tolerance = max(.2, .2 * observed) if usable else None
        error = abs(pred-observed) if usable else None
        rows.append(dict(zone=j, line_samples=int(keep.sum()), predicted_range_m=pred,
                         firmware_range_m=observed if usable else None,
                         abs_error_m=error, tolerance_m=tolerance,
                         agrees=bool(error <= tolerance) if usable else None,
                         footprint_predicted_range_m=float(zone_pred[j]) if np.isfinite(zone_pred[j]) else None))
    usable = [r for r in rows if r['agrees'] is not None]
    agree = sum(r['agrees'] for r in usable)
    status = 'UNSUPPORTED' if len(usable) < 3 else 'CONSISTENT' if agree == len(usable) else 'INCONSISTENT'
    return dict(status=status, intersected_zones=len(set(zone[inside].tolist())), valid_zones=len(usable), agreeing_zones=agree,
                line_samples_in_tof_fov=int(inside.sum()), zones=rows)


def frame_quality(r):
    accepted = ((r['anchor'] == 'tof' and r.get('quality', {}).get('accepted', False)) or
                (r['anchor'] == 'robust_tof' and r.get('robust_diagnostic', {}).get('accepted', False)))
    return dict(anchor=r['anchor'], accepted=bool(accepted), quality=r.get('quality', {}),
                robust_diagnostic=r.get('robust_diagnostic', {}))


def checks():
    K = np.array([[100.,0,50.],[0,100.,50.],[0,0,1.]])
    c = dict(x1=50,y1=20,x2=50,y2=80)
    profiles, raw = strip_sample(c, np.ones((100,100)), np.ones((100,100),bool), [1,0], K)
    assert profiles['center']['relative_span'] == 0 and profiles['center']['neighbor_continuity'] == 1
    # Exact synthetic ranges per covered zone and orthogonal range invariance.
    empty = tof_support(raw, np.full(64,np.nan), np.eye(3), np.ones(64))
    assert empty['status'] == 'UNSUPPORTED'
    dist = np.ones(64)
    for r in empty['zones']: dist[r['zone']] = r['predicted_range_m']
    matched = tof_support(raw, dist, np.eye(3), np.ones(64))
    assert matched['valid_zones'] >= 3 and matched['status'] == 'CONSISTENT'
    assert tof_support(raw, dist*2, np.eye(3), np.ones(64))['status'] == 'INCONSISTENT'
    assert not frame_quality(dict(anchor='reused',quality={'accepted':True}))['accepted']
    assert frame_quality(dict(anchor='robust_tof',robust_diagnostic={'accepted':True}))['accepted']
    return dict(constant_depth_profile=True, synthetic_tof_agreement=True, unsupported_not_negative=True, reused_not_accepted=True)


def main():
    start = time.perf_counter(); verified = checks(); OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CACHE/'cache_manifest.json').read_text())
    fitdata = json.loads((CACHE/'robust_readout.json').read_text())
    meta = {(k,r['i']):r for k,s in manifest['sessions'].items() for r in s['frames']}
    fits = {(k,r['i']):r for k,s in fitdata['sessions'].items() for r in s['arms']['robust_replace']}
    K = AtomCamera().K_upright
    Rct = np.asarray(json.loads((X.OUT/'camera_tof_extrinsic.json').read_text())['rotation'])
    rays = X.zone_rays(); totals = Counter(); sessions = {}; fingerprint = hashlib.sha256()
    with (SOURCE/'camera_chain_frames.jsonl').open() as src, (SOURCE/'geometry_frames.jsonl').open() as geo, (OUT/'frames.jsonl').open('w') as out:
        for line, gline in zip(src, geo, strict=True):
            if time.perf_counter()-start > 270: raise TimeoutError('300 second command wall reserve')
            row = json.loads(line); gr = json.loads(gline); key = row['key']; i = row['i']
            assert (key,i) == (gr['key'],gr['i']) and len(row['candidates']) == len(gr['candidates'])
            m = meta[key,i]; f = fits[key,i]; fq = frame_quality(f); counts = sessions.setdefault(key,Counter())
            counts['frames'] += 1; counts['fit_accepted_frames'] += fq['accepted']
            disp = valid = None; zp = np.full(64,np.nan)
            fit = f['fit']
            if m.get('cache') and fit is not None and np.isfinite(fit).all() and fit[0] > 1e-12:
                with np.load(CACHE/m['cache']) as p: disp,valid = p['disp'],p['valid']
                dz,zfac = S.zone_disparity(disp,K,Rct,rays)
                inv = (dz-fit[1])/fit[0]; ok = np.isfinite(inv)&(inv>1e-3)&np.isfinite(zfac)&(zfac>0)
                zp[ok] = 1/(inv[ok]*zfac[ok])
            for c,gc in zip(row['candidates'], gr['candidates'],strict=True):
                for n in ('x1','y1','x2','y2','rank'): assert c[n] == gc[n]
                fingerprint.update(json.dumps([key,i,*[c[n] for n in ('x1','y1','x2','y2','rank')]]).encode())
                c['geometry'] = gc['geometry']; g = c['geometry']['robust_replace']
                local = None; tof = dict(status='UNSUPPORTED',intersected_zones=0,valid_zones=0,agreeing_zones=0,zones=[])
                if disp is not None:
                    local,raw = strip_sample(c,disp,valid,fit,K)
                    tof = tof_support(raw,np.asarray(m['dist']),Rct,zp)
                range_ok = g['status']=='AVAILABLE' and g['domain_status']!='BELOW_NEAR_MODEL_RANGE'
                stable_depth = local is not None and local['center']['valid_points']>=20 and local['center']['relative_span']<=.5
                arms = dict(range_only=range_ok,fit_gate=range_ok and fq['accepted'],
                            tof_agreement=range_ok and tof['status']=='CONSISTENT',local_consistency=range_ok and stable_depth)
                raw_rules = dict(fit_accepted=fq['accepted'], tof_consistent=tof['status']=='CONSISTENT',
                                 local_consistent=stable_depth)
                c['confidence'] = dict(frame_fit=fq,local=local,tof=tof,raw_rules=raw_rules,arms=arms)
                counts['candidates'] += 1; counts['tof_'+tof['status']] += 1
                for arm,v in arms.items():
                    counts[arm+'_candidates'] += v
                    counts[arm+'_stable'] += v and c['stable']
                    counts[arm+'_stable_box'] += v and c['stable'] and any(x is True for x in g['query'])
            out.write(json.dumps(row,allow_nan=False)+'\n')
    for c in sessions.values(): totals.update(c)
    assert totals['frames']==1677 and len(sessions)==18
    result = dict(evidence='Consumed Development diagnostic; ToF consistency reuses scale anchors, not independent metric truth; thin targets need not dominate a zone return.',
                  rules=dict(range_only='AVAILABLE and not BELOW_NEAR_MODEL_RANGE; far kept for confidence, original query unchanged',
                             fit_gate='range_only AND accepted fresh tof/robust_tof fit',
                             tof_agreement='range_only AND >=3 valid intersected zones AND every zone abs error <= max(0.2m,20% firmware range)',
                             local_consistency='range_only AND >=20/40 center valid AND (p90-p10)/p50 <= 0.5; sides descriptive'),
                  sessions=sessions,totals=totals,checks=verified,candidate_identity_sha256=fingerprint.hexdigest(),wall_s=time.perf_counter()-start)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))


if __name__ == '__main__': main()
