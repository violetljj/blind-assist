"""Are in-view small targets signal-limited? (Development diagnostic, fast lane.)

For every audit positive that is visible in the current frame, rebuild the expected
H3 contribution of only the rays whose first hit is a label-causing object inside
the query box (same sensor model and SNR-tier gain as the observations, noise off),
then compare it with the public noise level v = 16*ambient + max(bias, 0):
  z_mf   sqrt(sum s^2/v): known-template matched filter, an upper bound for any
         single-frame detector of that return (ignores the occluded background);
  z_mf4, z_mf8  the same over the current and previous 3 / 7 frames (static world,
         object part inside the current box), the budget of S2 / S3 integration;
  z_win  best 1x1 / 2x2-zone, 1- or 2-bin window sum, a simple search statistic.
Saved v3 scores (S2@0.75, S3) are compared at the frozen calib thresholds.
Evaluator-only oracle; not a readout.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from cnh_route_sensor import SensorParameters, angular_rays, synthesize_response
from cnh_track_a_readout import BOXES
import cnh_track_a_scale_evaluate as se
import cnh_track_a_v13_sensor as sensor_module

DIRS, W = angular_rays(16)
EDGES = [0, 2, 5, 10, np.inf]


def window_z(s, v):
    best = 0.
    for size in (1, 2):
        S = sum(s[i:8-size+1+i, j:8-size+1+j] for i in range(size) for j in range(size))
        V = sum(v[i:8-size+1+i, j:8-size+1+j] for i in range(size) for j in range(size))
        best = max(best, float((S/np.sqrt(V)).max()), float(((S[..., :-1]+S[..., 1:])/np.sqrt(V[..., :-1]+V[..., 1:])).max()))
    return best


def unit_rows(geometry, sensor, readouts, unit, snr):
    split, records, step = se.unit_records(geometry, sensor, unit, -10, (3, 6, 12).index(snr))
    if split != 'audit':
        return []
    gain = json.loads((sensor/f'unit{unit:02d}-mount-10.json').read_text(encoding='utf-8'))['parameters'][str(snr)]['signal_counts']
    params = replace(SensorParameters(), signal_counts=gain, noise_scale=0.)
    empty = synthesize_response(np.full((8, 8, 256), np.inf), .5, 1., W, params=params, seed=0)['histogram']
    bias = np.load(readouts/'bias.npy')
    data = json.loads((geometry/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
    rho_of = {c['config']: {o['id']: o['rho'] for o in c['objects']} for c in data['configs']}
    with np.load(sensor/f'unit{unit:02d}-mount-10-oracle.npz') as f:
        dist_all, oid_all, cos_all = f['raydistance'], f['object_id'], f['raycos']
    with np.load(sensor/f'unit{unit:02d}-mount-10-observations.npz') as f:
        obs_config = f['config']
    with np.load(readouts/f'unit{unit:02d}.npz') as f:
        s2, mem, strata = f['S2__noisy@0.75'], f['memory__noisy'], f['strata']
        index = {(int(c), int(t)): n for n, (c, t) in enumerate(zip(f['config'], f['frame']))}
    out = []
    for rec in records:
        rows = np.flatnonzero(obs_config == rec['config'])[::step]
        frames = []
        for i, row in enumerate(rows):
            dist, oid, cos = dist_all[row], oid_all[row], cos_all[row]
            lut = rho_of[rec['config']]
            rho = np.full(oid.shape, .5)
            for k in np.unique(oid[oid >= 0]):
                rho[oid == k] = lut.get(int(k), .5)
            pose = rec['poses'][i]
            frames.append(dict(dist=dist, oid=oid, cos=cos, rho=rho, points=pose[:3, 3]+(DIRS@pose[:3, :3].T)*dist[..., None],
                               v=16*rec['ambient'][i][..., None]+np.maximum(bias, 0)))
        for i in range(len(rows)):
            if not rec['main'][i]:
                continue
            n = index[(rec['config'], i)]
            wq = rec['world_from_Q'][i]
            for q, (lo, hi) in enumerate(BOXES):
                ids = rec['contributors'][i][q]
                if rec['labels'][i, q] != 1 or not ids:
                    continue
                z2, now = [], None
                for j in range(i, max(-1, i-8), -1):             # current box, static world, frames i..i-7
                    f = frames[j]
                    q_pts = (f['points']-wq[:3, 3])@wq[:3, :3]
                    hit = np.isin(f['oid'], ids) & (q_pts >= lo).all(-1) & (q_pts <= hi).all(-1)
                    if j == i:
                        if not hit.any():
                            break                                 # invisible now: covered by the memory diagnostic
                        now = hit
                    if not hit.any():
                        z2.append(0.)
                        continue
                    s = synthesize_response(np.where(hit, f['dist'], np.inf), f['rho'], np.where(hit, f['cos'], 0.), W,
                                            params=params, seed=0)['histogram']-empty
                    s = s.reshape(8, 8, 16, 8).sum(-1)
                    z2.append(float((s**2/np.maximum(f['v'], 1e-9)).sum()))
                    if j == i:
                        zwin = window_z(s, f['v'])
                if now is None:
                    continue
                c = np.cumsum(z2)
                out.append(dict(unit=unit, config=rec['config'], frame=i, box=q, stratum=str(strata[n, q]),
                                z_mf=float(np.sqrt(c[0])), z_mf4=float(np.sqrt(c[min(3, len(c)-1)])),
                                z_mf8=float(np.sqrt(c[-1])), z_win=zwin,
                                solid_angle=float((W/W.sum()*now).sum()), range_m=float(np.median(frames[i]['dist'][now])),
                                s2=float(s2[n, q]), s3=float(max(s2[n, q], mem[n, q]))))
    return out


def summarize(root, result_json):
    """S2 recall at the frozen calib threshold, binned by the 4-frame ceiling z_mf4."""
    thr = {g: json.loads(Path(result_json).read_text(encoding='utf-8'))['arms']['S2/noisy'][g]['threshold'] for g in ('HEAD', 'BODY')}
    rows = [r for p in sorted(Path(root).glob('unit*.json')) for r in json.loads(p.read_text(encoding='utf-8'))]
    out = dict(units=len({r['unit'] for r in rows}), S2_threshold=thr, z_bins=[str(e) for e in EDGES])
    for g, boxes in (('HEAD', (0, 2, 4)), ('BODY', (1, 3, 5))):
        res = {}
        for st in ('tiny', 'realistic', 'wide', 'all'):
            rs = [r for r in rows if r['box'] in boxes and (st == 'all' or r['stratum'] == st)]
            if not rs:
                continue
            z1, z4, z8, zw, s2 = (np.array([r[k] for r in rs]) for k in ('z_mf', 'z_mf4', 'z_mf8', 'z_win', 's2'))
            det = s2 >= thr[g]
            bins = []
            for a, b in zip(EDGES[:-1], EDGES[1:]):
                m = (z4 >= a) & (z4 < b)
                bins.append(dict(z_mf4=f'[{a},{b})', n=int(m.sum()), S2_recall=float(det[m].mean()) if m.any() else None))
            res[st] = dict(n=len(rs), S2_recall=float(det.mean()),
                           median_z=dict(single=float(np.median(z1)), four=float(np.median(z4)), eight=float(np.median(z8)),
                                         window=float(np.median(zw))),
                           frac_below=dict(z4_lt_2=float(np.mean(z4 < 2)), z4_lt_5=float(np.mean(z4 < 5)), z8_lt_2=float(np.mean(z8 < 2))),
                           misses=int((~det).sum()), misses_z4_ge_5=int(((~det) & (z4 >= 5)).sum()),
                           misses_z4_lt_2=int(((~det) & (z4 < 2)).sum()), misses_z8_lt_2=int(((~det) & (z8 < 2)).sum()),
                           by_z_mf4=bins)
        out[g] = res
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--sensor', type=Path, required=True)
    p.add_argument('--readouts', type=Path, required=True)
    p.add_argument('--units', type=int, nargs='+', required=True)
    p.add_argument('--family', required=True)
    p.add_argument('--snr', type=int, default=6)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sensor_module.FAMILY = a.family
    a.out.mkdir(parents=True, exist_ok=True)
    for u in a.units:
        target = a.out/f'unit{u}.json'
        if not target.exists() and (a.readouts/f'unit{u:02d}.npz').exists():
            target.write_text(json.dumps(unit_rows(a.geometry, a.sensor, a.readouts, u, a.snr)), encoding='utf-8')
        print(u, flush=True)


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'summarize':
        print(json.dumps(summarize(sys.argv[2], sys.argv[3]), indent=1, ensure_ascii=False))
    else:
        main()
