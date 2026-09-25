"""Track A v1.3 forward sensor (fast lane), adapted from the unexecuted v1.2 draft.

H3 observations and evaluator-only first-hit geometry are stored separately;
readouts never open the oracle files. calib/audit units are rendered at 10 Hz
(even frames are the 5 Hz stream, same exposures); train units at 5 Hz.
Assumption (ASSUMED): per-frame integration identical at 5 and 10 Hz; hardware
bus throughput DEFERRED_PHASE2. Oracle arrays are written for audit units only.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
from cnh_route_sensor import SensorParameters, angular_rays, synthesize_response, RAW_BIN_M
from cnh_track_a_geometry import raycast
from cnh_track_a_fov import rx, rz

FAMILY = 'cnh-track-a-v13-20260926'
LEVELS = (3, 6, 12)
MOUNTS = (0, -10)


def seed_for(unit, config, purpose, **suffix):
    text = f'{FAMILY}|{unit}|{config}|{purpose}|' + json.dumps(suffix, sort_keys=True, separators=(',', ':'))
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def reference_parameters():
    """G3 sim reference: analytic gain per SNR tier, then one 200-draw check (no retries)."""
    distance = np.full((8, 8, 1), np.inf)
    distance[3, 3, 0] = 2.
    unit = replace(SensorParameters(), signal_counts=1., noise_scale=0.)
    a = synthesize_response(distance, .5, 1., 1., params=unit, seed=0)['histogram'][3, 3]
    b = synthesize_response(np.full_like(distance, np.inf), .5, 1., 1., params=unit, seed=0)['histogram'][3, 3]
    target_bin = int(np.floor(2./RAW_BIN_M))
    window = np.arange(target_bin-3, target_bin+4)
    increment, target_mean, background = float((a-b)[window].sum()), float(a[window].sum()), float(b[window].sum())
    params, rows = {}, []
    for level in LEVELS:
        def snr(g):
            return g*increment/np.sqrt(g*target_mean+2*unit.ambient_counts*len(window))
        lo, hi = 1e-3, 1e7
        for _ in range(40):
            mid = (lo+hi)/2
            lo, hi = (mid, hi) if snr(mid) < level else (lo, mid)
        gain = (lo+hi)/2
        rng = np.random.default_rng(seed_for(-1, -1, 'g3', snr=level))
        expected = a[window]*gain
        samples = (rng.poisson(expected+unit.ambient_counts, size=(200, len(window))) -
                   rng.poisson(unit.ambient_counts, size=(200, len(window)))).sum(1)
        empirical = (float(samples.mean())-background*gain)/float(samples.std(ddof=1))
        rows.append(dict(target_snr=level, signal_counts=gain, analytic_snr=snr(gain), empirical_snr=empirical,
                         draws=200, pass_gate=bool(abs(empirical/level-1) <= .1)))
        params[level] = replace(SensorParameters(), signal_counts=gain)
    return params, dict(pass_gate=all(r['pass_gate'] for r in rows), rows=rows, reference='zone(3,3), 2 m, rho .5, cos 1')


def _h3(raw):
    return np.asarray(raw).reshape(8, 8, 16, 8).sum(-1)


def render_config(c, mount, params, rate, keep_oracle):
    directions, weights = angular_rays(16)
    normalized = weights/weights.sum(-1, keepdims=True)
    objects = c['objects']
    triangles = np.concatenate([np.asarray(o['triangles_world'], float) for o in objects])
    ids = np.concatenate([np.full(len(o['triangles_world']), o['id'], int) for o in objects])
    rho = np.concatenate([np.full(len(o['triangles_world']), o['rho']) for o in objects])
    poses = c['world_from_Q_10hz'] if rate == 10 else c['world_from_Q']
    angles = c['angles_10hz'] if rate == 10 else c['angles']
    n = len(poses)
    obs = dict(hist=np.empty((3, n, 8, 8, 16), np.float32), ambient=np.empty((3, n, 8, 8), np.float32),
               world_from_tof=np.empty((n, 4, 4)), T_Q_tof=np.empty((n, 4, 4)), world_from_Q=np.asarray(poses))
    oracle = dict(noisefreeH3=np.zeros((n, 8, 8, 16)), expected_H3=np.empty((3, n, 8, 8, 16)),
                  raydistance=np.zeros((n, 8, 8, 256), np.float32), object_id=np.full((n, 8, 8, 256), -1, np.int16),
                  raycos=np.zeros((n, 8, 8, 256), np.float32)) if keep_oracle else None
    for t, worldq in enumerate(poses):
        pitch, roll, _ = angles[t]
        qt = np.eye(4)
        qt[:3, :3] = rx(pitch)@rz(roll)@rx(mount)
        wt = np.asarray(worldq)@qt
        hit = raycast(wt[:3, 3], directions@wt[:3, :3].T, triangles, ids, rho)
        obs['world_from_tof'][t], obs['T_Q_tof'][t] = wt, qt
        cos = np.clip(hit['cos'], 0, 1)
        for s, level in enumerate(LEVELS):
            seed = seed_for(c['unit'], c['config'], 'noise', t10=t if rate == 10 else 2*t, mount=mount, snr=level)
            response = synthesize_response(hit['distance'], hit['rho'], cos, weights, params=params[level], seed=seed)
            obs['hist'][s, t] = _h3(response['histogram'])
            obs['ambient'][s, t] = response['ambient']
            if keep_oracle:
                quiet = synthesize_response(hit['distance'], hit['rho'], cos, weights,
                                            params=replace(params[level], noise_scale=0.), seed=seed)
                oracle['expected_H3'][s, t] = _h3(quiet['histogram'])
        if keep_oracle:
            oracle['raydistance'][t] = np.where(hit['valid'], hit['distance'], 0)
            oracle['object_id'][t] = hit['object_id']
            oracle['raycos'][t] = cos
            bins = np.floor(np.where(hit['valid'], hit['distance'], 0)/(8*RAW_BIN_M)).astype(int)
            for y in range(8):
                for x in range(8):
                    good = hit['valid'][y, x] & (bins[y, x] < 16)
                    oracle['noisefreeH3'][t, y, x] = np.bincount(bins[y, x, good], weights=normalized[y, x, good], minlength=16)
    return obs, oracle


def run(geometry, output, unit, mount):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    params, g3 = reference_parameters()
    data = json.loads((Path(geometry)/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
    rate = 5 if data['split'] == 'train' else 10
    keep = data['split'] == 'audit'
    parts, oracles = [], []
    for c in data['configs']:
        obs, oracle = render_config(c, mount, params, rate, keep)
        parts.append(obs)
        oracles.append(oracle)
    n = len(parts[0]['T_Q_tof'])
    stacked = {k: np.concatenate([p[k] for p in parts], axis=1 if k in ('hist', 'ambient') else 0) for k in parts[0]}
    stacked.update(config=np.repeat([c['config'] for c in data['configs']], n), frame=np.tile(np.arange(n), len(parts)),
                   unit=np.full(len(parts)*n, unit), rate=np.array(rate), snr_levels=np.array(LEVELS))
    stem = f'unit{unit:02d}-mount{mount}'
    np.savez_compressed(output/f'{stem}-observations.npz', **stacked)
    if keep:
        np.savez_compressed(output/f'{stem}-oracle.npz',
                            **{k: np.concatenate([o[k] for o in oracles], axis=1 if k == 'expected_H3' else 0) for k in oracles[0]})
    write(output/f'{stem}.json', dict(unit=unit, mount=mount, split=data['split'], rate=rate, frames=len(parts)*n,
                                      G3=g3, parameters={str(k): asdict(v) for k, v in params.items()}, oracle=keep))
    return rate, len(parts)*n


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--unit', type=int, required=True)
    p.add_argument('--mount', type=int, choices=MOUNTS, required=True)
    a = p.parse_args()
    print(json.dumps(dict(unit=a.unit, mount=a.mount, rate_frames=run(a.geometry, a.output, a.unit, a.mount))))
