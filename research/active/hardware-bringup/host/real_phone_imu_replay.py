"""Robust ToF stack on phone-captured CNH+IMU segments with measured pose.

Development description. Segments are the 2026-10-12 phone captures (Atom IMU,
operator-pilot mounting); scene labels are Claude's reading of the RGB frames
and ToF distances, not user notes or ground truth. Frozen robust-stack models
and cuts are unchanged; only the public pose (pitch, roll, sensor height) varies.

Pose arms per segment:
  imu        segment-mean IMU pitch/roll, sensor at the head-height origin
  imu_h      as imu, sensor placed (1.65 - floor-fit height) below the origin
  floor_h    floor-fit pitch/roll and height (segments with a fitted floor)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import real_pilot_calibration as P
from atom_imu import load_mounting
from local_signed_bias_adapter import ContractReplay
from real_contract_v5 import adapter_v5

ROOT = P.ROOT
RAW = ROOT / 'artifacts.local/work/phone-imu-capture-20261012/raw'
OUT = ROOT / 'artifacts.local/work/phone-imu-capture-20261012'
MANIFEST = ROOT / 'artifacts.local/work/cnh-robust-stack-dev-20261012/real_manifest.json'
MOUNTING = ROOT / 'artifacts.local/hardware-bringup/atom-imu-v1/imu-mounting-operator-pilot.json'
SIM_FLOOR_M = 1.65  # floor slab depth below the sensor in the simulator background
# Claude's reading of RGB + ToF (start time CST -> scene, expectation).
LABELS = {
    '100702': ('hall empty, level', 'quiet'),
    '103924': ('hall empty, level', 'quiet'),
    '100754': ('hall empty, looking down', 'quiet'),
    '112815': ('pole + holder centre ~0.8 m, A', 'alarm'),
    '112845': ('pole right edge ~0.8 m (lateral ~0.3 m), A', 'near-pass'),
    '112911': ('pole left edge ~0.75 m (lateral ~0.3 m), A', 'near-pass'),
    '112946': ('pole left edge ~0.8 m, B', 'near-pass'),
    '113029': ('pole + holder centre ~0.7 m, B', 'alarm'),
    '113101': ('pole right edge ~0.7 m, B', 'near-pass'),
    '113143': ('holder walks in from 3.3 m and back, A', 'alarm when near'),
    '113214': ('holder walks in and back, B', 'alarm when near'),
    '104239': ('white wall 2.3 m', 'alarm'),
    '104500': ('white wall 0.5 m', 'alarm'),
    '104020': ('pillar 1.9 m', 'alarm'),
    '103707': ('office chair 0.8 m', 'alarm'),
    '104543': ('corridor, wall on left 1.2-2.8 m', 'UNKNOWN'),
    '104652': ('corridor, looking slightly down', 'UNKNOWN'),
    '104620': ('corridor, looking up at ceiling', 'UNKNOWN'),
}

EDGE = math.tan(math.pi / 8)
_c = (np.arange(8) + .5) / 8 * 2 * EDGE - EDGE
_U = np.stack(list(np.meshgrid(_c, _c, indexing='xy')) + [np.ones((8, 8))], -1)
_U /= np.linalg.norm(_U, axis=-1, keepdims=True)


def lines(path):
    with open(path, encoding='utf8') as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def load(directory, rotation):
    """Valid CNH frames in model zone order handled later; raw order kept here."""
    imu = {r['index']: r for r in lines(directory / 'imu.index.jsonl') if r.get('valid_frame')}
    raw = (directory / 'tof.raw.jsonl').read_bytes()
    H, A, D, S, seq, ms, pitch, roll = [], [], [], [], [], [], [], []
    for r in lines(directory / 'tof.index.jsonl'):
        if not r.get('valid_frame'):
            continue
        f = json.loads(raw[r['raw_byte_offset']:r['raw_byte_offset'] + r['raw_byte_length']])
        H.append(np.ldexp(np.asarray(f['hist_raw'], float), -np.asarray(f['hist_scaler'], int)))
        A.append(np.ldexp(np.asarray(f['ambient_raw'], float), -np.asarray(f['ambient_scaler'], int)))
        D.append(f['distance_mm']); S.append(f['target_status']); seq.append(f['seq']); ms.append(f['ms'])
        g = (imu.get((r.get('imu_alignment') or {}).get('imu_record_index')) or {}).get('gravity_unit')
        if g is None:
            pitch.append(np.nan); roll.append(np.nan); continue
        t = rotation @ np.asarray(g, float); t /= np.linalg.norm(t)
        pitch.append(math.degrees(math.asin(t[2]))); roll.append(math.degrees(math.atan2(-t[0], -t[1])))
    return dict(H=np.array(H), A=np.array(A), D=np.array(D, float), S=np.array(S), seq=np.array(seq, np.int64),
                ms=np.array(ms, np.int64), pitch=np.array(pitch), roll=np.array(roll))


def floor_fit(d):
    """Median-frame floor plane -> (height m, pitch deg (neg down), roll deg, inliers) or None."""
    dist = np.where(np.isin(d['S'], (5, 6, 9, 10)), d['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1]
    m = np.nanmedian(dist, 0)
    ok = np.isfinite(m) & (m > 600); ok[:4] = False
    if ok.sum() < 8:
        return None
    u, r = _U[ok], m[ok] / 1000

    def res(p):
        cr, sr = math.cos(p[2]), math.sin(p[2])
        y = u[:, 0] * sr + u[:, 1] * cr
        return r * (y * math.cos(p[1]) + u[:, 2] * math.sin(p[1])) - p[0]
    s = least_squares(res, [1., math.radians(20), 0.], loss='soft_l1', f_scale=.03)
    keep = np.abs(res(s.x)) < .05
    if keep.sum() < 8:
        return None
    s = least_squares(lambda p: res(p)[keep], s.x)
    return float(s.x[0]), -math.degrees(s.x[1]), math.degrees(s.x[2]), int(keep.sum())


def summarize(result, distance):
    ready = result['ready']
    out = dict(ready=int(ready.sum()))
    for q, name in enumerate(('HEAD', 'BODY')):
        g = result['S_grade'][ready, q]
        out[name] = dict(strong=float((g == 2).mean()), light=float((g == 1).mean()),
                         m3_median=float(np.median(result['m3_score'][ready, q])), s_median=float(np.median(result['s_score'][ready, q])))
    if ready.any():
        near = np.nanmin(np.where(np.isfinite(distance), distance, np.inf)[:, :, 2:6].reshape(len(distance), -1), 1)[ready]
        anyalert = (result['S_grade'][ready] > 0).any(1)
        out['alert_by_centre_range'] = {k: (float(anyalert[sel].mean()) if sel.any() else None, int(sel.sum()))
                                        for k, sel in (('<1.2m', near < 1200), ('1.2-2m', (near >= 1200) & (near < 2000)), ('>=2m', near >= 2000))}
    return out


def main(only, manifest=MANIFEST, tag=''):
    mounting = load_mounting(MOUNTING)
    rotation = np.array(mounting['tof_from_imu_rotation'])
    arrays = np.load(P.OUT / 'instrument_arrays.npz'); bias = arrays['bias'].reshape(64, 16)
    noise = P.read('instrument_parameters.json')['noise']
    engine = ContractReplay(str(manifest))
    report = {}
    try:
        for directory in sorted(RAW.iterdir()):
            key = f"{int(directory.name[17:19]) + 8:02d}{directory.name[19:23]}"
            if key not in LABELS or (only and key not in only):
                continue
            d = load(directory, rotation)
            z = adapter_v5(d['H'], d['A'], bias, noise, None, True, False)
            dist = np.where(np.isin(d['S'], (5, 6, 9, 10)), d['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1]
            fit = floor_fit(d)
            p, r = float(np.nanmean(d['pitch'])), float(np.nanmean(d['roll']))
            if fit and (abs(fit[1] - p) > 15 or not .5 <= fit[0] <= 2. or abs(fit[2]) > 20):
                fit = None  # a person, pole or wall fitted as floor
            arms = {'imu': (p, r, 0.)}
            if fit:
                arms['imu_h'] = (p, r, SIM_FLOOR_M - fit[0])
                arms['floor_h'] = (fit[1], fit[2], SIM_FLOOR_M - fit[0])
            entry = dict(scene=LABELS[key][0], expect=LABELS[key][1], session=directory.name, frames=len(z),
                         imu_pitch=p, imu_pitch_sd=float(np.nanstd(d['pitch'])), imu_roll=r,
                         floor=dict(zip(('height_m', 'pitch_deg', 'roll_deg', 'inliers'), fit)) if fit else None, arms={})
            for arm, (pitch, roll, dh) in arms.items():
                engine.engine.configure_geometry(pitch, roll, dh)
                entry['arms'][arm] = dict(pose=[pitch, roll, dh], **summarize(engine.infer(z, d['seq'], d['ms']), dist))
                a = entry['arms'][arm]
                print(f"{key} {arm:7s} pitch {pitch:+5.1f} roll {roll:+4.1f} dh {dh:.2f} | HEAD s/l {100*a['HEAD']['strong']:3.0f}/{100*a['HEAD']['light']:3.0f}"
                      f"  BODY s/l {100*a['BODY']['strong']:3.0f}/{100*a['BODY']['light']:3.0f}"
                      f"  m3 H/B {a['HEAD']['m3_median']:5.1f}/{a['BODY']['m3_median']:5.1f} | {LABELS[key][0]} -> {LABELS[key][1]}", flush=True)
            report[key] = entry
    finally:
        engine.close()
    name = f'replay{tag or "_robust"}.json' if not only else f"replay{tag}_{'_'.join(only)}.json"
    (OUT / name).write_text(json.dumps(report, indent=1), encoding='utf8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--only', nargs='*', default=[])
    ap.add_argument('--manifest', default=str(MANIFEST)); ap.add_argument('--tag', default='')
    a = ap.parse_args(); main(a.only, a.manifest, a.tag)
