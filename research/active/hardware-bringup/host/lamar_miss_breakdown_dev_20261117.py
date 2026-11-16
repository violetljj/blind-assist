"""Break down why the single-frame head-heading baseline missed occupied travel corridors on LaMAR (EXPLORE diagnostic).

Method fixed before the run in LAMAR_MISS_BREAKDOWN_DEV_20261117.md. Reuses the original comparison's labels, per-time
baseline decisions, native ToF, LUT/extrinsics, NavVis mesh and walk windows; changes no baseline, threshold or label.
Usage: lamar_miss_breakdown_dev_20261117.py <lamar work dir> <output dir>
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from arkit_selective_heading_dev_20261116 import BANDS, FAR, HALF_WIDTH, MIN_SOLID_ANGLE, NEAR, samples
from lamar_native_corridor_dev_20261116 import DepthSensor, episode_pairs
from lamar_navvis_corridor_dev_20261116 import MAX_RANGE, NavVis, walk_windows

ORDER = ["direction_offset", "out_of_view", "occluded", "too_near", "no_return", "range_error", "frame_difference",
         "confirmation_delay"]   # funnel order: later = closer to being reported
MIN_TOF = .5


def band_masks(xyz, origin, forward, half_width):
    xy = xyz[:, :2] - origin[:2]
    z = xy @ forward
    x = xy @ np.array([forward[1], -forward[0]])
    y = origin[2] - xyz[:, 2]
    planar = (np.abs(x) <= half_width) & (z >= NEAR) & (z <= FAR)
    return [planar & (y >= lo) & (y < hi) for lo, hi in BANDS]


def in_fov(sensor, pose, pts):
    coords = (pts - pose[:3, 3]) @ pose[:3, :3]
    ratio = coords[:, :2] / np.maximum(coords[:, 2:], .1)
    mismatch, _ = sensor.tree.query(ratio)
    return (coords[:, 2] > .1) & (mismatch <= .008)


def classify(sensor, scan, pose, origin, head, native, head_label, travel_label):
    sub = (sensor.valid_pixel & np.isfinite(sensor.weights))[::2, ::2]
    rays = sensor.rays[::2, ::2][sub] @ pose[:3, :3].T
    w = 4. * sensor.weights[::2, ::2][sub]
    ref = scan.distance(pose[:3, 3], rays)
    hit = (ref > .08) & (ref < MAX_RANGE)
    ref_xyz = pose[:3, 3] + rays * np.where(hit, ref, 0.)[:, None]
    meas = native[::2, ::2][sub]
    meas_ok = sensor.valid(native)[::2, ::2][sub]
    meas_xyz = pose[:3, 3] + rays * meas[:, None]
    ref_bands = band_masks(ref_xyz, origin, head, HALF_WIDTH)
    meas_bands = band_masks(meas_xyz, origin, head, HALF_WIDTH)
    grid = samples(origin, head, HALF_WIDTH)
    out = []
    for b in range(2):
        if travel_label[b] != 1:
            continue
        info = dict(band=b, head_label=int(head_label[b]))
        rb = ref_bands[b] & hit
        info['V_ref'] = float(w[rb].sum())
        agree = meas_ok & (np.abs(meas - ref) <= np.maximum(.10, .05 * ref))
        info['V_ok'] = float(w[rb & agree].sum()); info['V_none'] = float(w[rb & ~meas_ok].sum())
        info['V_near'] = float(w[rb & ~meas_ok & (ref < MIN_TOF)].sum()); info['V_bad'] = float(w[rb & meas_ok & ~agree].sum())
        info['V_meas'] = float(w[meas_bands[b] & meas_ok].sum())
        info['fov_share'] = float(in_fov(sensor, pose, grid[b]).mean())
        if head_label[b] != 1:
            reason = 'direction_offset'
        elif info['V_ref'] < MIN_SOLID_ANGLE:
            reason = 'out_of_view' if info['fov_share'] < .5 else 'occluded'
        elif info['V_ok'] < MIN_SOLID_ANGLE:
            if info['V_near'] > .5 * info['V_none'] and info['V_none'] > 0:
                reason = 'too_near'
            else:
                reason = 'no_return' if info['V_none'] >= info['V_bad'] else 'range_error'
        else:
            reason = 'confirmation_delay' if info['V_meas'] >= MIN_SOLID_ANGLE else 'frame_difference'
        info['reason'] = reason
        out.append(info)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('work', type=Path); ap.add_argument('out', type=Path)
    a = ap.parse_args()
    geom = json.loads((a.work / 'screen-hololens/summary.json').read_text(encoding='utf-8'))
    comp = json.loads((a.work / 'screen-hololens/causal_comparison.json').read_text(encoding='utf-8'))
    base = {(p['session'], str(p['timestamp_us'])): np.array(p['baseline'], bool) for p in comp['predictions']}
    with np.load(a.work / 'screen-hololens/cropped_mesh.npz', allow_pickle=False) as m:
        scan = NavVis(m['vertices'], m['faces'])
    a.out.mkdir(parents=True, exist_ok=False)
    records, episodes = [], []
    for visit in geom['visits']:
        name = visit['visit']; native_path = a.work / 'hololens/CAB/native_depth' / f'{name}.npz'
        if not native_path.exists():
            continue
        with np.load(a.work / 'hololens/CAB/prepared' / f'{name}.npz', allow_pickle=False) as src:
            data = {k: src[k] for k in ('ts', 'timestamp_us', 'T_wc', 'T_local', 'component', 'ref_gap_s', 'depth_index')}
            sensor = DepthSensor(src['depth_lut'], src['T_depth_to_head'])
        win = {r[0]: r for r in walk_windows(data)[0]}
        with np.load(native_path, allow_pickle=False) as nat:
            frames = dict(zip(nat['depth_rows'].tolist(), nat['depth_mm']))
        for k, ep in enumerate(episode_pairs(visit['rows'], 'travel_occupied')):
            keys = [(name, row['timestamp']) for row in ep]
            if not all(key in base for key in keys):
                continue
            detected = any((base[key] & (np.asarray(row['travel']) == 1)).any() for key, row in zip(keys, ep))
            reasons = []
            for row in ep:
                dr = row['depth_row']; _, i, _, _, head, travel, _ = win[dr]
                origin = data['T_wc'][i, :3, 3]
                if not np.allclose(origin.round(4), row['head_origin'], atol=1e-3):
                    raise ValueError(f'origin mismatch {name} {dr}')
                pose = data['T_wc'][i] @ sensor.head_from_depth
                native = frames[dr].astype(np.float32) * .001
                for info in classify(sensor, scan, pose, origin, head, native, row['head'], row['travel']):
                    info.update(session=name, timestamp=row['timestamp'], episode=k, detected=detected,
                                baseline=base[(name, row['timestamp'])].tolist())
                    records.append(info); reasons.append(info['reason'])
            episodes.append(dict(session=name, episode=k, detected=detected, rows=len(ep),
                                 main_reason=max(reasons, key=ORDER.index) if reasons else None,
                                 reasons=dict(Counter(reasons))))
    summary = {}
    for flag, label in ((False, 'missed'), (True, 'detected')):
        eps = [e for e in episodes if e['detected'] == flag]
        summary[label] = dict(episodes=len(eps), main_reason=dict(Counter(e['main_reason'] for e in eps)),
                              time_band_reason=dict(Counter(r['reason'] for r in records if r['detected'] == flag)))
    (a.out / 'breakdown.json').write_text(json.dumps(dict(summary=summary, episodes=episodes, records=records), indent=1),
                                          encoding='utf-8')
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
