"""Narrow-FoV replay on LaMAR: does drawing the corridor along the travel direction still help when the native depth is
cropped to the mainline ToF field of view? Rules fixed before the run in LAMAR_NARROW_FOV_DEV_20261117.md.
Usage: lamar_narrow_fov_dev_20261117.py <lamar work dir> <output dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from arkit_selective_heading_dev_20261116 import HALF_WIDTH, MIN_SOLID_ANGLE
from lamar_native_corridor_dev_20261116 import DepthSensor, episode_pairs
from lamar_navvis_corridor_dev_20261116 import walk_windows
from lamar_miss_breakdown_dev_20261117 import band_masks

ARMS = {'A_45_level': (45., 0.), 'B_45_down30': (45., 30.), 'C_80_level': (80., 0.), 'W_native': (None, 0.)}
KINDS = ('travel_occupied', 'head_occupied_travel_empty', 'other_clear')


def fov_mask(sensor, sub, fov_deg, down_deg):
    rays = sensor.rays[::2, ::2][sub] @ sensor.head_from_depth[:3, :3].T   # head frame: x right, y down, z forward
    if fov_deg is None:
        return np.ones(len(rays), bool)
    p = np.deg2rad(down_deg)
    z_ = rays @ np.array([0., np.sin(p), np.cos(p)]); y_ = rays @ np.array([0., np.cos(p), -np.sin(p)]); x_ = rays[:, 0]
    half = np.deg2rad(fov_deg / 2)
    return (z_ > 0) & (np.abs(np.arctan2(x_, z_)) <= half) & (np.abs(np.arctan2(y_, z_)) <= half)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('work', type=Path); ap.add_argument('out', type=Path)
    a = ap.parse_args()
    geom = json.loads((a.work / 'screen-hololens/summary.json').read_text(encoding='utf-8'))
    comp = json.loads((a.work / 'screen-hololens/causal_comparison.json').read_text(encoding='utf-8'))
    have = {(p['session'], str(p['timestamp_us'])) for p in comp['predictions']}
    a.out.mkdir(parents=True, exist_ok=False)
    tally = {f'{arm}|{d}|{m}': {k: [0, 0] for k in KINDS} for arm in ARMS for d in ('head', 'travel') for m in ('single', 'two_of_three')}
    checks = dict(y_axis_world_z=[], missing_history=0, frames=0)
    band_tally = {key: [[0, 0], [0, 0]] for key in tally}   # added after the run (descriptive): per height band, occupied-travel episodes
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
        sub = (sensor.valid_pixel & np.isfinite(sensor.weights))[::2, ::2]
        w = 4. * sensor.weights[::2, ::2][sub]
        fovs = {arm: fov_mask(sensor, sub, *spec) for arm, spec in ARMS.items()}
        cloud_cache = {}

        def cloud(f):
            if f not in cloud_cache:
                j = data['depth_index'][f]; depth = frames[f].astype(np.float32) * .001
                pose = data['T_wc'][j] @ sensor.head_from_depth
                rays = sensor.rays[::2, ::2][sub] @ pose[:3, :3].T
                cloud_cache[f] = (pose[:3, 3] + rays * depth[::2, ::2][sub][:, None], sensor.valid(depth)[::2, ::2][sub])
            return cloud_cache[f]

        def usable(f, ref):
            if f not in frames or data['depth_index'][f] < 0 or not np.isfinite(data['T_wc'][data['depth_index'][f]]).all():
                return False
            j, jr = data['depth_index'][f], data['depth_index'][ref]
            return data['component'][j] == data['component'][jr] and abs(data['ts'][jr] - data['ts'][j]) <= 1.2

        for kind in KINDS:
            for ep in episode_pairs(visit['rows'], kind):
                if not all((name, row['timestamp']) in have for row in ep):
                    continue
                hit = {key: False for key in tally}
                bhit = {key: np.zeros(2, bool) for key in tally}; bpos = np.zeros(2, bool)
                for row in ep:
                    r = row['depth_row']; _, i, _, _, head, travel, _ = win[r]
                    origin = data['T_wc'][i, :3, 3]
                    checks['y_axis_world_z'].append(float(data['T_wc'][i, 2, 1])); checks['frames'] += 1
                    hist = [f for f in (r - 2, r - 1, r) if usable(f, r)]
                    checks['missing_history'] += 3 - len(hist)
                    mask = (np.asarray(row['travel']) == 1) if kind == 'travel_occupied' else np.ones(2, bool)
                    if kind == 'travel_occupied':
                        bpos |= mask
                    for d, vec in (('head', head), ('travel', travel)):
                        per_frame = {}
                        for f in hist:
                            xyz, ok = cloud(f); bands = band_masks(xyz, origin, vec, HALF_WIDTH)
                            per_frame[f] = {arm: np.array([w[bm & ok & fm].sum() >= MIN_SOLID_ANGLE for bm in bands])
                                            for arm, fm in fovs.items()}
                        for arm in ARMS:
                            single = per_frame[r][arm] if r in per_frame else np.zeros(2, bool)
                            votes = sum((per_frame[f][arm] for f in per_frame), np.zeros(2, int))
                            for m, alert in (('single', single), ('two_of_three', votes >= 2)):
                                hit[f'{arm}|{d}|{m}'] |= bool((alert & mask).any())
                                bhit[f'{arm}|{d}|{m}'] |= alert & mask
                for key, v in hit.items():
                    tally[key][kind][0] += int(v); tally[key][kind][1] += 1
                    if kind == 'travel_occupied':
                        for b in range(2):
                            band_tally[key][b][0] += int(bhit[key][b]); band_tally[key][b][1] += int(bpos[b])
    yz = np.array(checks['y_axis_world_z'])
    summary = dict(head_y_axis_world_z_median=float(np.median(yz)), missing_history_frames=checks['missing_history'],
                   rows=checks['frames'], tally=tally, band_tally_descriptive=band_tally)
    (a.out / 'narrow_fov.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print('头坐标 y 轴的世界 z 分量中位（应接近 -1，表示 y 朝下）: %.3f' % summary['head_y_axis_world_z_median'],
          '| 缺历史帧', checks['missing_history'], '/', 3 * checks['frames'])
    print('臂|方向|口径 -> 正前检出 / 头障碍空走廊误报 / 其他空走廊误报')
    for key, t in tally.items():
        bt = band_tally[key]
        print(f'{key:32s}', ' / '.join(f'{t[k][0]}/{t[k][1]}' for k in KINDS), f'| 头部带 {bt[0][0]}/{bt[0][1]} 身体带 {bt[1][0]}/{bt[1][1]}')


if __name__ == '__main__':
    main()
