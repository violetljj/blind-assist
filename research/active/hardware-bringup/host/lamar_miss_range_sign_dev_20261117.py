"""Post-hoc (descriptive) for LAMAR_MISS_BREAKDOWN_DEV_20261117.md: sign of measured - reference range on the
reference pixels of each 'range_error' time-band (farther = ToF saw through where the scan has a surface)."""
import json, sys
from pathlib import Path
import numpy as np
from arkit_selective_heading_dev_20261116 import HALF_WIDTH
from lamar_native_corridor_dev_20261116 import DepthSensor
from lamar_navvis_corridor_dev_20261116 import MAX_RANGE, NavVis, walk_windows
from lamar_miss_breakdown_dev_20261117 import band_masks

work, out = Path(sys.argv[1]), Path(sys.argv[2])
bd = json.loads((out / 'breakdown.json').read_text(encoding='utf-8'))
geom = json.loads((work / 'screen-hololens/summary.json').read_text(encoding='utf-8'))
rows = {(v['visit'], r['timestamp']): r for v in geom['visits'] for r in v['rows']}
with np.load(work / 'screen-hololens/cropped_mesh.npz', allow_pickle=False) as m:
    scan = NavVis(m['vertices'], m['faces'])
cache, res = {}, []
for rec in bd['records']:
    if rec['reason'] != 'range_error':
        continue
    s = rec['session']
    if s not in cache:
        with np.load(work / 'hololens/CAB/prepared' / f'{s}.npz', allow_pickle=False) as src:
            data = {k: src[k] for k in ('ts', 'timestamp_us', 'T_wc', 'T_local', 'component', 'ref_gap_s', 'depth_index')}
            sensor = DepthSensor(src['depth_lut'], src['T_depth_to_head'])
        with np.load(work / 'hololens/CAB/native_depth' / f'{s}.npz', allow_pickle=False) as nat:
            frames = dict(zip(nat['depth_rows'].tolist(), nat['depth_mm']))
        cache[s] = (data, sensor, {r[0]: r for r in walk_windows(data)[0]}, frames)
    data, sensor, win, frames = cache[s]
    row = rows[(s, rec['timestamp'])]; _, i, _, _, head, _, _ = win[row['depth_row']]
    pose = data['T_wc'][i] @ sensor.head_from_depth; origin = data['T_wc'][i, :3, 3]
    native = frames[row['depth_row']].astype(np.float32) * .001
    sub = (sensor.valid_pixel & np.isfinite(sensor.weights))[::2, ::2]
    rays = sensor.rays[::2, ::2][sub] @ pose[:3, :3].T; w = 4. * sensor.weights[::2, ::2][sub]
    ref = scan.distance(pose[:3, 3], rays); hit = (ref > .08) & (ref < MAX_RANGE)
    rb = band_masks(pose[:3, 3] + rays * np.where(hit, ref, 0.)[:, None], origin, head, HALF_WIDTH)[rec['band']] & hit
    meas = native[::2, ::2][sub]; ok = sensor.valid(native)[::2, ::2][sub]
    bad = rb & ok & (np.abs(meas - ref) > np.maximum(.10, .05 * ref))
    d = (meas - ref)[bad]
    res.append(dict(session=s, timestamp=rec['timestamp'], band=rec['band'], detected=rec['detected'],
                    farther_share=float(w[bad][d > 0].sum() / w[bad].sum()) if bad.any() else None,
                    median_diff_m=float(np.median(d)) if len(d) else None, ref_median_m=float(np.median(ref[rb]))))
(out / 'range_error_sign.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
for r in res:
    print(r['session'][-12:], r['timestamp'][-6:], '带', r['band'], '已检出' if r['detected'] else '漏报',
          '更远占比 %.2f' % r['farther_share'], '差值中位 %.2f m' % r['median_diff_m'], '参考距离中位 %.2f m' % r['ref_median_m'])
