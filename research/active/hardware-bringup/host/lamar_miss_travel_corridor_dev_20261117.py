"""Post-hoc (descriptive, ideal upper bound) for LAMAR_MISS_BREAKDOWN_DEV_20261117.md: for 'direction_offset' time-bands,
would the current single ToF frame already reach the solid-angle threshold if the corridor followed the (future-derived)
travel direction instead of the head heading? Also: is the travel corridor inside the ToF field of view?"""
import json, sys
from pathlib import Path
import numpy as np
from arkit_selective_heading_dev_20261116 import HALF_WIDTH, MIN_SOLID_ANGLE, samples
from lamar_native_corridor_dev_20261116 import DepthSensor
from lamar_navvis_corridor_dev_20261116 import MAX_RANGE, NavVis, walk_windows
from lamar_miss_breakdown_dev_20261117 import band_masks, in_fov

work, out = Path(sys.argv[1]), Path(sys.argv[2])
bd = json.loads((out / 'breakdown.json').read_text(encoding='utf-8'))
geom = json.loads((work / 'screen-hololens/summary.json').read_text(encoding='utf-8'))
rows = {(v['visit'], r['timestamp']): r for v in geom['visits'] for r in v['rows']}
with np.load(work / 'screen-hololens/cropped_mesh.npz', allow_pickle=False) as m:
    scan = NavVis(m['vertices'], m['faces'])
cache, res = {}, []
for rec in bd['records']:
    if rec['reason'] != 'direction_offset':
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
    row = rows[(s, rec['timestamp'])]; _, i, _, _, head, travel, yaw = win[row['depth_row']]
    pose = data['T_wc'][i] @ sensor.head_from_depth; origin = data['T_wc'][i, :3, 3]
    native = frames[row['depth_row']].astype(np.float32) * .001
    sub = (sensor.valid_pixel & np.isfinite(sensor.weights))[::2, ::2]
    rays = sensor.rays[::2, ::2][sub] @ pose[:3, :3].T; w = 4. * sensor.weights[::2, ::2][sub]
    ref = scan.distance(pose[:3, 3], rays); hit = (ref > .08) & (ref < MAX_RANGE)
    b = rec['band']
    rb = band_masks(pose[:3, 3] + rays * np.where(hit, ref, 0.)[:, None], origin, travel, HALF_WIDTH)[b] & hit
    ok = sensor.valid(native)[::2, ::2][sub]
    mb = band_masks(pose[:3, 3] + rays * native[::2, ::2][sub][:, None], origin, travel, HALF_WIDTH)[b] & ok
    res.append(dict(session=s, timestamp=rec['timestamp'], band=b, detected=rec['detected'], yaw_deg=float(yaw),
                    travel_fov_share=float(in_fov(sensor, pose, samples(origin, travel, HALF_WIDTH)[b]).mean()),
                    V_ref_travel=float(w[rb].sum()), V_meas_travel=float(w[mb].sum()),
                    meas_reaches_threshold=bool(w[mb].sum() >= MIN_SOLID_ANGLE)))
(out / 'direction_offset_travel.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
for flag, label in ((False, '漏报'), (True, '已检出')):
    r = [x for x in res if x['detected'] == flag]
    if r:
        print(label, '时刻带', len(r), '| 当帧实测在行进走廊达标', sum(x['meas_reaches_threshold'] for x in r),
              '| 行进走廊采样点在视场内比例中位 %.2f' % np.median([x['travel_fov_share'] for x in r]),
              '| 头-行进夹角中位 %.1f°' % np.median([x['yaw_deg'] for x in r]))
for x in res:
    if not x['detected']:
        print(' ', x['session'][-12:], x['timestamp'][-6:], '带', x['band'], '夹角 %.0f°' % x['yaw_deg'], '视场内 %.2f' % x['travel_fov_share'],
              '参考可见 %.4f' % x['V_ref_travel'], '实测 %.4f' % x['V_meas_travel'], '达标' if x['meas_reaches_threshold'] else '未达标')
