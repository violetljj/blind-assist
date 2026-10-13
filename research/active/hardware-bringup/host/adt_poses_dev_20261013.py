"""Per prepared ADT frame: camera position and horizontal heading in the scene frame (run in the projectaria venv).

Needed to place predicted obstacle blobs in a world-fixed frame (motion-consistency rules). The corridor frame of frame i
is: origin = camera position, y = scene -Y (down), z = heading (horizontal part of the camera optical axis),
x = y x z -- exactly corridor_from_scene in adt_prep_dev_20261013.
Output: <prep>/<sequence>/poses.npz (pos [N,3], fwd [N,3], speed [N] horizontal m/s over +-0.5 s).
Usage: adt_poses_dev_20261013.py <adt data dir> <prep dir>
"""
import sys
from pathlib import Path

import numpy as np
from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider, AriaDigitalTwinDataProvider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration as C

data, prep = Path(sys.argv[1]), Path(sys.argv[2])
for seq in sorted(p for p in prep.iterdir() if (p / 'frames.npz').exists() and not (p / 'poses.npz').exists()):
    if not (data / seq.name).exists():
        print('skip (no raw data)', seq.name); continue
    dp = AriaDigitalTwinDataProvider(AriaDigitalTwinDataPathsProvider(str(data / seq.name)).get_datapaths())
    raw = dp.get_aria_camera_calibration(StreamId('214-1'))
    lin = C.rotate_camera_calib_cw90deg(C.get_linear_camera_calibration(640, 480, 425.0, 'pinhole', raw.get_transform_device_camera()))
    T_dc = lin.get_transform_device_camera()
    with np.load(seq / 'frames.npz') as Z:
        ts = Z['t_ns']
    pos, fwd = [], []
    for t in ts:
        M = (dp.get_aria_3d_pose_by_timestamp_ns(int(t)).data().transform_scene_device @ T_dc).to_matrix()
        f = M[:3, 2].copy(); f[1] = 0; f /= np.linalg.norm(f)
        pos.append(M[:3, 3]); fwd.append(f)
    pos, fwd = np.array(pos), np.array(fwd); tt = ts / 1e9; speed = np.full(len(ts), np.nan)
    for i in range(len(ts)):
        a, b = np.searchsorted(tt, tt[i] - .5), min(np.searchsorted(tt, tt[i] + .5, 'right') - 1, len(ts) - 1)
        if b > a:
            d = pos[b] - pos[a]; d[1] = 0; speed[i] = np.linalg.norm(d) / (tt[b] - tt[a])
    np.savez(seq / 'poses.npz', pos=pos, fwd=fwd, speed=speed)
    print(seq.name, len(ts), 'walking>0.3: %.0f%%' % (100 * np.nanmean(speed > .3)), flush=True)
