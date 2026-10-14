"""Full scene<-camera poses for the frames already prepared by adt_prep_dev_20261013.py (projectaria-tools venv).

The 2026-10-13 prep stored only the corridor rotation and camera height; the cane-reach labels need translations
to fuse neighbouring frames. Same camera definition (linear 640x480 pinhole rotated CW 90 deg) and the same
timestamps (frames.npz t_ns); no image decoding. Output: <prep>/<sequence>/poses.npz (T_sc [N,4,4], t_ns).
Usage: adt_poses_dev_20261015.py <adt data dir> <prep dir>
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

F, W_RAW, H_RAW = 425.0, 640, 480


def poses(args):
    seq_dir, prep_seq = args
    from projectaria_tools.core import calibration as C
    from projectaria_tools.core.stream_id import StreamId
    from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider, AriaDigitalTwinDataProvider
    dp = AriaDigitalTwinDataProvider(AriaDigitalTwinDataPathsProvider(str(seq_dir)).get_datapaths())
    raw = dp.get_aria_camera_calibration(StreamId('214-1'))
    lin = C.get_linear_camera_calibration(W_RAW, H_RAW, F, 'pinhole', raw.get_transform_device_camera())
    T_dc = C.rotate_camera_calib_cw90deg(lin).get_transform_device_camera()
    with np.load(prep_seq / 'frames.npz') as Z:
        ts = Z['t_ns']
    out = np.stack([(dp.get_aria_3d_pose_by_timestamp_ns(int(t)).data().transform_scene_device @ T_dc).to_matrix() for t in ts])
    np.savez(prep_seq / 'poses.npz', T_sc=out, t_ns=ts)
    return prep_seq.name, len(ts)


if __name__ == '__main__':
    data, prep = Path(sys.argv[1]), Path(sys.argv[2])
    jobs = [(data / p.name, p) for p in sorted(prep.iterdir()) if (p / 'frames.npz').exists() and (data / p.name).exists()]
    with ProcessPoolExecutor(min(13, len(jobs))) as ex:
        for name, n in ex.map(poses, jobs):
            print(name, n, flush=True)
