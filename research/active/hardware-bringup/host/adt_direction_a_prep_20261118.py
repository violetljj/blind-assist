"""ADT -> wide pinhole frames for direction A indoors (DIRECTION_A_INDOOR_ADT_DEV_20261118.md); run in the projectaria-tools venv.

Per sequence, every STRIDE-th RGB frame (30 fps -> 2 Hz): Aria RGB fisheye -> linear pinhole 512x512, F = 256 px
(90 deg x 90 deg), rotated CW 90 deg to upright; GT depth reprojected the same way (mm); gravity "down" in that camera
frame from the ADT ground-truth device pose (scene frame is Y-up); cam_y = camera scene height (for the true floor).
Output: <out>/<sequence>.npz (rgb uint8 RGB, depth uint16 mm, down float32 Nx3, cam_y, t_ns) and <sequence>.json (K, frames).
ADT license: non-commercial research; outputs stay in ignored artifacts, only aggregate numbers are reported.
Usage: adt_direction_a_prep_20261118.py <adt data dir> <out dir> [sequence ...]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STRIDE, F, SIDE = 15, 256.0, 512


def prep(args):
    seq_dir, out = args
    from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider, AriaDigitalTwinDataProvider
    from projectaria_tools.core.stream_id import StreamId
    from projectaria_tools.core import calibration as C
    dp = AriaDigitalTwinDataProvider(AriaDigitalTwinDataPathsProvider(str(seq_dir)).get_datapaths())
    sid = StreamId('214-1'); raw = dp.get_aria_camera_calibration(sid)
    lin = C.get_linear_camera_calibration(SIDE, SIDE, F, 'pinhole', raw.get_transform_device_camera())
    rot = C.rotate_camera_calib_cw90deg(lin); T_dc = rot.get_transform_device_camera()
    t0, t1 = dp.get_start_time_ns(), dp.get_end_time_ns()
    ts = [t for t in dp.get_aria_device_capture_timestamps_ns(sid) if t0 <= t <= t1][::STRIDE]
    rgbs, deps, downs, cys, tt = [], [], [], [], []
    for t in ts:
        img, dep, pose = dp.get_aria_image_by_timestamp_ns(t, sid), dp.get_depth_image_by_timestamp_ns(t, sid), dp.get_aria_3d_pose_by_timestamp_ns(t)
        if not (img.is_valid() and dep.is_valid() and pose.is_valid()):
            continue
        im = np.rot90(C.distort_by_calibration(img.data().to_numpy_array(), lin, raw), 3)
        dz = np.rot90(C.distort_depth_by_calibration(dep.data().to_numpy_array(), lin, raw), 3)
        M = (pose.data().transform_scene_device @ T_dc).to_matrix(); R_sc = M[:3, :3]        # camera -> scene
        rgbs.append(np.ascontiguousarray(im)); deps.append(np.ascontiguousarray(dz).astype(np.uint16))
        downs.append((R_sc.T @ np.array([0., -1., 0.])).astype(np.float32)); cys.append(float(M[1, 3])); tt.append(int(t))
    name = Path(seq_dir).name; o = Path(out); o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o / f'{name}.npz', rgb=np.array(rgbs), depth=np.array(deps), down=np.array(downs), cam_y=np.array(cys), t_ns=np.array(tt))
    fx, fy, cx, cy = [float(v) for v in rot.get_projection_params()][:4]
    meta = dict(sequence=name, frames=len(tt), stride=STRIDE, image_size=[int(v) for v in rot.get_image_size()], K=[fx, fy, cx, cy])
    (o / f'{name}.json').write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == '__main__':
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    seqs = [data / s for s in sys.argv[3:]] or sorted(p for p in data.iterdir() if p.is_dir() and '10s_sample' not in p.name)
    seqs = [s for s in seqs if not (out / f'{s.name}.npz').exists()]
    with ProcessPoolExecutor(min(4, len(seqs) or 1)) as ex:
        for m in ex.map(prep, [(s, out) for s in seqs]):
            print(m['sequence'], m['frames'], m['image_size'], [round(v, 1) for v in m['K']], flush=True)
