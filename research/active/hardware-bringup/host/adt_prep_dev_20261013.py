"""ADT -> Atom-like upright pinhole frames with GT depth and gravity frame (run in the projectaria-tools venv).

Per sequence, every STRIDE-th RGB frame (~5 Hz): Aria RGB fisheye -> linear pinhole (F px, 640x480 in raw orientation)
-> rotated CW 90 deg to an upright 480x640 image; GT depth reprojected the same way (z in metres); the rotation from the
upright camera to a gravity-aligned corridor frame (y down, z = horizontal heading) and the camera height above the floor.
Output: <out>/<sequence>/frames.npz (rgb uint8, depth uint16 mm, R_cc, height, t_ns) and meta.json.
ADT license: non-commercial research; outputs stay in ignored artifacts and are never redistributed.
Usage: adt_prep_dev_20261013.py <adt data dir> <out dir> [sequence ...]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STRIDE, F, W_RAW, H_RAW = 6, 425.0, 640, 480


def corridor_from_scene(R_sc):
    y = np.array([0., -1., 0.])                                    # ADT scene frame is Y-up
    fwd = R_sc[:, 2] - y * (R_sc[:, 2] @ y); fwd /= np.linalg.norm(fwd)
    x = np.cross(y, fwd)
    return np.stack([x, y, fwd])                                    # rows: corridor axes in scene coordinates


def prep(args):
    seq_dir, out = args
    from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider, AriaDigitalTwinDataProvider
    from projectaria_tools.core.stream_id import StreamId
    from projectaria_tools.core import calibration as C
    dp = AriaDigitalTwinDataProvider(AriaDigitalTwinDataPathsProvider(str(seq_dir)).get_datapaths())
    sid = StreamId('214-1'); raw = dp.get_aria_camera_calibration(sid)
    lin = C.get_linear_camera_calibration(W_RAW, H_RAW, F, 'pinhole', raw.get_transform_device_camera())
    rot = C.rotate_camera_calib_cw90deg(lin)
    T_dc = rot.get_transform_device_camera()
    t0, t1 = dp.get_start_time_ns(), dp.get_end_time_ns()
    ts = [t for t in dp.get_aria_device_capture_timestamps_ns(sid) if t0 <= t <= t1][::STRIDE]
    rgbs, deps, Rs, hs, tt = [], [], [], [], []
    for t in ts:
        img = dp.get_aria_image_by_timestamp_ns(t, sid); dep = dp.get_depth_image_by_timestamp_ns(t, sid)
        pose = dp.get_aria_3d_pose_by_timestamp_ns(t)
        if not (img.is_valid() and dep.is_valid() and pose.is_valid()):
            continue
        im = C.distort_by_calibration(img.data().to_numpy_array(), lin, raw)
        dz = C.distort_depth_by_calibration(dep.data().to_numpy_array(), lin, raw)
        im, dz = np.rot90(im, 3), np.rot90(dz, 3)                    # clockwise, matches rotate_camera_calib_cw90deg
        M = (pose.data().transform_scene_device @ T_dc).to_matrix()
        R_sc, p = M[:3, :3], M[:3, 3]
        rgbs.append(np.ascontiguousarray(im)); deps.append(np.ascontiguousarray(dz).astype(np.uint16))
        Rs.append(corridor_from_scene(R_sc) @ R_sc); hs.append(float(p[1])); tt.append(int(t))
    o = Path(out) / Path(seq_dir).name; o.mkdir(parents=True, exist_ok=True)
    K = rot.get_projection_params()
    np.savez_compressed(o / 'frames.npz', rgb=np.array(rgbs), depth=np.array(deps), R_cc=np.array(Rs), cam_y=np.array(hs), t_ns=np.array(tt))
    meta = dict(sequence=Path(seq_dir).name, frames=len(tt), stride=STRIDE, image_size=[int(v) for v in rot.get_image_size()],
                projection_params=[float(v) for v in K], note='cam_y is the camera scene-Y; floor level estimated at eval time')
    (o / 'meta.json').write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == '__main__':
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    seqs = [data / s for s in sys.argv[3:]] or sorted(p for p in data.iterdir() if p.is_dir())
    seqs = [s for s in seqs if (s / '.download_status.json').exists() and not (Path(out) / s.name / 'frames.npz').exists()]
    with ProcessPoolExecutor(min(6, len(seqs) or 1)) as ex:
        for m in ex.map(prep, [(s, out) for s in seqs]):
            print(m['sequence'], m['frames'], m['image_size'], [round(v, 1) for v in m['projection_params']], flush=True)
