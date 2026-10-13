"""ASE scenes -> the same upright pinhole frame format as adt_prep_dev_20261013 (run in the projectaria-tools venv).

ASE: Z-up world, 704x704 fisheye RGB, ray depth in mm, 10 Hz trajectory. Every 2nd frame (5 Hz). Ray depth -> z depth in
the fisheye camera, then reprojected like ADT; R_cc from the GT pose with gravity -Z; cam_y holds the camera height (tz).
Usage: ase_prep_dev_20261013.py <scenes dir> <out dir> [n_workers]
"""
import csv, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

STRIDE, F, W_RAW, H_RAW = 2, 425.0, 640, 480


def qmat(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def corridor_from_world(R_wc):
    y = np.array([0., 0., -1.])                                     # ASE world is Z-up
    fwd = R_wc[:, 2] - y * (R_wc[:, 2] @ y); fwd /= np.linalg.norm(fwd)
    return np.stack([np.cross(y, fwd), y, fwd])


def prep(args):
    scene, out = args
    from PIL import Image
    from projectaria_tools.projects import ase
    from projectaria_tools.core import calibration as C
    raw = ase.get_ase_rgb_calibration()
    lin = C.get_linear_camera_calibration(W_RAW, H_RAW, F, 'pinhole', raw.get_transform_device_camera())
    rot = C.rotate_camera_calib_cw90deg(lin); T_dc = raw.get_transform_device_camera().to_matrix()
    n = int(raw.get_image_size()[0]); zfac = np.zeros((n, n))
    for v in range(n):                                              # ray depth -> z depth factor per fisheye pixel
        for u in range(n):
            r = raw.unproject_no_checks(np.array([u + .5, v + .5]))
            zfac[v, u] = r[2] / np.linalg.norm(r)
    rows = list(csv.DictReader(open(scene / 'trajectory.csv')))
    rgbs, deps, Rs, hs, tt = [], [], [], [], []
    for i in range(0, len(rows), STRIDE):
        fr, fd = scene / 'rgb' / f'vignette{i:07d}.jpg', scene / 'depth' / f'depth{i:07d}.png'
        if not (fr.exists() and fd.exists()):
            continue
        r = rows[i]
        T_wd = np.eye(4); T_wd[:3, :3] = qmat([float(r[k]) for k in ('qx_world_device', 'qy_world_device', 'qz_world_device', 'qw_world_device')])
        T_wd[:3, 3] = [float(r[k]) for k in ('tx_world_device', 'ty_world_device', 'tz_world_device')]
        T_wc = T_wd @ T_dc
        im = np.array(Image.open(fr)); d = np.clip(np.round(np.array(Image.open(fd)).astype(np.float32) * zfac), 0, 65535).astype(np.uint16)
        im = C.distort_by_calibration(im, lin, raw); dz = C.distort_depth_by_calibration(d, lin, raw)
        im, dz = np.rot90(im, 3), np.rot90(dz, 3)
        rgbs.append(np.ascontiguousarray(im)); deps.append(np.clip(np.ascontiguousarray(dz), 0, 65535).astype(np.uint16))
        tt.append(int(float(r['tracking_timestamp_us']) * 1000)); hs.append(float(T_wc[2, 3]))
        Rs.append(T_wc[:3, :3])
    # upright camera rotation: rotated calib's T_device_camera
    T_dc_up = rot.get_transform_device_camera().to_matrix()[:3, :3]
    R_dev = [Rw @ T_dc[:3, :3].T for Rw in Rs]                      # world_from_device
    R_cc = [corridor_from_world(Rd @ T_dc_up) @ (Rd @ T_dc_up) for Rd in R_dev]
    o = Path(out) / f'ase_{scene.name}'; o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o / 'frames.npz', rgb=np.array(rgbs), depth=np.array(deps), R_cc=np.array(R_cc), cam_y=np.array(hs), t_ns=np.array(tt))
    meta = dict(sequence=o.name, frames=len(tt), stride=STRIDE, image_size=[int(v) for v in rot.get_image_size()],
                projection_params=[float(v) for v in rot.get_projection_params()], source='ASE (synthetic)')
    (o / 'meta.json').write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == '__main__':
    scenes = sorted((p for p in Path(sys.argv[1]).iterdir() if (p / 'trajectory.csv').exists()), key=lambda p: int(p.name))
    out = Path(sys.argv[2]); nw = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    todo = [s for s in scenes if not (out / f'ase_{s.name}' / 'frames.npz').exists()]
    with ProcessPoolExecutor(nw) as ex:
        for m in ex.map(prep, [(s, out) for s in todo]):
            print(m['sequence'], m['frames'], flush=True)
