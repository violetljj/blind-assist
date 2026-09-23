"""Ideal ray hints and fixed upstream VPP adapter; no evaluator access."""
import hashlib
import importlib.util
import math
from pathlib import Path
import numpy as np
from numba import njit


@njit
def seed_numba(seed):
    np.random.seed(seed)


def frame_seed(panel, frame):
    return int.from_bytes(hashlib.sha256((panel+'/'+frame).encode()).digest()[:4], 'little') % (2**31)


def ray_hints(ranges, valid, rig):
    """Radial ranges at known centre directions, colocated with left camera."""
    assert rig['width'] == 640 and rig['height'] == 360 and rig['tof_hfov_deg'] == 45
    r, valid = np.asarray(ranges).reshape(64), np.asarray(valid).reshape(64).astype(bool)
    f = 320/math.tan(math.radians(rig['hfov_deg']/2))
    disparity = np.zeros((360, 640), np.float32)
    seeds = []
    for zone in range(64):
        if not (valid[zone] and np.isfinite(r[zone]) and .5 <= r[zone] <= 4.): continue
        row, col = divmod(zone, 8)
        az = math.radians(-22.5+(col+.5)*45/8)
        el = math.radians(22.5-(row+.5)*45/8)
        direction = np.array([1., math.tan(az), math.tan(el)])
        xyz = direction/np.linalg.norm(direction)*float(r[zone])
        uf, vf = 320+f*xyz[1]/xyz[0], 180-f*xyz[2]/xyz[0]
        u, v = int(round(uf)), int(round(vf))
        d = f*rig['baseline_m']/xyz[0]
        if not (0 <= u < 640 and 0 <= v < 360 and 0 <= u-d <= 639): continue
        assert disparity[v, u] == 0, 'Unexpected centre-ray raster collision'
        disparity[v, u] = d
        seeds.append(dict(zone=zone,u=u,v=v,u_float=float(uf),v_float=float(vf),
            disparity=float(disparity[v,u]),range_m=float(r[zone]),axial_z_m=float(xyz[0]),xyz_camera=xyz.tolist()))
    return disparity, seeds


class Projector:
    def __init__(self, source, digest):
        path = Path(source)/'vpp_standalone.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        spec = importlib.util.spec_from_file_location('ba_frozen_upstream_vpp', path)
        self.module = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.module)

    def apply(self, left, right, hints, seed):
        seed_numba(seed)
        a,b = self.module.vpp(left, right, hints, wsize=3, blending=.4,
            method='rnd', uniform_color=False, interpolate=True, left2right=True,
            use_distance_patch=False, use_bilateral_patch=False, g_occ=None)
        if not np.any(hints):
            assert np.array_equal(a,left) and np.array_equal(b,right)
        return a,b
