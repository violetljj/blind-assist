"""Export a consumed-frame geometry parity fixture; does not run an evaluator or change thresholds.

Provide an existing DAV2 disparity .npy in calibrated upright (640,480) resolution and its old
capture session/ToF frame index. The binary fixture is read by AtomRgbGeometryFixtureTest via
BLINDASSIST_ATOM_RGB_FIXTURE. Payloads must stay under artifacts.local. This checks geometry
on the supplied model output, not the model itself, scene accuracy, or fresh confirmation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

from atom_camera import AtomCamera, ROOT

HOST = Path(__file__).parent


def reference_functions(filename, names, namespace):
    """Execute the original function bodies, without loading unrelated model/evaluator modules."""
    tree = ast.parse((HOST / filename).read_text(encoding='utf8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names):
        raise ValueError('Missing reference function')
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), namespace)


class RecordingRng:
    def __init__(self):
        self.rng = np.random.default_rng(20261013)
        self.calls = []

    def choice(self, *args, **kwargs):
        result = self.rng.choice(*args, **kwargs)
        self.calls.append(result)
        return result


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def main(args):
    session = Path(args.session)
    # This tool is intentionally restricted to the already-consumed 2026-10-12 hall batch.
    if not session.name.startswith('capture-20261012T') or session.name[17:23] not in (
            '072853', '072939', '073020', '073138', '073245', '073554', '073635', '073740', '073821'):
        raise ValueError('Only the already-consumed 2026-10-12 hall sessions are admitted')
    out = Path(args.out)
    if not out.resolve().is_relative_to((ROOT / 'artifacts.local').resolve()):
        raise ValueError('Fixture payload must be under artifacts.local')
    tof_rows = [r for r in rows(session / 'tof.index.jsonl') if r.get('valid_frame')]
    record = tof_rows[args.tof_index]
    raw = (session / 'tof.raw.jsonl').read_bytes()
    tof = json.loads(raw[record['raw_byte_offset']:record['raw_byte_offset'] + record['raw_byte_length']])
    rgb = rows(session / 'rgb.jsonl')
    j = min(range(len(rgb)), key=lambda i: abs(rgb[i]['received_elapsed_ns'] - record['received_elapsed_ns']))
    if abs(rgb[j]['received_elapsed_ns'] - record['received_elapsed_ns']) > 1e8:
        raise ValueError('No RGB within frozen 100 ms alignment window')
    imu = {r['index']: r for r in rows(session / 'imu.index.jsonl') if r.get('valid_frame')}
    gravity = imu[record['imu_alignment']['imu_record_index']]['gravity_unit']
    # load_mounting owns the on-disk schema (including any wrapper).
    from atom_imu import load_mounting
    mounting = load_mounting(ROOT / 'artifacts.local/hardware-bringup/atom-imu-v1/imu-mounting-operator-pilot.json')
    t = np.asarray(mounting['tof_from_imu_rotation']) @ gravity
    t /= np.linalg.norm(t)
    pitch, roll = math.degrees(math.asin(t[2])), math.degrees(math.atan2(-t[0], -t[1]))
    cam = AtomCamera(); K = cam.K_upright
    image = cam.undistort_upright(cv2.imread(str(session / rgb[j]['file']))[:, :, ::-1])
    disp = np.asarray(np.load(args.disparity), np.float32)
    h, w = disp.shape
    if (w, h) != cam.upright_size:
        raise ValueError('Disparity must already have calibrated upright output resolution')
    mask = image.sum(-1) > 0
    ext = ROOT / 'artifacts.local/work/atom-camera-calibration-20261012/camera_tof_extrinsic.json'
    rct = np.asarray(json.loads(ext.read_text())['rotation'])
    ns = dict(np=np, math=math, EDGE=math.tan(math.pi / 8))
    reference_functions('atom_tof_extrinsic_dev_20261012.py', ['zone_rays'], ns)
    reference_functions('real_rgb_query_feasibility_dev_20261012.py', ['rot'], ns)
    reference_functions('real_rgb_small_model_dev_20261012.py', ['zone_disparity', 'fit_affine', 'floor_anchor'], ns)
    dz, zfac = ns['zone_disparity'](disp, K, rct, ns['zone_rays']())
    distance = np.asarray(tof['distance_mm'], float)[::-1] / 1000
    status = np.asarray(tof['target_status'])[::-1]
    valid = np.isin(status, (5, 6, 9, 10)) & (distance > .3) & (distance < 4)
    tof_fit = ns['fit_affine'](dz, np.where(valid, 1 / (distance * zfac), np.nan))
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    rng = RecordingRng()
    floor_fit = ns['floor_anchor'](disp, dirs, ns['rot'](pitch, roll), rct, 1.35, rng)
    fit = tof_fit or floor_fit
    counts = [0, 0]
    if fit:
        inv = (disp.reshape(-1) - fit[1]) / fit[0]
        z = np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), np.inf)
        ok = np.isfinite(z) & mask.reshape(-1)
        pts = (ns['rot'](pitch, roll) @ (rct.T @ (dirs[ok] * z[ok, None]).T)).T
        for q, (lo, hi) in enumerate(((-.2, .42), (.42, .9))):
            counts[q] = int(((np.abs(pts[:, 0]) <= .3) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .3) & (pts[:, 2] <= 2)).sum())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        def ints(values): f.write(np.asarray(values, '>i4').tobytes())
        def doubles(values): f.write(np.asarray(values, '>f8').tobytes())
        def optional_fit(value):
            f.write(struct.pack('>?', value is not None))
            if value is not None: doubles(value)
        ints([0x41524731, w, h])
        doubles([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        doubles(rct.ravel()); doubles([pitch, roll])
        f.write(disp.astype('>f4').tobytes()); f.write(mask.astype('u1').tobytes())
        doubles(tof['distance_mm']); ints(tof['target_status'])
        doubles(dz); doubles(zfac); optional_fit(tof_fit); optional_fit(floor_fit)
        ints([len(rng.calls[0]) if rng.calls else 0])
        if rng.calls: ints(rng.calls[0]); ints(np.asarray(rng.calls[1:]).ravel())
        ints(counts)
    receipt = dict(evidence='consumed Development frame; geometry parity only', session=session.name,
                   tof_index=args.tof_index, rgb_index=j, rgb_file=rgb[j]['file'],
                   disparity=str(Path(args.disparity)), disparity_sha256=hashlib.sha256(Path(args.disparity).read_bytes()).hexdigest(),
                   fixture_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), tof_fit=tof_fit, floor_fit=floor_fit,
                   pixels=counts, floor_samples=len(rng.calls[0]) if rng.calls else 0,
                   source_sha256={name: hashlib.sha256((HOST / name).read_bytes()).hexdigest() for name in (
                       'real_rgb_small_model_dev_20261012.py', 'real_rgb_query_feasibility_dev_20261012.py',
                       'atom_tof_extrinsic_dev_20261012.py')})
    out.with_suffix('.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--disparity', required=True); p.add_argument('--session', required=True)
    p.add_argument('--tof-index', type=int, required=True); p.add_argument('--out', required=True)
    main(p.parse_args())
