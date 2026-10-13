"""Synthetic Android arithmetic oracles from the frozen Python function bodies.

No model, camera, capture, labels or evaluator is loaded. RANSAC draws are recorded and handed
to Kotlin; this does not assert equivalence of Java and NumPy random generators. Binary payloads
stay under artifacts.local and are consumed by AtomRgbNumericOracleTest.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HOST = Path(__file__).resolve().parent
ROOT = HOST.parents[3]
SOURCES = (
    'atom_tof_extrinsic_dev_20261012.py',
    'real_rgb_query_feasibility_dev_20261012.py',
    'real_rgb_small_model_dev_20261012.py',
    'pose_envelope.py',
    'frozen_fusion_eval_v1.py',
    'frozen_fusion_eval_v2.py',
)


def functions(filename, names, namespace):
    tree = ast.parse((HOST / filename).read_text(encoding='utf8'))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError(f'Missing reference functions: {filename}/{names}')
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), namespace)


def readout_code():
    """Use the original frozen readout statements, adding only pixel-count observation."""
    tree = ast.parse((HOST / 'frozen_fusion_eval_v1.py').read_text(encoding='utf8'))
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.If)
               and ast.unparse(n.test) == 'fit is not None']
    if len(matches) != 1:
        raise ValueError('Frozen readout block changed; inspect before adapting this exporter')
    body = copy.deepcopy(matches[0].body)
    loop = next(n for n in body if isinstance(n, ast.For))
    loop.body += ast.parse('counts.append(int(inside.sum()))').body
    return compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                   'frozen_fusion_eval_v1.py:readout', 'exec')


class Draws:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.calls = []

    def choice(self, *args, **kwargs):
        value = self.rng.choice(*args, **kwargs)
        self.calls.append(value)
        return value


class Writer:
    def __init__(self, stream):
        self.stream = stream

    def ints(self, values): self.stream.write(np.asarray(values, '>i4').tobytes())
    def doubles(self, values): self.stream.write(np.asarray(values, '>f8').tobytes())
    def floats(self, values): self.stream.write(np.asarray(values, '>f4').tobytes())
    def flags(self, values): self.stream.write(np.asarray(values, 'u1').tobytes())
    def text(self, value):
        encoded = value.encode('ascii')
        self.stream.write(struct.pack('>H', len(encoded)) + encoded)

    def fit(self, value):
        self.flags([value is not None])
        if value is not None: self.doubles(value)


def main(args):
    out = Path(args.out)
    if not out.resolve().is_relative_to((ROOT / 'artifacts.local').resolve()):
        raise ValueError('Oracle payload must stay under artifacts.local')
    ns = dict(np=np, math=math, EDGE=math.tan(math.pi / 8),
              PITCH_MAX_DEG=10.0, ROLL_MAX_DEG=25.0, K=2, N=3)
    functions('atom_tof_extrinsic_dev_20261012.py', ['zone_rays', 'euler'], ns)
    functions('real_rgb_query_feasibility_dev_20261012.py', ['rot'], ns)
    functions('real_rgb_small_model_dev_20261012.py', ['zone_disparity', 'fit_affine', 'floor_anchor'], ns)
    functions('pose_envelope.py', ['in_envelope'], ns)
    functions('frozen_fusion_eval_v2.py', ['confirm'], ns)
    code = readout_code()
    boxes = {'HEAD': (-.2, .42), 'BODY': (.42, .9)}
    rays = ns['zone_rays']()

    affine = []
    def add_affine(name, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        affine.append(dict(name=name, x=x, y=y, fit=ns['fit_affine'](y, x)))
    x = np.linspace(.25, 2.0, 32)
    y = 3.4 * x - .7
    add_affine('trimmed_outliers', x, y + np.where(np.arange(32) % 9 == 0, 12., 0.))
    add_affine('invalid_observations', np.r_[x, np.nan, np.inf], np.r_[y, 2., 3.])
    add_affine('five_usable', x[:5], y[:5])
    add_affine('negative_slope', x, -y)
    for n in (6, 7, 19, 64):
        add_affine(f'constant_inverse_{n}', np.full(n, .7), np.linspace(1.8, 2.2, n))
    add_affine('rank_one_at_numpy_cutoff', .7 + np.arange(16) * 1e-16, np.linspace(1.8, 2.2, 16))
    add_affine('small_but_resolved_spread', .7 + np.arange(16) * 1e-5, 2 + np.arange(16) * 3e-5)

    cases = []
    width, height = 128, 192
    k = np.array([[140., 0., 61.25], [0., 141., 88.5], [0., 0., 1.]])
    rct = ns['euler'](10.5, 14.5, -1.7)
    vv, uu = np.mgrid[:height, :width]

    def dirs_for(w, h, camera_k):
        v, u = np.mgrid[:h, :w]
        return np.stack([(u - camera_k[0, 2]) / camera_k[0, 0],
                         (v - camera_k[1, 2]) / camera_k[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)

    def readout(c, fit):
        if fit is None: return [0, 0], [False, False]
        env = dict(np=np, fit=fit, disp=c['disp'], dirs=c['dirs'], Rct=c['rct'],
                   F=SimpleNamespace(rot=ns['rot'], BOXES=boxes),
                   x={'pitch': [c['pitch']], 'roll': [c['roll']]}, i=0,
                   up=np.repeat(c['mask'][:, :, None].astype('u1'), 3, axis=2),
                   MIN_PX=150, Z_MAX=2.0, rgb_alert=[False, False], counts=[])
        with np.errstate(invalid='ignore', divide='ignore'):
            exec(code, env)
        pose_ok = bool(ns['in_envelope'](c['pitch'], c['roll']))
        return env['counts'], [bool(v and pose_ok) for v in env['rgb_alert']]

    def add_case(name, disp, *, mask=None, camera_k=k, rotation=rct, pitch=-23., roll=11.,
                 sensor_height=1.35, distance=None, status=None, manual_fit=None,
                 reset=True, process=True):
        d = np.asarray(disp, np.float32)
        h, w = d.shape
        c = dict(name=name, disp=d, mask=np.ones_like(d, bool) if mask is None else mask,
                 k=camera_k, rct=rotation, pitch=pitch, roll=roll, height=sensor_height,
                 dirs=dirs_for(w, h, camera_k), distance=np.zeros(64) if distance is None else distance,
                 status=np.zeros(64, int) if status is None else status, reset=reset, process=process)
        with np.errstate(invalid='ignore', divide='ignore'):
            dz, zfac = ns['zone_disparity'](d, camera_k, rotation, rays)
            ranges = np.asarray(c['distance'])[::-1] / 1000.
            valid = np.isin(np.asarray(c['status'])[::-1], (5, 6, 9, 10)) & (ranges > .3) & (ranges < 4.)
            c['tof_fit'] = ns['fit_affine'](dz, np.where(valid, 1 / (ranges * zfac), np.nan))
            rng = Draws(20261013 + len(cases))
            c['floor_fit'] = ns['floor_anchor'](d, c['dirs'], ns['rot'](pitch, roll), rotation, sensor_height, rng)
        c.update(dz=dz, factor=zfac, draws=rng.calls, readout_fit=manual_fit or c['tof_fit'] or c['floor_fit'])
        c['pixels'], c['alarms'] = readout(c, c['readout_fit'])
        cases.append(c)
        return c

    disparity = 1.8 + .014 * uu + .019 * vv + .12 * np.sin(uu / 9.)
    dz, zfac = ns['zone_disparity'](disparity.astype(np.float32), k, rct, rays)
    ranges = 1000. * 2.4 / ((dz - .3) * zfac)
    ranges[::11] *= .55
    raw_distance = ranges[::-1].copy()
    raw_status = np.resize([5, 6, 9, 10], 64)
    raw_status[::13] = 0
    add_case('tof_trimmed_rotated', disparity, distance=raw_distance, status=raw_status,
             mask=(uu + vv) % 7 != 0)

    directions = dirs_for(width, height, k)
    corridor_rays = (ns['rot'](-23., 11.) @ (rct.T @ directions.T)).T
    floor_disp = (3.7 * corridor_rays[:, 1] / 1.42 + .4).reshape(height, width)
    corrupted_floor = floor_disp.copy()
    corrupted_floor[(uu * 7 + vv * 11) % 10 < 2] += 1.6
    add_case('floor_with_20pct_outliers', corrupted_floor, sensor_height=1.42)
    # A single state sequence verifies fallback and that gated frames do not update prior scale.
    add_case('state_tof', disparity, distance=raw_distance, status=raw_status)
    add_case('state_floor', corrupted_floor, sensor_height=1.42, reset=False)
    add_case('state_previous_no_anchors', np.full((height, width), np.nan), reset=False)
    add_case('state_gated_good_tof', disparity, distance=raw_distance, status=raw_status,
             pitch=10.01, reset=False)
    add_case('state_previous_after_gate', np.full((height, width), np.nan), reset=False)
    add_case('state_reset_has_no_previous', np.full((height, width), np.nan))

    for pitch, roll in ((0., 0.), (-39., -17.), (10., 25.)):
        d = (1.7 + .1 * np.sin(uu / 3.) + .2 * np.cos(vv / 5.)).astype(np.float32)
        d[0, :5] = [np.nan, np.inf, -np.inf, .2, .2001]
        add_case(f'rotated_readout_{pitch}_{roll}', d, pitch=pitch, roll=roll,
                 mask=(uu + 3 * vv) % 9 != 0, manual_fit=(2.123456789, .234567891), process=False)

    # A constant optical plane at 2 m reveals FP32 inverse/reciprocal differences at the
    # inclusive far face. Weak Python scalar promotion is deliberate here, not model output.
    edge_k = np.array([[1000., 0., 10.], [0., 1000., 0.], [0., 0., 1.]])
    for count in (149, 150, 400):
        mask = np.arange(400).reshape(20, 20) < count
        add_case(f'far_face_threshold_{count}', np.full((20, 20), np.float32(.8)),
                 mask=mask, camera_k=edge_k, rotation=np.eye(3), pitch=0., roll=0.,
                 manual_fit=(1.20000003, .2), process=False)
    for name, value in (('inside', np.nextafter(np.float32(.8), np.float32(np.inf))),
                        ('outside', np.nextafter(np.float32(.8), np.float32(-np.inf)))):
        add_case(f'far_face_one_ulp_{name}', np.full((20, 20), value), camera_k=edge_k,
                 rotation=np.eye(3), pitch=0., roll=0., manual_fit=(1.20000003, .2), process=False)

    last = None
    for c in cases:
        if c['reset']: last = None
        pose_ok = bool(ns['in_envelope'](c['pitch'], c['roll']))
        fit = None
        anchor = 'NONE'
        if pose_ok:
            if c['tof_fit'] is not None: fit, anchor = c['tof_fit'], 'TOF'
            elif c['floor_fit'] is not None: fit, anchor = c['floor_fit'], 'FLOOR'
            elif last is not None: fit, anchor = last, 'PREVIOUS'
            if fit is not None: last = fit
        c.update(process_fit=fit, anchor=anchor, pose_ok=pose_ok)
        c['process_pixels'], c['process_alarms'] = readout(c, fit)

    temporal = []
    for head_bits in range(32):
        for body_bits in range(32):
            h = [(head_bits >> i) & 1 for i in range(5)]
            b = [(body_bits >> i) & 1 for i in range(5)]
            temporal.append((h, b, ns['confirm'](h), ns['confirm'](b)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as stream:
        w = Writer(stream)
        w.ints([0x4d464e31, len(affine)])
        for c in affine:
            w.text(c['name']); w.ints([len(c['x'])]); w.doubles(c['x']); w.doubles(c['y']); w.fit(c['fit'])
        w.ints([len(cases)])
        for c in cases:
            h, width = c['disp'].shape
            w.text(c['name']); w.ints([width, h]); w.flags([c['reset'], c['process']])
            w.doubles([c['k'][0, 0], c['k'][1, 1], c['k'][0, 2], c['k'][1, 2]])
            w.doubles(c['rct'].ravel()); w.doubles([c['pitch'], c['roll'], c['height']])
            w.floats(c['disp']); w.flags(c['mask']); w.doubles(c['distance']); w.ints(c['status'])
            w.doubles(c['dz']); w.doubles(c['factor']); w.fit(c['tof_fit']); w.fit(c['floor_fit'])
            w.ints([len(c['draws'][0]) if c['draws'] else 0])
            if c['draws']: w.ints(c['draws'][0]); w.ints(c['draws'][1:])
            w.fit(c['readout_fit']); w.ints(c['pixels']); w.flags(c['alarms'])
            w.text(c['anchor']); w.flags([c['pose_ok']]); w.fit(c['process_fit'])
            w.ints(c['process_pixels']); w.flags(c['process_alarms'])
        w.ints([len(temporal), 5])
        for h, b, hc, bc in temporal:
            for step in zip(h, b, hc, bc): w.flags(step)
    report = dict(evidence='synthetic fixed-input arithmetic oracle; no scene/model/eval evidence',
                  numpy=np.__version__, affine_cases=len(affine), geometry_cases=len(cases),
                  temporal_sequences=len(temporal), temporal_steps=sum(len(v[0]) for v in temporal),
                  fixture_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), fixture_bytes=out.stat().st_size,
                  sources_sha256={name: hashlib.sha256((HOST / name).read_bytes()).hexdigest() for name in SOURCES},
                  affine=[dict(name=c['name'], expected_fit=c['fit']) for c in affine],
                  geometry=[dict(name=c['name'], tof_fit=c['tof_fit'], floor_fit=c['floor_fit'],
                                 pixels=c['pixels'], alarms=c['alarms'], process_anchor=c['anchor'],
                                 process_pixels=c['process_pixels'], floor_samples=len(c['draws'][0]) if c['draws'] else 0)
                            for c in cases],
                  limitations=['same recorded RANSAC samples, not Java/NumPy RNG equivalence',
                               'synthetic arithmetic, no accuracy or new confirmation claims'])
    out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k: report[k] for k in ('affine_cases', 'geometry_cases', 'temporal_sequences',
                                           'temporal_steps', 'fixture_bytes', 'fixture_sha256')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    main(parser.parse_args())
