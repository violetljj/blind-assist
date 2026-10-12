"""Training-only per-quadrature-ray first-visible geometry supervision.

No new observations or electronics synthesis. valid denotes a geometric first
hit within the retained radial domain, not sensor validity or absence of danger.
Public ray/weight/query helpers never read scene geometry or surface labels.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import cnh_surface_distribution_data as D

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-ray-surface-20261002'
LABEL_ROOT = OUT/'labels/train'
OBSERVATIONS = D.OBSERVATIONS
RUN_ID = 'CNH_RAY_SURFACE_20261002'
UNITS = list(range(93000, 93096))
CONFIGS = np.arange(22)
FRAMES = np.arange(3, 16)
R = .0375348*128
LABEL_SHAPE = (22, 13, 128, 128)
QUERY_LOW, QUERY_HIGH, EPS = D.QUERY_LOW, D.QUERY_HIGH, D.EPS
sha, read, create_json = D.sha, D.read, D.create_json


def to_microzones(array):
    """global[y,x,...] -> [micro_y,micro_x,64,...], preserving ray order."""
    array = np.asarray(array)
    if array.shape[:2] != (128, 128):
        raise ValueError('Expected global128x128 ray axes')
    tail = array.shape[2:]
    return array.reshape(16, 8, 16, 8, *tail).transpose(0, 2, 1, 3, *range(4, 4+len(tail))).reshape(16, 16, 64, *tail)


@lru_cache(maxsize=1)
def public_rays():
    """Return float64 rays[128,128,3], weights[128,128]; each8x8tile sums1."""
    micro_rays, micro_weights = D.public_rays()
    rays = micro_rays.reshape(16, 16, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(128, 128, 3)
    weights = micro_weights.reshape(16, 16, 8, 8).transpose(0, 2, 1, 3).reshape(128, 128)
    rays.setflags(write=False); weights.setflags(write=False)
    return rays, weights


def rays128():
    return public_rays()[0]


def weights128():
    return public_rays()[1]


def encode_radial(radial):
    """Half normalized depth + independent bool validity; invalid depth is zero."""
    radial = np.asarray(radial, np.float64)
    if radial.shape != (128, 128):
        raise ValueError('Expected128x128 first-hit radial distances')
    valid = np.isfinite(radial) & (radial > 0) & (radial < R)
    depth = np.zeros(radial.shape, np.float16)
    depth[valid] = (radial[valid]/R).astype(np.float16)
    return depth, valid


def first_visible_radial(boxes, sensor_pose):
    """Geometry-label helper: first opaque surface across ALL training boxes."""
    import cnh_proposal_attribution_scenes as S
    pose = np.asarray(sensor_pose, np.float64)
    hits = S.raycast_boxes(pose[:3, 3], rays128()@pose[:3, :3].T, boxes)
    return hits['distance']


def exact_query_mass(radial, sensor_to_travel):
    """Public same-ray hard query integral [2,16,16], for geometry checks only.

    Inputs are radial predictions/values and a rigid public transform. Keeping
    ray identity avoids the old angle/range marginal product. Invalid rays have
    zero geometric mass, which is not a clear-space observation.
    """
    radial = np.asarray(radial, np.float64)
    transform = np.asarray(sensor_to_travel, np.float64)
    if radial.shape != (128, 128) or transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Expected128x128radial and finite4x4transform')
    rays, weights = public_rays()
    valid = np.isfinite(radial) & (radial > 0) & (radial < R)
    points = (rays@transform[:3, :3].T)*np.where(valid, radial, 0)[..., None]+transform[:3, 3]
    result = []
    for lo, hi in zip(QUERY_LOW, QUERY_HIGH):
        inside = valid & ((points >= lo+EPS) & (points <= hi-EPS)).all(-1)
        result.append(to_microzones(inside*weights).sum(-1))
    return np.stack(result)


def specification():
    return dict(units=UNITS, configs=CONFIGS.tolist(), frames=FRAMES.tolist(), shape=list(LABEL_SHAPE),
        R_m=R, depth_dtype='float16', valid_dtype='bool',
        layout='global y=zone_y*16+sub_y,x=zone_x*16+sub_x; original8x8zones each16x16 quadrature rays',
        depth='radial/R; invalid depth0; validity stored separately because valid near-R depth can round to1',
        valid='finite and0<radial<R; geometric in-domain first hit, not sensor SNR/validity or clear-space evidence',
        weights='Original solid angle normalized within each global8x8tile (one old16x16microzone)',
        query='Same public HEAD/BODY boxes and1e-8interior; no radial-bin-center replacement',
        training_only=True, synthesize_response=False, new_observation=False)


def source_identity():
    value = D.source_identity()
    value[str(Path(__file__))] = sha(__file__)
    return value


def identity():
    plan_path = OUT/'PLAN.json'
    if not plan_path.exists():
        raise RuntimeError('Parent PLAN required before full training-label generation')
    return dict(plan_sha256=sha(plan_path), source_sha256=source_identity(), specification=specification())


def identity_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check():
    import cnh_proposal_attribution_scenes as S
    import cnh_surface_factorization_probe as F
    started = time.perf_counter()
    original, _ = S.ray_grid()
    expected = original.reshape(8, 8, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(128, 128, 3)
    rays, weights = public_rays()
    assert np.array_equal(rays, expected)
    assert np.array_equal(to_microzones(rays), D.public_rays()[0])
    assert np.array_equal(to_microzones(weights), D.public_rays()[1])
    assert np.allclose(to_microzones(weights).sum(-1), 1., atol=1e-14)
    front = dict(lo=[-4., -4., 1.], hi=[4., 4., 1.1], rho=.1)
    back = dict(lo=[-4., -4., 2.], hi=[4., 4., 2.1], rho=.9)
    radial = first_visible_radial([front], np.eye(4))
    assert np.array_equal(radial, first_visible_radial([back, front], np.eye(4)))
    assert np.allclose(radial, 1/rays[..., 2], atol=1e-12)
    rng = np.random.default_rng(2026100236)
    radial = rng.uniform(.01, R, (128, 128))
    radial.flat[:6] = [np.inf, np.nan, 0., -1., R, R-1e-8]
    depth, valid = encode_radial(radial)
    assert np.array_equal(valid.ravel()[:6], [False]*5+[True])
    assert np.all(depth[~valid] == 0) and depth.ravel()[5] == 1.
    half_error = float(np.max(np.abs(depth[valid].astype(float)*R-radial[valid])))
    assert half_error <= R*2**-12+1e-12
    # EXACT-arm equality uses continuous radial and the same original ray weights.
    # No old probe run/load function, scene, evaluation cohort or score is read.
    transforms = [np.eye(4), np.array([[1., 0, 0, .03], [0, 1., 0, -.04], [0, 0, 1., .02], [0, 0, 0, 1.]])]
    errors = []
    for transform in transforms:
        old = F.masses(to_microzones(radial), transform, np.zeros((2, 16, 16, 129)))[..., 0]
        new = exact_query_mass(radial, transform)
        errors.append(float(np.max(np.abs(old-new))))
        assert np.allclose(old, new, rtol=0, atol=1e-14)
    result = dict(status='PASS', source_sha256=source_identity(), specification=specification(),
        checks=['global128ray ordering matches frozen original and old microzones', 'per-microzone solid angle mass1',
            'first-visible nearest occlusion', 'geometric validity and invaliddepth0',
            'half normalized radial quantization bound', 'old marginal probe EXACT mass parity'],
        original_exact_mass_max_abs_errors=errors, sampled_max_half_radial_error_m=half_error,
        half_radial_error_bound_m=R*2**-12,
        check_helper_sha256={str(Path(F.__file__)): sha(F.__file__)},
        payload_bytes_estimate=int(np.prod(LABEL_SHAPE))*3*len(UNITS),
        training_units_processed=0, observation_synthesis_calls=0, elapsed_s=time.perf_counter()-started)
    create_json(OUT/'data_check.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def run(chunk):
    import cnh_near_range as NR
    k, n = map(int, chunk.split('/'))
    if not 0 <= k < n <= 96:
        raise ValueError('Expected zero-based k/n')
    frozen = identity(); digest = identity_sha(frozen)
    proof = read(OUT/'data_check.json')
    if proof['status'] != 'PASS' or proof['source_sha256'] != frozen['source_sha256']:
        raise ValueError('Matching synthetic geometry check required')
    chunk_path = OUT/f'labels_chunk_{k}of{n}.json'
    manifest = dict(chunk=chunk, units=UNITS[k::n], identity_sha256=digest, identity=frozen)
    if chunk_path.exists():
        if read(chunk_path) != manifest:
            raise ValueError('Chunk belongs to changed inputs')
    else:
        create_json(chunk_path, manifest)
    for unit in UNITS[k::n]:
        observation = OBSERVATIONS/f'unit{unit}.npz'
        observation_sha = sha(observation)
        folder = LABEL_ROOT/f'unit{unit}'; receipt_path = folder/'receipt.json'
        if receipt_path.exists():
            old = read(receipt_path)
            if old['status'] != 'COMPLETE' or old['identity_sha256'] != digest or old['observation_sha256'] != observation_sha:
                raise ValueError('Completed label unit has changed identity')
            if any(sha(folder/name) != checksum for name, checksum in old['output_sha256'].items()):
                raise ValueError('Completed label data changed')
            continue
        started = time.perf_counter()
        scenes = NR.scenes_for(unit)
        metadata = D.unit_metadata(unit, scenes)
        folder.mkdir(parents=True, exist_ok=True)
        names = ('depth.npy', 'valid.npy')
        partials = [folder/name.replace('.npy', '.partial.npy') for name in names]
        if any(p.exists() for p in [*partials, *[folder/name for name in names]]):
            raise FileExistsError('Preserve incomplete labels before restart')
        arrays = []
        max_error = 0.; valid_count = 0
        try:
            arrays.append(np.lib.format.open_memmap(partials[0], mode='w+', dtype=np.float16, shape=LABEL_SHAPE))
            arrays.append(np.lib.format.open_memmap(partials[1], mode='w+', dtype=np.bool_, shape=LABEL_SHAPE))
            for config, scene in enumerate(scenes):
                for fi, frame in enumerate(FRAMES):
                    radial = first_visible_radial(scene['boxes'], scene['poses'][frame])
                    depth, valid = encode_radial(radial)
                    arrays[0][config, fi], arrays[1][config, fi] = depth, valid
                    valid_count += int(valid.sum())
                    if valid.any():
                        max_error = max(max_error, float(np.max(np.abs(depth[valid].astype(float)*R-radial[valid]))))
            for array in arrays:
                array.flush()
        finally:
            for array in arrays:
                array._mmap.close()
        if max_error > R*2**-12+1e-12:
            raise ValueError('Unexpected normalized half error')
        with (folder/'metadata.npz').open('xb') as stream:
            np.savez(stream, **metadata)
        for path, name in zip(partials, names):
            path.rename(folder/name)
        if identity() != frozen or sha(observation) != observation_sha:
            raise ValueError('Inputs changed while building labels; preserve incomplete receipt')
        result = dict(status='COMPLETE', unit=unit, split='train', plan_sha256=frozen['plan_sha256'],
            identity_sha256=digest, source_sha256=frozen['source_sha256'], observation_sha256=observation_sha,
            output_sha256={name: sha(folder/name) for name in (*names, 'metadata.npz')},
            shape=list(LABEL_SHAPE), depth_dtype='float16', valid_dtype='bool', R_m=R,
            configs=CONFIGS.tolist(), frames=FRAMES.tolist(), ray_label_count=int(np.prod(LABEL_SHAPE)),
            valid_ray_count=valid_count, max_half_radial_error_m=max_error,
            sensor_to_travel_shape=[13, 4, 4], elapsed_s=time.perf_counter()-started, observation_synthesis_calls=0)
        create_json(receipt_path, result)
        print('COMPLETE ray labels', unit, round(result['elapsed_s'], 2), flush=True)


def finalize():
    frozen = identity(); digest = identity_sha(frozen)
    hashes, bytes_total, seconds = {}, 0, 0.
    for unit in UNITS:
        folder = LABEL_ROOT/f'unit{unit}'; receipt_path = folder/'receipt.json'
        receipt = read(receipt_path)
        if receipt['status'] != 'COMPLETE' or receipt['identity_sha256'] != digest or receipt['unit'] != unit or receipt['shape'] != list(LABEL_SHAPE):
            raise ValueError('Training ray labels incomplete or identity changed')
        if receipt['observation_sha256'] != sha(OBSERVATIONS/f'unit{unit}.npz'):
            raise ValueError('Original observation identity changed')
        for name, checksum in receipt['output_sha256'].items():
            if sha(folder/name) != checksum:
                raise ValueError('Completed ray labels changed')
            bytes_total += (folder/name).stat().st_size
        hashes[f'train/unit{unit}/receipt.json'] = sha(receipt_path)
        seconds += receipt['elapsed_s']
    result = dict(status='COMPLETE', identity_sha256=digest, plan_sha256=frozen['plan_sha256'],
        source_sha256=frozen['source_sha256'], specification=specification(), units=UNITS,
        unit_receipt_sha256=hashes, output_bytes=bytes_total, sum_unit_elapsed_s=seconds,
        training_only=True, observation_synthesis_calls=0)
    create_json(OUT/'labels_receipt.json', result)
    print('COMPLETE96train ray labels', bytes_total, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'run', 'finalize'))
    parser.add_argument('--chunk', default='0/1')
    args = parser.parse_args()
    {'check': check, 'run': lambda: run(args.chunk), 'finalize': finalize}[args.stage]()
