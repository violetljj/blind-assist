"""Exact frozen expected electronics without discarded shot-noise sampling.

Task-local acceleration: retained sensor/renderer files are never modified.
Signal histogram, pulse, neighbour exchange, residual xtalk and aggregation
are unchanged. Pose blocks bound transient memory, not scientific quadrature.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import cnh_displacement_ceiling_render as R

S = R.S
F = sys.modules[S.synthesize_response.__module__]
OUT = Path(__file__).resolve().parents[4]/'artifacts.local/work/cnh-pose-marginal-reference-20261004/performance'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf8')


def quiet_electronics(radial, rho, cosine, weights, params):
    """Same floating-point signal operations as frozen synthesize_response."""
    params = params.validate()
    distance, rho, cosine, weights, valid = F._ray_inputs(radial, rho, cosine, weights)
    safe_distance = np.where(valid, distance, 1)
    energy = params.signal_counts*rho*cosine*weights/np.maximum(safe_distance, .05)**2
    histogram, _, _ = F._histogram(distance, energy, valid, F.RAW_BINS,
                                 F.RAW_BIN_M, params.range_zero_m, 'numpy', None)
    matrix = F._pulse_matrix(params)
    signal = histogram@matrix
    leak = params.neighbour_leak/4
    old = signal.copy()
    signal[..., 1:, :, :] += leak*(old[..., :-1, :, :]-old[..., 1:, :, :])
    signal[..., :-1, :, :] += leak*(old[..., 1:, :, :]-old[..., :-1, :, :])
    signal[..., :, 1:, :] += leak*(old[..., :, :-1, :]-old[..., :, 1:, :])
    signal[..., :, :-1, :] += leak*(old[..., :, 1:, :]-old[..., :, :-1, :])
    xtalk_bin = int(np.floor((params.crosstalk_range_m-params.range_zero_m)/F.RAW_BIN_M))
    xtalk = np.zeros(F.RAW_BINS)
    if 0 <= xtalk_bin < F.RAW_BINS:
        xtalk = params.signal_counts*params.crosstalk_fraction*matrix[xtalk_bin]
    expectation = signal+xtalk
    ambient = np.full(expectation.shape[:-1], params.ambient_counts*params.output_gain)
    return expectation*params.output_gain, ambient


def _block(poses, boxes, sub, directions, weights, quarter):
    hits = [S.raycast_boxes(p[:3, 3], directions@p[:3, :3].T, boxes) for p in poses]
    shape = (len(poses), 8, 8, sub, sub)
    distance = np.stack([h['distance'] for h in hits]).reshape(shape)
    reflectance = np.stack([h['rho'] for h in hits]).reshape(shape)
    cosine = np.stack([h['cos'] for h in hits]).reshape(shape)
    object_id = np.stack([h['object_id'] for h in hits]).astype(np.int64)
    del hits
    w = weights.reshape(8, 8, sub, sub)
    fine = np.empty((len(poses), 16, 16, 16), np.float64)
    fine_ambient = np.empty((len(poses), 16, 16), np.float64)
    half = sub//2
    for qy in range(2):
        for qx in range(2):
            ys, xs = slice(half*qy, half*(qy+1)), slice(half*qx, half*(qx+1))
            qw = w[:, :, ys, xs].reshape(8, 8, half*half)
            fraction = qw.sum(-1)/weights.sum(-1)
            ray_shape = (len(poses), 8, 8, half*half)
            rho = reflectance[:, :, :, ys, xs].reshape(ray_shape)*(4*fraction[None, :, :, None])
            if rho.max() > 1:
                raise ValueError('Quarter reflectance scaling exceeds frozen sensor domain')
            expectation, ambient = quiet_electronics(distance[:, :, :, ys, xs].reshape(ray_shape), rho,
                cosine[:, :, :, ys, xs].reshape(ray_shape), qw, quarter)
            fine[:, qy::2, qx::2] = expectation.reshape(len(poses), 8, 8, 16, 8).sum(-1)
            fine_ambient[:, qy::2, qx::2] = ambient
    return (fine.reshape(len(poses), 8, 2, 8, 2, 16).sum(axis=(2, 4)),
            fine_ambient.reshape(len(poses), 8, 2, 8, 2).sum(axis=(2, 4)), object_id)


def expected(scene, sub=16, *, pose_batch=16):
    """Drop-in R.expected schema; exact sub16/sub32, bounded pose workspace.

    pose_batch controls transient memory only. No photon draws or likelihood
    approximation. Scalar ray/pulse budgets and float64 outputs stay frozen.
    """
    if sub not in (16, 32) or int(pose_batch) != pose_batch or pose_batch < 1:
        raise ValueError('Declared sub16/32 and positive integer pose_batch required')
    poses = np.asarray(scene['poses'], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('Expected finite nonempty sensor pose sequence')
    parameters, _ = S.nominal_parameters()
    if parameters.output_gain != 1. or parameters.noise_scale != 1.:
        raise ValueError('Frozen gain1/noiseScale1 required')
    quarter = replace(parameters, signal_counts=parameters.signal_counts/4,
                      ambient_counts=parameters.ambient_counts/4, noise_scale=0.)
    directions, weights = S.angular_rays(sub)
    mean = np.empty((len(poses), 8, 8, 16), np.float64)
    ambient = np.empty((len(poses), 8, 8), np.float64)
    object_id = np.empty((len(poses), 8, 8, sub*sub), np.int64)
    for start in range(0, len(poses), int(pose_batch)):
        stop = min(start+int(pose_batch), len(poses))
        mean[start:stop], ambient[start:stop], object_id[start:stop] = _block(
            poses[start:stop], scene['boxes'], sub, directions, weights, quarter)
    if not np.isfinite(mean).all() or np.any(mean < 0) or not np.isfinite(ambient).all() or np.any(ambient < 0):
        raise ValueError('Invalid frozen expectation/background')
    return dict(schema=R.SCHEMA, expectation=mean, ambient=ambient, object_id=object_id,
                sensor_params=asdict(parameters), sub=sub,
                count_semantics='8x8x16; each radial bin sums4 angular quadrants x8 raw radial bins=32 aggregate terms')


def memory():
    """Windows process peak working set, read without external dependencies."""
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t), ('PrivateUsage', ctypes.c_size_t)]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set_bytes=int(counters.WorkingSetSize), peak_working_set_bytes=int(counters.PeakWorkingSetSize),
                private_bytes=int(counters.PrivateUsage), peak_pagefile_bytes=int(counters.PeakPagefileUsage))


def child(backend):
    with np.load(OUT/'poses.npz') as z:
        poses = z['poses']
    boxes = json.loads((OUT/'boxes.json').read_text(encoding='utf8'))
    renderer = R.expected if backend == 'frozen' else expected
    before = memory(); times = []
    for _ in range(3):
        tick = time.perf_counter()
        result = renderer(dict(poses=np.tile(poses, (15, 1, 1)), boxes=boxes), sub=16)
        times.append(time.perf_counter()-tick)
    after = memory()
    np.savez_compressed(OUT/f'{backend}_output.npz', expectation=result['expectation'], ambient=result['ambient'], object_id=result['object_id'])
    save(OUT/f'{backend}_timing.json', dict(backend=backend, seconds=times, median_s=float(np.median(times)),
        memory_before=before, memory_after=after, poses=120, unique_engineering_poses=8, repeats=3,
        memory_definition='Separate fresh process for each backend; OS cumulative peak working set includes imports and input/output arrays'))


def check():
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT.parents[1]/'cnh-displacement-ceiling-20261003'
    unit = 110001
    truth_path, observation_path = source/'truth'/f'unit{unit}.json', source/'observations'/f'unit{unit}.npz'
    truth = json.loads(truth_path.read_text(encoding='utf8'))
    with np.load(observation_path) as z:
        sensor, noisy = z['sensor'], z['noisy']
    perturbed = sensor[13]@np.linalg.inv(noisy[0, 13])@noisy[0, [0, 5, 10, 13]]
    poses = np.concatenate([sensor[[3, 7, 11, 15]], perturbed])
    np.savez_compressed(OUT/'poses.npz', poses=poses)
    save(OUT/'boxes.json', truth['boxes'][0])
    for backend in ('frozen', 'expected_only'):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', 'child', '--backend', backend], check=True)
    with np.load(OUT/'frozen_output.npz') as a, np.load(OUT/'expected_only_output.npz') as b:
        error = float(np.max(np.abs(a['expectation']-b['expectation'])))
        bitwise = np.array_equal(a['expectation'], b['expectation'])
        np.testing.assert_allclose(a['expectation'], b['expectation'], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(a['ambient'], b['ambient'])
        np.testing.assert_array_equal(a['object_id'], b['object_id'])
    # Native8 call checks full schema and the un-tiled unique physical poses.
    original = R.expected(dict(poses=poses, boxes=truth['boxes'][0]), sub=16)
    quiet = expected(dict(poses=poses, boxes=truth['boxes'][0]), sub=16)
    for key in ('schema', 'sensor_params', 'sub', 'count_semantics'):
        assert original[key] == quiet[key], key
    np.testing.assert_array_equal(original['expectation'], quiet['expectation'])
    np.testing.assert_array_equal(original['ambient'], quiet['ambient'])
    base, fast = (json.loads((OUT/f'{b}_timing.json').read_text(encoding='utf8')) for b in ('frozen', 'expected_only'))
    receipt = dict(status='PASS', role='Engineering-only8fixed poses repeated15 times; no scientific cohort/photons/inference/likelihood',
        expectation_max_abs_error=error, expectation_array_equal=bitwise, ambient_array_equal=True, object_id_array_equal=True,
        complete_schema_equal=True, native8_expectation_array_equal=True,
        source_sha256={str(Path(__file__).resolve()): sha(__file__), **R.source_sha256()},
        engineering_input_sha256={str(p): sha(p) for p in [truth_path, observation_path, OUT/'poses.npz', OUT/'boxes.json']},
        timings=dict(frozen=base, expected_only=fast), speedup=base['median_s']/fast['median_s'],
        absolute_peak_working_set_reduction_bytes=base['memory_after']['peak_working_set_bytes']-fast['memory_after']['peak_working_set_bytes'],
        preserved=dict(sub=16, raw_radial_bins=128, coarse32_aggregation=True, pose_float64=True,
                       pulse_leak_xtalk=True, frozen_sensor_sources_unmodified=True))
    save(OUT/'verification.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'expectation_max_abs_error', 'expectation_array_equal', 'speedup', 'absolute_peak_working_set_reduction_bytes', 'timings')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['check', 'child'], default='check')
    parser.add_argument('--backend', choices=['frozen', 'expected_only'])
    args = parser.parse_args()
    (check if args.stage == 'check' else lambda: child(args.backend))()
