"""Task-local exact background hit reuse across lateral target hypotheses.

Frozen Q/S/R files stay unchanged. The sensor hook is installed only while
expected() runs and restored in finally. Single renderer call per process;
independent multiprocessing workers own independent bounded caches.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import cnh_pose_expected_only as Q

S = Q.S
_RAYCAST = S.raycast_boxes  # Retain original function, never call patched name.
_CACHE = OrderedDict()
_LOCK = threading.RLock()
_STATS = dict(background_hits=0, background_misses=0, target_calls=0)
CACHE_SIZE = 128
OUT = Q.OUT/'background_cache'


def clear_cache():
    _CACHE.clear()
    for key in _STATS:
        _STATS[key] = 0


def cache_info():
    return dict(**_STATS, entries=len(_CACHE), max_entries=CACHE_SIZE,
                cached_array_bytes=sum(a.nbytes for h in _CACHE.values() for a in h.values()))


def _key(origin, directions, background):
    digest = hashlib.sha256()
    o, d = np.asarray(origin, dtype=np.float64), np.asarray(directions, dtype=np.float64)
    digest.update(o.tobytes()); digest.update(str(d.shape).encode()); digest.update(d.tobytes())
    for box in background:
        digest.update(np.asarray([*box['lo'], *box['hi'], box['rho']], dtype=np.float64).tobytes())
    return digest.digest()


def raycast_cached(origin, directions, boxes):
    if len(boxes) < 2:
        return _RAYCAST(origin, directions, boxes)
    key = _key(origin, directions, boxes[1:])
    if key in _CACHE:
        background = _CACHE.pop(key)
        _CACHE[key] = background
        _STATS['background_hits'] += 1
    else:
        background = _RAYCAST(origin, directions, boxes[1:])
        _CACHE[key] = background
        if len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
        _STATS['background_misses'] += 1
    target = _RAYCAST(origin, directions, boxes[:1])
    _STATS['target_calls'] += 1
    # Original target is visited first and retains exact-distance ties.
    take = target['valid'] & (target['distance'] <= background['distance'])
    background_ids = np.where(background['object_id'] >= 0, background['object_id']+1, -1)
    return dict(distance=np.where(take, target['distance'], background['distance']),
                rho=np.where(take, target['rho'], background['rho']),
                cos=np.where(take, target['cos'], background['cos']),
                object_id=np.where(take, target['object_id'], background_ids),
                valid=np.where(take, target['valid'], background['valid']))


def expected(scene, sub=16, *, pose_batch=16):
    """Same Q.expected schema/budget; LRU128 persists across candidates."""
    with _LOCK:
        previous = S.raycast_boxes
        S.raycast_boxes = raycast_cached
        try:
            return Q.expected(scene, sub=sub, pose_batch=pose_batch)
        finally:
            S.raycast_boxes = previous


def child(backend):
    with np.load(Q.OUT/'poses.npz') as z:
        poses = np.tile(z['poses'], (16, 1, 1))
    candidates = json.loads((OUT/'candidates.json').read_text(encoding='utf8'))
    renderer = Q.expected if backend == 'Q' else expected
    before = Q.memory(); elapsed = []; cpu_times = []
    clear_cache()
    for ci, boxes in enumerate(candidates):
        tick = time.perf_counter(); cpu = time.process_time()
        result = renderer(dict(poses=poses, boxes=boxes), sub=16, pose_batch=16)
        cpu_times.append(time.process_time()-cpu); elapsed.append(time.perf_counter()-tick)
        np.savez_compressed(OUT/f'{backend}_candidate{ci}.npz', expectation=result['expectation'],
                            ambient=result['ambient'], object_id=result['object_id'])
        del result
    Q.save(OUT/f'{backend}_timing.json', dict(backend=backend, candidate_seconds=elapsed, total_s=sum(elapsed),
        candidate_cpu_seconds=cpu_times, total_cpu_s=sum(cpu_times),
        memory_before=before, memory_after=Q.memory(), cache=cache_info(), poses_per_candidate=128,
        unique_engineering_poses=8, candidates=5, OMP_BLAS_threads=1,
        peak_definition='Fresh process; cumulative OS peak working set includes imports and output serialization'))


def check():
    OUT.mkdir(parents=True, exist_ok=True)
    source = Q.OUT.parents[1]/'cnh-displacement-ceiling-20261003'
    truth_path = source/'truth/unit110001.json'
    truth = json.loads(truth_path.read_text(encoding='utf8'))
    Q.save(OUT/'candidates.json', [truth['boxes'][i] for i in (0, 1, 4, 5, 6)])
    for backend in ('Q', 'Q2'):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', 'child', '--backend', backend], check=True)
    error = 0.
    for ci in range(5):
        with np.load(OUT/f'Q_candidate{ci}.npz') as a, np.load(OUT/f'Q2_candidate{ci}.npz') as b:
            error = max(error, float(np.max(np.abs(a['expectation']-b['expectation']))))
            for key in ('expectation', 'ambient', 'object_id'):
                np.testing.assert_array_equal(a[key], b[key])
    toy = [dict(lo=[-.1, -.1, 1.], hi=[.1, .1, 1.2], rho=.3),
           dict(lo=[-.1, -.1, 1.], hi=[.1, .1, 1.2], rho=.8)]
    directions = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    clear_cache()
    a, b = _RAYCAST(np.zeros(3), directions, toy), raycast_cached(np.zeros(3), directions, toy)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert b['object_id'].tolist() == [0, -1, -1]
    # A retained renderer exception must still restore the shared sensor hook.
    previous = S.raycast_boxes
    try:
        expected(dict(poses=np.empty((0, 4, 4)), boxes=toy))
    except ValueError:
        pass
    assert S.raycast_boxes is previous
    timings = {key: json.loads((OUT/f'{key}_timing.json').read_text()) for key in ('Q', 'Q2')}
    receipt = dict(status='PASS', role='Engineering-only fixed8poses x5candidates x16repetition; no new scientific cohort or scores',
        all_full_expectation_array_equal=True, expectation_max_abs_error=error, all_ambient_array_equal=True,
        all_object_id_array_equal=True, target_first_exact_tie=True, missing_id_minus1=True, finally_restored=True,
        timings=timings, batch_speedup=timings['Q']['total_s']/timings['Q2']['total_s'],
        batch_cpu_speedup=timings['Q']['total_cpu_s']/timings['Q2']['total_cpu_s'],
        engineering_cpu_s=sum(t['total_cpu_s'] for t in timings.values()),
        peak_working_set_delta_Q2_minus_Q_bytes=timings['Q2']['memory_after']['peak_working_set_bytes']-timings['Q']['memory_after']['peak_working_set_bytes'],
        source_sha256={str(Path(__file__).resolve()): Q.sha(__file__), str(Path(Q.__file__).resolve()): Q.sha(Q.__file__), **Q.R.source_sha256()},
        input_sha256={str(p): Q.sha(p) for p in [truth_path, Q.OUT/'poses.npz', OUT/'candidates.json']},
        limits=['Engineering repeats have8unique pose keys; production LRU128 can retain more background arrays',
                'Concurrent science workers affect wall timing; CPU time and process peaks reported separately',
                'Candidate backgrounds must match; changing backgrounds automatically produces another cache key'])
    Q.save(OUT/'verification.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'expectation_max_abs_error', 'batch_speedup', 'batch_cpu_speedup', 'engineering_cpu_s', 'peak_working_set_delta_Q2_minus_Q_bytes', 'timings')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['check', 'child'], default='check')
    parser.add_argument('--backend', choices=['Q', 'Q2'])
    args = parser.parse_args()
    (check if args.stage == 'check' else lambda: child(args.backend))()
