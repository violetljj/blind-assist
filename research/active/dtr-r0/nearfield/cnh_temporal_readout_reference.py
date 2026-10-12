"""Fresh pilot's privileged D reference; CPU rendering precedes all GPU scoring.

This evaluator-only module never constructs learned features or trains a model.
The full scene, target properties and true current anchor remain privileged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import time

import numpy as np
from scipy.special import logsumexp

import cnh_displacement_ceiling_render as R
import cnh_unknown_target_reference_evaluate as U

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-temporal-readout-20261004'
UNITS = [u for u in range(221001, 221073) if u % 3 in (0, 1)][:48]
FRAMES = np.arange(3, 16)
TIDS = [0, 1, 4, 5, 6]
PAST = np.concatenate([np.arange(max(0, f-7), f+1) for f in FRAMES])
START = np.cumsum([0] + [min(8, int(f)+1) for f in FRAMES])


def inputs(out, unit):
    return {key: out / key / 'evaluation' / f'unit{unit}.{ext}'
            for key, ext in [('observations', 'npz'), ('truth', 'json'), ('templates', 'npz')]}


def hashes(paths):
    return {str(p): U.sha(p) for p in paths}


def require_plan(out):
    plan = U.read(out / 'PLAN.json')
    for key in ('eval_units', 'evaluation_units', 'fresh_evaluation_units'):
        if key in plan and plan[key] != UNITS:
            raise ValueError('Fresh reference cohort differs from frozen PLAN: '+key)
    return U.sha(out / 'PLAN.json')


def check_budget():
    # The parent runner owns the single computation deadline, including workers.
    import cnh_temporal_readout as C
    C.check_budget()


def render_one(out_string, unit, lam):
    check_budget()
    out = Path(out_string)
    tick = time.monotonic()
    paths = inputs(out, unit)
    before = hashes(paths.values())
    truth = U.read(paths['truth'])
    with np.load(paths['observations'], allow_pickle=False) as z:
        sensor, noisy, ambient = z['sensor'], z['noisy'], z['ambient']
        if z['hist'].shape != (7, 4, 16, 8, 8, 16):
            raise ValueError('Fresh reference requires all7 branches and K4')
    if sensor.shape != (16, 4, 4) or noisy.shape != (4, 16, 4, 4):
        raise ValueError('Invalid fresh pose axes')
    if lam == 0:
        noisy = np.repeat(sensor[None], 4, axis=0)
    elif lam != 1:
        raise ValueError('Only true-relative check and estimated-relative D are declared')
    expected = np.empty((4, 5, len(PAST), 8, 8, 16), np.float64)
    for k in range(4):
        belief = np.concatenate([sensor[f] @ np.linalg.inv(noisy[k, f]) @ noisy[k, max(0, f-7):f+1]
                                 for f in FRAMES])
        for j, di in enumerate(TIDS):
            check_budget()
            clean = R.expected(dict(poses=belief, boxes=truth['boxes'][di]))
            np.testing.assert_array_equal(clean['ambient'], ambient[PAST])
            expected[k, j] = clean['expectation']
    if hashes(paths.values()) != before:
        raise ValueError('Fresh scene input changed during reference rendering')
    folder = out / 'reference' / ('lambda0' if lam == 0 else 'belief')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'unit{unit}.npz'
    if path.exists():
        raise FileExistsError('Retain existing reference templates: '+str(path))
    np.savez(path, expected=expected, past=PAST, offsets=START)
    return dict(unit=unit, lambda_value=lam, seconds=time.monotonic()-tick,
                path=str(path), sha256=U.sha(path), input_sha256=before)


def render(out=OUT, workers=4):
    check_budget()
    plan_sha = require_plan(out)
    folder = out / 'reference'
    folder.mkdir(exist_ok=True)
    receipt_path = folder / 'render_receipt.json'
    if receipt_path.exists():
        raise FileExistsError('Inspect completed rendering instead of restarting')
    started = time.monotonic()
    records = []
    jobspec = [(u, 1) for u in UNITS] + [(u, 0) for u in UNITS[:2]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(render_one, str(out), u, lam) for u, lam in jobspec]
        for future in as_completed(jobs):
            records.append(future.result())
            print('REFERENCE RENDER', len(records), '/', len(jobspec), flush=True)
    U.save(receipt_path, dict(status='COMPLETE', units=UNITS, plan_sha256=plan_sha,
        seconds=time.monotonic()-started, workers=workers, records=records,
        source_sha256=hashes([Path(__file__), Path(R.__file__)]) | R.source_sha256(),
        all_rendering_finished_before_gpu=True))


def barrier(out):
    receipt = U.read(out / 'reference/render_receipt.json')
    if receipt['status'] != 'COMPLETE' or receipt['units'] != UNITS:
        raise ValueError('All fresh D and lambda0 templates must finish before GPU use')
    if receipt['plan_sha256'] != require_plan(out):
        raise ValueError('PLAN changed since rendering')
    for record in receipt['records']:
        if U.sha(record['path']) != record['sha256'] or hashes(map(Path, record['input_sha256'])) != record['input_sha256']:
            raise ValueError('Reference or frozen scene input changed')
    return receipt


def score_expected(hist, expected, ambient):
    """Same exact float64 Skellam mixture for either re-rendered or true templates."""
    import cnh_unknown_target_gpu as G
    result = np.empty((7, 4, 13), np.float64)
    for k in range(4):
        means = expected[k] if expected.ndim == 6 else expected
        ll = G.skellam_logpmf_signed_gpu(hist[:, k, PAST][:, None], means[None],
                8 * ambient[PAST][None, None, ..., None]).sum(axis=(-3, -2, -1))
        for ai in range(13):
            total = ll[:, :, START[ai]:START[ai+1]].sum(-1)
            result[:, k, ai] = logsumexp(total[:, :2], axis=1)-np.log(2)-logsumexp(total[:, 2:], axis=1)+np.log(3)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite exact reference scores')
    return result


def check(out=OUT):
    check_budget()
    barrier(out)
    path = out / 'reference/lambda0_check.json'
    if path.exists():
        if U.read(path)['status'] != 'PASS':
            raise ValueError('Inspect prior failed parity check')
        return U.read(path)
    started = time.monotonic()
    checks = []
    for unit in UNITS[:2]:
        check_budget()
        paths = inputs(out, unit)
        with np.load(paths['observations']) as z, np.load(paths['templates']) as t, np.load(out / 'reference/lambda0' / f'unit{unit}.npz') as l:
            truth_expected = t['expected'][TIDS][:, PAST]
            true_score = score_expected(z['hist'], truth_expected, z['ambient'])
            zero_score = score_expected(z['hist'], l['expected'], z['ambient'])
            error = float(np.max(np.abs(zero_score-true_score)))
            if error >= 1e-8:
                raise ValueError(f'lambda0 score path mismatch {unit}: {error}; D must not continue')
            checks.append(dict(unit=unit, maximum_score_difference=error,
                maximum_expectation_difference=float(np.max(np.abs(l['expected']-truth_expected[None])))))
    result = dict(status='PASS', units=checks, criterion='Every score in two fresh scenes max difference<1e-8',
                  seconds=time.monotonic()-started, render_receipt_sha256=U.sha(out / 'reference/render_receipt.json'))
    U.save(path, result)
    return result


def score(out=OUT):
    check_budget()
    receipt = barrier(out)
    parity = check(out)
    if parity['render_receipt_sha256'] != U.sha(out / 'reference/render_receipt.json'):
        raise ValueError('Parity receipt belongs to different rendered templates')
    target = out / 'reference/all_scores.npz'
    if target.exists():
        raise FileExistsError('Do not overwrite D scores')
    started = time.monotonic()
    scores = []
    for unit in UNITS:
        check_budget()
        with np.load(inputs(out, unit)['observations']) as z, np.load(out / 'reference/belief' / f'unit{unit}.npz') as t:
            scores.append(score_expected(z['hist'], t['expected'], z['ambient']))
        print('REFERENCE SCORE', len(scores), '/48', flush=True)
    np.savez_compressed(target, units=UNITS, D=np.stack(scores), frames=FRAMES)
    U.save(out / 'reference/scores_receipt.json', dict(status='COMPLETE', units=UNITS,
        score_sha256=U.sha(target), plan_sha256=require_plan(out), seconds=time.monotonic()-started,
        rendering_seconds=receipt['seconds'], render_receipt_sha256=U.sha(out / 'reference/render_receipt.json'),
        check_sha256=U.sha(out / 'reference/lambda0_check.json'), source_sha256=U.sha(__file__),
        limits=['Exact target/background/current true anchor privileged', 'Plug-in estimated relative motion, not pose marginalization']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['render', 'check', 'score'], required=True)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    for key, sub in [('TEMP', 'tmp'), ('TMP', 'tmp'), ('CUPY_CACHE_DIR', 'cupy-cache')]:
        cache = args.out / sub
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(cache)
    if args.stage == 'render':
        render(args.out, args.workers)
    elif args.stage == 'check':
        check(args.out)
    else:
        score(args.out)
