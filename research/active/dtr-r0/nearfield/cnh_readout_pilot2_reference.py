"""Privileged continuous-scene training teacher and fresh pilot2 D reference.

Shared pure helpers render canonical targets and score exact signed Skellam
mixtures. Teachers remain separate from network inputs. No old payload is edited.
Each requested CPU phase seals all its templates before that phase can use GPU.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from pathlib import Path
import os
import time

import numpy as np
from scipy.special import expit, logsumexp

import cnh_displacement_ceiling_render as R
import cnh_unknown_target_reference_evaluate as U
import cnh_displacement_ceiling as D0

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-readout-pilot2-20261005'
FRAMES = np.arange(3, 16)
DELTAS = np.array([1., 2., -10., -15., -20.])
TIDS = [0, 1, 4, 5, 6]
PAST = np.concatenate([np.arange(max(0, int(f)-7), int(f)+1) for f in FRAMES])
START = np.cumsum([0] + [min(8, int(f)+1) for f in FRAMES])
PARITY_UNITS = [110001, 110008]
ORIGINAL = ROOT / 'artifacts.local/work/cnh-displacement-ceiling-20261003'
FROZEN_ORACLE = ROOT / 'artifacts.local/work/cnh-unknown-target-reference-20261004/scores/all_scores.npz'


def common():
    import cnh_readout_pilot2 as C
    return C


def frozen():
    c = common()
    p = c.load_plan()
    c.check_budget()
    if not np.array_equal(c.FRAMES, FRAMES) or len(c.TRAIN_UNITS) != 480 or len(c.EVAL_UNITS) != 48:
        raise ValueError('Pilot2 teacher/fresh cohort or retained frames differ')
    return c, p, c.plan_sha()


def sources(split, unit):
    c = common()
    base = Path(c.OLD) if split == 'teacher' else Path(c.OUT) if split == 'evaluation' else ORIGINAL
    folder = 'train' if split == 'teacher' else 'evaluation' if split == 'evaluation' else None
    if folder is None:
        return dict(observations=base / 'observations' / f'unit{unit}.npz',
                    truth=base / 'truth' / f'unit{unit}.json',
                    templates=base / 'templates' / f'unit{unit}.npz', oracle=FROZEN_ORACLE)
    paths = dict(observations=base / 'observations' / folder / f'unit{unit}.npz',
                 truth=base / 'truth' / folder / f'unit{unit}.json')
    if split == 'evaluation':
        paths['templates'] = base / 'templates' / folder / f'unit{unit}.npz'
    return paths


def source_hashes():
    return {str(Path(__file__)): U.sha(__file__), str(Path(R.__file__)): U.sha(R.__file__)} | R.source_sha256()


def canonical_boxes(truth):
    """Move only target x; recover actual target first, with no nuisance redraw."""
    actual = truth['boxes'][0]
    target = actual[0]
    width = float(target['hi'][0])-float(target['lo'][0])
    centre = (float(target['hi'][0])+float(target['lo'][0]))/2
    if width <= 0 or centre == 0:
        raise ValueError('Target width/side cannot be recovered')
    side = 1 if centre > 0 else -1

    def shifted(delta):
        boxes = copy.deepcopy(actual)
        inner = .30-float(delta)/100
        low, high = ((inner, inner+width) if side == 1 else (-inner-width, -inner))
        boxes[0]['lo'][0], boxes[0]['hi'][0] = low, high
        if boxes[1:] != actual[1:]:
            raise ValueError('Canonical construction changed background')
        for box in boxes[1:]:
            overlap = np.minimum(boxes[0]['hi'], box['hi'])-np.maximum(boxes[0]['lo'], box['lo'])
            if np.all(overlap > 0):
                raise ValueError('Canonical target/background interpenetration; retain scene, do not silently redraw')
        return boxes

    restored = shifted(truth['intrusion_cm'][0])
    restoration = max(float(np.max(np.abs(np.asarray(restored[0][key])-np.asarray(target[key])))) for key in ('lo', 'hi'))
    if restoration > 1e-12 or restored[0]['rho'] != target['rho']:
        raise ValueError('Actual target reconstruction failed')
    candidates = [shifted(delta) for delta in DELTAS]
    # Fresh7 original candidate templates offer an independent geometry identity.
    if len(truth['boxes']) == 7:
        for candidate, idx in zip(candidates, TIDS):
            original = truth['boxes'][idx]
            if candidate[1:] != original[1:] or candidate[0]['rho'] != original[0]['rho']:
                raise ValueError('Canonical candidate differs from retained fresh truth')
            for key in ('lo', 'hi'):
                np.testing.assert_allclose(candidate[0][key], original[0][key], atol=1e-12, rtol=0)
        # Preserve saved float literals for the independent frozen-path parity.
        candidates = [copy.deepcopy(truth['boxes'][idx]) for idx in TIDS]
    return candidates, dict(side=side, width_m=width, actual_delta_cm=float(truth['intrusion_cm'][0]),
                           reconstruction_max_abs_m=restoration, candidate_delta_cm=DELTAS.tolist())


def anchored_history_poses(sensor, noisy, lam=1):
    sensor, noisy = np.asarray(sensor), np.asarray(noisy)
    if sensor.shape != (16, 4, 4) or noisy.ndim != 4 or noisy.shape[1:] != (16, 4, 4):
        raise ValueError('Expected sensor[16,4,4], noisy[K,16,4,4]')
    if lam == 0:
        noisy = np.repeat(sensor[None], len(noisy), axis=0)
    elif lam != 1:
        raise ValueError('Only frozen estimated relative motion and lambda0 parity are supported')
    return np.stack([np.concatenate([sensor[f] @ np.linalg.inv(path[f]) @ path[max(0, int(f)-7):f+1]
                                    for f in FRAMES]) for path in noisy])


def render_expected(sensor, noisy, ambient, candidates, lam=1):
    """Same original renderer, float64 templates; no label or photon resampling."""
    poses = anchored_history_poses(sensor, noisy, lam)
    result = np.empty((len(poses), 5, len(PAST), 8, 8, 16), np.float64)
    for k, sequence in enumerate(poses):
        for j, boxes in enumerate(candidates):
            common().check_budget()
            clean = R.expected(dict(poses=sequence, boxes=boxes))
            np.testing.assert_array_equal(clean['ambient'], ambient[PAST])
            result[k, j] = clean['expectation']
    return result


def template_path(split, unit):
    return Path(common().OUT) / 'reference' / (split+'_templates') / f'unit{unit}.npz'


def render_one(split, unit, lam):
    c, _, digest = frozen()
    tick = time.monotonic()
    paths = sources(split, unit)
    before = {str(p): U.sha(p) for p in paths.values()}
    destination = template_path(split, unit)
    record_path = destination.with_suffix('.json')
    if record_path.exists():
        record = U.read(record_path)
        if record['plan_sha256'] != digest or record['input_sha256'] != before or record['source_sha256'] != source_hashes() or U.sha(destination) != record['sha256']:
            raise ValueError('Existing template receipt differs from frozen inputs/source')
        return record
    if destination.exists():
        raise FileExistsError('Unreceipted template needs inspection: '+str(destination))
    truth = U.read(paths['truth'])
    candidates, geometry_check = canonical_boxes(truth)
    with np.load(paths['observations'], allow_pickle=False) as z:
        sensor, noisy, ambient = z['sensor'], z['noisy'], z['ambient']
        hist_shape = z['hist'].shape
    expected_shape = (1, 2, 16, 8, 8, 16) if split == 'teacher' else (7, 4, 16, 8, 8, 16)
    if hist_shape != expected_shape:
        raise ValueError('Observation branch/replica count changed')
    expected = render_expected(sensor, noisy, ambient, candidates, lam)
    expectation_parity = None
    if split == 'parity':
        with np.load(paths['templates'], allow_pickle=False) as old:
            exact = old['expected'][TIDS][:, PAST]
            expectation_parity = float(np.max(np.abs(expected-exact[None])))
    if any(U.sha(p) != h for p, h in before.items()):
        raise ValueError('Scene input changed during CPU reference rendering')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez(stream, expected=expected, past=PAST, offsets=START, delta_cm=DELTAS)
    temporary.replace(destination)
    record = dict(status='COMPLETE', split=split, unit=int(unit), lambda_value=lam,
        seconds=time.monotonic()-tick, path=str(destination), sha256=U.sha(destination),
        plan_sha256=digest, input_sha256=before, source_sha256=source_hashes(),
        geometry_check=geometry_check, expectation_parity_max_abs=expectation_parity)
    c.save(record_path, record)
    return record


def render(split='all', workers=4):
    c, _, digest = frozen()
    wanted = ['teacher', 'evaluation'] if split == 'all' else [split]
    if set(wanted)-{'teacher', 'evaluation'} or workers not in (1, 2, 3, 4):
        raise ValueError('Only declared CPU phases and at most4workers')
    folder = Path(c.OUT) / 'reference'
    folder.mkdir(parents=True, exist_ok=True)
    tick = time.monotonic()
    jobspec = [('parity', u, 0) for u in PARITY_UNITS]
    for phase in wanted:
        jobspec += [(phase, int(u), 1) for u in (c.TRAIN_UNITS if phase == 'teacher' else c.EVAL_UNITS)]
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(render_one, phase, unit, lam) for phase, unit, lam in jobspec]
        for future in as_completed(jobs):
            records.append(future.result())
            if len(records) % 12 == 0 or len(records) == len(jobspec):
                print('PILOT2 REFERENCE RENDER', split, len(records), '/', len(jobspec), round(time.monotonic()-tick, 1), flush=True)
    for phase in ['parity']+wanted:
        path = folder / f'render_{phase}.json'
        selected = sorted([r for r in records if r['split'] == phase], key=lambda r: r['unit'])
        receipt = dict(status='COMPLETE', split=phase, units=[r['unit'] for r in selected], records=selected,
            plan_sha256=digest, workers=workers, elapsed_phase_wall_s=time.monotonic()-tick,
            requested_phase=split, requested_phase_all_templates_complete=True, source_sha256=source_hashes())
        if path.exists():
            old = U.read(path)
            if old['plan_sha256'] != digest or old['records'] != selected:
                raise ValueError('Sealed rendering receipt differs')
        else:
            c.save(path, receipt)
    if split == 'all':
        c.save(folder / 'render_all.json', dict(status='COMPLETE', plan_sha256=digest,
            requested_phase='all', all_templates_complete=True, seconds=time.monotonic()-tick,
            units={'teacher':list(c.TRAIN_UNITS), 'evaluation':list(c.EVAL_UNITS)}, workers=workers))


def barrier(phase):
    c, _, digest = frozen()
    receipt = U.read(Path(c.OUT) / 'reference' / f'render_{phase}.json')
    expected = PARITY_UNITS if phase == 'parity' else list(c.TRAIN_UNITS) if phase == 'teacher' else list(c.EVAL_UNITS)
    if receipt['status'] != 'COMPLETE' or receipt['units'] != expected or receipt['plan_sha256'] != digest:
        raise ValueError('Entire requested CPU phase must finish before GPU scoring')
    if receipt['requested_phase'] == 'all':
        all_receipt = U.read(Path(c.OUT) / 'reference/render_all.json')
        if all_receipt['status'] != 'COMPLETE' or all_receipt['plan_sha256'] != digest:
            raise ValueError('Combined teacher/evaluation rendering barrier is incomplete')
    for record in receipt['records']:
        if U.sha(record['path']) != record['sha256'] or any(U.sha(path) != h for path, h in record['input_sha256'].items()):
            raise ValueError('Sealed template/input changed before scoring')
    return receipt


def score_exact(hist, expected, ambient):
    """Generalizes the original exactGPU path solely in branch andK axes."""
    import cnh_unknown_target_gpu as G
    hist, expected, ambient = np.asarray(hist), np.asarray(expected), np.asarray(ambient)
    if hist.ndim != 6 or hist.shape[2:] != (16, 8, 8, 16) or expected.shape != (hist.shape[1], 5, len(PAST), 8, 8, 16):
        raise ValueError('Exact scorer input axes differ')
    score = np.empty((hist.shape[0], hist.shape[1], 13), np.float64)
    for k in range(hist.shape[1]):
        common().check_budget()
        ll = G.skellam_logpmf_signed_gpu(hist[:, k, PAST][:, None], expected[k][None],
                8 * ambient[PAST][None, None, ..., None]).sum(axis=(-3, -2, -1))
        for ai in range(13):
            total = ll[:, :, START[ai]:START[ai+1]].sum(-1)
            score[:, k, ai] = logsumexp(total[:, :2], axis=1)-np.log(2)-logsumexp(total[:, 2:], axis=1)+np.log(3)
    if not np.isfinite(score).all():
        raise FloatingPointError('Exact teacher scores must be finite')
    return score


def check():
    c, _, digest = frozen()
    barrier('parity')
    path = Path(c.OUT) / 'reference/lambda0_check.json'
    if path.exists():
        old = U.read(path)
        if old['status'] != 'PASS' or old['plan_sha256'] != digest or old['frozen_oracle_sha256'] != U.sha(FROZEN_ORACLE):
            raise ValueError('Prior lambda0 receipt differs')
        return old
    tick = time.monotonic()
    with np.load(FROZEN_ORACLE) as frozen_scores:
        oracle = frozen_scores['oracle8']
    rows = []
    for unit in PARITY_UNITS:
        paths = sources('parity', unit)
        with np.load(paths['observations']) as observation, np.load(template_path('parity', unit)) as rendered:
            score = score_exact(observation['hist'], rendered['expected'], observation['ambient'])
        difference = float(np.max(np.abs(score-oracle[D0.UNITS.index(unit)])))
        if difference >= 1e-8:
            raise ValueError(f'lambda0 frozen-oracle difference {difference}; teacher path must stop')
        rows.append(dict(unit=unit, maximum_score_difference=difference))
    result = dict(status='PASS', plan_sha256=digest, units=rows, seconds=time.monotonic()-tick,
        frozen_oracle_sha256=U.sha(FROZEN_ORACLE), criterion='All scores for >=2original scenes reproduce frozenoracle8, maxdiff<1e-8',
        source_sha256=source_hashes())
    c.save(path, result)
    return result


def atomic_scores(path, **arrays):
    if path.exists():
        raise FileExistsError('Do not overwrite sealed reference scores')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.partial')
    with partial.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    partial.replace(path)


def score(split='all'):
    c, _, digest = frozen()
    wanted = ['teacher', 'evaluation'] if split == 'all' else [split]
    for phase in wanted:
        barrier(phase)
    check()
    for phase in wanted:
        tick = time.monotonic()
        units = list(c.TRAIN_UNITS if phase == 'teacher' else c.EVAL_UNITS)
        result = []
        for idx, unit in enumerate(units):
            c.check_budget()
            with np.load(sources(phase, unit)['observations']) as observation, np.load(template_path(phase, unit)) as rendered:
                result.append(score_exact(observation['hist'], rendered['expected'], observation['ambient']))
            if (idx+1) % 24 == 0 or idx+1 == len(units):
                print('PILOT2 EXACT SCORE', phase, idx+1, '/', len(units), flush=True)
        result = np.stack(result)
        folder = Path(c.OUT) / ('teacher' if phase == 'teacher' else 'reference')
        folder.mkdir(exist_ok=True)
        if phase == 'teacher':
            llr = result[:, 0]
            path = folder / 'train_scores.npz'
            atomic_scores(path, units=units, llr=llr, frames=FRAMES)
            rowpath = folder / 'rows.npz'
            atomic_scores(rowpath, unit=np.repeat(units, 26), variant=np.zeros(len(units)*26, np.int64),
                replica=np.tile(np.repeat(np.arange(2), 13), len(units)), frame=np.tile(FRAMES, len(units)*2), llr=llr.ravel())
            files = [path, rowpath]
            # Describe teacher targets but never construct or alter network labels here.
            valid, hard = [], []
            for unit in units:
                truth = U.read(sources(phase, unit)['truth'])
                categories = np.asarray(truth['categories'])[0, FRAMES, int(truth['group'])]
                valid.append(np.repeat((categories != 'pass0-10cm')[None], 2, axis=0))
                hard.append(np.repeat(np.char.startswith(categories.astype(str), 'contact')[None], 2, axis=0))
            valid, hard = np.stack(valid), np.stack(hard)
            q = expit(llr)
            active = q[valid]
            summary = dict(target_group_valid_rows=int(valid.sum()), target_group_pass_masked=int((~valid).sum()),
                sigmoid_quantiles=np.quantile(active, [0, .01, .1, .25, .5, .75, .9, .99, 1]).tolist(),
                soft_0p01_0p99=int(((active > .01) & (active < .99)).sum()),
                hard_disagreement=int(((q >= .5) != hard)[valid].sum()),
                absolute_soft_hard_difference_mean=float(np.abs(q-hard)[valid].mean()),
                scope='Targetgroup hard-valid rows only; q is model probability atT1, labels/weights remain parent-owned')
        else:
            path = folder / 'all_scores.npz'
            atomic_scores(path, units=units, D=result, frames=FRAMES)
            files, summary = [path], None
        c.save(folder / 'scores_receipt.json', dict(status='COMPLETE', split=phase, plan_sha256=digest,
            units=units, seconds=time.monotonic()-tick, outputs={str(p):U.sha(p) for p in files},
            score_sha256=U.sha(path),
            source_sha256=source_hashes(), lambda0_check_sha256=U.sha(Path(c.OUT) / 'reference/lambda0_check.json'),
            teacher_summary=summary, limits='Known target/background/current trueanchor; estimated-relative plugin; no sampled photons changed'))


def geometry_check():
    """Cheap engineering check without rendering, GPU or teacher probabilities."""
    c = common()
    count, worst = 0, 0.
    for unit in c.TRAIN_UNITS:
        _, receipt = canonical_boxes(U.read(sources('teacher', unit)['truth']))
        count += 1
        worst = max(worst, receipt['reconstruction_max_abs_m'])
    print('CANONICAL GEOMETRY PASS', count, 'max_reconstruction_m', worst, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['geometry-check', 'render', 'check', 'score'])
    parser.add_argument('--split', default='all', choices=['teacher', 'evaluation', 'all'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    c = common()
    c.setup()
    if args.stage == 'geometry-check':
        geometry_check()
    elif args.stage == 'render':
        render(args.split, args.workers)
    elif args.stage == 'check':
        check()
    else:
        score(args.split)
