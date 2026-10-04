"""Location-marginal R: frozen prior, bounded FP64 templates and exact mixtures.

Inputs deliberately contain only photons, estimated motion, privileged current
anchors and known backgrounds. Target truth is retained for the separate
evaluator and never copied into the candidate prior or likelihood inputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

import cnh_readout_pilot2_reference as P

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
OUT = ROOT/'artifacts.local/work/cnh-location-reference-20261004'
SHAPES = [(w, None, d) for w in (.09, .11) for d in (.09, .15)]+[(.02, .04, .04)]
SHAPE_WEIGHTS = [.225]*4+[.1]
COARSE_SHAPES = [(.10, None, .09), (.10, None, .15), (.02, .04, .04)]
COARSE_WEIGHTS = [.45, .45, .1]
RHOS = [.22+(.65-.22)*(i+.5)/3 for i in range(3)]
Z = [.7+.2*i for i in range(10)]
Z_SENS = [.6+.2*i for i in range(11)]
FRAMES = P.FRAMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8', newline='\n') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def check_deadline(out=OUT):
    if time.time() >= read(Path(out)/'budget.json')['deadline_unix_s']:
        raise TimeoutError('Authorized two-hour measurement budget reached')


def candidates(plan, sensitivity=False):
    """No observation, target identity, label or cohort outcome enters the prior."""
    prior = plan['prior']
    zs = prior['halfstep_z'] if sensitivity else prior['front_z']
    zw = ([.05]+[.1]*9+[.05]) if sensitivity else [1/len(zs)]*len(zs)
    boxes, metadata = [], []
    for side in (-1, 1):
        for band in (0, 1):
            for z, wz in zip(zs, zw):
                for si, (shape, ws) in enumerate(zip(prior['shapes'], prior['shape_weights'])):
                    w, h, depth = shape
                    h = (.36, .34)[band] if h is None else h
                    yc = (.08, .67)[band]
                    for delta in [1., 2., -10., -15., -20.]:
                        a = .30-delta/100
                        xl, xh = (a, a+w) if side == 1 else (-a-w, -a)
                        boxes.append(dict(lo=[xl, yc-h/2, z], hi=[xh, yc+h/2, z+depth]))
                        metadata.append(dict(side=side, band=band, front_z=z, shape=si,
                            delta=delta, prior_weight=.25*wz*ws*(.5 if delta > 0 else 1/3)))
    return boxes, metadata


def prepare():
    """Strip target truth once, preserving all immutable photon replicas."""
    if (OUT/'inputs/receipt.json').exists():
        return read(OUT/'inputs/receipt.json')
    old_plan = read(OLD/'PLAN.json')
    records = []
    for unit in old_plan['eval_units']:
        check_deadline()
        obs = OLD/f'observations/evaluation/unit{unit}.npz'
        truth = OLD/f'truth/evaluation/unit{unit}.json'
        tr = read(truth)
        background = tr['boxes'][0][1:]
        if any(branch[1:] != background for branch in tr['boxes']):
            raise ValueError('Background differs across controlled branches')
        dst = OUT/f'inputs/unit{unit}.npz'
        dst.parent.mkdir(parents=True, exist_ok=True)
        with np.load(obs, allow_pickle=False) as z:
            np.savez(dst, hist=z['hist'], ambient=z['ambient'], sensor=z['sensor'], noisy=z['noisy'])
        bg = OUT/f'inputs/unit{unit}.json'
        save(bg, dict(unit=unit, background=background, privileges='known background including FLOOR/BACK; true current anchor'))
        records.append(dict(unit=unit, original_observation=str(obs), original_observation_sha256=sha(obs),
            original_truth=str(truth), original_truth_sha256=sha(truth), observation=str(dst), observation_sha256=sha(dst),
            background=str(bg), background_sha256=sha(bg)))
    receipt = dict(status='COMPLETE', units=old_plan['eval_units'], records=records,
        target_fields_copied=False, input_contract='hist/ambient/sensor anchors/noisy relative paths and background only',
        old_plan_sha256=sha(OLD/'PLAN.json'), original_observations_resampled=False)
    save(OUT/'inputs/receipt.json', receipt)
    return receipt


def mix(ll, metadata):
    """Accumulate fixed candidate/rho likelihood over time before class mixing."""
    weights = np.array([m['prior_weight'] for m in metadata])[:, None]/len(RHOS)
    weights = np.broadcast_to(weights, ll.shape[:2])
    inside = np.array([m['delta'] > 0 for m in metadata])[:, None]
    inside = np.broadcast_to(inside, weights.shape)
    weighted = ll+np.log(weights)[..., None, None, None]
    def evidence(mask):
        mass = weights[mask].sum()
        if mass <= 0:
            raise ValueError('Empty class prior')
        return logsumexp(weighted[mask], axis=0)-np.log(mass)
    any_score = evidence(inside)-evidence(~inside)
    query = []
    for band in (0, 1):
        target_band = np.array([m['band'] == band for m in metadata])[:, None]
        positive = inside & target_band
        query.append(evidence(positive)-evidence(~positive))
    return any_score, np.stack(query, axis=-1)


def run(out=OUT, sensitivity=False):
    from cnh_location_reference_gpu import ExpectedRenderer
    from cnh_location_reference_likelihood import ExactWindowScorer
    out = Path(out)
    plan = read(out/'PLAN.json')
    ph = sha(out/'PLAN.json')
    for source, digest in plan['source_sha256'].items():
        if sha(source) != digest:
            raise ValueError('Frozen source changed: '+source)
    if sensitivity:
        selection = read(out/'sensitivity_selection.json')
        if selection['plan_sha256'] != ph or selection['units'] not in [plan['units'], plan['sensitivity_fallback_units']]:
            raise ValueError('Sensitivity cohort must be a prospectively frozen profile')
        plan = dict(plan, units=selection['units'])
    destination = out/('sensitivity_scores' if sensitivity else 'scores')
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'all_scores.npz').exists():
        raise FileExistsError('Scores already sealed; evaluate existing output')
    boxes, metadata = candidates(plan, sensitivity)
    any_all, query_all, records = [], [], []
    started = time.monotonic()
    for ui, unit in enumerate(plan['units']):
        check_deadline(out)
        path = OUT/f'inputs/unit{unit}.npz'
        bg = OUT/f'inputs/unit{unit}.json'
        input_record = next(x for x in plan['inputs']['records'] if x['unit'] == unit)
        if sha(path) != input_record['observation_sha256'] or sha(bg) != input_record['background_sha256']:
            raise ValueError('Frozen sanitized input changed')
        chunk = destination/f'unit{unit}.npz'
        rp = chunk.with_suffix('.json')
        if rp.exists():
            record = read(rp)
            if record['plan_sha256'] != ph or sha(chunk) != record['sha256']:
                raise ValueError('Existing chunk has inconsistent receipt')
            with np.load(chunk) as z:
                any_all.append(z['R_any']); query_all.append(z['Rquery'])
            records.append(record)
            continue
        if chunk.exists():
            raise FileExistsError('Unreceipted partial score needs inspection')
        background = read(bg)['background']
        # Prior was geometrically checked before PLAN; never silently reject a nuisance.
        for box in boxes:
            for b in background:
                if np.all(np.minimum(box['hi'], b['hi'])-np.maximum(box['lo'], b['lo']) > 0):
                    raise ValueError('Fixed prior intersects known background; no hidden deletion')
        with np.load(path, allow_pickle=False) as z:
            replicas = plan['replica_indices']
            hist, ambient, sensor, noisy = z['hist'][:, replicas], z['ambient'], z['sensor'], z['noisy'][replicas]
        poses = P.anchored_history_poses(sensor, noisy).reshape(-1, 4, 4)
        tick = time.monotonic()
        renderer = ExpectedRenderer(poses, background)
        engine = None
        render_s, likelihood_s = 0., 0.
        all_ll = np.empty((len(boxes), len(RHOS), 7, plan['K'], 13), np.float64)
        try:
            background_expected = renderer.background_expected(return_device=True)
            engine = ExactWindowScorer(hist, ambient, background_expected)
            render_s += time.monotonic()-tick
            iterator = renderer.iter_render(boxes, candidate_batch=plan['candidate_batch'], pose_batch=32,
                return_device=True, deadline_check=lambda: check_deadline(out))
            while True:
                check_deadline(out)
                tick = time.monotonic()
                try:
                    begin, endpoints = next(iterator)
                except StopIteration:
                    break
                render_s += time.monotonic()-tick
                tick = time.monotonic()
                values, timing = engine.scores(endpoints)
                likelihood_s += time.monotonic()-tick
                all_ll[begin:begin+len(values)] = values
                del endpoints, values
            any_score, query_score = mix(all_ll, metadata)
            if not np.isfinite(any_score).all() or not np.isfinite(query_score).all():
                raise ValueError('Nonfinite R score, retained as error')
            np.savez(chunk, R_any=any_score, Rquery=query_score, candidate_window_relative_ll=all_ll)
            record = dict(status='COMPLETE', unit=unit, plan_sha256=ph, sha256=sha(chunk),
                render_s=render_s, likelihood_s=likelihood_s, seconds=render_s+likelihood_s,
                candidates=len(boxes), metadata=renderer.metadata, likelihood=dict(
                    backend='CuPy FP64 exact signed Skellam series', relative_likelihood=True,
                    background_cache_s=engine.background_seconds, last_tile=timing))
            save(rp, record)
            any_all.append(any_score); query_all.append(query_score); records.append(record)
            print('R_SCENE', ui+1, '/', len(plan['units']), unit, 'render_s', round(render_s, 2),
                'likelihood_s', round(likelihood_s, 2), 'elapsed_s', round(time.monotonic()-started, 2), flush=True)
        finally:
            if engine is not None:
                engine.close()
            renderer.close()
    np.savez(destination/'all_scores.npz', units=plan['units'], frames=FRAMES,
        R_any=np.stack(any_all), Rquery=np.stack(query_all))
    receipt = dict(status='COMPLETE', plan_sha256=ph, score_sha256=sha(destination/'all_scores.npz'),
        sensitivity=sensitivity, unit_records=records, elapsed_s=time.monotonic()-started,
        render_s=sum(r['render_s'] for r in records), likelihood_s=sum(r['likelihood_s'] for r in records),
        template_pose_endpoints=len(plan['units'])*len(boxes)*plan['K']*len(P.PAST)*2)
    save(destination/'receipt.json', receipt)
    print('R_SCORES_SEALED', receipt['elapsed_s'], flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'run'], required=True)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        run(args.out, args.sensitivity)
