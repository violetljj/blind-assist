"""Frozen dual-sensor M3 alarm evaluation on consumed pilot2 Development.

Truth enters only this evaluator. Calibration uses whole-scene clear cost;
evaluation, thresholds, smoothing and all denominators remain separate.
"""
import argparse
from pathlib import Path
import time

import numpy as np

import cnh_location_reference_evaluate as L
import cnh_temporal_readout_evaluate as T
import cnh_three_level_sequence as SE
import cnh_unknown_target_reference_evaluate as U

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-dual-sensor-alarm-20261005'
FRAMES = np.arange(3, 16)
THRESHOLD = .8557642486787612
BOOT_SEED = 2026100515
BOOTSTRAPS = 2000
ARMS = ('single', 'dual_frozen', 'dual_matched', 'alternating_frozen')
BRANCHES = {'shallow': [0, 1], 'deep': [2], 'outside5_10cm': [3, 4], 'outside15_20cm': [5, 6]}


def load_baseline():
    scores, scenes, categories, visible, refs, covered, receipt = L.load_baselines()
    rows, _ = T.rows_for(L.OLD, 'fresh_evaluation')
    raw, _ = T.prediction(L.OLD, 'fresh_evaluation', 'M3')
    both, keys = T.ordered_episodes(raw, rows, fresh=True)
    units = [s['unit'] for s in scenes]
    if keys != [(u, d, k) for u in units for d in range(7) for k in range(4)]:
        raise ValueError('Original two-query baseline identity differs')
    both = both.reshape(48, 7, 4, 13, 2)
    np.testing.assert_array_equal(scores['M3'], select_target(both, scenes))
    all_categories = []
    for scene in scenes:
        truth = U.read(L.OLD/'truth/evaluation'/f"unit{scene['unit']}.json")
        scene['mode'] = int(truth['mode'])
        all_categories.append(np.asarray(truth['categories'])[:, FRAMES, :])
    if set(s['mode'] for s in scenes) != {0, 1}:
        raise ValueError('Frozen pilot2 contains only modes0/1')
    return dict(units=np.asarray(units), scenes=scenes, scores=scores['M3'], both=both,
                categories=categories, all_categories=np.asarray(all_categories),
                covered=covered, refs=refs, receipt=receipt)


def select_target(both, scenes):
    return np.stack([both[i, ..., s['group']] for i, s in enumerate(scenes)])


def smooth_full(raw):
    raw = np.asarray(raw, np.float64)
    if raw.shape[-2:] != (13, 2) or not np.isfinite(raw).all():
        raise ValueError('Full-rate raw scores must be finite, with all13 retained frames')
    return SE.smooth(raw.reshape(-1, 1, 13, 2)).reshape(raw.shape)


def smooth_alternating(raw):
    """Update only on new samples; latest score stays valid at most0.4s.

    The input is newly inferred from the sampled observations, never subsampled
    full-rate logits. Initial absent sensor score is -inf, so it cannot alarm.
    """
    raw = np.asarray(raw, np.float64)
    if raw.shape != (2, 7, 4, 13, 2):
        raise ValueError('Alternating sensor/branch/K/frame/query axes differ')
    result = np.full(raw.shape, -np.inf, np.float64)
    weights = np.asarray([1., 2., 4., 8., 16.])
    for sensor in range(2):
        schedule = FRAMES % 2 == sensor  # L=even, R=odd absolute physical frame
        observed = np.isfinite(raw[sensor]).all(axis=(0, 1, 3))
        missing = np.isnan(raw[sensor]).all(axis=(0, 1, 3))
        if not np.array_equal(observed, schedule) or not np.array_equal(missing, ~schedule):
            raise ValueError('Alternating finite/NaN frames differ from frozen L-even/R-odd schedule')
        history, last = [], None
        for t in range(13):
            if schedule[t]:
                history.append(t)
                last = t
                ids = history[-5:]
                w = weights[-len(ids):]
                current = np.tensordot(raw[sensor, :, :, ids, :], w/w.sum(), axes=([0], [0]))
            if last is not None and (t-last)*.2 <= .4+1e-12:
                result[sensor, :, :, t, :] = current
    return result


def matched_ge_threshold(values, target):
    """Largest whole tied-score alarm set within integer cost, using>=."""
    values = np.asarray(values, np.float64).ravel()
    if not len(values) or not np.isfinite(values).all() or not 0 <= target <= len(values):
        raise ValueError('Nonempty finite calibration clear scores and valid integer target required')
    unique, sizes = np.unique(values, return_counts=True)
    unique, sizes = unique[::-1], sizes[::-1]
    n_groups = int((np.cumsum(sizes) <= target).sum())
    # Smallest representable >= cutoff above the next excluded tied group.
    # Taking the minimum included clear score would unnecessarily raise the
    # alarm cutoff for positive episodes while preserving the same clear cost.
    threshold = (float(np.nextafter(unique[n_groups], np.inf))
                 if n_groups < len(unique) else float(unique[-1]))
    actual = int((values >= threshold).sum())
    assert actual <= target
    assert n_groups == len(unique) or actual+int(sizes[n_groups]) > target
    return dict(threshold=threshold, operator='>=', allowed_stops=int(target), actual_stops=actual,
                n=len(values), unattained_stops=int(target-actual),
                threshold_ties=int((values == threshold).sum()),
                rule='Largest complete tied-score groups within single clear budget; lowest >= cutoff above next excluded tie')


def split_masks(plan, units):
    calibration = plan.get('calibration_units', plan.get('calib_units'))
    evaluation = plan.get('evaluation_units', plan.get('eval_units'))
    if calibration is None or evaluation is None:
        splits = plan.get('split', plan.get('splits', {}))
        calibration = splits.get('calibration', splits.get('calib'))
        evaluation = splits.get('evaluation', splits.get('eval'))
    if (len(calibration or []) != 24 or len(evaluation or []) != 24
            or len(set(calibration+evaluation)) != 48
            or set(calibration+evaluation) != set(units.tolist())):
        raise ValueError('PLAN must partition all48 whole scenes into calibration24/evaluation24')
    return {'all': np.ones(len(units), bool),
            'calibration': np.isin(units, calibration), 'evaluation': np.isin(units, evaluation)}


def boot_weights(keep, seed):
    ids = np.flatnonzero(keep)
    rng = np.random.default_rng(seed)
    return np.asarray([np.bincount(rng.choice(ids, len(ids), replace=True), minlength=len(keep))
                       for _ in range(BOOTSTRAPS)])


def group_masks(scenes):
    masks = L.groups_for(scenes)
    for mode in range(3):
        masks[f'mode{mode}'] = np.array([s['mode'] == mode for s in scenes])
        for fov in ('FOV_OUT', 'FOV_IN'):
            masks[f'mode{mode}/{fov}'] = masks[f'mode{mode}'] & masks[fov]
    return masks


def pooled(flags, den, keep, boot):
    axes = tuple(range(1, flags.ndim))
    metric, draws = U.pooled_rate(flags.sum(axes), den.sum(axes), keep, boot)
    metric['valid_bootstrap_draws'] = int(np.isfinite(draws).sum())
    metric['zero_denominator_bootstrap_draws'] = int((~np.isfinite(draws)).sum())
    return metric, draws


def paired(flags, baseline, den, keep, boot):
    a, draws_a = pooled(flags & den, den, keep, boot)
    b, draws_b = pooled(baseline & den, den, keep, boot)
    return dict(delta_stops=a['stops']-b['stops'], n=a['n'],
                delta_rate=a['rate']-b['rate'] if a['rate'] is not None else None,
                ci95=U.interval(draws_a-draws_b),
                rescues=int((flags & ~baseline & den)[keep].sum()),
                losses=int((baseline & ~flags & den)[keep].sum()))


def source_counts(sensor_scores, threshold, first, stopped, keep):
    """Dominant score at first report; tie means exact equal max scores."""
    if sensor_scores is None:
        return {'single': int(stopped[keep].sum())}
    left = np.take_along_axis(sensor_scores[:, 0], first[..., None], -1)[..., 0]
    right = np.take_along_axis(sensor_scores[:, 1], first[..., None], -1)[..., 0]
    chosen = stopped & keep[:, None, None]
    return dict(L=int((chosen & (left > right)).sum()), R=int((chosen & (right > left)).sum()),
                tie=int((chosen & (left == right)).sum()),
                both_above_threshold=int((chosen & (left >= threshold) & (right >= threshold)).sum()))


def joint_clear_den(data):
    clear = np.all(data['all_categories'][:, [5, 6]] == 'clear', axis=(-2, -1))
    return np.broadcast_to(clear[..., None], (len(clear), 2, 4))


def decision(sequence):
    cells = sequence['evaluation']
    out = cells['FOV_OUT']['dual_matched']['branches']['shallow']['paired_minus_single']
    inside = cells['FOV_IN']['dual_matched']['branches']['shallow']['paired_minus_single']
    clear = cells['all']['dual_matched']['joint_clear']['paired_minus_single']
    gain, loss = out['delta_rate'], inside['delta_rate']
    if gain is None or loss is None or not clear['n']:
        branch = 'NOT_EVALUABLE'
    elif gain < .1-1e-12 or loss < -.03-1e-12:
        branch = 'DUAL_ALARM_NOT_SUPPORTED_SIM'
    elif gain >= .3-1e-12 and loss >= -.03-1e-12 and clear['delta_stops'] <= 1:
        branch = 'DUAL_ALARM_SUPPORTED_SIM'
    else:
        branch = 'MIXED'
    return dict(branch=branch, FOV_OUT_shallow_delta_rate=gain, FOV_OUT_n=out['n'],
                FOV_IN_shallow_delta_rate=loss, FOV_IN_n=inside['n'],
                clear_delta_stops=clear['delta_stops'], clear_n=clear['n'],
                evaluation_clear_tolerance_stops=1, clear_cost_pass=clear['delta_stops'] <= 1,
                rule='Evaluation matched dual: OUT>=+30pp, IN>=-3pp, joint-clear<=single+1; negative if OUT<+10pp or IN<-3pp; otherwise MIXED')


def evaluate(out=OUT):
    out = Path(out)
    started = time.monotonic()
    if (out/'result.json').exists():
        raise FileExistsError('Completed dual alarm result is immutable')
    plan_path = out/'PLAN.json'
    plan = U.read(plan_path)
    plan_sha = U.sha(plan_path)
    if (plan.get('frames') != FRAMES.tolist() or plan.get('K') != 4
            or plan.get('threshold') != THRESHOLD or plan.get('operator') != '>='
            or plan.get('splay_deg') != [-15,15] or plan.get('pitch_deg') != -10):
        raise ValueError('Frozen frame/K/threshold/extrinsic PLAN differs from this evaluator')
    data = load_baseline()
    units, scenes = data['units'], data['scenes']
    if plan['units'] != units.tolist():
        raise ValueError('PLAN units differ from original full48 pilot2 order')
    splits = split_masks(plan, units)
    for split in ('calibration','evaluation'):
        for group in (0,1):
            for mode in (0,1):
                for context in ('none','panel'):
                    count = sum(bool(splits[split][i]) and s['group'] == group
                        and s['mode'] == mode and s['context'] == context for i,s in enumerate(scenes))
                    if count != 3:
                        raise ValueError('Frozen whole-scene split must retain3 scenes per height/mode/context stratum')
    full, alternating, hashes = [], [], {}
    for unit in units:
        path = out/'scores'/f'unit{unit}.npz'
        before = U.sha(path)
        with np.load(path, allow_pickle=False) as z:
            if 'frames' in z and not np.array_equal(z['frames'], FRAMES):
                raise ValueError('Score retained-frame identity differs')
            if 'unit' in z and int(z['unit']) != unit:
                raise ValueError('Score unit identity differs')
            if 'sensors' in z and z['sensors'].tolist() not in (['L', 'R'], ['LEFT', 'RIGHT']):
                raise ValueError('Sensor order must be L/R')
            raw, alt = z['raw_full'].copy(), z['raw_alternating'].copy()
        if raw.shape != (2, 7, 4, 13, 2):
            raise ValueError('Full-rate scores must retain2sensor/7delta/K4/13frame/2query axes')
        full.append(smooth_full(raw))
        alternating.append(smooth_alternating(alt))
        if before != U.sha(path):
            raise ValueError('Score changed while evaluating: '+str(path))
        hashes[str(path)] = before
    sensor_full, sensor_alt = np.stack(full), np.stack(alternating)
    both = {'single': data['both'], 'dual_frozen': sensor_full.max(1),
            'dual_matched': sensor_full.max(1), 'alternating_frozen': sensor_alt.max(1)}
    clear_den = joint_clear_den(data)
    peak = {arm: val[:, [5, 6]].max(axis=(-2, -1)) for arm, val in both.items()}
    calibration_den = clear_den & splits['calibration'][:, None, None]
    target = int((peak['single'][calibration_den] >= THRESHOLD).sum())
    thresholds = {arm: dict(threshold=THRESHOLD, operator='>=', origin='Frozen original M3 threshold')
                  for arm in ARMS}
    thresholds['dual_matched'] = matched_ge_threshold(peak['dual_matched'][calibration_den], target)
    thresholds['dual_matched']['origin'] = 'This frozen whole-scene calibration split; one global clear-cost threshold'
    ranges = np.asarray([s['front_range_m'] for s in scenes])
    selected = {arm: select_target(val, scenes) for arm, val in both.items()}
    ledgers = {arm: L.events(selected[arm], thresholds[arm], ranges, data['covered']) for arm in ARMS}
    joint_stopped = {arm: peak[arm] >= thresholds[arm]['threshold'] for arm in ARMS}
    sensor_target = {arm: None if arm == 'single' else np.stack([
        select_target((sensor_alt if arm == 'alternating_frozen' else sensor_full)[:, s], scenes)
        for s in range(2)], axis=1) for arm in ARMS}
    groups, sequence = group_masks(scenes), {}
    for split_index, (split, split_keep) in enumerate(splits.items()):
        boot = boot_weights(split_keep, BOOT_SEED+split_index)
        sequence[split] = {}
        for group, group_keep in groups.items():
            keep = split_keep & group_keep
            cells = sequence[split][group] = {}
            for arm in ARMS:
                event, base = ledgers[arm], ledgers['single']
                cell = cells[arm] = dict(scenes=int(keep.sum()), branches={})
                for name, ids in BRANCHES.items():
                    den = np.ones((len(units), len(ids), 4), bool)
                    valid = np.broadcast_to(data['covered'][:, ids, None], den.shape)
                    stopped, _ = pooled(event['stopped'][:, ids], den, keep, boot)
                    timely, _ = pooled(event['timely'][:, ids] & valid, valid, keep, boot)
                    sensor = sensor_target[arm]
                    cell['branches'][name] = dict(first_stops=stopped, timely=timely,
                        censored_n=int((~valid)[keep].sum()),
                        paired_minus_single=paired(event['timely'][:, ids], base['timely'][:, ids], valid, keep, boot),
                        first_report_source=source_counts(None if sensor is None else sensor[:, :, ids],
                            thresholds[arm]['threshold'], event['first_index'][:, ids], event['stopped'][:, ids], keep))
                metric, _ = pooled(joint_stopped[arm] & clear_den, clear_den, keep, boot)
                minutes = metric['n']*13*.2/60
                cell['joint_clear'] = dict(**metric, proxy_minutes=minutes,
                    first_stops_per_proxy_minute=metric['stops']/minutes if minutes else None,
                    denominator_unit='physical episodes: both HEAD/BODY queries clear in all13frames; either query stops; outside15/20cm',
                    paired_minus_single=paired(joint_stopped[arm], joint_stopped['single'], clear_den, keep, boot))
    np.testing.assert_equal(sequence['calibration']['all']['dual_matched']['joint_clear']['stops'],
                            thresholds['dual_matched']['actual_stops'])
    if U.sha(plan_path) != plan_sha:
        raise ValueError('Frozen PLAN changed during evaluation')
    evaluation_dir = out/'evaluation'
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = evaluation_dir/'ledger.npz'
    np.savez_compressed(ledger_path, units=units, frames=FRAMES,
        full_sensor_smoothed=sensor_full, alternating_sensor_smoothed=sensor_alt,
        both_categories=data['all_categories'], covered=data['covered'], joint_clear_den=clear_den,
        **{arm+'_'+field: e[field] for arm, e in ledgers.items() for field in ('alarm','stopped','timely','first_index','first_range')},
        **{arm+'_both_query_scores': value for arm, value in both.items()},
        **{arm+'_joint_clear_stopped': value for arm, value in joint_stopped.items()})
    result = dict(status='COMPLETE', task='Dual +/-15deg alarm; consumed synthetic Development',
        units=units.tolist(), scenes=scenes, split_units={name: units[mask].tolist() for name, mask in splits.items()},
        thresholds=thresholds, sequence=sequence, decision=decision(sequence),
        bootstrap=dict(n=BOOTSTRAPS, seed=BOOT_SEED, unit='Whole paired scenes; within each split; thresholds fixed'),
        smoothing=dict(full='Original causal trailing5 retained-frame weighted logits, [1,2,4,8,16]',
            alternating='New observed logits only, trailing5 observations; L even/R odd absolute frame; hold latest <=0.4s',
            fusion='Max of two separately smoothed sensor scores for each public query'),
        baseline_identity=data['receipt'], provenance=dict(plan_sha256=plan_sha, score_sha256=hashes,
            evaluator_sha256=U.sha(__file__), ledger_sha256=U.sha(ledger_path)),
        evaluation_s=time.monotonic()-started,
        limits=['Reused synthetic Development; geometric rendering/photon assumptions, no real-user safety claim',
            'FOV labels frozen from original single sensor; mode2 unavailable, denominator0',
            'K4 realizations are correlated; confidence intervals resample whole scenes',
            'One threshold calibrated on24 scenes; evaluation clear cost must independently meet +1stop tolerance',
            'Both-query all-object-clear outer15/20cm first stops are a proxy cost, not natural walking burden',
            '2.5Hz alternating is a pressure test, not measured dual hardware rate',
            'No measured dual SNR, crosstalk, energy or hardware alarm performance'])
    L.save(out/'result.json', result)
    rows = ['split,group,arm,scenes,shallow_stops,shallow_n,deep_stops,deep_n,joint_clear_stops,joint_clear_n']
    for split, groups_result in sequence.items():
        for group, cells in groups_result.items():
            for arm, cell in cells.items():
                a, b, c = cell['branches']['shallow']['timely'], cell['branches']['deep']['timely'], cell['joint_clear']
                rows.append(f"{split},{group},{arm},{cell['scenes']},{a['stops']},{a['n']},{b['stops']},{b['n']},{c['stops']},{c['n']}")
    (evaluation_dir/'counts.csv').write_text('\n'.join(rows)+'\n', encoding='utf8')
    return result


def check():
    for values, budget, expected in [([2.,2.,1.,0.],1,0), ([2.,2.,1.,0.],2,2), ([2.,2.,1.,0.],4,4)]:
        spec = matched_ge_threshold(values, budget)
        assert spec['actual_stops'] == expected
        assert int((np.asarray(values) >= spec['threshold']).sum()) == expected
    assert matched_ge_threshold([2.,2.,1.,0.],2)['threshold'] == np.nextafter(1., np.inf)
    raw = np.full((2,7,4,13,2), np.nan)
    for s in range(2):
        for t, frame in enumerate(FRAMES):
            if frame % 2 == s:
                raw[s,:,:,t,:] = t
    value = smooth_alternating(raw)
    assert np.isneginf(value[0,:,:,0]).all()
    assert np.array_equal(value[0,:,:,1], value[0,:,:,2])
    assert np.array_equal(value[1,:,:,0], value[1,:,:,1])
    np.testing.assert_allclose(value[0,:,:,3], (1.*8+3.*16)/24)
    finite = np.arange(2*7*4*13*2).reshape(2,7,4,13,2).astype(float)
    full = smooth_full(finite)
    np.testing.assert_array_equal(full[...,0,:], finite[...,0,:])
    np.testing.assert_allclose(full[...,1,:], (finite[...,0,:]*8+finite[...,1,:]*16)/24)
    # Hand-set first reports distinguish dominant source, exact tie, and a
    # simultaneous two-sensor alarm without treating it as an exact-score tie.
    sensor = np.zeros((1,2,2,2,3))
    sensor[0,0,0,0,1], sensor[0,1,0,0,1] = 2., 1.5
    sensor[0,0,0,1,2], sensor[0,1,0,1,2] = 0., 2.
    sensor[0,:,1,0,0] = 2.
    first = np.array([[[1,2],[0,0]]])
    stopped = np.array([[[True,True],[True,False]]])
    sources = source_counts(sensor, 1., first, stopped, np.array([True]))
    assert sources == dict(L=1,R=1,tie=1,both_above_threshold=2)
    empty = source_counts(sensor,1.,first,stopped,np.array([False]))
    assert not any(empty.values())
    synthetic_truth = np.full((1,7,13,2), 'clear')
    synthetic_truth[0,6,12,1] = 'contact0-2cm'
    joint_den = joint_clear_den({'all_categories':synthetic_truth})
    assert joint_den.shape == (1,2,4) and joint_den[0,0].all() and not joint_den[0,1].any()
    def decision_case(gain, inside, clear):
        sequence = {'evaluation': {group: {'dual_matched': {
            'branches': {'shallow': {'paired_minus_single': {'delta_rate':change,'n':48}}},
            'joint_clear': {'paired_minus_single': {'delta_stops':clear,'n':192}}}}
            for group,change in [('FOV_OUT',gain),('FOV_IN',inside),('all',0.)]}}
        return decision(sequence)['branch']
    assert decision_case(.3,-.03,1) == 'DUAL_ALARM_SUPPORTED_SIM'
    assert decision_case(.3,0.,2) == 'MIXED'
    assert decision_case(.1,0.,0) == 'MIXED'
    assert decision_case(.099,0.,0) == 'DUAL_ALARM_NOT_SUPPORTED_SIM'
    assert decision_case(.9,-.031,0) == 'DUAL_ALARM_NOT_SUPPORTED_SIM'
    print('PASS >= threshold whole ties; alternating observed-only smoothing/hold/initial missing; full causal smoothing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
    else:
        print(evaluate(args.out)['decision'])
