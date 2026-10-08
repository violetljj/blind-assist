"""CPU-only frozen query-perturbation evaluation; no model fitting or file writes.

Raw axes are [model, exact/E1, branch(0 single,1/2 dual), clip,13,2height].
FA intervals are .2-second proxies in independent two-second clear clips.
Clip episodes and maximal positive runs reset at every clip; neither is a
continuous-session event rate. Same cost means intervals only, not episodes.
"""
from __future__ import annotations

import numpy as np

QUERIES = ('exact', 'E1')
SENSORS = ('single', 'dual')
MODELS = ('frozen_M3_5', 'frozen_M3_seed0', 'Control', 'Aug')
FA_RATE = .025


def interval_budget(control_clips):
    """floor(.025 * clips * 10), integer algebra avoids float floor drift."""
    return int(control_clips)//4


def load_metadata(rows):
    """Optional read-only adapter to the inherited evaluator geometry ledger."""
    import cnh_tristate_event_dev as E
    # Same geometry definition as E.load_geometry, but allow a subset and cache
    # npz members once (the inherited loader requires every source row).
    n = len(rows)
    lookup = {(r['unit'], r['config']): i for i, r in enumerate(rows)}
    if len(lookup) != n:
        raise ValueError('Duplicate unit/config metadata key')
    category = np.full((n, 2), 'MISSING', dtype='<U20')
    clear = np.zeros((n, 2), bool)
    covered = np.zeros((n, 2), bool)
    fraction = np.full((n, 2), np.nan)
    for path in E.GEOMETRY:
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in ('unit', 'config', 'query', 'ref_category',
                                   'clear_all', 'covered', 'reference_fraction')}
        for j, (u, c) in enumerate(zip(d['unit'], d['config'])):
            i = lookup.get((int(u), int(c)))
            if i is None:
                continue
            q = int(d['query'][j])
            if category[i, q] != 'MISSING':
                raise ValueError('Duplicate source geometry query')
            category[i, q] = d['ref_category'][j]
            clear[i, q] = d['clear_all'][j]
            covered[i, q] = d['covered'][j]
            fraction[i, q] = d['reference_fraction'][j]
    if (category == 'MISSING').any():
        raise ValueError('Missing source geometry query')
    np.testing.assert_array_equal(covered[:, 0], covered[:, 1])
    np.testing.assert_allclose(fraction[:, 0], fraction[:, 1], equal_nan=True)
    contact = (covered & np.isin(category, E.CATS)).any(1)
    return dict(unit=np.array([r['unit'] for r in rows]),
                config=np.array([r['config'] for r in rows]),
                contact=contact, control=clear.all(1),
                deadline=E.causal_index(fraction[:, 0]))


def _metadata(metadata, n):
    out = {k: np.asarray(metadata[k]) for k in ('unit', 'config', 'contact', 'control', 'deadline')}
    if any(v.shape != (n,) for v in out.values()):
        raise ValueError('Metadata must have one value per clip')
    for k in ('contact', 'control'):
        if not np.isin(out[k], (0, 1)).all():
            raise ValueError(f'{k} must be boolean')
        out[k] = out[k].astype(bool)
    if (out['contact'] & out['control']).any():
        raise ValueError('Contact and strict-clear control clips overlap')
    d = out['deadline'][out['contact']]
    if not np.isfinite(d).all() or not np.equal(d, np.floor(d)).all() or ((d < 0) | (d > 12)).any():
        raise ValueError('Contact output deadline must be integer 0..12')
    # Invalid/noncontact deadlines never enter an event test.
    out['deadline'] = np.where(out['contact'], out['deadline'], -1).astype(int)
    for k, default in (('evaluation', out['unit'] >= 99000), ('calibration', out['unit'] < 99000)):
        out[k] = np.asarray(metadata.get(k, default), dtype=bool)
        if out[k].shape != (n,):
            raise ValueError(f'{k} must have one boolean per clip')
    if (out['evaluation'] & out['calibration']).any():
        raise ValueError('Calibration and evaluation overlap')
    return out


def fuse(raw):
    """Original R.smooth algebra per branch/height, then fuse, never reverse."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 6 or raw.shape[1:3] != (2, 3) or raw.shape[-2:] != (13, 2):
        raise ValueError('Expected raw[models,2queries,3branches,N,13,2height]')
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite raw score')
    sm = np.empty_like(raw)
    # Identical float64 operation order to cnh_tristate_dev.smooth.
    for f in range(13):
        begin = max(0, f-4)
        w = 2.**np.arange(f-begin+1)
        sm[..., f, :] = np.sum(raw[..., begin:f+1, :]*w[:, None], axis=-2)/w.sum()
    height = sm.max(-1)
    return np.stack((height[:, :, 0], height[:, :, 1:3].max(2)), axis=2)


def select_threshold(values, budget):
    """Lowest float64 >= threshold within budget, rejecting whole boundary tie."""
    v = np.asarray(values, dtype=float).ravel()
    if not v.size or not np.isfinite(v).all():
        raise ValueError('Finite nonempty clear-control score sample required')
    if isinstance(budget, (bool, np.bool_)) or int(budget) != budget or not 0 <= budget <= len(v):
        raise ValueError('Integer budget outside available intervals')
    budget = int(budget)
    boundary = None if budget == len(v) else float(np.sort(v)[len(v)-budget-1])
    theta = -np.inf if boundary is None else float(np.nextafter(boundary, np.inf))
    actual = int((v >= theta).sum())
    predecessor = None if boundary is None else int((v >= boundary).sum())
    if actual > budget or (predecessor is not None and predecessor <= budget):
        raise AssertionError('Whole-tie minimality failed')
    return dict(threshold=theta, target_fa_intervals=budget, actual_fa_intervals=actual,
                residual_fa_intervals=budget-actual, boundary_score=boundary,
                boundary_tie_count=0 if boundary is None else int((v == boundary).sum()),
                predecessor_fa_intervals=predecessor,
                cost_status='COST_MATCHED' if actual == budget else 'NOT_COST_MATCHED')


def _event_vectors(score, theta, g, selected):
    alarm = np.asarray(score) >= theta
    eligible = g['contact'] & selected
    outputs = np.arange(13)[None]
    before = (outputs <= g['deadline'][:, None]) & eligible[:, None]
    active = alarm & before
    main = active & (outputs >= 2)
    return alarm, main.any(1), active.any(1), np.where(main.any(1), main.argmax(1), -1)


def summarize(score, theta, metadata, selection=None):
    """One fixed-threshold result. Main timely includes outputs2..deadline."""
    score = np.asarray(score, dtype=float)
    if score.ndim != 2 or score.shape[1] != 13 or not np.isfinite(score).all():
        raise ValueError('Expected finite score[N,13]')
    g = _metadata(metadata, len(score))
    selected = g['evaluation'] if selection is None else np.asarray(selection, bool)
    if selected.shape != (len(score),):
        raise ValueError('Selection shape differs from clips')
    alarm, timely, any_before, first = _event_vectors(score, theta, g, selected)
    ctl = g['control'] & selected
    main = alarm[ctl, 2:12]
    runs = int(main[:, 0].sum() + (main[:, 1:] & ~main[:, :-1]).sum())
    intervals = int(main.sum())
    clips = int(ctl.sum())
    episodes = int(main.any(1).sum())
    events = g['contact'] & selected
    rr = np.flatnonzero(events)
    return dict(events=int(events.sum()), timely=int(timely.sum()),
                any_before=int(any_before.sum()), startup_only_timely=int((any_before & ~timely).sum()),
                false_alarm=dict(interval_count=intervals, interval_denominator=clips*10,
                    interval_rate=intervals/(clips*10) if clips else None,
                    alarm_seconds_proxy=intervals*.2, control_seconds_proxy=clips*2.,
                    mergeclip_episode_count=episodes, control_clip_denominator=clips,
                    maximal_positive_run_count=runs,
                    interpretation='Independent clips; interval cost only. Runs and mergeclip episodes are descriptive, reset per clip.'),
                event_ledger=[dict(unit=int(g['unit'][i]), config=int(g['config'][i]),
                    deadline_output=int(g['deadline'][i]), first_main_output=int(first[i]),
                    timely=bool(timely[i]), any_before=bool(any_before[i])) for i in rr])


def paired(candidate, reference):
    """Paired timely flags in a shared event cohort, including losses separately."""
    a, b = np.asarray(candidate, bool), np.asarray(reference, bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError('Paired event vectors differ')
    rescued, lost = int((a & ~b).sum()), int((~a & b).sum())
    return dict(rescues=rescued, losses=lost, net=rescued-lost,
                both_timely=int((a & b).sum()), both_not_timely=int((~a & ~b).sum()))


def decision(e1_aug_control, e1_aug_frozen5, exact_aug_frozen5, cost_matched=True):
    """Only the three authorized count conditions; no added E1-loss gate."""
    conditions = dict(E1_Aug_gt_Control=e1_aug_control['net'] >= 1,
                      E1_Aug_gt_frozen_M3_5=e1_aug_frozen5['net'] >= 1,
                      exact_paired_preserved=exact_aug_frozen5['losses'] <= 2 and exact_aug_frozen5['net'] >= -2)
    status = 'NOT_COST_MATCHED' if not cost_matched else 'PASS' if all(conditions.values()) else 'DO_NOT_ADVANCE'
    return dict(status=status, conditions=conditions,
                interpretation='Single seeded systems versus five-seed ensemble; first screening is not mechanism attribution.')


def evaluate(raw, metadata, model_names=MODELS):
    """Primary evaluation whole-tie match plus calibration-only secondary.

    Primary selection uses only evaluation strict-clear scores. Each model and
    each query has its own threshold under the same integer interval budget. A tie
    residual invalidates cost matching instead of being silently tolerated.
    Each budget is floor(.025 * its strict-clear clips * 10). Secondary selects
    on calibration clips only, then holds theta on evaluation.
    """
    names = tuple(model_names)
    if len(set(names)) != len(names) or not set(MODELS).issubset(names):
        raise ValueError(f'Unique model_names must include {MODELS}')
    score = fuse(raw)
    if len(names) != len(score):
        raise ValueError('Model axis differs from model_names')
    g = _metadata(metadata, score.shape[3])
    ec, cc = g['evaluation'] & g['control'], g['calibration'] & g['control']
    eval_controls, cal_controls = int(ec.sum()), int(cc.sum())
    if not eval_controls or not cal_controls:
        raise ValueError('Both calibration and evaluation require strict-clear clips')
    eval_budget, cal_budget = interval_budget(eval_controls), interval_budget(cal_controls)
    result = dict(schema='cnh_query_perturb_evaluation_v1',
        primary_contract=dict(eval_fa_interval_budget=eval_budget,
            eval_control_clips=eval_controls, eval_control_intervals=eval_controls*10,
            target_fa_rate=FA_RATE, budget_formula='floor(.025 * strict-clear clips * 10)',
            alarm_seconds_proxy_budget=eval_budget*.2, control_seconds_proxy=eval_controls*2.,
            selection='Eval strict-clear scores only, independent arm/query thresholds, whole ties, no contact objective',
            same_cost=f'{eval_budget} positive .2-second intervals; not equal episode or positive-run counts',
            timely='Outputs2..deadline inclusive; input=output+3; any-before separate',
            startup_hold='Outputs0:2 excluded from main timely and clear-cost; output12 may be timely but is outside clear-cost span2:12'),
        secondary_contract=dict(calibration_fa_interval_budget=cal_budget,
            calibration_control_clips=cal_controls, calibration_intervals=cal_controls*10,
            target_fa_rate=FA_RATE, budget_formula='floor(.025 * strict-clear clips * 10)',
            selection='Calibration clear only; hold threshold for evaluation actual FA'), sensors={})
    for si, sensor in enumerate(SENSORS):
        primary, secondary, flags, matches = {}, {}, {}, {}
        for mi, model in enumerate(names):
            for qi, query in enumerate(QUERIES):
                x = score[mi, qi, si]
                key = f'{model}/{query}'
                th = select_threshold(x[ec, 2:12], eval_budget)
                primary[key] = dict(threshold_selection=th, **summarize(x, th['threshold'], g))
                flags[model, query] = _event_vectors(x, th['threshold'], g, g['evaluation'])[1]
                matches[model, query] = th['cost_status'] == 'COST_MATCHED'
                ct = select_threshold(x[cc, 2:12], cal_budget)
                secondary[key] = dict(threshold_selection=ct, evaluation=summarize(x, ct['threshold'], g),
                    calibration=summarize(x, ct['threshold'], g, g['calibration']))
        pairs = {}
        for query in QUERIES:
            for a, b in (('Aug', 'Control'), ('Aug', 'frozen_M3_5'), ('Aug', 'frozen_M3_seed0'), ('Control', 'frozen_M3_5')):
                event = g['contact'] & g['evaluation']
                pairs[f'{query}/{a}-vs-{b}'] = paired(flags[a, query][event], flags[b, query][event])
        required = (('Aug', 'E1'), ('Control', 'E1'), ('frozen_M3_5', 'E1'), ('Aug', 'exact'), ('frozen_M3_5', 'exact'))
        dec = decision(pairs['E1/Aug-vs-Control'], pairs['E1/Aug-vs-frozen_M3_5'],
                       pairs['exact/Aug-vs-frozen_M3_5'], all(matches[k] for k in required))
        result['sensors'][sensor] = dict(primary=primary, secondary=secondary, paired=pairs, decision=dec)
    return result
