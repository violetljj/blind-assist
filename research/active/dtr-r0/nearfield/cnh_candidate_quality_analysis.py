"""Candidate quality-dose diagnostic on repaired v2 consumed Development.

Explicit saved split: calib96..127; audit128..191 except INCOMPLETE143.
Reuse prior metric/selection functions without changing their module globals.
This is descriptive privileged simulation, not fresh or camera evidence.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cnh_position_prior_analysis as prior

FAMILY = 'cnh-track-a-scale-v2-20260926'
ARMS = ('G0', 'G1', 'FA1', 'FA3', 'FA10', 'SHIFT025', 'SHIFT05', 'SHIFT1',
        'DILATE05', 'DILATE1', 'DROP20', 'DROP40', 'COMBO_MODERATE',
        'COMBO_SEVERE', 'BGFA1', 'BGFA3', 'BGFA10', 'G4prime_local')
POLICIES = ARMS[:-1]+('G4prime',)
AXES = {'extra_candidates': ('FA1', 'FA3', 'FA10'),
        'angular_offset': ('SHIFT025', 'SHIFT05', 'SHIFT1'),
        'dilation': ('DILATE05', 'DILATE1'), 'drop_probability': ('DROP20', 'DROP40'),
        'background_candidates': ('BGFA1', 'BGFA3', 'BGFA10'),
        'combined': ('COMBO_MODERATE', 'COMBO_SEVERE')}


def load(root, require_complete=True):
    seqs = {'calib': [], 'eval': []}
    units = {'calib': [], 'eval': []}
    parity = {}
    for path in sorted(Path(root).glob('unit*.npz'), key=lambda p: int(p.stem[4:])):
        unit = int(path.stem[4:])
        with np.load(path, allow_pickle=False) as packed:
            f = {key: packed[key] for key in packed.files}
            split = str(f['split'])
            if split not in ('calib', 'audit'):
                raise ValueError(f'{path}: unexpected split {split}')
            if (split == 'calib' and unit not in range(96, 128)) or (split == 'audit' and (unit not in range(128, 192) or unit == 143)):
                raise ValueError(f'{path}: outside authorized repaired v2 cohort')
            if 'family' in f and str(f['family']) != FAMILY:
                raise ValueError(f'{path}: wrong family')
            outsplit = 'calib' if split == 'calib' else 'eval'
            units[outsplit].append(unit)
            absolute = float(f['G0_consistency_max_abs'])
            relative = float(f['G0_consistency_max_rel'])
            if not np.isfinite([absolute, relative]).all() or min(absolute, relative) < 0 or relative > 1e-5:
                raise ValueError(f'{path}: invalid or failed G0 reference parity')
            parity[str(unit)] = dict(split=outsplit, max_abs=absolute, max_rel=relative)
            frame, cfg = f['frame'], f['config']
            for key in ('labels', 'witness', 'strata', *ARMS):
                if f[key].shape != (len(frame), 6):
                    raise ValueError(f'{path}: wrong {key} shape')
            for arm in ARMS:
                if np.isnan(f[arm]).any() or np.isposinf(f[arm]).any():
                    raise ValueError(f'{path}: invalid {arm} score')
            for c in np.unique(cfg):
                ix = np.flatnonzero((cfg == c) & (frame >= 3))
                ix = ix[np.argsort(frame[ix])]
                if not np.array_equal(frame[ix], np.arange(3, 12)):
                    raise ValueError(f'{path}: incomplete or duplicated evaluated frames')
                seqs[outsplit].append(dict(unit=unit, y=f['labels'][ix], w=f['witness'][ix],
                    strata=f['strata'][ix].astype(str), s={a: f[a][ix] for a in ARMS}))
    if require_complete and (units['calib'] != list(range(96, 128)) or units['eval'] != [u for u in range(128, 192) if u != 143]):
        raise ValueError('Incomplete authorized cohort; do not analyze partial outputs')
    if not all(seqs.values()):
        raise ValueError('Both splits required')
    summary = dict(reference='Newly generated unmodified GPU2 S2 on repaired v2 inputs; no historical v2 readout files existed',
        tolerance_max_rel=1e-5, per_unit=parity)
    for split in ('all', 'eval'):
        rows = [r for r in parity.values() if split == 'all' or r['split'] == split]
        summary[split] = dict(units=len(rows), max_abs=max(r['max_abs'] for r in rows), max_rel=max(r['max_rel'] for r in rows))
    return seqs, units, summary


def pack(seqs, boxes):
    # prior.pack defines and validates labels, witnesses and strata. Its fixed
    # score-name list receives harmless aliases; actual policy arrays replace it.
    aliases = [dict(sq, s={a: sq['s']['G0'] for a in prior.ARMS}) for sq in seqs]
    p = prior.pack(aliases, boxes)
    scores = {}
    for arm in ARMS:
        score = np.full(p['y'].shape, -np.inf)
        i = 0
        for sq in seqs:
            for q in boxes:
                score[i, :len(sq['y'])] = sq['s'][arm][:, q]
                i += 1
        scores[arm] = score
    scores['G4local'] = scores['G4prime_local']
    p['s'] = scores
    return p


def quality_checks(results):
    checks = {}
    for group in prior.GROUPS:
        ref = results[f'{group}/G0/0.10']['eval']
        ideal = results[f'{group}/G1/0.10']['eval']['tiny']['timely']['rate']
        base = ref['tiny']['timely']['rate']
        base_fa = ref['all']['false_alert']['rate']
        gain = ideal-base if ideal is not None and base is not None else None
        rows = {}
        for arm in POLICIES:
            row = results[f'{group}/{arm}/0.10']['eval']
            timely = row['tiny']['timely']['rate']
            fa = row['all']['false_alert']['rate']
            delta = timely-base if timely is not None and base is not None else None
            retention = delta/gain if gain is not None and gain > 0 and delta is not None else None
            excess = fa-base_fa if fa is not None and base_fa is not None else None
            over = bool(excess > .02) if excess is not None else None
            rows[arm] = dict(tiny_timely=row['tiny']['timely'], actual_false_alert=row['all']['false_alert'],
                gain_over_G0=delta, retention_of_G1_gain=retention, false_alert_delta_over_G0=excess,
                false_alert_exceeds_G0_by_2pp=over,
                meets_quality_spec=bool(retention >= .5 and not over) if retention is not None and over is not None else None)
        checks[group] = dict(G1_gain_over_G0=gain, policies=rows,
            axis_tested_values_meeting_spec={axis: [a for a in arms if rows[a]['meets_quality_spec'] is True] for axis, arms in AXES.items()})
    return checks


def analyze(root):
    seqs, units, parity = load(root)
    out = dict(source_family=FAMILY, scope='Repaired v2 consumed Development; privileged simulation, not real camera performance or formal reproduction',
        unit_ids=units, excluded_original_incomplete=[143], G0_reference_parity=parity,
        selection=dict(single_grid='linspace(1,12,441)+infinity; lowest feasible threshold',
            G4prime_grid='local linspace(1,12,45)+infinity; global same plus budget-specific G0 threshold',
            G4prime_objective='per-group pooled calib near timely count; ties fewer FA, larger global threshold, larger local threshold',
            quality_spec='At10pct: tiny gain retention>=0.5 AND eval overall FA<=G0 overall FA+0.02. Undefined retention when G1 gain<=0.',
            scope_of_spec='Separate tested axis values only; no interpolation or simultaneous guarantee. Combined arms are separate tests.',
            size_FA='Empty pairs lack target size; report group-wide FA only'), results={})
    for group, boxes in prior.GROUPS.items():
        ca, ev = pack(seqs['calib'], boxes), pack(seqs['eval'], boxes)
        for arm in POLICIES:
            internal = 'G4' if arm == 'G4prime' else arm
            for budget in prior.BUDGETS:
                thresholds = prior.select(ca, internal, budget)
                out['results'][f'{group}/{arm}/{budget:.2f}'] = dict(thresholds=prior.threshold_json(thresholds),
                    calib=prior.evaluate(ca, internal, thresholds), eval=prior.evaluate(ev, internal, thresholds))
    # Flag actual FA differences for every policy and budget, not just main gate.
    for key, row in out['results'].items():
        group, arm, budget = key.split('/')
        base_fa = out['results'][f'{group}/G0/{budget}']['eval']['all']['false_alert']['rate']
        fa = row['eval']['all']['false_alert']['rate']
        delta = fa-base_fa if fa is not None and base_fa is not None else None
        row['eval_false_alert_delta_over_G0'] = delta
        row['eval_false_alert_exceeds_G0_by_2pp'] = bool(delta > .02) if delta is not None else None
    out['quality_at_10pct'] = quality_checks(out['results'])
    g4 = {k: v['thresholds'] for k, v in out['results'].items() if '/G4prime/' in k}
    disabled = all(t[1] == 'disabled' for t in g4.values())
    out['G4prime_decision'] = dict(thresholds=g4, local_disabled_all_six=disabled,
        abandon_this_exact_route=disabled, statement='Stop this exact G4prime recipe if local disabled at all six points; no universal claim about temporal priors.')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores')
    parser.add_argument('output')
    args = parser.parse_args()
    result = analyze(args.scores)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['G4prime_decision'], indent=2))


if __name__ == '__main__':
    main()
