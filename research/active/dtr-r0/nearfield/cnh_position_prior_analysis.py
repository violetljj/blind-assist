"""Consumed-v3 position-prior diagnostic. No formal or camera claims.

Input: unitNNN.npz with config/frame, labels/witness/strata [N,6], and
G0,G1,G2,G3a,G3b,G3c,G3d,G4local scores [N,6]. Even unit IDs calibrate;
odd IDs evaluate. All events, thresholds and distance bins use frame >=3.
An event is a sequence x query with positive frames; its first alert counts
only on positive frames, as in the frozen v4 analysis. Empty pairs have no
positive frame. Stratum is the modal positive-frame stratum (v4 convention).
G4 jointly selects global/local thresholds, separately for HEAD and BODY,
maximizing pooled calibration near-event timely count subject to FA budget;
ties minimize false alerts, then prefer larger global, then local thresholds.
Its bounded grid is 45 evenly spaced values from 1 to 12 plus infinity
(disabled branch). The global grid additionally includes the selected G0
calibration threshold for that budget, preserving the G0-only feasible point.
Other arms use 441 values plus infinity.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ARMS = ('G0', 'G1', 'G2', 'G3a', 'G3b', 'G3c', 'G3d', 'G4local')
GROUPS = {'HEAD': (0, 2, 4), 'BODY': (1, 3, 5)}
BUDGETS = (.05, .1, .2)
GRID = np.r_[np.linspace(1., 12., 441), np.inf]
JOINT_GRID = np.r_[np.linspace(1., 12., 45), np.inf]
BINS = ((.3, 1.), (1., 1.5), (1.5, 2.), (2., 3.))


def load(root):
    seqs = {'calib': [], 'eval': []}
    consistency = {}
    for p in sorted(Path(root).glob('unit*.npz')):
        unit = int(p.stem[4:])
        with np.load(p, allow_pickle=False) as f:
            cfg, frame = f['config'], f['frame']
            for arm in ARMS:
                if f[arm].shape != (len(frame), 6):
                    raise ValueError(f'{p}: invalid {arm} shape')
                if np.isnan(f[arm]).any() or np.isposinf(f[arm]).any():
                    raise ValueError(f'{p}: NaN or positive infinity in {arm}')
            if 'G0_consistency_max_abs' in f:
                consistency[str(unit)] = float(f['G0_consistency_max_abs'])
            for c in np.unique(cfg):
                m = np.flatnonzero((cfg == c) & (frame >= 3))
                m = m[np.argsort(frame[m])]
                if not len(m):
                    continue
                if len(np.unique(frame[m])) != len(m):
                    raise ValueError(f'{p}: duplicate frame in config {c}')
                seqs['calib' if unit % 2 == 0 else 'eval'].append(dict(
                    unit=unit, y=f['labels'][m], w=f['witness'][m],
                    strata=f['strata'][m].astype(str), s={a: f[a][m] for a in ARMS}))
    if not all(seqs.values()):
        raise ValueError('Both even calibration and odd evaluation units required')
    return seqs, consistency


def pack(seqs, boxes):
    """Pad sequence/query rows so threshold selection remains vectorized."""
    n, t = len(seqs)*len(boxes), max(len(s['y']) for s in seqs)
    y = np.zeros((n, t), bool)
    w = np.full((n, t), np.nan)
    st = np.full((n, t), '', dtype='U32')
    scores = {a: np.full((n, t), -np.inf) for a in ARMS}
    strata, units = [], []
    i = 0
    for sq in seqs:
        for q in boxes:
            k = len(sq['y'])
            y[i, :k] = sq['y'][:, q] == 1
            w[i, :k] = sq['w'][:, q]
            st[i, :k] = sq['strata'][:, q]
            if np.any(y[i] & ~np.isfinite(w[i])):
                raise ValueError('Positive-frame witness must be finite')
            strata.append(Counter(st[i, y[i]].tolist()).most_common(1)[0][0] if y[i].any() else '')
            units.append(sq['unit'])
            for a in ARMS:
                scores[a][i, :k] = sq['s'][a][:, q]
            i += 1
    return dict(y=y, w=w, frame_strata=st, strata=np.array(strata),
                unit=np.array(units), empty=~y.any(1),
                near=(y & (w <= 1.)).any(1), s=scores)


def first_hits(p, arm, grid):
    hits = p['s'][arm][None, :, :] >= grid[:, None, None]
    false = hits.any(2) & p['empty'][None, :]
    positive = hits & p['y'][None, :, :]
    first = np.where(positive.any(2), positive.argmax(2), p['y'].shape[1])
    return first, false


def timely(p, first):
    w = np.c_[p['w'], np.full(len(p['w']), np.nan)]
    return (w[np.arange(len(w)), first] >= 1.) & p['near']


def select(p, arm, budget):
    if not p['empty'].any():
        raise ValueError('No empty calibration pairs: FA budget cannot be established')
    if arm != 'G4':
        fa = ((p['s'][arm].max(1)[None, :] >= GRID[:, None]) & p['empty']).sum(1)
        valid = np.flatnonzero(fa / p['empty'].sum() <= budget)
        if not len(valid):
            raise ValueError('No threshold satisfies budget, including disabled branch')
        return (float(GRID[valid[0]]),)
    if not p['near'].any():
        raise ValueError('No near calibration events for G4 objective')
    global_grid = np.unique(np.r_[JOINT_GRID, select(p, 'G0', budget)[0]])
    fg, ag = first_hits(p, 'G0', global_grid)
    fl, al = first_hits(p, 'G4local', JOINT_GRID)
    best = None
    for i, tg in enumerate(global_grid):
        for j, tl in enumerate(JOINT_GRID):
            false = int((ag[i] | al[j]).sum())
            if false / p['empty'].sum() > budget:
                continue
            count = int(timely(p, np.minimum(fg[i], fl[j])).sum())
            key = (count, -false, float(tg), float(tl))
            if best is None or key > best:
                best = key
    if best is None:
        raise ValueError('G4 budget unreachable')
    return best[2:]


def rate(num, den):
    return dict(numerator=int(num), denominator=int(den), rate=float(num/den) if den else None)


def evaluate(p, arm, thresholds):
    a = p['s']['G0' if arm == 'G4' else arm] >= thresholds[0]
    if arm == 'G4':
        a |= p['s']['G4local'] >= thresholds[1]
    hit = a & p['y']
    first = np.where(hit.any(1), hit.argmax(1), p['y'].shape[1])
    ti = timely(p, first)
    detected = hit.any(1)
    out = {}
    for st in ('all', 'tiny', 'realistic', 'wide'):
        mask = np.ones(len(a), bool) if st == 'all' else p['strata'] == st
        ev, near = mask & ~p['empty'], mask & p['near']
        row = dict(event_alert=rate((detected & ev).sum(), ev.sum()),
                   timely=rate((ti & near).sum(), near.sum()),
                   late=rate((detected & ~ti & near).sum(), near.sum()),
                   never=rate((~detected & near).sum(), near.sum()))
        # Empty pairs have no positive target stratum; do not invent size-specific FA.
        if st == 'all':
            row['false_alert'] = rate((a.any(1) & p['empty']).sum(), p['empty'].sum())
        row['distance_frames'] = {}
        for low, high in BINS:
            m = p['y'] & (p['w'] >= low) & (p['w'] < high)
            if st != 'all':
                m &= p['frame_strata'] == st
            row['distance_frames'][f'[{low},{high})'] = rate((a & m).sum(), m.sum())
        out[st] = row
    return out


def threshold_json(t):
    return ['disabled' if np.isinf(x) else float(x) for x in t]


def analyze(root):
    seqs, consistency = load(root)
    out = dict(scope='Consumed v3 Development; privileged mechanism diagnostic, not camera or deployable results',
               unit_ids={k: sorted({s['unit'] for s in v}) for k, v in seqs.items()},
               G0_consistency_max_abs_by_unit=consistency,
               selection=dict(single_grid='linspace(1,12,441) + infinity',
                              G4_grid='local: linspace(1,12,45) + infinity; global: same plus selected G0 calibration threshold for this budget',
                              G4_objective='per group: maximum pooled near timely count; tie fewer FA, larger global threshold, larger local threshold',
                              size_specific_FA='undefined: empty pairs have no target size'), results={})
    for group, boxes in GROUPS.items():
        ca, ev = pack(seqs['calib'], boxes), pack(seqs['eval'], boxes)
        for arm in (*ARMS[:-1], 'G4'):
            for budget in BUDGETS:
                t = select(ca, arm, budget)
                out['results'][f'{group}/{arm}/{budget:.2f}'] = dict(
                    thresholds=threshold_json(t), calib=evaluate(ca, arm, t), eval=evaluate(ev, arm, t))
    decisions = {}
    for group in GROUPS:
        rows = {a: out['results'][f'{group}/{a}/0.10']['eval']['tiny']['timely']['rate']
                for a in ('G0', 'G1', 'G3b', 'G4')}
        if any(v is None for v in rows.values()):
            decisions[group] = dict(status='NOT_EVALUABLE')
            continue
        gain = rows['G1']-rows['G0']
        g4 = rows['G4']-rows['G0']
        perturb = rows['G3b']-rows['G0']
        decisions[group] = dict(G1_gain= gain, G3b_gain=perturb, G4_gain=g4,
            G4_to_G1_gain_ratio=g4/gain if gain > 0 else None,
            G3b_to_G1_gain_ratio=perturb/gain if gain > 0 else None,
            rgb_candidate_gate=bool(gain >= .10 and perturb >= .5*gain),
            tof_temporal_priority_gate=bool(gain > 0 and g4 >= .5*gain),
            position_downgrade_gate=bool(gain < .05))
    out['descriptive_decision_checks_10pct_tiny'] = decisions
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('readouts')
    parser.add_argument('out')
    args = parser.parse_args()
    result = analyze(args.readouts)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['descriptive_decision_checks_10pct_tiny'], indent=2))


if __name__ == '__main__':
    main()
