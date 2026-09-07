"""NF-G7 cached-prediction evaluation; controlled synthetic Development only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[4]
PARTS = ('BODY', 'HEAD')
HORIZONS = (1, 2, 3)
SEEDS = (17, 29, 43)
PARENTS = ('single_frame', 'ordinary_video', 'temporal_structure')
ARMS = PARENTS + ('ordinary_video_repeated_history', 'temporal_structure_repeated_history')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(value, binary=False):
    if not isinstance(value, list) or len(value) != 2 or any(not isinstance(r, list) or len(r) != 3 for r in value):
        raise ValueError('Expected BODY/HEAD x 1/2/3 second matrix')
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1
           or (binary and x not in (0, 1)) for row in value for x in row):
        raise ValueError('Invalid probability/binary matrix')
    if any(a > b for row in value for a, b in zip(row, row[1:])):
        raise ValueError('Horizons must be cumulative monotone')
    return value


def validate_dataset(dataset):
    samples = dataset['samples']
    ids = [s['sample_id'] for s in samples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate samples')
    groups, clips = {}, {}
    for s in samples:
        if s['split'] not in ('train', 'val', 'test'):
            raise ValueError('Invalid split')
        for table, key in ((groups, s['group_id']), (clips, s['clip_id'])):
            if key in table and table[key] != s['split']:
                raise ValueError('Group/clip crosses split')
            table[key] = s['split']
    if set(groups.values()) != {'train', 'val', 'test'}:
        raise ValueError('Missing partition')
    return samples


def validate_predictions(payload, samples, learned):
    if payload.get('complete') is False:
        raise ValueError('Incomplete prediction cache')
    if payload.get('target_order', list(PARTS)) != list(PARTS) or payload.get('horizons_s', list(HORIZONS)) != list(HORIZONS):
        raise ValueError('Prediction target order mismatch')
    lookup = {s['sample_id']: s for s in samples}
    expected = {(a, seed) for a in ARMS for seed in SEEDS} if learned else {('mde_contact', 0)}
    result = {k: {} for k in expected}
    for row in payload['rows']:
        key, sid = (row['arm'], row['seed']), row['sample_id']
        if key not in expected or sid not in lookup or sid in result[key]:
            raise ValueError('Unexpected or duplicate prediction identity')
        if row['split'] != lookup[sid]['split']:
            raise ValueError('Prediction split mismatch')
        matrix(row['scores'], binary=not learned)
        if 'states' in row:
            states = row['states']
            if learned or len(states) != 2 or any(len(r) != 3 for r in states):
                raise ValueError('Invalid baseline states')
            if any(x not in ('UNKNOWN', 'POSITIVE', 'NEGATIVE', 'ALERT', 'CLEAR', 'CONTACT', 'NO_CONTACT', 'KNOWN', 'OBSTACLE', 'NO_NEAR_OBSERVED') for r in states for x in r):
                raise ValueError('Unknown baseline state vocabulary')
            for p in range(2):
                for h in range(3):
                    state, score = states[p][h], row['scores'][p][h]
                    if state in ('POSITIVE', 'ALERT', 'CONTACT', 'OBSTACLE') and score != 1 or state in ('NEGATIVE', 'CLEAR', 'NO_CONTACT', 'NO_NEAR_OBSERVED', 'UNKNOWN') and score != 0:
                        raise ValueError('Baseline state/score contradiction')
        result[key][sid] = row
    if any(set(rows) != set(lookup) for rows in result.values()):
        raise ValueError('Every arm/seed requires exact full sample coverage')
    return result


def threshold(rows, samples, trainval_targets):
    negatives = [rows[s['sample_id']]['scores'][p][h] for s in samples if s['split'] == 'val'
                 for p in range(2) for h in range(3) if trainval_targets[s['sample_id']][p][h] == 0]
    if not negatives:
        raise ValueError('No validation negative cells')
    # The next representable float is strictly above the maximum, even at 1.
    return math.nextafter(max(negatives), math.inf)


def add_ensembles(predictions, samples):
    for arm in ARMS:
        predictions[arm, 'ensemble'] = {
            s['sample_id']: dict(sample_id=s['sample_id'], split=s['split'], arm=arm, seed='ensemble',
                scores=[[statistics.mean(predictions[arm, seed][s['sample_id']]['scores'][p][h] for seed in SEEDS)
                         for h in range(3)] for p in range(2)]) for s in samples}


def select_thresholds(predictions, samples, trainval):
    values = {(arm, seed): threshold(predictions[arm, seed], samples, trainval)
              for arm in PARENTS for seed in (*SEEDS, 'ensemble')}
    for arm in ARMS:
        if arm.endswith('_repeated_history'):
            for seed in (*SEEDS, 'ensemble'):
                values[arm, seed] = values[arm.removesuffix('_repeated_history'), seed]
    values['mde_contact', 0] = .5
    return values


def evaluate(rows, samples, targets, contacts, operating_point, frame_times=None):
    test = [s for s in samples if s['split'] == 'test']
    cells = {f'{part}@{h}s': dict(TP=0, FP=0, FN=0, TN=0, UNKNOWN=0,
                                  unknown_positive=0, unknown_negative=0, squared_error_sum=0., known_cells=0)
             for part in PARTS for h in HORIZONS}
    clips = {}
    for s in test:
        sid, cid = s['sample_id'], s['clip_id']
        clip = clips.setdefault(cid, dict(clip_id=cid, group_id=s['group_id'], pure_negative=True,
                    false_alert=False, first_alarm=None, unknown_cells=0, samples=0, parts={part: dict(opportunity=False,
                    recovered=False, correct_warnings=[]) for part in PARTS}))
        clip['samples'] += 1
        for p, part in enumerate(PARTS):
            truth = targets[sid][p]
            episode = clip['parts'][part]
            episode['opportunity'] |= bool(truth[2])
            clip['pure_negative'] &= not any(truth)
            for h in range(3):
                cell = cells[f'{part}@{HORIZONS[h]}s']
                score = rows[sid]['scores'][p][h]
                unknown = rows[sid].get('states', [['KNOWN']*3 for _ in PARTS])[p][h] == 'UNKNOWN'
                if unknown:
                    cell['UNKNOWN'] += 1
                    cell['unknown_positive' if truth[h] else 'unknown_negative'] += 1
                    clip['unknown_cells'] += 1
                    continue
                alert = score >= operating_point
                cell[('TP' if truth[h] else 'FP') if alert else ('FN' if truth[h] else 'TN')] += 1
                cell['squared_error_sum'] += (score-truth[h])**2
                cell['known_cells'] += 1
                if alert:
                    alarm = dict(sample_id=sid, time_s=(frame_times or {}).get(sid), part=part,
                                 horizon_s=HORIZONS[h], correct=bool(truth[h]))
                    previous = clip['first_alarm']
                    if previous is None or (alarm['time_s'] is not None and
                            (previous['time_s'] is None or alarm['time_s'] < previous['time_s'])):
                        clip['first_alarm'] = alarm
                if alert and not truth[h]:
                    clip['false_alert'] = True
                if h == 2 and alert and truth[h]:
                    episode['recovered'] = True
                    episode['correct_warnings'].append(dict(sample_id=sid,
                        time_s=(frame_times or {}).get(sid), simulated_lead_s=contacts[sid][p]))
    for cell in cells.values():
        cell['Brier_known_cells'] = cell.pop('squared_error_sum')/cell['known_cells'] if cell['known_cells'] else None
        cell['total_cells'] = cell['known_cells']+cell['UNKNOWN']
        cell['missed_or_unknown_positive'] = cell['FN']+cell['unknown_positive']
        positive_count = cell['TP']+cell['missed_or_unknown_positive']
        cell['all_opportunity_recall'] = cell['TP']/positive_count if positive_count else None
    leads, recovered, opportunities = [], [], []
    for cid, clip in clips.items():
        for part, ep in clip['parts'].items():
            ident = f'{cid}/{part}'
            if ep['opportunity']:
                opportunities.append(ident)
            warnings = ep.pop('correct_warnings')
            if warnings:
                # Observation order uses simulated current-frame time; caller data order is fallback.
                warnings.sort(key=lambda w: w['time_s'] if w['time_s'] is not None else math.inf)
                ep['first_correct_warning'] = warnings[0]
                ep['simulated_lead_s'] = warnings[0]['simulated_lead_s']
                recovered.append(ident)
                if ep['simulated_lead_s'] is not None:
                    leads.append(ep['simulated_lead_s'])
            else:
                ep['first_correct_warning'] = None
                ep['simulated_lead_s'] = None
    false_clips = sorted(cid for cid, c in clips.items() if c['false_alert'])
    pure = sorted(cid for cid, c in clips.items() if c['pure_negative'])
    return dict(threshold=operating_point, test_samples=len(test), test_clips=len(clips), cells=cells,
        opportunity_episodes=sorted(opportunities), recovered_episodes=sorted(recovered),
        contact_opportunities=len(opportunities), recovered=len(recovered),
        all_false_alert_clips=false_clips, all_false_alert_clip_count=len(false_clips),
        pure_negative_clips=pure, pure_negative_false_alert_clips=sorted(set(pure)&set(false_clips)),
        pure_negative_false_alert_clip_count=len(set(pure)&set(false_clips)),
        unknown_cells=sum(c['UNKNOWN'] for c in cells.values()),
        simulated_lead_s_mean=statistics.mean(leads) if leads else None,
        simulated_lead_s_median=statistics.median(leads) if leads else None,
        recovered_with_available_lead=len(leads), clips=clips)


def structural_criterion(evaluations, seed):
    structure = evaluations[f'temporal_structure/{seed}']
    recovered = set(structure['recovered_episodes'])
    comparisons = {}
    for arm in ('single_frame', 'ordinary_video'):
        other = evaluations[f'{arm}/{seed}']
        additional = sorted(recovered-set(other['recovered_episodes']))
        lost = sorted(set(other['recovered_episodes'])-recovered)
        added_false = sorted(set(structure['all_false_alert_clips'])-set(other['all_false_alert_clips']))
        comparisons[arm] = dict(additional_recovered_episodes=additional, lost_recovered_episodes=lost,
            added_false_alert_clips=added_false, net_recovery=structure['recovered']-other['recovered'],
            false_alert_clip_count_change=structure['all_false_alert_clip_count']-other['all_false_alert_clip_count'],
            passes=bool(additional) and structure['recovered'] > other['recovered']
                and structure['all_false_alert_clip_count'] <= other['all_false_alert_clip_count'])
    repeated = evaluations[f'temporal_structure_repeated_history/{seed}']
    loss = sorted(recovered-set(repeated['recovered_episodes']))
    repeat_pass = bool(loss) and structure['recovered'] > repeated['recovered']
    return dict(comparisons=comparisons, repeated_history_lost_recovered_episodes=loss,
                repeated_history_net_recovery_loss=structure['recovered']-repeated['recovered'],
                repeated_history_loss=repeat_pass,
                retain_criterion_met=all(c['passes'] for c in comparisons.values()) and repeat_pass)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'learned', 'baseline', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    artifacts = (ROOT/'artifacts.local').resolve()
    if output == artifacts or not output.is_relative_to(artifacts) or output.exists():
        raise ValueError('Output must be fresh and under canonical artifacts.local')
    capture = args.capture.resolve()
    paths = dict(dataset=capture/'model/dataset.json', trainval=capture/'training/labels.json',
                 learned=args.learned.resolve(), baseline=args.baseline.resolve())
    hashes = {name: sha(path) for name, path in paths.items()}
    dataset = read(paths['dataset'])
    samples = validate_dataset(dataset)
    predictions = validate_predictions(read(paths['learned']), samples, True)
    predictions.update(validate_predictions(read(paths['baseline']), samples, False))
    trainval = read(paths['trainval'])['targets']
    if set(trainval) != {s['sample_id'] for s in samples if s['split'] != 'test'}:
        raise ValueError('Trainval label coverage mismatch/test label contamination')
    for value in trainval.values():
        matrix(value, True)
    add_ensembles(predictions, samples)
    thresholds = select_thresholds(predictions, samples, trainval)
    # All prediction caches validated and hashed, thresholds fixed BEFORE test truth access.
    paths.update(labels=capture/'evaluator/labels.json', specification=capture/'evaluator/spec.json')
    hashes.update({name: sha(paths[name]) for name in ('labels', 'specification')})
    truth = read(paths['labels'])
    targets, contacts = truth['targets'], truth['first_contact_s']
    ids = {s['sample_id'] for s in samples}
    if set(targets) != ids or set(contacts) != ids:
        raise ValueError('Evaluator label coverage mismatch')
    for sid, target in targets.items():
        matrix(target, True)
        if sid in trainval and target != trainval[sid]:
            raise ValueError('Trainval/evaluator labels disagree')
        lead = contacts[sid]
        if len(lead) != 2 or any(v is not None and (not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0) for v in lead):
            raise ValueError('Invalid simulated first contact time')
        for p, v in enumerate(lead):
            if v is not None and target[p] != [int(v <= h) for h in HORIZONS]:
                raise ValueError('Contact time and targets disagree')
    spec = read(paths['specification'])
    clip_metadata = {c['clip_id']: c for c in spec['clips']}
    if set(clip_metadata) != {s['clip_id'] for s in samples}:
        raise ValueError('Clip metadata coverage mismatch')
    for sample in samples:
        clip = clip_metadata[sample['clip_id']]
        if clip['split'] != sample['split'] or clip['group_id'] != sample['group_id']:
            raise ValueError('Clip metadata split/group mismatch')
    frames = {int(f['sample_index']): f for f in dataset['frames']}
    times = {s['sample_id']: float(frames[int(s['frame_indices'][-1])]['time_s']) for s in samples}
    evaluations = {f'{arm}/{seed}': evaluate(rows, samples, targets, contacts, thresholds[arm, seed], times)
                   for (arm, seed), rows in sorted(predictions.items(), key=lambda item: (item[0][0], str(item[0][1])))}
    criterion = {str(seed): structural_criterion(evaluations, seed) for seed in (*SEEDS, 'ensemble')}
    passes = [v['retain_criterion_met'] for v in criterion.values()]
    verdict = 'CRITERION_MET_CONTROLLED_DEVELOPMENT_ONLY' if all(passes) else ('MIXED_SEED_ENSEMBLE_NO_PROMOTION' if any(passes) else 'NO_STRUCTURAL_RETENTION_GAIN_THIS_IMPLEMENTATION_SOURCE_BUDGET')
    for name, path in paths.items():
        if sha(path) != hashes[name]:
            raise ValueError('Input changed while evaluating: '+name)
    result = dict(schema='nf-g7-contact-retina-evaluation-v1', verdict=verdict, evaluations=evaluations,
        structural_criterion=criterion, clip_metadata=clip_metadata,
        input_sha256=hashes, input_paths={k: str(v) for k, v in paths.items()},
        evaluator_sha256=sha(Path(__file__)), threshold_rule='one scalar per arm/seed or ensemble; nextafter(max validation negative cell); alert >= threshold; repeated history reuses parent',
        unknown_rule='excluded from known-cell confusion and Brier; explicit positive/negative UNKNOWN counts; no warning and no episode recovery. Baseline Brier is partially censored to known cells and is not an equal-coverage score versus full-coverage learners; all six part/horizon cell denominators are reported',
        false_alert_rule='any incorrect part or premature horizon positive in any test window, including contact clips',
        episode_rule='clip plus BODY/HEAD with any 3-second positive window; recovery requires a correct 3-second alert',
        claims='Controlled synthetic Development; simulated lead only; no real cadence, avoidance, natural generalization or statistical significance claim',
        criterion_rule='strict extra episode recovery count over both learners, no increase in falsely alerted clip count, and net recovery loss under repeated history; newly false-alerted clip identities remain descriptive')
    output.mkdir(parents=True)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['# NF-G7 cached contact evaluation', '', verdict, '', result['claims'], '',
             '| Arm/seed | Recovered / opportunities | All falsely alerted clips | Pure-negative FP clips | UNKNOWN cells | Mean simulated lead (s) |',
             '| --- | --- | --- | --- | --- | --- |']
    for key, ev in evaluations.items():
        lead = ev['simulated_lead_s_mean']
        lines.append(f"| {key} | {ev['recovered']}/{ev['contact_opportunities']} | {ev['all_false_alert_clip_count']}/{ev['test_clips']} | {ev['pure_negative_false_alert_clip_count']}/{len(ev['pure_negative_clips'])} | {ev['unknown_cells']} | {lead if lead is not None else 'null'} |")
    lines.extend(['', result['threshold_rule'], '', result['unknown_rule'], '', result['false_alert_rule'], '',
                  'Detailed per-cell TP/FP/FN/TN, Brier, per-clip/part leads, clip identities and seed/ensemble retention comparisons are in result.json.'])
    for key, ev in evaluations.items():
        lines.extend(['', '## '+key, '', f"Threshold: {ev['threshold']!r}", '',
                      '| Part/horizon | TP | FP | FN | TN | UNKNOWN (+ / -) | Missed or unknown positive | All opportunity recall | Brier known cells |',
                      '| --- | --- | --- | --- | --- | --- | --- | --- | --- |'])
        for cell_name, cell in ev['cells'].items():
            lines.append(f"| {cell_name} | {cell['TP']} | {cell['FP']} | {cell['FN']} | {cell['TN']} | {cell['UNKNOWN']} ({cell['unknown_positive']} / {cell['unknown_negative']}) | {cell['missed_or_unknown_positive']} | {cell['all_opportunity_recall']} | {cell['Brier_known_cells']} |")
        lines.extend(['', '| Clip | Part | Opportunity | Recovered | First correct sample | Simulated lead (s) | Any false alert in clip |',
                      '| --- | --- | --- | --- | --- | --- | --- |'])
        for cid, clip in sorted(ev['clips'].items()):
            for part, ep in clip['parts'].items():
                warning = ep['first_correct_warning']
                lines.append(f"| {cid} | {part} | {ep['opportunity']} | {ep['recovered']} | {warning['sample_id'] if warning else 'null'} | {ep['simulated_lead_s'] if ep['simulated_lead_s'] is not None else 'null'} | {clip['false_alert']} |")
    lines.extend(['', '## Structural criterion', '', '| Seed | Gain vs single | Gain vs video | Repetition recovery loss | Criterion |',
                  '| --- | --- | --- | --- | --- |'])
    for seed, row in criterion.items():
        lines.append(f"| {seed} | {row['comparisons']['single_frame']['passes']} | {row['comparisons']['ordinary_video']['passes']} | {row['repeated_history_net_recovery_loss']} | {row['retain_criterion_met']} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), verdict=verdict)))


if __name__ == '__main__':
    main()
