"""Same-source coarse depth-anchor side probe, never a real ToF experiment.

Even checkerboard sensor pixels form 8x8 median anchors; only odd pixels are
scored. A single scalar is applied per frame. The disjoint samples still share
sensor errors and this probe cannot establish an independent ToF increment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from rgb_body_query_3rscan import sample_prediction, sensor_labels
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import confusion, ratios, rays, ray_interval

CAL_SCALE = 1.012320716490867
ARMS = ('raw', 'cal_global', 'frame_anchor')
COUNTS = ('tp', 'fn', 'fp', 'tn')
PAIR_KEYS = ('positive_rescue', 'positive_loss', 'free_removed', 'free_added')


def frame_anchor(reference, predicted, observed, bins=8, min_points=16):
    yy, xx = np.indices(reference.shape)
    even = (xx + yy) % 2 == 0
    valid = observed & even & np.isfinite(reference) & (reference > 0)
    valid &= np.isfinite(predicted) & (predicted > 0)
    by = yy * bins // reference.shape[0]
    bx = xx * bins // reference.shape[1]
    cells, logs = [], []
    for y in range(bins):
        for x in range(bins):
            mask = valid & (by == y) & (bx == x)
            n = int(mask.sum())
            row = dict(y=y, x=x, paired_anchor_points=n, available=n >= min_points)
            if n >= min_points:
                truth = float(np.median(reference[mask].astype(np.float64)))
                estimate = float(np.median(predicted[mask].astype(np.float64)))
                log_ratio = float(np.log(truth / estimate))
                logs.append(log_ratio)
                row.update(reference_median=truth, prediction_median=estimate,
                           ratio=truth / estimate, log_ratio=log_ratio)
            cells.append(row)
    scale = float(np.exp(np.median(logs))) if logs else None
    return scale, cells, valid, ~even


def counts_paired(labels, baseline, candidate):
    pos, free = labels == 1, labels == 0
    result = dict(positive_rescue=int((pos & ~baseline & candidate).sum()),
                  positive_loss=int((pos & baseline & ~candidate).sum()),
                  free_removed=int((free & baseline & ~candidate).sum()),
                  free_added=int((free & ~baseline & candidate).sum()))
    a, b = confusion(labels, baseline), confusion(labels, candidate)
    assert result['positive_rescue'] - result['positive_loss'] == b['tp'] - a['tp']
    assert result['free_added'] - result['free_removed'] == b['fp'] - a['fp']
    return result


def grouped(records, count_keys):
    groups = defaultdict(list)
    for r in records:
        groups['all'].append(r)
        for prefix, value in [('split', r['split']), ('environment', r['environment']),
                              ('band', r['distance_band']), ('query', r['query']),
                              ('split_band', r['split'] + '/' + r['distance_band']),
                              ('environment_band', r['environment'] + '/' + r['distance_band'])]:
            groups[prefix + '/' + value].append(r)
    output = {}
    for key, rows in groups.items():
        valid = [r for r in rows if r['available']]
        c = {k: sum(r[k] for r in valid) for k in count_keys}
        output[key] = dict(records=len(rows), evaluated_records=len(valid),
                           unavailable_records=len(rows)-len(valid),
                           frames=len({(r['scan'], r['frame']) for r in rows}), **c)
        if count_keys == COUNTS:
            output[key].update(ratios(c))
            output[key]['positive_units'] = c['tp'] + c['fn']
            output[key]['free_units'] = c['fp'] + c['tn']
    return output


def fixtures():
    truth = np.full((16, 16), 6., np.float32)
    pred = np.full((16, 16), 2., np.float32)
    observed = np.ones_like(truth, bool)
    scale, cells, anchor, score = frame_anchor(truth, pred, observed, bins=2, min_points=16)
    assert scale == np.exp(np.median([np.log(3.)] * 4))
    assert np.isclose(scale, 3.) and sum(c['available'] for c in cells) == 4
    assert not (anchor & score).any()
    truth[:] = np.nan
    assert frame_anchor(truth, pred, observed, bins=2, min_points=16)[0] is None
    labels = np.array([[1, 1, 0, 0, 2, 255]], np.uint8)
    raw = np.array([[0, 1, 1, 0, 1, 1]], bool)
    fixed = np.array([[1, 0, 0, 1, 0, 0]], bool)
    pair = counts_paired(labels, raw, fixed)
    assert list(pair.values()) == [1, 1, 1, 1]
    return dict(median_ratio_fixture=True, empty_anchor_unavailable=True,
                disjoint_sample_fixture=True, paired_count_identity_fixture=True)


def run(source, output, budget_s=240):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists():
        raise FileExistsError('Preserve previous anchor probe output')
    manifest_path, predictions_path = source/'dataset_manifest.json', source/'depthpro/predictions.json'
    manifest = json.loads(manifest_path.read_text('utf-8-sig'))
    predictions = json.loads(predictions_path.read_text('utf-8-sig'))
    plan = dict(cpu_budget_s=budget_s, gpu_budget_s=0, download_budget_bytes=0,
                source_manifest_sha256=sha(manifest_path), prediction_manifest_sha256=sha(predictions_path),
                source_code_sha256=sha(__file__), arms=list(ARMS), cal_scale=CAL_SCALE,
                anchor='even (u+v)%2 == 0 native observed sensor rays; fixed 8x8 bins; >=16 finite positive paired points per bin; each reference/prediction median uses identical points',
                scale='one frame scalar exp(median(log(reference_bin_median/prediction_bin_median))) over all usable bins; no query/outcome fitting',
                score='odd checkerboard only; regenerate original first-return query labels; outside odd sample domain IGNORE, UNKNOWN never negative',
                unavailable='frame_anchor NOT_EVALUABLE if no usable bins; omit unavailable pairs and report coverage',
                frozen='old 96 frames and 15 queries, optical Z metres, closed geometry intervals; no head/threshold training',
                interpretation='same-source measured-depth anchors; disjoint pixels can share systematic errors; not real ToF or mixed-return simulation; no whole-box/body/event claim',
                decision='check whether frame-scale anchors change direct geometry scores, preserving rescue/loss and all slices; no gain gate or prerequisite for frozen RGB candidate')
    write(output/'plan.json', plan)
    check = fixtures()
    refs = {(r['scan'], r['frame']): r for r in manifest['rows']}
    anchor_rows, records, pairs = [], [], []
    shared_reference_checks = 0
    direct_checks = 0
    for p in predictions['rows']:
        if time.perf_counter()-start >= budget_s:
            raise TimeoutError('Anchor CPU budget reached')
        r = refs[p['scan'], p['frame']]
        assert all(r[k] == p[k] for k in ('environment', 'split', 'rgb_sha256'))
        assert sha(p['path']) == p['sha256'] and sha(r['reference_path']) == r['reference_sha256']
        with np.load(p['path']) as f:
            depth = f['depth']
        with np.load(r['reference_path']) as f:
            sampled = sample_prediction(depth, f['map_x'], f['map_y'])
            truth, observed, k, old_labels = f['depth'], f['observed'], f['depth_K'], f['labels']
        scale, cells, anchor_mask, odd = frame_anchor(truth, sampled, observed)
        assert not (anchor_mask & odd).any()
        anchor_rows.append(dict(scan=r['scan'], frame=r['frame'], environment=r['environment'],
                                split=r['split'], scale=scale, available=scale is not None,
                                usable_bins=sum(c['available'] for c in cells),
                                anchor_paired_points=int(anchor_mask.sum()), cells=cells,
                                scoring_sample_pixels=int((observed & odd).sum()),
                                prediction_sha256=p['sha256'], reference_sha256=r['reference_sha256']))
        rx, ry = rays(k, truth.shape)
        for i, q in enumerate(manifest['queries']):
            labels, info = sensor_labels(truth, k, q, observed)
            assert np.array_equal(labels, old_labels[i])
            shared_reference_checks += 1
            labels[~odd] = 255
            meta = dict(scan=r['scan'], frame=r['frame'], environment=r['environment'], split=r['split'],
                        query=q['name'], distance_band=f'{q["low"][2]:g}-{q["high"][2]:g}m',
                        positive_units=int((labels == 1).sum()), free_units=int((labels == 0).sum()),
                        unknown_units=int((labels == 2).sum()))
            entry, exit, reachable = ray_interval(rx, ry, q)
            model_predictions = {}
            for arm, s in zip(ARMS, (1., CAL_SCALE, scale)):
                if s is None:
                    records.append(dict(**meta, arm=arm, available=False, scale=None,
                                        status='NOT_EVALUABLE', **{c:None for c in COUNTS}))
                    continue
                scaled = sampled*s
                predicted = np.isfinite(scaled) & reachable & (scaled >= entry) & (scaled <= exit)
                direct = np.isfinite(scaled)
                for coordinate, low, high in zip((rx*scaled, ry*scaled, scaled), q['low'], q['high']):
                    direct &= (coordinate >= low) & (coordinate <= high)
                assert np.array_equal(predicted, direct)
                direct_checks += 1
                model_predictions[arm] = predicted
                records.append(dict(**meta, arm=arm, available=True, scale=s, status='EVALUABLE',
                                    **confusion(labels, predicted)))
            for baseline, candidate in [('raw', 'cal_global'), ('raw', 'frame_anchor'), ('cal_global', 'frame_anchor')]:
                available = candidate in model_predictions
                pair = counts_paired(labels, model_predictions[baseline], model_predictions[candidate]) if available else {c:None for c in PAIR_KEYS}
                pairs.append(dict(**meta, baseline=baseline, candidate=candidate, available=available, **pair))
    summaries = {arm:grouped([r for r in records if r['arm']==arm], COUNTS) for arm in ARMS}
    pair_summaries = {a+'__'+b:grouped([r for r in pairs if r['baseline']==a and r['candidate']==b], PAIR_KEYS)
                      for a,b in [('raw','cal_global'), ('raw','frame_anchor'), ('cal_global','frame_anchor')]}
    for name, rows in pair_summaries.items():
        a, b = name.split('__')
        if all(r['available'] for r in records):
            for key, row in rows.items():
                assert row['positive_rescue']-row['positive_loss'] == summaries[b][key]['tp']-summaries[a][key]['tp']
                assert row['free_added']-row['free_removed'] == summaries[b][key]['fp']-summaries[a][key]['fp']
    for filename, rows in [('frame_anchors.json', anchor_rows), ('records.json', records), ('paired_records.json', pairs)]:
        write(output/filename, rows)
    for filename, rows in [('summary_all_slices.csv', [dict(arm=a, slice=k, **v) for a,g in summaries.items() for k,v in g.items()]),
                           ('paired_all_slices.csv', [dict(comparison=a,slice=k,**v) for a,g in pair_summaries.items() for k,v in g.items()])]:
        with (output/filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    check.update(all_frame_anchor_score_intersections_zero=True, regenerated_old_reference_checks=shared_reference_checks,
                 direct_xyz_checks=direct_checks, paired_count_identities=True)
    write(output/'focused_checks.json', check)
    result = dict(status='COMPLETE', frames=len(anchor_rows), available_frames=sum(r['available'] for r in anchor_rows),
                  summary=summaries, paired_summary=pair_summaries, wall_s=time.perf_counter()-start,
                  checks=check, interpretation=plan['interpretation'])
    write(output/'evaluation.json', result)
    (output/'executed_anchor_probe.py').write_bytes(Path(__file__).read_bytes())
    write(output/'completion_receipt.json', dict(status='COMPLETE', source_sha256=sha(__file__),
          plan_sha256=sha(output/'plan.json'), evaluation_sha256=sha(output/'evaluation.json'),
          wall_s=result['wall_s'], cpu_budget_s=budget_s, gpu_allocation_s=0, downloads=0, resources_retained=[]))
    print(json.dumps(dict(frames=result['frames'], available_frames=result['available_frames'],
                         validation={a:summaries[a]['split/validation'] for a in ARMS},
                         paired_validation={a:g['split/validation'] for a,g in pair_summaries.items()},
                         checks=check, wall_s=result['wall_s'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=240)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.budget_s)
