"""Independent cuboid/entry-native truth, saved readout and event audit.

No model, simulator, current evaluator or label implementation is imported.
The prior audit supplies generic counters only; native verification is 48/576.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from local_support_audit import Checks, active_journal, digest, metric_recount, read

REPO = Path(__file__).resolve().parents[4]
ARMS = ('A_current', 'raw', 'local', 'raw_standalone', 'local_standalone')
AXES = ('trajectory', 'size_level', 'shape', 'layer', 'layout_relation', 'base_group_id')
QUERIES = np.array([[x-.3, x+.3, lo, hi] for lo, hi in ((.42, .9), (-.2, .42))
                    for x in (-.3, 0., .3)], np.float32)
EDGES = np.array([.3, .75, 1.25, 1.75, 2.25, 2.75, 3.], np.float32)
CUTOFFS = dict(raw=.776778260888226, local=.7959699076137833)
MODELS = dict(raw='c87050130956f6142298d9701e8f445eba9d3712912f361080fe732e267f72eb',
              local='1e6a4345353887effebb2aa798ace9dcdc5a46111166486179402bfcd4cc610e')
SELECTION = 'fa4ab0845aab5eb62a2c3dadacc82bc073c73b2e52da198527609384e6b4a753'
FEATURE_SOURCE = 'c0bd6dfaf391c4aab5455de988ea922975a5b933898b0d9aa9bd94315d694dcf'


def cuboid_truth(objects, camera):
    """Intersect each rendered solid separately, preserving composite holes."""
    boxes = []
    for obj in objects:
        c, h = (np.asarray(obj[k], np.float64) for k in ('render_bounds_center_m', 'render_bounds_extent_m'))
        mid = np.array([c[1]-camera['y'], camera['z']-c[2], c[0]-camera['x']])
        boxes.append((mid-h[[1, 2, 0]], mid+h[[1, 2, 0]]))
    distances, classes = np.full(6, np.nan, np.float32), np.full(6, 6, np.int64)
    for q, (left, right, top, bottom) in enumerate(QUERIES):
        hits = [max(lo[2], float(EDGES[0])) for lo, hi in boxes
                if max(lo[0], left) <= min(hi[0], right) and max(lo[1], top) <= min(hi[1], bottom)
                and max(lo[2], EDGES[0]) <= min(hi[2], EDGES[-1])]
        if hits:
            distances[q] = min(hits)
            classes[q] = min(5, int(np.count_nonzero(distances[q] > EDGES[1:])))
    return classes, distances, boxes


def native_support(depth, boxes):
    if depth.shape != (360, 640):
        raise ValueError('Native optical Z must remain 360x640')
    z = np.asarray(depth, np.float32)
    valid = np.isfinite(z) & (z > 0)
    z = np.where(valid, z, 0)
    f = 320/np.tan(np.deg2rad(50))
    x = z*((np.arange(640)+.5-320)/f)[None, :]
    y = z*((np.arange(360)+.5-180)/f)[:, None]
    declared = np.zeros(z.shape, bool)
    for lo, hi in boxes:
        declared |= ((x >= lo[0]-.02) & (x <= hi[0]+.02) & (y >= lo[1]-.02)
                     & (y <= hi[1]+.02) & (z >= lo[2]-.02) & (z <= hi[2]+.02))
    counts, unexplained, nearest = [], [], []
    for left, right, top, bottom in QUERIES:
        occupied = valid & (z >= EDGES[0]) & (z <= EDGES[-1])
        occupied &= (x >= left) & (x <= right) & (y >= top) & (y <= bottom)
        counts.append(int(occupied.sum())); unexplained.append(int((occupied & ~declared).sum()))
        nearest.append(float(z[occupied].min()) if occupied.any() else None)
    return dict(visible_pixels=counts, unexplained_pixels=unexplained, visible_nearest=nearest,
                query_label_valid=[v == 0 for v in unexplained])


def independent_gate(rows, metrics, strata, admissible):
    arms, meta = metrics['arms'], {r['clip_id']: r for r in rows}
    old = {(e['clip_id'], e['start_frame']): e for e in arms['A_current']['events']}
    timing = dict(opportunity_events=0, benefit_events=0, recovered_events=0,
                  by_trajectory={t: dict(events=0, opportunities=0, benefits=0) for t in strata['trajectory']})
    preserved = all(not r['predictions']['A_current']['alert'] or r['predictions']['local']['alert'] for r in rows)
    for event in arms['local']['events']:
        prior = old[event['clip_id'], event['start_frame']]
        t0, t1 = prior['first_alert_time_s'], event['first_alert_time_s']
        opportunity = t0 is None or t0 >= event['entry_time_s']+.2-1e-9
        benefit = t1 is not None and (t0 is None or t1 <= t0-.2+1e-9)
        preserved &= t0 is None or (t1 is not None and t1 <= t0+1e-9)
        timing['opportunity_events'] += int(opportunity); timing['benefit_events'] += int(benefit)
        timing['recovered_events'] += int(t0 is None and t1 is not None)
        sub = timing['by_trajectory'][meta[event['clip_id']]['trajectory']]
        sub['events'] += 1; sub['opportunities'] += int(opportunity); sub['benefits'] += int(benefit)
    def deltas(m, ref):
        a, b = (m['arms'][key] for key in ('local', ref))
        return dict(FP_delta=a['frames']['all_known']['FP']-b['frames']['all_known']['FP'],
                    false_segment_delta=a['false_alert_segment_count']-b['false_alert_segment_count'])
    costs = {t: deltas(m, 'A_current') for t, m in strata['trajectory'].items()}
    cost_ok = all(d['FP_delta'] <= 2 and d['false_segment_delta'] <= 1
                  for d in [deltas(metrics, 'A_current'), *costs.values()])
    count = lambda m, a: m['arms'][a]['frames']['all_known']
    layer_ok = all(count(m, 'local')['recall'] >= count(m, 'raw')['recall']-.05-1e-9 for m in strata['layer'].values())
    improved = sum(count(m, 'local')['TP'] > count(m, 'raw')['TP'] for m in strata['base_group_id'].values())
    dr = deltas(metrics, 'raw')
    raw_ok = (count(metrics, 'local')['recall']-count(metrics, 'raw')['recall'] >= .05-1e-9
              and dr['FP_delta'] <= 1 and dr['false_segment_delta'] <= 1 and improved >= 4 and layer_ok)
    timing_ok = timing['benefit_events'] >= 8 and all(s['benefits'] >= 3 for s in timing['by_trajectory'].values()) and preserved
    timing_status = 'NOT_EVALUABLE' if timing['opportunity_events'] < 8 else 'PASS' if timing_ok else 'FAIL'
    status = ('NOT_EVALUABLE' if not admissible else 'FAIL' if not cost_ok or not raw_ok or timing_status == 'FAIL'
              else timing_status)
    return dict(status=status, source_admissible=admissible, timing_status=timing_status, timing=timing,
                A_preserved=bool(preserved), costs_pass=cost_ok, RAW_comparison_pass=raw_ok,
                layer_noninferior_to_RAW=layer_ok, trajectory_costs=costs)


def audit(root, result):
    root, result = Path(root), Path(result)
    governance, check, inputs = active_journal(REPO, result), Checks(), {}
    def stage(name): return root.with_name(root.name+'-'+name)
    def sealed(path):
        inputs[str(path.resolve())] = digest(path)
        return inputs[str(path.resolve())]
    def load(path): sealed(path); return read(path)
    pred, prepared, evaluator = stage('predictions'), stage('prepared'), stage('capture')/'evaluator'
    saved = load(stage('evaluated')/'metrics.json'); saved_rows = load(stage('evaluated')/'frame-results.json')
    for name, sha in load(stage('evaluated')/'output-seal.json')['files'].items():
        check.check(sealed(stage('evaluated')/name) == sha, 'seal', 'evaluated/'+name)
    seal, freeze = load(pred/'prediction-seal.json'), load(pred/'freeze.json')
    required = {'freeze.json', 'features.npz', 'probabilities.npz', 'baseline.json', 'identities.json',
                'selection.json', 'costs.json', 'public-identity-checks.json', 'backend.json'}
    check.check(required <= seal['hashes'].keys(), 'seal', 'Complete prediction payload')
    for name, sha in seal['hashes'].items(): check.check(sealed(pred/name) == sha, 'seal', name)
    check.same(seal, dict(status='PASS', frames=576, queries=3456, fits=0, cutoff_selections=0,
                         evaluation_labels_opened=False, thresholds=CUTOFFS), 'seal', 'Prediction contract')
    check.same(freeze, dict(frozen_model_hashes=MODELS, frozen_selection_sha256=SELECTION,
        thresholds=CUTOFFS, A_threshold=.4071309640537889, feature_width=961, fits=0,
        cutoff_selections=0, evaluation_labels_opened=False), 'seal', 'Frozen recipe')
    check.check(digest(pred/'selection.json') == SELECTION, 'seal', 'Original selection bytes')
    check.check(seal['thresholds'] == freeze['thresholds'] == CUTOFFS and freeze['A_threshold'] == .4071309640537889,
                'seal', 'Exact original cutoffs')
    sources = {k.replace('\\', '/'): v for k, v in freeze['source_hashes'].items()}
    check.check(sources.get('research/active/dtr-r0/nearfield/inherit_spatial_model.py') == FEATURE_SOURCE,
                'code', 'Original 961-feature implementation anchor')
    check.check(np.allclose(freeze['query_boxes'], QUERIES, rtol=0, atol=3e-8), 'seal', 'Fixed queries')
    for name, sha in freeze['source_hashes'].items():
        check.check(sealed(REPO/name) == sha == sealed(pred/'source-snapshot'/name), 'code', name)
    manifest = load(prepared/'materialization.json')
    check.check(digest(prepared/'materialization.json') == freeze['materialization_sha256'], 'seal', 'Materialization')
    check.check(sealed(root/'plan/stability-protocol.md') == freeze['protocol_sha256'], 'seal', 'Protocol')
    for name, sha in freeze['input_hashes'].items(): check.check(manifest['hashes'][name] == sha, 'seal', name)
    for name in ('evaluation.npz', 'visibility-audit.json', 'source-admission.json'):
        check.check(sealed(prepared/'labels'/name) == manifest['hashes']['labels/'+name], 'seal', name)
    check.check(digest(prepared/'labels/evaluation.npz') == freeze['expected_evaluation_labels_sha256'], 'seal', 'Label promise')
    check.check(digest(pred/'identities.json') == manifest['hashes']['observations/identities.json'], 'seal', 'Public identities')
    ids, baseline = load(pred/'identities.json'), load(pred/'baseline.json')
    labels = dict(np.load(prepared/'labels/evaluation.npz', allow_pickle=False))
    probability = dict(np.load(pred/'probabilities.npz', allow_pickle=False))
    geometry, spec = load(evaluator/'geometry.json'), load(root/'plan/spec.json')
    check.check(digest(root/'plan/spec.json') == manifest['source_spec_sha256'], 'seal', 'Source plan')
    visibility, admission = load(prepared/'labels/visibility-audit.json'), load(prepared/'labels/source-admission.json')
    check.check(len(ids) == len(baseline) == len(geometry) == len(visibility) == len(saved_rows) == len(spec['cases']) == 576,
                'source', 'Full frame denominator')
    for obj in (labels, probability): check.check(np.array_equal(obj['indices'], np.arange(576)), 'source', 'Indices')
    check.check(labels['classes'].shape == labels['distances'].shape == labels['valid'].shape == (576, 6), 'source', 'Six labels')
    for arm in ('raw', 'local'):
        p = probability[arm]
        check.check(p.shape == (576, 6) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'source', 'Probability '+arm)
    rows, subset, clips, object_errors = [], [], defaultdict(list), []
    for i, (meta, a, geo, case, vis) in enumerate(zip(ids, baseline, geometry, spec['cases'], visibility)):
        tag = str(i); clips[meta['clip_id']].append(meta['frame_in_clip'])
        check.check(meta['index'] == geo['sample_index'] == vis['index'] == i and meta['id'] == geo['id'] == case['name'], 'source', tag)
        for key in ('clip_id', 'frame_in_clip', 'time_s', 'type_id', *AXES): check.same(meta[key], case[key], 'source', tag+'/'+key)
        check.check(abs(meta['time_s']-.2*meta['frame_in_clip']) < 1e-9, 'source', tag+'/clock')
        check.same(geo['declared_camera'], case['camera'], 'geometry', tag+'/camera')
        check.check(all(case['camera'][k] == 0 for k in ('pitch', 'yaw', 'roll')), 'geometry', tag+'/axes')
        check.check(max(abs(geo['actual_camera_location_m'][j]-case['camera'][k]) for j, k in enumerate(('x', 'y', 'z'))) <= .002,
                    'geometry', tag+'/camera-location')
        check.check(len(geo['objects']) == 3 and {o['name'] for o in geo['objects']} == {'target_a', 'target_b', 'background'}, 'geometry', tag+'/union')
        errors = []
        for obj, plan in zip(geo['objects'], case['objects']):
            check.check(obj['name'] == plan['name'] and obj['mesh_path'].split('.')[-1] == 'Cube', 'geometry', tag+'/cube')
            c = float(np.max(np.abs(np.asarray(obj['render_bounds_center_m'])-plan['center_m'])))
            s = float(np.max(np.abs(2*np.asarray(obj['render_bounds_extent_m'])-plan['size_m'])))
            errors.append(dict(name=obj['name'], center_error_m=c, size_error_m=s, within_2mm=c <= .002 and s <= .002))
        object_errors.extend(errors); check.same(vis['object_bounds'], errors, 'geometry', tag+'/bounds')
        classes, distances, boxes = cuboid_truth(geo['objects'], case['camera'])
        check.check(np.array_equal(classes, labels['classes'][i]) and np.allclose(distances, labels['distances'][i], rtol=0, atol=0, equal_nan=True), 'geometry', tag+'/classes-distance')
        check.check(np.array_equal(labels['valid'][i], np.asarray(vis['unexplained_pixels']) == 0), 'source', tag+'/validity')
        if meta['frame_in_clip'] == 3:
            path = evaluator/geo['native_path']; check.check(sealed(path) == geo['native_sha256'] == vis['source_native_sha256'], 'native', tag+'/hash')
            native = native_support(np.load(path, allow_pickle=False), boxes)
            check.same(vis, native, 'native', tag+'/counts'); subset.append(dict(index=i, **native))
        truth = bool((classes[[1, 4]] < 6).any()) if labels['valid'][i, [1, 4]].all() else None
        check.same(a, meta['baseline'], 'readout', tag+'/sealed-A')
        check.check(a['threshold'] == .4071309640537889 and a['unknown'] == (a['definite_zones'] == 0), 'readout', tag+'/A-unknown')
        flags, scores = dict(A_current=bool(a['alert'])), {}
        for arm in ('raw', 'local'):
            scores[arm] = float(max(probability[arm][i, 1], probability[arm][i, 4]))
            flags[arm+'_standalone'] = scores[arm] >= CUTOFFS[arm]
            flags[arm] = bool(a['alert'] or flags[arm+'_standalone'])
        row = {key: meta[key] for key in ('id', 'clip_id', 'frame_in_clip', 'time_s', 'index', 'type_id', *AXES)}
        row.update(truth=truth, boundary=meta['layout_relation'] == 'BOUNDARY', query_truth=(classes < 6).tolist(),
            query_valid=labels['valid'][i].tolist(), frame_scores=scores,
            query_probabilities={arm: probability[arm][i].tolist() for arm in ('raw', 'local')},
            predictions={arm: dict(alert=bool(flag), unknown=bool(a['unknown']), ambiguous=bool(flag and a['unknown'])) for arm, flag in flags.items()})
        check.same(saved_rows[i], row, 'readout', tag); rows.append(row)
    check.check(len(clips) == len(subset) == 48 and all(sorted(v) == list(range(12)) for v in clips.values()), 'source', '48 complete clips and entry subset')
    check.same(admission, dict(frames=576, queries=3456, valid_queries=int(labels['valid'].sum()),
        invalid_queries=int((~labels['valid']).sum()), rendered_objects=1728,
        max_center_error_m=max(e['center_error_m'] for e in object_errors), max_size_error_m=max(e['size_error_m'] for e in object_errors)), 'source', 'Admission accounting')
    check.check(admission['admissible'] == all(admission['checks'].values()), 'source', 'Admission status')
    metrics = metric_recount(rows, ARMS)
    strata = {key: {v: metric_recount([r for r in rows if r[key] == v], ARMS) for v in sorted({r[key] for r in rows})} for key in AXES}
    clip_rows = {c: [r for r in rows if r['clip_id'] == c] for c in clips}
    intended = [v for v in clip_rows.values() if v[0]['layout_relation'] in ('INSIDE', 'BOUNDARY')]
    source_checks = dict(complete_frames=len(rows) == 576, complete_clips=len(clips) == 48,
        complete_groups=len(strata['base_group_id']) == 8,
        complete_trajectories=set(strata['trajectory']) == {'approach_return', 'approach_dwell_return'} and
            all(m['frames'] == 288 and m['clips'] == 24 for m in strata['trajectory'].values()),
        all_query_labels_valid=labels['valid'].size == 3456 and bool(labels['valid'].all()),
        all_objects_bounded_2mm=len(object_errors) == 1728 and all(e['within_2mm'] for e in object_errors),
        intended_positive_clips_present=len(intended) == 32,
        all_intended_clips_have_known_positive_event=all(any(r['truth'] is True for r in clip) for clip in intended))
    check.same(admission, dict(checks=source_checks, admissible=all(source_checks.values()),
        status='PASS' if all(source_checks.values()) else 'NOT_EVALUABLE'), 'source', 'Independent admission clauses')
    check.same(saved['metrics'], metrics, 'metrics', 'All frames/events'); check.same(saved['strata'], strata, 'metrics', 'All strata')
    gate = independent_gate(rows, metrics, strata, admission['admissible'])
    check.same(saved['stability'], gate, 'gate', 'Frozen three clauses')
    source_hashes = {str(p.relative_to(REPO)): digest(p) for p in (Path(__file__), Path(__file__).with_name('local_support_audit.py'))}
    answer = dict(status='PASS', checks=dict(check.counts), frames=576, clips=48, queries=3456,
        native_subset=dict(rule='Every clip frame_in_clip == 3, including OUTSIDE controls', frames=48, queries=288, rows=subset),
        independent_metrics=metrics, gate=gate, input_seal=inputs, source_hashes=source_hashes, governance=governance,
        backend='CPU NumPy '+np.__version__+' TASK_NOT_GPU_SUITABLE', fits=0, model_replays=0,
        limits=['Native support verified at 48 fixed entry frames; other validity flags use sealed visibility counts.',
                'Cuboid truth, readouts and event metrics cover all 576 frames; no depth interpolation.',
                'Source/model seals prove recorded identity, not independent model or feature replay.',
                'Generic metric counters shared with earlier independent audit; no current evaluator imported.',
                'Controlled posed Development; no physical motion, natural scene or product evidence.'])
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.open('x', encoding='utf-8') as stream: json.dump(answer, stream, indent=2, allow_nan=False)
    print('LOCAL_STABILITY_AUDIT_PASS', flush=True)
    return answer


def synthetic_checks():
    camera = dict(x=0., y=0., z=0.)
    def cube(x, y, z, hx=.05, hy=.05, hz=.05):
        return dict(render_bounds_center_m=[z, x, -y], render_bounds_extent_m=[hz, hx, hy])
    # A composite enclosing box crosses centre BODY, but neither separate cube does.
    classes, _, _ = cuboid_truth([cube(-.4, .6, 1.), cube(.4, .6, 1.)], camera)
    assert classes[1] == 6 and classes[0] < 6 and classes[2] < 6
    classes, distance, boxes = cuboid_truth([cube(0., .6, .8, hz=.05)], camera)
    assert classes[1] == 0 and distance[1] == np.float32(.75)
    depth = np.full((360, 640), np.nan, np.float32); depth[240, 320] = 1.
    result = native_support(depth, [(np.array([-.1, .2, .9]), np.array([.1, .3, 1.1]))])
    assert sum(result['visible_pixels']) > 0 and sum(result['unexplained_pixels']) == 0
    bad = native_support(depth, [])
    assert sum(bad['visible_pixels']) == sum(bad['unexplained_pixels']) > 0
    rows = []
    for clip in range(8):
        for frame in range(2):
            flags = dict(A_current=frame == 1, raw=frame == 1, local=True,
                         raw_standalone=False, local_standalone=frame == 0)
            rows.append(dict(id=f'{clip}/{frame}', clip_id=str(clip), frame_in_clip=frame, time_s=frame*.2,
                truth=True, boundary=False, trajectory=str(clip%2), base_group_id=str(clip), layer='BODY',
                predictions={a: dict(alert=f, unknown=False, ambiguous=False) for a, f in flags.items()}))
    def gate():
        m = metric_recount(rows, ARMS)
        strata = {k: {v: metric_recount([r for r in rows if r[k] == v], ARMS) for v in {r[k] for r in rows}}
                  for k in ('trajectory', 'base_group_id', 'layer')}
        return independent_gate(rows, m, strata, True)
    assert gate()['status'] == 'PASS'
    rows[0]['predictions']['local']['alert'] = False
    assert gate()['status'] == 'FAIL' and gate()['timing']['benefit_events'] == 7
    rows[0]['predictions']['local']['alert'] = True
    for row in rows:
        for arm in ('A_current', 'raw'): row['predictions'][arm]['alert'] = row['frame_in_clip'] == 0
    assert gate()['status'] == 'NOT_EVALUABLE' and gate()['timing']['opportunity_events'] == 0
    return dict(status='PASS', cases=7, scientific_inputs_read=False)
