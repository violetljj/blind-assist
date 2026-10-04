"""Evaluator-only source visibility diagnostic; cached M3 operating point.

No photons, inference, training, threshold change or protected-test access.
Actual exposures and the interpolated physical deadline are reported separately.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import cnh_margin_confirm as MC
import cnh_proposal_attribution_scenes as S
import cnh_sequence_observed_geometry as G

OUT = MC.SS.WORK/'cnh-memory-reality-20261004/source_visibility'
COHORTS = {'source96000': MC.SS.WORK/'cnh-observed-sequence-20261002',
           'supplement94000': MC.SS.WORK/'cnh-sequence-transfer-20261002'}
STATES = ('VISIBLE', 'FOV_OUTSIDE', 'OCCLUDED', 'NO_QUADRATURE_RAY',
          'OUT_OF_RADIAL_WINDOW', 'UNKNOWN')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def interpolate_sensor(a, b, alpha):
    p = np.eye(4)
    p[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))([alpha]).as_matrix()[0]
    p[:3, 3] = a[:3, 3]+alpha*(b[:3, 3]-a[:3, 3])
    return p


def frustum_overlap(box, sensor):
    """Continuous positive-area target surface/FOV overlap, no ray sampling."""
    triangles = (S.box_mesh(box['lo'], box['hi'])-sensor[:3, 3])@sensor[:3, :3]
    edge = np.tan(np.pi/8)
    planes = [(np.array([0., 0., 1.]), -1e-10),
              (np.array([1., 0., edge]), 0.), (np.array([-1., 0., edge]), 0.),
              (np.array([0., 1., edge]), 0.), (np.array([0., -1., edge]), 0.)]
    for triangle in triangles:
        poly = list(triangle)
        for normal, offset in planes:
            clipped = []
            if not poly:
                break
            for start, end in zip(poly, poly[1:]+poly[:1]):
                ds, de = float(start@normal+offset), float(end@normal+offset)
                if ds >= 0:
                    clipped.append(start)
                if (ds >= 0) != (de >= 0):
                    clipped.append(start+ds/(ds-de)*(end-start))
            poly = clipped
        for i in range(1, len(poly)-1):
            if np.linalg.norm(np.cross(poly[i]-poly[0], poly[i+1]-poly[0])) > 1e-14:
                return True
    return False


def visibility(boxes, sensor, body, query, directions, weights):
    world = directions@sensor[:3, :3].T
    full = S.raycast_boxes(sensor[:3, 3], world, boxes)
    alone = S.raycast_boxes(sensor[:3, 3], world, boxes[:1])
    params, _ = S.nominal_parameters()
    minimum, maximum = params.range_zero_m, params.range_zero_m+.0375348*128
    in_window = full['valid'] & (full['distance'] >= minimum) & (full['distance'] < maximum)
    visible = (full['object_id'] == 0) & in_window
    standalone = alone['valid'] & (alone['distance'] >= minimum) & (alone['distance'] < maximum)
    overlap = frustum_overlap(boxes[0], sensor)
    if visible.any():
        state = 'VISIBLE'
    elif not overlap:
        state = 'FOV_OUTSIDE'
    elif not alone['valid'].any():
        state = 'NO_QUADRATURE_RAY'
    elif not standalone.any():
        state = 'OUT_OF_RADIAL_WINDOW'
    else:
        state = 'OCCLUDED'
    safe_distance = np.where(full['valid'], full['distance'], 0.)
    points = sensor[:3, 3]+safe_distance[..., None]*world/np.linalg.norm(world, axis=-1)[..., None]
    local = (points-body[:3, 3])@body[:3, :3]
    in_query = ((local >= S.QUERY_LOW[query]+1e-8) & (local <= S.QUERY_HIGH[query]-1e-8)).all(-1)
    query_visible = visible & in_query
    metrics = dict(state=state, continuous_frustum_overlap=bool(overlap),
                   target_rays=int(visible.sum()), target_only_rays=int(standalone.sum()),
                   blocked_target_rays=int((standalone & ~visible).sum()),
                   target_solid_angle_fraction=float(weights[visible].sum()/weights.sum()),
                   query_target_rays=int(query_visible.sum()),
                   query_target_solid_angle_fraction=float(weights[query_visible].sum()/weights.sum()))
    return metrics, full['object_id'].astype(np.int16), visible, query_visible


def load_cohort(name, folder):
    required = ['geometry.npz', 'rows.json', 'geometry_receipt.json', 'result.json']
    if not all((folder/f).is_file() for f in required):
        return None, dict(status='NOT_EVALUABLE', reason='Missing same-contract cached geometry/scores/result')
    receipt = read(folder/'geometry_receipt.json')
    if receipt['status'] != 'COMPLETE':
        raise ValueError('Incomplete geometry '+name)
    hashes = {str(folder/f): sha(folder/f) for f in required}
    for fn, field in [('geometry.npz', 'geometry_sha256'), ('rows.json', 'rows_sha256')]:
        assert hashes[str(folder/fn)] == receipt[field], (name, fn)
    with np.load(folder/'geometry.npz') as cache:
        g = {k: cache[k] for k in cache.files}
    rows, result = read(folder/'rows.json'), read(folder/'result.json')
    assert result['status'] == 'COMPLETE'
    assert [(r['unit'], r['config'], r['query']) for r in rows] == list(zip(g['unit'], g['config'], g['query']))
    threshold = float(result['cells']['M3']['threshold'])
    if name == 'source96000':
        old = read(MC.OUT/'sequence_result.json')
        files = [MC.OUT/'frame_scores_M3_early.npz', MC.OUT/'frame_scores_M3.npz']
        for path in files:
            assert sha(path) == old['provenance']['input_sha256'][path.name]
        with np.load(files[0]) as early, np.load(files[1]) as late:
            raw = np.stack([np.concatenate([early[str(u)], late[str(u)]], axis=1)
                            for s, units in MC.SPLITS.items() for u in units])
        hashes[str(MC.OUT/'sequence_result.json')] = sha(MC.OUT/'sequence_result.json')
    else:
        files = [folder/'frame_scores_M3.npz']
        score_receipt = read(folder/'scores_receipt.json')
        assert score_receipt['status'] == 'COMPLETE'
        assert sha(files[0]) == score_receipt['output_sha256']['M3']
        with np.load(files[0]) as cache:
            raw = cache['logit']
            assert np.array_equal(cache['frames'], np.arange(3, 16))
            assert np.array_equal(cache['units'], np.arange(94000, 94096))
        hashes[str(folder/'scores_receipt.json')] = sha(folder/'scores_receipt.json')
        assert threshold == read(COHORTS['source96000']/'result.json')['cells']['M3']['threshold']
    for path in files:
        hashes[str(path)] = sha(path)
    assert raw.shape[1:] == (40, 13, 2) and np.isfinite(raw).all()
    smooth = np.empty_like(raw, dtype=float)
    for frame in range(13):
        w = np.array([1., 2., 4., 8., 16.])[-min(frame+1, 5):]
        smooth[:, :, frame] = np.tensordot(raw[:, :, frame+1-len(w):frame+1], w/w.sum(), axes=([2], [0]))
    score = smooth.transpose(0, 1, 3, 2).reshape(-1, 13)
    assert score.shape == g['frame_ranges'].shape
    keep = (g['split'] == 'evaluation') & g['covered'] & (g['ref_category'] == 'contact0-2cm')
    alarm = score >= threshold
    stopped = alarm.any(1)
    first = alarm.argmax(1)
    timely = stopped & (g['frame_ranges'][np.arange(len(first)), first] >= .9)
    metrics = result['cells']['M3']['metrics']['contact0-2cm']
    assert int(keep.sum()) == metrics['n'] and int((keep & stopped).sum()) == metrics['stops']
    assert int((keep & timely).sum()) == metrics['timely_stops']
    inputs = receipt.get('inputs', {})
    expected_sources = inputs.get('original_source_sha256', inputs.get('dependency_sha256', {}))
    for filename, digest in expected_sources.items():
        assert sha(Path(__file__).with_name(filename)) == digest, filename
    return (g, rows, score, threshold, keep, stopped, timely, first), dict(
        status='EVALUABLE', input_sha256=hashes, geometry_source_bindings=expected_sources,
        m3_point_parity=dict(n=int(keep.sum()), stopped=int((keep & stopped).sum()), timely=int((keep & timely).sum())))


def classification(sequence, endpoint):
    if endpoint:
        return 'visible_at_endpoint'
    return 'saw_before_then_lost' if any(sequence) else 'never_seen'


def summarize(events):
    def group(items):
        return dict(n=len(items), m3_timely=sum(e['m3_timely'] for e in items),
                    actual_last_predeadline=dict(Counter(e['actual_class'] for e in items)),
                    retained_frames3plus=dict(Counter(e['retained_class'] for e in items)),
                    interpolated_deadline=dict(Counter(e['deadline_class'] for e in items)),
                    current_m3_history=dict(Counter(e['m3_history_class'] for e in items)),
                    query_support_last_predeadline=dict(Counter(e['query_actual_class'] for e in items)))
    per_frame = []
    for f in range(16):
        eligible = [e for e in events if f <= e['last_predeadline_frame']]
        v = [e['frames'][f] for e in eligible]
        per_frame.append(dict(frame=f, time_s=f*.2, n=len(v), visible=sum(a['target_rays'] > 0 for a in v),
                              states=dict(Counter(a['state'] for a in v)),
                              mean_target_fraction=float(np.mean([a['target_solid_angle_fraction'] for a in v])) if v else None,
                              query_supported=sum(a['query_target_rays'] > 0 for a in v)))
    return dict(all=group(events), by_mode={str(m): group([e for e in events if e['mode'] == m]) for m in range(3)},
                timely=group([e for e in events if e['m3_timely']]),
                missed_timely=group([e for e in events if not e['m3_timely']]), per_frame_before_deadline=per_frame,
                endpoint_states=dict(Counter(e['frames'][e['last_predeadline_frame']]['state'] for e in events)))


def run():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'result.json').exists():
        raise FileExistsError('Inspect existing result; no silent rerun')
    (OUT/'source_base.py').write_bytes(Path(__file__).read_bytes())
    directions, weights = S.angular_rays(16)
    cohorts, all_events = {}, {}
    for name, folder in COHORTS.items():
        loaded, provenance = load_cohort(name, folder)
        if loaded is None:
            cohorts[name] = provenance
            continue
        g, rows, score, threshold, keep, stopped, timely, first = loaded
        events, originals = [], {}
        for index in np.flatnonzero(keep):
            row = rows[index]
            u, c, q = (int(row[k]) for k in ('unit', 'config', 'query'))
            if u not in originals:
                originals[u] = (MC.scenes_for(u) if name == 'source96000' else MC.NR.scenes_for(u))
            scene = originals[u][c]
            assert np.allclose(scene['travel'][3:16], g['frame_poses'][index], atol=1e-12, rtol=0)
            vertices = G.corners(scene['boxes'][0])
            ranges = np.array([G.target_front(vertices, p) for p in scene['travel'][3:16]])
            assert np.allclose(ranges, g['frame_ranges'][index], atol=1e-12, rtol=0)
            assert abs(G.target_front(vertices, g['reference_pose'][index])-.9) < 1e-10
            frame_metrics, ids, visibles, supports = [], [], [], []
            for f in range(16):
                m, oid, v, support = visibility(scene['boxes'], scene['poses'][f], scene['travel'][f], q, directions, weights)
                frame_metrics.append(m); ids.append(oid); visibles.append(v); supports.append(support)
            fraction = float(g['reference_fraction'][index])
            last = int(np.floor(fraction+1e-10))
            left, right = int(g['crossing_left_frame'][index]), int(g['crossing_right_frame'][index])
            alpha = float(g['interpolation_alpha'][index])
            sensor = interpolate_sensor(scene['poses'][left], scene['poses'][right], alpha) if left != right else scene['poses'][left]
            deadline, did, dv, ds = visibility(scene['boxes'], sensor, g['reference_pose'][index], q, directions, weights)
            seen = [bool(v.any()) for v in visibles]
            query_seen = [bool(v.any()) for v in supports]
            event = dict(row_index=int(index), unit=u, config=c, query=q, query_name=('HEAD', 'BODY')[q],
                         target_group=int(scene['group']), family=scene['family'], mode=int(scene['mode']),
                         target_intrusion_m=-float(scene['margin']), final_range_m=float(ranges[-1]),
                         m3_timely=bool(timely[index]), m3_stopped=bool(stopped[index]),
                         m3_first_alarm_frame=int(first[index]+3) if stopped[index] else None,
                         frozen_threshold=threshold, max_smoothed_score=float(score[index].max()),
                         deadline_fraction=fraction, deadline_time_since_frame3_s=float(g['reference_time_s'][index]),
                         last_predeadline_frame=last, frames=frame_metrics, interpolated_deadline=deadline,
                         actual_class=classification(seen[:last], seen[last]),
                         retained_class=classification(seen[3:last], seen[last]),
                         m3_history_class=classification(seen[max(0, last-7):last], seen[last]),
                         query_actual_class=classification(query_seen[:last], query_seen[last]),
                         deadline_class=classification(seen[:last+1], bool(dv.any())),
                         first_seen_frame=next((f for f, yes in enumerate(seen[:last+1]) if yes), None),
                         last_seen_frame=next((f for f in range(last, -1, -1) if seen[f]), None),
                         visible_predeadline_frames=sum(seen[:last+1]),
                         total_predeadline_frames=last+1,
                         visible_predeadline_fraction=sum(seen[:last+1])/(last+1))
            events.append(event)
            payload = OUT/name/f'unit{u}_config{c}_query{q}.npz'
            payload.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(payload, object_id=np.stack(ids), target_visible=np.stack(visibles),
                                query_target_visible=np.stack(supports), deadline_object_id=did,
                                deadline_target_visible=dv, deadline_query_target_visible=ds,
                                true_sensor_poses=scene['poses'], true_travel_poses=scene['travel'],
                                sensor_directions=directions, solid_angle_weights=weights,
                                frozen_smoothed_m3=score[index], frames=np.arange(16))
            event['payload_sha256'] = sha(payload)
            print(name, u, c, q, event['actual_class'], 'timely', event['m3_timely'], flush=True)
        all_events[name] = events
        save(OUT/name/'events.json', events)
        cohorts[name] = dict(status='COMPLETE', provenance=provenance, summary=summarize(events),
                             missed_timely_details=[e for e in events if not e['m3_timely']],
                             event_file=str(OUT/name/'events.json'))
    result = dict(status='COMPLETE', role='Consumed synthetic Development; evaluator/source truth only',
                  elapsed_s=time.perf_counter()-started, cohorts=cohorts,
                  source_sha256={p.name: sha(p) for p in [Path(__file__), Path(S.__file__), Path(MC.__file__), Path(MC.NR.__file__), Path(G.__file__)]},
                  definitions=dict(visible='At least one first-visible target ray inside direct raw histogram radial window; solid-angle fraction uses complete sensor FOV denominator',
                      actual_endpoint='Last real exposure at/before .9m crossing, frames0..last; causal M3 history max(0,last-7)..last separately',
                      interpolated_endpoint='Physical .9m pose; diagnostic only, not an acquired exposure',
                      unknown='Never-seen at quadrature16 is not absence of continuous surface; NO_QUADRATURE_RAY remains separate',
                      query_support='First-visible target points within current body query x±.30m, height band, z(.3,3)m'),
                  limits=['Geometry visibility does not establish sufficient photon SNR or memory benefit',
                          'Finite 16x16 rays per zone; tiny surfaces can miss quadrature',
                          'First hit in radial window measures direct return support, not pulse/crosstalk leakage',
                          'Two source cohorts remain separate; no protected test or new inference'])
    save(OUT/'result.json', result)
    check()
    supplement()
    print('COMPLETE elapsed_s', result['elapsed_s'], flush=True)


def check():
    """One payload-level cross-check independent of aggregation branches."""
    started = time.perf_counter()
    result = read(OUT/'result.json')
    point = {}
    for name, cohort in result['cohorts'].items():
        if cohort['status'] != 'COMPLETE':
            continue
        events = read(cohort['event_file'])
        visible_count = lost_count = never_count = 0
        for e in events:
            path = OUT/name/f"unit{e['unit']}_config{e['config']}_query{e['query']}.npz"
            assert sha(path) == e['payload_sha256']
            with np.load(path) as a:
                oid, visible, weights = a['object_id'], a['target_visible'], a['solid_angle_weights']
                assert oid.shape == visible.shape == (16, 8, 8, 256)
                assert np.all((oid == 0)[visible])
                n = e['last_predeadline_frame']
                endpoint = bool(visible[n].any()); prior = bool(visible[:n].any())
                visible_count += endpoint; lost_count += prior and not endpoint; never_count += not prior and not endpoint
                for f in range(16):
                    assert int(visible[f].sum()) == e['frames'][f]['target_rays']
                    assert abs(float(weights[visible[f]].sum()/weights.sum())-e['frames'][f]['target_solid_angle_fraction']) < 1e-14
        counts = cohort['summary']['all']['actual_last_predeadline']
        assert (visible_count, lost_count, never_count) == tuple(counts.get(k, 0) for k in ('visible_at_endpoint', 'saw_before_then_lost', 'never_seen'))
        point[name] = dict(n=len(events), visible_at_endpoint=int(visible_count), saw_before_then_lost=int(lost_count), never_seen=int(never_count),
                          m3_timely=sum(e['m3_timely'] for e in events))
    assert point['source96000']['n'] == 31 and point['source96000']['m3_timely'] == 26
    source_miss = {(e['unit'], e['config'], e['query']) for e in read(OUT/'source96000/events.json') if not e['m3_timely']}
    assert source_miss == {(96004, 3, 1), (96031, 36, 1), (96057, 26, 1), (96073, 11, 1), (96082, 35, 0)}
    save(OUT/'verification.json', dict(status='PASS', result_sha256=sha(OUT/'result.json'), point_reconstruction=point,
        source_five_miss_identity_match=True, frozen_m3_count_parity=True, observed_travel_front_cache_parity=True,
        payload_visible_mask_and_solid_angle_reconstruction=True, elapsed_s=time.perf_counter()-started))


def supplement():
    """Read retained payloads only: history ages and deadline contributors.

    This separate receipt preserves the completed rendering producer and hash.
    No raycasting, observations, scores or original result are changed.
    """
    started = time.perf_counter()
    base = read(OUT/'result.json')
    assert sha(OUT/'source_base.py') == base['source_sha256'][Path(__file__).name]
    cohorts, hashes = {}, {str(OUT/'result.json'): sha(OUT/'result.json'),
                           str(OUT/'source_base.py'): sha(OUT/'source_base.py')}
    for name, cell in base['cohorts'].items():
        if cell['status'] != 'COMPLETE':
            cohorts[name] = dict(status='NOT_EVALUABLE')
            continue
        folder = COHORTS[name]
        manifest_path = MC.OUT/'scene_manifest.json' if name == 'source96000' else folder/'scene_manifest.json'
        manifest = {(r['unit'], r['config']): r for r in read(manifest_path)}
        geometry_receipt = read(folder/'geometry_receipt.json')
        expected_manifest = (geometry_receipt['inputs']['scene_manifest_sha256'] if name == 'source96000'
                             else geometry_receipt['scene_manifest_sha256'])
        assert sha(manifest_path) == expected_manifest
        hashes[str(manifest_path)] = sha(manifest_path)
        with np.load(folder/'geometry.npz') as geometry:
            reference_poses = geometry['reference_pose']
        events = read(cell['event_file'])
        enriched = []
        for e in events:
            scene = manifest[(e['unit'], e['config'])]
            pose = reference_poses[e['row_index']]
            categories = {str(i): G.surface_category((S.box_mesh(b['lo'], b['hi'])-pose[:3, 3])@pose[:3, :3], e['query'])
                          for i, b in enumerate(scene['boxes'])}
            contributors = [int(i) for i, category in categories.items() if category.startswith('contact')]
            assert contributors, (name, e['unit'], e['config'], e['query'])
            # Per-object reconstruction must agree with the frozen union truth.
            assert all(categories[str(i)] == 'contact0-2cm' for i in contributors)
            last = e['last_predeadline_frame']
            payload = OUT/name/f"unit{e['unit']}_config{e['config']}_query{e['query']}.npz"
            assert sha(payload) == e['payload_sha256']
            with np.load(payload) as a:
                observed = a['target_visible'].any(axis=(1, 2, 3))
            raw_start = max(0, last-7)
            # First evaluated smoothed logit is frame3; each of its raw logits
            # sees up to eight exposures. Latest mature smooth5 footprint is12.
            smooth_start = max(0, max(3, last-4)-7)
            first_seen, last_seen = e['first_seen_frame'], e['last_seen_frame']
            raw_seen = np.flatnonzero(observed[raw_start:last+1])+raw_start
            smooth_seen = np.flatnonzero(observed[smooth_start:last+1])+smooth_start
            extra = dict(unit=e['unit'], config=e['config'], query=e['query'], mode=e['mode'], family=e['family'],
                         m3_timely=e['m3_timely'], m3_stopped=e['m3_stopped'], m3_first_alarm_frame=e['m3_first_alarm_frame'],
                         actual_class=e['actual_class'], query_actual_class=e['query_actual_class'],
                         deadline_object_categories=categories, deadline_contact_contributor_ids=contributors,
                         object0_is_contact_contributor=0 in contributors,
                         target_visibility_as_shallow_cause=('EVALUABLE' if 0 in contributors else 'UNKNOWN_TARGET_IS_NOT_CONTACT_CONTRIBUTOR'),
                         first_seen_frame=first_seen, last_seen_frame=last_seen, last_predeadline_frame=last,
                         last_seen_to_last_predeadline_s=(last-last_seen)*.2 if last_seen is not None else None,
                         last_seen_to_interpolated_deadline_s=(e['deadline_fraction']-last_seen)*.2 if last_seen is not None else None,
                         raw8_start_frame=raw_start, raw8_visible_frames=raw_seen.tolist(), raw8_contains_target=bool(len(raw_seen)),
                         smooth_up_to12_start_frame=smooth_start, smooth_up_to12_visible_frames=smooth_seen.tolist(),
                         smooth_up_to12_contains_target=bool(len(smooth_seen)),
                         endpoint_state=e['frames'][last]['state'],
                         max_predeadline_target_solid_angle_fraction=max(x['target_solid_angle_fraction'] for x in e['frames'][:last+1]),
                         last_seen_target_rays=e['frames'][last_seen]['target_rays'] if last_seen is not None else None,
                         last_seen_query_target_rays=e['frames'][last_seen]['query_target_rays'] if last_seen is not None else None)
            enriched.append(extra)
        cohorts[name] = dict(status='COMPLETE', n=len(enriched),
                             actual_shallow_target_contributor_n=sum(e['object0_is_contact_contributor'] for e in enriched),
                             target_visibility_as_shallow_cause=dict(Counter(e['target_visibility_as_shallow_cause'] for e in enriched)),
                             raw8_contains_target_n=sum(e['raw8_contains_target'] for e in enriched),
                             smooth_up_to12_contains_target_n=sum(e['smooth_up_to12_contains_target'] for e in enriched),
                             missed_timely_details=[e for e in enriched if not e['m3_timely']], events=enriched)
    result = dict(status='COMPLETE', role='Post-render cached diagnostic supplement; evaluator-only',
                  elapsed_s=time.perf_counter()-started, source_sha256=sha(__file__), input_sha256=hashes, cohorts=cohorts,
                  limits=['raw8 membership means geometric support was inside input exposure footprint; does not establish photon SNR or recovered correspondence',
                          'smooth footprint combines five causal raw8 logits, up to12 exposures; this is not a single12-frame voxel memory',
                          'object0 visibility is not visibility of a panel-caused shallow surface; contributor-mismatch events stay in the full denominator as UNKNOWN',
                          'query_target support in base events is in each exposure current body query, not the deadline-anchored physical sliver'])
    save(OUT/'history_contributors_receipt.json', result)
    lines = ['# Source visibility diagnostic', '',
             'Consumed synthetic Development; frozen cached M3 scores and .9m deadline. No inference/training.', '',
             f"CPU geometry render/analysis: {base['elapsed_s']:.6f}s; payload/count cross-check PASS; supplement {result['elapsed_s']:.6f}s.", '']
    for name, cell in base['cohorts'].items():
        if cell['status'] != 'COMPLETE':
            lines += [f'{name}: NOT_EVALUABLE', '']; continue
        summary, extra = cell['summary']['all'], cohorts[name]
        lines += [f"## {name}", '', f"n={summary['n']}; M3 timely={summary['m3_timely']}; target visibility at last actual predeadline exposure={summary['actual_last_predeadline']}; endpoint states={cell['summary']['endpoint_states']}.",
                  f"Target object0 is an actual deadline contact contributor in {extra['actual_shallow_target_contributor_n']}/{extra['n']}; other events are UNKNOWN for target-based shallow-cause interpretation.",
                  f"Raw8 footprint still includes target support in {extra['raw8_contains_target_n']}/{extra['n']}; smooth5/raw8 combined footprint in {extra['smooth_up_to12_contains_target_n']}/{extra['n']}.", '',
                  '|unit/config/query|M3 timely|endpoint|last seen→last actual / physical deadline s|raw8/smooth12 visible frames|',
                  '|---|---|---|---|---|']
        for e in extra['missed_timely_details']:
            lines.append(f"|{e['unit']}/{e['config']}/{e['query']}|{e['m3_timely']}|{e['endpoint_state']}|{e['last_seen_to_last_predeadline_s']:.3f} / {e['last_seen_to_interpolated_deadline_s']:.3f}|{e['raw8_visible_frames']} / {e['smooth_up_to12_visible_frames']}|")
        lines += ['']
    lines += ['Geometric visibility is necessary support, not measured photon detectability or evidence that memory training will improve alerts. Four source missed-timely cases lose the target outside FOV but all still have visible exposures inside raw8: this alone does not support a too-short-window explanation.', '',
              'Per-frame/solid-angle and interpolated deadline fields: result.json and cohort/events.json. Per-object deadline contributors and history ages: history_contributors_receipt.json. Interpolated deadline is not an acquired exposure.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps({k: {field: value for field, value in v.items() if field not in ('events', 'missed_timely_details')} for k, v in cohorts.items()}, indent=2), flush=True)
    print('supplement_elapsed_s', result['elapsed_s'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['run', 'check', 'supplement'], default='run')
    args = parser.parse_args()
    {'run': run, 'check': check, 'supplement': supplement}[args.stage]()
