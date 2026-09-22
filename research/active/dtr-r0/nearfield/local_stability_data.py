"""Frozen stability observations and full union-of-cuboids evaluator labels."""
from collections import defaultdict
from pathlib import Path
import time

import cv2
import numpy as np

from query_occupancy_data import (stage_path, read, write, sha, new_stage_directory,
    observation_tokens, camera_bounds, geometric_labels, visible_labels, QUERIES, EDGES)


def source_admission(identities, audits, classes, valid):
    """Keep every row; source inadequacy is separate from materialization success."""
    classes, valid = np.asarray(classes), np.asarray(valid, bool)
    assert classes.shape == valid.shape == (len(identities), 6)
    positive = (classes[:, [1, 4]] < len(EDGES)-1).any(axis=1)
    known = valid[:, [1, 4]].all(axis=1)
    clips, trajectories = defaultdict(list), defaultdict(list)
    for i, row in enumerate(identities):
        clips[row['clip_id']].append(i)
        trajectories[row['trajectory']].append(i)
    clip_rows = []
    for clip_id, indices in sorted(clips.items()):
        first = identities[indices[0]]
        occupied = positive[indices] & known[indices]
        events = int(np.count_nonzero(occupied & ~np.r_[False, occupied[:-1]]))
        intended = first['layout_relation'] in ('INSIDE', 'BOUNDARY')
        clip_rows.append(dict(clip_id=clip_id, trajectory=first['trajectory'],
            base_group_id=first['base_group_id'], layout_relation=first['layout_relation'],
            frames=len(indices), positive_frames=int(occupied.sum()),
            unknown_frames=int((~known[indices]).sum()), known_positive_events=events,
            intended_positive=intended, positive_event_admissible=not intended or events > 0))
    counts = {name: dict(frames=len(indices),
        clips=len({identities[i]['clip_id'] for i in indices}),
        positive_frames=int((positive[indices] & known[indices]).sum()),
        negative_frames=int((~positive[indices] & known[indices]).sum()),
        unknown_frames=int((~known[indices]).sum()),
        valid_queries=int(valid[indices].sum())) for name, indices in sorted(trajectories.items())}
    objects = [obj for row in audits for obj in row['object_bounds']]
    checks = dict(complete_frames=len(identities) == len(audits) == 576,
        complete_clips=len(clips) == 48 and all(len(v) == 12 for v in clips.values()),
        complete_groups=len({r['base_group_id'] for r in identities}) == 8,
        complete_trajectories=set(counts) == {'approach_return', 'approach_dwell_return'}
            and all(v['frames'] == 288 and v['clips'] == 24 for v in counts.values()),
        all_query_labels_valid=valid.size == 3456 and bool(valid.all()),
        all_objects_bounded_2mm=len(objects) == 1728 and all(o['within_2mm'] for o in objects),
        intended_positive_clips_present=sum(r['intended_positive'] for r in clip_rows) == 32,
        all_intended_clips_have_known_positive_event=all(r['positive_event_admissible'] for r in clip_rows))
    return dict(status='PASS' if all(checks.values()) else 'NOT_EVALUABLE',
        admissible=all(checks.values()), checks=checks, frames=len(identities),
        queries=int(valid.size), valid_queries=int(valid.sum()), invalid_queries=int((~valid).sum()),
        rendered_objects=len(objects),
        max_center_error_m=max((o['center_error_m'] for o in objects), default=None),
        max_size_error_m=max((o['size_error_m'] for o in objects), default=None),
        counts_by_trajectory=counts, clips=clip_rows,
        truth_scope='UNION_OF_EACH_RENDERED_CUBOID_SEPARATELY_NO_ENCLOSING_BOX',
        missing_support_policy='RETAIN_ALL_FRAMES_AND_UNKNOWN')


def materialize(root):
    """Called by the governed runner; root is the source stage, not -prepared."""
    from ba_camera_corridor import sample_native
    from tof_fov45_core import boxes45, simulate
    from tof_corridor_calibration import score_frame, decide
    from local_stability_spec import check_spec

    root = Path(root)
    cap = stage_path(root, 'capture')
    spec = read(root/'plan/spec.json')
    check_spec(spec)
    receipt, launch = read(cap/'receipt.json'), read(cap/'launch-receipt.json')
    assert receipt['status'] == 'PASS' and read(cap/'process-release.json')['released']
    assert receipt['source_unchanged'] and receipt['task_actors_released']
    assert receipt['spec_sha256'] == launch['spec_sha256'] == sha(root/'plan/spec.json')
    assert receipt['protocol_sha256'] == launch['protocol_sha256'] == sha(root/'plan/protocol.json')
    assert sha(cap/'evaluator/spec.json') == sha(root/'plan/spec.json') or read(cap/'evaluator/spec.json') == spec
    manifests = read(cap/'observations/manifest.json')['frames']
    geometry = read(cap/'evaluator/geometry.json')
    cases = spec['cases']
    assert len(cases) == len(manifests) == len(geometry) == spec['frames'] == 576
    out = stage_path(root, 'prepared')
    new_stage_directory(out)
    (out/'observations').mkdir()
    (out/'labels').mkdir()
    boxes, n = boxes45(), len(cases)
    rgb = np.lib.format.open_memmap(out/'observations/rgb.npy', mode='w+', dtype=np.uint8, shape=(n, 3, 180, 320))
    tof = np.lib.format.open_memmap(out/'observations/tof.npy', mode='w+', dtype=np.float32, shape=(n, 64, 6))
    labels = dict(classes=[], distances=[], mask=[], coverage=[], valid=[])
    identities, audits, start = [], [], time.perf_counter()
    try:
        for i, (case, row, geo) in enumerate(zip(cases, manifests, geometry)):
            assert case['split'] == 'evaluation'
            assert case['name'] == row['id'] == geo['id'] and row['sample_index'] == geo['sample_index'] == i
            assert case['clip_id'] == row['clip_id'] == geo['clip_id']
            assert case['frame_in_clip'] == row['frame_in_clip'] == geo['frame_in_clip']
            assert case['time_s'] == row['time_s'] == geo['nominal_time_s']
            assert case['camera'] == geo['declared_camera']
            assert max(abs(geo['actual_camera_location_m'][j]-case['camera'][k]) for j, k in enumerate(('x', 'y', 'z'))) <= .002
            assert len(case['objects']) == len(geo['objects']) == 3
            assert {o['name'] for o in case['objects']} == {'target_a', 'target_b', 'background'}
            bounds_audit = []
            for planned, actual in zip(case['objects'], geo['objects']):
                assert planned['name'] == actual['name'] and planned['kind'] == 'cube'
                assert actual['mesh_path'].split('.')[-1] == 'Cube'
                center_error = float(np.max(np.abs(np.asarray(planned['center_m'])-actual['render_bounds_center_m'])))
                size_error = float(np.max(np.abs(np.asarray(planned['size_m'])-2*np.asarray(actual['render_bounds_extent_m']))))
                bounds_audit.append(dict(name=planned['name'], center_error_m=center_error,
                    size_error_m=size_error, within_2mm=center_error <= .002 and size_error <= .002))
            rp, dp = cap/'observations'/row['rgb_path'], cap/'evaluator'/geo['native_path']
            assert sha(rp) == row['rgb_sha256'] == geo['rgb_sha256'] and sha(dp) == geo['native_sha256']
            image = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
            assert image.shape == (360, 640, 3)
            rgb[i] = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
            depth = np.load(dp, allow_pickle=False)
            values, _ = simulate(sample_native(depth), 'local-stability/'+case['sensor_noise_key'], boxes)
            tof[i] = observation_tokens(values, boxes)
            baseline = decide(score_frame(boxes, values), .4071309640537889)
            # Each rendered cuboid stays separate, including the background.
            bounds = camera_bounds(geo['objects'], case['camera'])
            classes, distances = geometric_labels(bounds)
            visible = visible_labels(depth, bounds)
            valid = np.asarray(visible['unexplained']) == 0
            for key, value in dict(classes=classes, distances=distances, mask=visible['mask'],
                    coverage=visible['coverage'], valid=valid).items():
                labels[key].append(value)
            identities.append(dict(index=i, id=case['name'], split='evaluation', clip_id=case['clip_id'],
                frame_in_clip=case['frame_in_clip'], time_s=case['time_s'],
                base_group_id=case['base_group_id'], type_id=case['type_id'], layer=case['layer'],
                trajectory=case['trajectory'], size_level=case['size_level'], shape=case['shape'],
                layout_relation=case['layout_relation'], rgb_sha256=row['rgb_sha256'], baseline=baseline))
            audits.append(dict(index=i, source_native_sha256=geo['native_sha256'],
                visible_pixels=visible['count'], unexplained_pixels=visible['unexplained'],
                visible_nearest=visible['visible_nearest'], query_label_valid=valid.tolist(),
                object_bounds=bounds_audit))
            if i % 144 == 0:
                print('MATERIALIZE', i, '/', n, flush=True)
        rgb.flush()
        tof.flush()
    finally:
        del rgb, tof
    arrays = {k: np.asarray(v, np.float32 if k in ('mask', 'coverage', 'distances')
              else np.int64 if k == 'classes' else bool) for k, v in labels.items()}
    np.savez_compressed(out/'labels/evaluation.npz', indices=np.arange(n), **arrays)
    write(out/'observations/identities.json', identities)
    write(out/'labels/visibility-audit.json', audits)
    admission = source_admission(identities, audits, arrays['classes'], arrays['valid'])
    write(out/'labels/source-admission.json', admission)
    hashes = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
    report = dict(status='PASS', frames=n, counts=dict(evaluation=n),
        source_admission=admission, counts_by_trajectory=admission['counts_by_trajectory'],
        queries=QUERIES.tolist(), bin_edges_m=EDGES.tolist(),
        source_spec_sha256=sha(root/'plan/spec.json'), hashes=hashes,
        invalid_queries=admission['invalid_queries'], elapsed_s=time.perf_counter()-start,
        backend='TASK_NOT_GPU_SUITABLE')
    write(out/'materialization.json', report)
    return report
