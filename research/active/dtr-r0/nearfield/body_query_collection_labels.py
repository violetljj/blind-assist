"""CUDA evaluator labels/ownership for completed counterfactual collections.

No model instance, weights, inference, optimizer or role-specific fitting data.
Every capture/case remains in the manifest, including rejection/mismatch rows.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

BASIC = {'CLEAR': [0, 0], 'BODY_ONLY': [1, 0], 'HEAD_ONLY': [0, 1], 'BOTH': [1, 1]}
EXTRA = ('LOW', 'ABOVE', 'LATERAL_OUT', 'FAR_OUT')
ROLES = ('TRAIN_ONLY', 'DEV_ONLY', 'EVAL_ONLY')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def confined(root, relative):
    path = (root/relative).resolve(strict=True)
    require(path.is_relative_to(root), 'Capture path escape: '+str(relative))
    return path


def expected_near(variant):
    if variant in BASIC:
        return BASIC[variant]
    if variant in EXTRA:
        return [0, 0]
    raise ValueError('Unknown intended condition: '+str(variant))


def floor_acceptance(case, probes):
    matches = [p for p in probes if p.get('case') == case.get('name')]
    probe = matches[0] if len(matches) == 1 else {}
    point = probe.get('point_m')
    hit = probe.get('hit') is True and isinstance(point, list) and len(point) == 3
    z_error = abs(point[2]-case['floor_z_m']) if hit else None
    xy_error = max(abs(point[j]-case['camera'][axis]) for j, axis in enumerate(('x', 'y'))) if hit else None
    return dict(accepted=bool(hit and z_error <= .02 and xy_error <= .02),
        unique_probe=len(matches) == 1, hit=hit, declared_floor_z_m=case['floor_z_m'],
        point_m=point, z_error_m=z_error, camera_xy_error_m=xy_error, tolerance_m=.02,
        continuous_floor_patch_enabled=case.get('floor_check') is True,
        scope='Camera-point floor probe only; not a continuous floor-patch certificate')


def verify_capture(root):
    """Read-only lifecycle and hash binding, before loading any native tensor."""
    root = root.resolve(strict=True)
    paths = {key: confined(root, name) for key, name in dict(receipt='receipt.json',
        completion='completion.json', release='process-release.json',
        health='render-resource-health.json', integrity='source-integrity.json',
        spec='source/spec.json', world='world-verification.json', log='editor.log').items()}
    meta = {k: read(v) for k, v in paths.items() if k != 'log'}
    receipt, completion, release, health, integrity, spec, world = [meta[k] for k in
        ('receipt', 'completion', 'release', 'health', 'integrity', 'spec', 'world')]
    require(receipt['status'] == completion['status'] == world['status'] == 'PASS', 'Incomplete capture or native verification')
    require(receipt['source_unchanged'] is True and integrity['unchanged'] is True and
        integrity['before'] == integrity['after'], 'Capture source changed')
    require(release['released'] is True and not release.get('survivors', []) and
        release.get('tracking_complete', True) is True, 'Capture processes not fully released')
    require(health['schema'] == 'city-render-resource-health-v1' and health['ready_data_eligible'] is True and
        not any(health['counts'].values()), 'Render-resource health is not eligible')
    require(health['log_sha256'] == sha(paths['log']), 'Render health log hash mismatch')
    require(sha(paths['spec']) == receipt['spec_sha256'] == world['source_spec_sha256'], 'Spec hash mismatch')
    require(world['receipt_sha256'] == sha(paths['receipt']), 'World verification receipt hash mismatch')
    require(world['schema'] == 'city-pcg-world-support-v1' and world['mask_order'] == ['BODY', 'HEAD'] and
        world['query_range_m'] == 3., 'Unexpected native world-label contract')
    require(len(spec['cases']) == len(world['rows']) == receipt['frame_count'] == completion['frames'] and
        len(spec['cases']) > 0, 'Capture frame coverage mismatch')
    require(len({c['name'] for c in spec['cases']}) == len(spec['cases']), 'Duplicate capture case names')
    calibration = world['calibration']
    require((calibration['width'], calibration['height'], calibration['horizontal_fov_degrees']) == (640, 360, 100.),
        'World optical calibration mismatch')
    for name in ('map_sha256_before', 'map_sha256_after'):
        if name in receipt:
            require(receipt[name] == spec['map_sha256'], 'Map identity changed')
    payload = confined(root, 'payload-hashes.json')
    require(sha(payload) == completion['payload_hashes_sha256'], 'Payload manifest hash mismatch')
    inputs = {str(p): sha(p) for p in paths.values()}
    inputs[str(payload)] = sha(payload)
    for relative, digest in read(payload).items():
        path = confined(root, relative)
        require(sha(path) == digest, 'Capture payload changed: '+relative)
        inputs[str(path)] = digest
    for i, (case, row) in enumerate(zip(spec['cases'], world['rows'])):
        require(row['sample_index'] == i and row['name'] == case['name'] and row['camera'] == case['camera'],
            'Native verification frame identity mismatch')
    return spec, world, inputs, receipt.get('native_floor_probes', [])


def ownership_fields(cells, known, grid, projection_valid, weights):
    """Exact-ray ownership and original maxpool/bilinear local memberships.

    local_mask is exactly the R1/probe supervision mask: exclude any UNKNOWN
    native ray in any positively weighted20x20 block, including positive blocks.
    local_known_mass weights wholly known blocks; the supplementary
    local_native_known_fraction_mass weights the actual known-pixel fractions.
    local_membership_mass is maxpool/bilinear block presence, not pixel density.
    """
    import torch
    import torch.nn.functional as F
    require(tuple(cells.shape) == (12, 360, 640) and tuple(known.shape) == (360, 640), 'Ownership dimensions mismatch')
    require(cells.device == known.device == grid.device == projection_valid.device == weights.device,
        'Ownership inputs must share device')
    xx = ((grid[..., 0]+1)*640/2).floor().long().clamp(0, 639)
    yy = ((grid[..., 1]+1)*360/2).floor().long().clamp(0, 359)
    ci = torch.arange(12, device=cells.device)[:, None]
    ray_owned = cells[ci, yy, xx] & projection_valid
    ray_known = known[yy, xx] & projection_valid
    pooled = F.max_pool2d(cells.float()[:, None], 20).flatten(1)
    w = weights.reshape(12, 27, 576)
    local = torch.einsum('cpk,ck->cp', w, pooled)*projection_valid
    known_fraction = F.avg_pool2d(known.float()[None, None], 20).flatten()
    mass = w.sum(-1)*projection_valid
    unknown_blocks = F.max_pool2d((~known).float()[None, None], 20).flatten()
    unknown_footprint = torch.einsum('cpk,k->cp', w, unknown_blocks)
    known_mass = torch.einsum('cpk,k->cp', w, 1-unknown_blocks)*projection_valid
    native_known_mass = torch.einsum('cpk,k->cp', w, known_fraction)*projection_valid
    unknown_mass = (mass-known_mass).clamp_min(0)
    fully_known = projection_valid & (unknown_footprint == 0)
    any_known = projection_valid & (native_known_mass > 0)
    ray_mask = torch.full_like(ray_owned, -1, dtype=torch.int8)
    ray_mask[ray_known] = 0; ray_mask[ray_owned] = 1
    return dict(ray_owned=ray_owned, ray_known=ray_known, ray_mask=ray_mask,
        local_membership_mass=local, local_known_mass=known_mass,
        local_unknown_mass=unknown_mass, local_projection_mass=mass,
        local_native_known_fraction_mass=native_known_mass,
        local_fully_known=fully_known, local_any_known=any_known,
        local_mask=fully_known, local_label=local >= .5)


def summarize(records):
    """Count all declared rows; admission never hides incomplete/mismatched groups."""
    axes = {}
    for axis in ('region_id', 'family', 'condition', 'declared_range', 'source_role'):
        summaries = {}
        for value in sorted({str(r.get(axis, 'MISSING')) for r in records}):
            rows = [r for r in records if str(r.get(axis, 'MISSING')) == value]
            good = [r for r in rows if r.get('status') == 'PASS']
            summaries[value] = dict(declared_frames=len(rows), labeled_frames=len(good), failed_frames=len(rows)-len(good),
                near_positives=[sum(r['near'][h] == 1 for r in good) for h in (0, 1)],
                intent_matches=sum(r['intent_matches'] for r in good),
                intent_mismatches=sum(not r['intent_matches'] for r in good),
                positive_query_cells=sum(sum(v > 0 for v in r['raw_counts']) for r in good))
        axes[axis] = summaries
    grouped = defaultdict(list)
    for r in records:
        grouped[(str(r.get('region_id', 'MISSING')), str(r.get('group_id', 'MISSING')))].append(r)
    groups = []
    for (region, group), rows in grouped.items():
        family = str(rows[0].get('family'))
        required = set(BASIC) | (set(EXTRA) if family == 'crossbar' else set())
        actual = Counter(r.get('condition') for r in rows)
        complete = set(actual) == required and all(n == 1 for n in actual.values())
        consistent = all(len({r.get(k) for r in rows}) == 1 for k in ('family', 'source_role', 'declared_range'))
        good = [r for r in rows if r.get('status') == 'PASS']
        expected = [r for r in rows if r.get('condition') in ('HEAD_ONLY', 'BOTH')]
        floor_ok = all(r.get('floor_probe', {}).get('accepted') is True for r in rows)
        # Actual native >=3 positives per HEAD range; never infer from declared_range.
        near_frames = [r['frame_id'] for r in good if sum(r['raw_counts'][6:9]) >= 3]
        far_frames = [r['frame_id'] for r in good if sum(r['raw_counts'][9:12]) >= 3]
        groups.append(dict(region_id=region, group_id=group, family=family,
            source_role=rows[0].get('source_role'), declared_range=rows[0].get('declared_range'),
            declared_frames=len(rows), labeled_frames=len(good), expected_conditions=sorted(required),
            actual_conditions=dict(actual), complete=complete, metadata_consistent=consistent,
            intent_mismatch_frames=[r['frame_id'] for r in good if not r['intent_matches']],
            failed_frames=[r['frame_id'] for r in rows if r.get('status') != 'PASS'],
            floor_probe_accepted=floor_ok,
            complete_group_accepted=bool(complete and consistent and floor_ok and len(good) == len(rows) and all(r['intent_matches'] for r in good)),
            HEAD_actual_positive_frames=sum(r['near'][1] == 1 for r in good),
            HEAD_expected_positive_frames=len(expected),
            HEAD_expected_positive_observed_frames=sum(r.get('status') == 'PASS' and r['near'][1] == 1 for r in expected),
            HEAD_near_positive_frames=near_frames, HEAD_far_positive_frames=far_frames,
            HEAD_near_positive_group=bool(near_frames), HEAD_far_positive_group=bool(far_frames)))
    coverage = {}
    for region in sorted({g['region_id'] for g in groups}):
        rows = [g for g in groups if g['region_id'] == region]
        coverage[region] = dict(groups=len(rows), accepted_groups=sum(g['complete_group_accepted'] for g in rows),
            HEAD_actual_near_positive_groups=sum(g['HEAD_near_positive_group'] for g in rows),
            HEAD_actual_far_positive_groups=sum(g['HEAD_far_positive_group'] for g in rows),
            by_family={family: dict(groups=sum(g['family'] == family for g in rows),
                near_positive_groups=sum(g['family'] == family and g['HEAD_near_positive_group'] for g in rows),
                far_positive_groups=sum(g['family'] == family and g['HEAD_far_positive_group'] for g in rows))
                for family in sorted({g['family'] for g in rows})})
    return dict(counts=axes, groups=groups, region_HEAD_range_coverage=coverage,
        complete_group_acceptance=bool(groups) and all(g['complete_group_accepted'] for g in groups))


def run(captures, output):
    import numpy as np
    import torch
    from body_query_labels import labels
    from body_query_ownership_audit import membership
    from body_query_model import CALIBRATION, fixed_projection, projection_weights
    from city_data import pool_support
    require(torch.cuda.is_available(), 'CUDA required for native geometry; no CPU fallback')
    started = time.perf_counter()
    artifacts = Path(os.environ.get('BLINDASSIST_ARTIFACTS', Path(__file__).resolve().parents[4]/'artifacts.local')).resolve(strict=True)
    out = output.resolve()
    require(not out.exists() and out != artifacts and out.is_relative_to(artifacts), 'Fresh canonical artifact output required')
    roots = [p.resolve(strict=True) for p in captures]
    require(len(set(roots)) == len(roots), 'Duplicate capture input')
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    grid, projection_valid, xyz = fixed_projection()
    weights = projection_weights(grid).cuda()
    grid, projection_valid = grid.cuda(), projection_valid.cuda()
    projection = out/'evaluator/projection.npz'
    projection.parent.mkdir(parents=True)
    np.savez_compressed(projection, grid=grid.cpu().numpy(), valid=projection_valid.cpu().numpy(),
        coordinates=xyz.numpy(), weights=weights.cpu().numpy())
    frame_dir = out/'evaluator/frames'; frame_dir.mkdir()
    records, capture_rows, hashes, output_hashes = [], [], {}, {projection.relative_to(out).as_posix(): sha(projection)}
    write(out/'manifest.json', dict(schema='body-query-collection-labels-v1', status='BUILDING', captures=[str(p) for p in roots]))
    try:
        with torch.inference_mode():
            for root in roots:
                spec = {'cases': []}
                capture_record = dict(capture=str(root))
                probes = []
                try:
                    spec = read(root/'source/spec.json')
                    spec, world, inputs, probes = verify_capture(root)
                    hashes.update(inputs)
                    capture_record['status'] = 'PASS'
                    capture_error = None
                except Exception as exc:
                    capture_error = repr(exc)
                    capture_record.update(status='FAIL', error=capture_error)
                    world = {'rows': []}
                capture_record['declared_frames'] = len(spec.get('cases', []))
                capture_rows.append(capture_record)
                for index, case in enumerate(spec.get('cases', [])):
                    condition = case.get('condition', {})
                    record = dict(frame_id=len(records), capture=str(root), sample_index=index, name=case.get('name'),
                        group_id=case.get('group_id'), region_id=case.get('region_id', spec.get('region_id')),
                        source_role=case.get('source_role', spec.get('source_role')), family=condition.get('family'),
                        condition=case.get('variant_id'), declared_range=case.get('declared_range', condition.get('declared_range')))
                    try:
                        require(capture_error is None, 'Capture integrity rejected: '+str(capture_error))
                        require(record['source_role'] in ROLES and record['region_id'] and record['group_id'] and
                            record['family'] in ('crossbar', 'cabinet', 'oblique_rod', 'hanging_sign') and
                            record['declared_range'] in ('near', 'far'), 'Incomplete or unsupported grouping metadata')
                        expected = expected_near(record['condition'])
                        record['floor_probe'] = floor_acceptance(case, probes)
                        row = world['rows'][index]
                        native_path = confined(root, f'evaluator/native/{index:04d}.npy')
                        rgb_path = confined(root, f'model/sample/{index:04d}.png')
                        mask_path = confined(root, row['mask_path'])
                        for path, key in ((native_path, 'native_sha256'), (rgb_path, 'rgb_sha256'), (mask_path, 'mask_sha256')):
                            require(sha(path) == row[key], 'Frame payload hash mismatch: '+str(path))
                            hashes[str(path)] = row[key]
                        native = np.load(native_path, allow_pickle=False)
                        require(native.shape == (360, 640) and native.dtype == np.float32, 'Native depth dimensions/dtype mismatch')
                        tensor = torch.from_numpy(native).cuda()
                        label = labels(tensor, case['camera'], case['floor_z_m'])
                        cells, known, _ = membership(tensor)
                        require(torch.equal(cells.sum((-2, -1)), label['raw_counts']), 'Ownership/raw-count partition mismatch')
                        support = label['support'].cpu().numpy()
                        require(np.array_equal(support, np.load(mask_path, allow_pickle=False)), 'Native support including UNKNOWN changed')
                        near = label['near'].cpu().tolist()
                        require(near == row['body_head_visible_targets'], 'Native near label changed')
                        require(label['raw_counts'].reshape(2, 6).sum(1).cpu().tolist() == row['visible_pixels_per_height'],
                            'Native visible-pixel counts changed')
                        ownership = ownership_fields(cells, known, grid, projection_valid, weights)
                        arrays = {k: v.cpu().numpy() for k, v in label.items()}
                        arrays.update({k: v.cpu().numpy() for k, v in ownership.items()})
                        arrays['native_known'] = known.cpu().numpy()
                        arrays['support_pooled'] = pool_support(label['support'][None])[0].cpu().numpy()
                        target = frame_dir/f'{record["frame_id"]:06d}.npz'
                        np.savez_compressed(target, **arrays)
                        relative = target.relative_to(out).as_posix()
                        output_hashes[relative] = sha(target)
                        record.update(status='PASS', labels=dict(path=relative, sha256=output_hashes[relative]),
                            near=near, raw_counts=label['raw_counts'].cpu().tolist(), counts=label['counts'].cpu().tolist(),
                            expected_near=expected, intent_matches=near == expected,
                            unknown_native_pixels=int((~known).sum()), native_sha256=row['native_sha256'],
                            rgb_sha256=row['rgb_sha256'], mask_sha256=row['mask_sha256'],
                            exact_owned_rays=int(ownership['ray_owned'].sum()), exact_known_rays=int(ownership['ray_known'].sum()),
                            local_positive_samples=int(ownership['local_label'].sum()),
                            local_unknown_samples=int((~ownership['local_mask']).sum()))
                    except Exception as exc:
                        record.update(status='FAIL', error=repr(exc))
                    records.append(record)
                    write(out/'progress.json', dict(stage='LABELING', processed_frames=len(records), capture=str(root), sample_index=index))
        torch.cuda.synchronize()
        result = summarize(records)
        result.update(captures=capture_rows, records=records, status='PASS' if all(r['status'] == 'PASS' for r in records) and
            all(c['status'] == 'PASS' for c in capture_rows) and records else 'FAIL',
            frames=len(records), labeled_frames=sum(r['status'] == 'PASS' for r in records),
            mismatches=[r['frame_id'] for r in records if r.get('status') == 'PASS' and not r['intent_matches']],
            seconds=time.perf_counter()-started, training_steps=0, model_inference_frames=0,
            scope='Native evaluator labels; all roles metadata only. Intent mismatch is retained, not relabeled or silently dropped.')
        result['complete_group_acceptance'] &= result['status'] == 'PASS'
        write(out/'evaluator/index.json', dict(records=records, query_order='BODY/HEAD x near/far x left/center/right;27 lattice samples per query',
            near='>=3 visible native pixels in the union of six cells per head; no visible witness is not certified empty space',
            local_membership='20x20 maxpool witness then fixed bilinear weights; not occupancy probability',
            local_known_mass='Weighted wholly known20x20 blocks; supplemental local_native_known_fraction_mass weights known-pixel fractions',
            local_mask='Boolean R1/probe supervision mask: false if any contributing block contains UNKNOWN or projection invalid',
            local_label='local_membership_mass >= .5; use only where local_mask is true'))
        write(out/'result.json', result)
        for path, digest in hashes.items(): require(sha(path) == digest, 'Input changed during labeling: '+path)
        source_files = ('body_query_collection_labels.py', 'body_query_labels.py', 'body_query_ownership_audit.py',
            'body_query_model.py', 'worlds_verify.py', 'city_data.py', 'contact_retina_spec.py')
        manifest = dict(schema='body-query-collection-labels-v1', status=result['status'],
            complete_group_acceptance=result['complete_group_acceptance'], calibration=CALIBRATION,
            captures=capture_rows, inputs=hashes, arrays=output_hashes,
            records_sha256=sha(out/'evaluator/index.json'), result_sha256=sha(out/'result.json'),
            source_sha256={name: sha(Path(__file__).with_name(name)) for name in source_files},
            backend='CUDA', device=torch.cuda.get_device_name(), torch_version=torch.__version__,
            seconds=result['seconds'], frames=len(records), training_steps=0, model_inference_frames=0,
            authority='EVALUATOR_ONLY_NATIVE_LABELS_NO_TRAINING_ADMISSION')
        write(out/'manifest.json', manifest)
        write(out/'receipt.json', dict(status=result['status'], manifest_sha256=sha(out/'manifest.json'),
            result_sha256=manifest['result_sha256'], backend='CUDA', device=manifest['device'],
            frames=len(records), seconds=result['seconds'], complete_group_acceptance=result['complete_group_acceptance']))
        return result
    except BaseException as exc:
        write(out/'failure.json', dict(status='FAIL', error=repr(exc), processed_frames=len(records), records=records, captures=capture_rows))
        write(out/'manifest.json', dict(schema='body-query-collection-labels-v1', status='FAIL',
            error=repr(exc), captures=capture_rows, inputs=hashes, arrays=output_hashes))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.captures, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'frames', 'labeled_frames', 'mismatches', 'complete_group_acceptance', 'seconds')}))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
