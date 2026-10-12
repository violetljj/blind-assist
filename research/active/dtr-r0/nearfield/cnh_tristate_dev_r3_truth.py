"""R3 evaluator-only truth join, after calibration and online decisions seal.

Labels retain the full 0..2.1m corridor, including unexamined near/edge space.
FOV and unexamined-space attribution uses finite contact witnesses; absence of
a witness is unresolved, not proof of no overlap or of a readout failure.
"""
from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import numpy as np

import cnh_tristate_dev as R1
import cnh_tristate_dev_r2 as R2
import cnh_tristate_dev_r3_geometry as G

OUT = R1.WORK / 'cnh-tristate-dev-r3-20261006'
FRAMES = R1.FRAMES
RANGE = R2.RANGE
ARMS = ((0,), (-15, 15))


def deadline():
    if time.time() >= R1.read(OUT / 'PLAN.json')['deadline_unix']:
        raise TimeoutError('R3 60minute wall limit; preserve completed evidence')


def shrunk_fov(points, pose, margin):
    local = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    edge = np.tan(np.deg2rad(22.5 - margin))
    z = local[:, 2]
    return ((z > 0) & (np.abs(local[:, 0]) <= edge * z + R1.EPS)
            & (np.abs(local[:, 1]) <= edge * z + R1.EPS)
            & (np.linalg.norm(local, axis=1) <= RANGE + R1.EPS))


def nominal_mask(points, poses, frame, margin):
    return G.core_mask(points, poses, frame, margin)


def fresh_true(points, sensor, frame, angles):
    fresh = np.zeros(len(points), bool)
    for h in range(frame - 3, frame + 1):
        for angle in angles:
            fresh |= R1.fov(points, sensor[h] @ R1.extrinsic(angle), RANGE)
    return fresh


def unchecked_witnesses(points, poses, frame, margin):
    """Finite-witness overlap proxy; never removes contact from main error."""
    direction = R2.direction(poses, frame)
    if direction is None:
        return dict(unchecked=True, near=False, edge=False, side=False, unresolved=True)
    right = np.array([direction[1], -direction[0]])
    relative = points - poses[frame, :3, 3]
    forward = relative[:, [0, 2]] @ direction
    lateral = relative[:, [0, 2]] @ right
    inside = ((forward >= .9 - R1.EPS) & (forward <= 2.1 + R1.EPS)
              & (np.abs(lateral) <= .29 + R1.EPS)
              & (relative[:, 1] >= R1.HEIGHTS[0][0] - R1.EPS)
              & (relative[:, 1] <= R1.HEIGHTS[-1][1] + R1.EPS))
    nominal = nominal_mask(points, poses, frame, margin)
    unexamined = ~inside | ~nominal
    return dict(unchecked=bool(unexamined.any()),
                near=bool(((forward >= -R1.EPS) & (forward < .9 - R1.EPS)).any()),
                edge=bool((inside & ~nominal).any()),
                side=bool((np.abs(lateral) > .29 + R1.EPS).any()), unresolved=False)


def evaluate(helper_module=None):
    tick = time.monotonic()
    if (OUT / 'truth.npz').exists() or (OUT / 'truth_receipt.json').exists():
        raise FileExistsError('R3 evaluator output already exists; preserve sealed evidence')
    calibration_path = OUT / 'calibration.json'
    calibration = R1.read(calibration_path)
    margin = float(calibration['selected_m'])
    receipt = R1.read(OUT / 'online_receipt.json')
    # This join cannot run before saved online decisions and frozen calibration.
    if receipt['online_sha256'] != R1.sha(OUT / 'online.npz'):
        raise ValueError('Online decision receipt mismatch')
    if receipt['rows_sha256'] != R1.sha(OUT / 'rows.json'):
        raise ValueError('Online row receipt mismatch')
    if 'calibration_sha256' in receipt and receipt['calibration_sha256'] != R1.sha(calibration_path):
        raise ValueError('Calibration changed after online decision seal')
    helper = importlib.import_module(helper_module) if helper_module else None
    rows = R1.read(OUT / 'rows.json')
    with np.load(OUT / 'online.npz') as z:
        poses = z['poses']
        scores = z['score']
        gate = z['gate']
        thresholds = z['thresholds']
    if poses.shape != (len(rows), 16, 4, 4):
        raise ValueError('Expected one saved causal estimated trajectory per row')
    if gate.shape != (len(rows), 13, 3, 2):
        raise ValueError('Gate must be [rows,13,margin_minus/main/plus,arms]')
    candidate = (scores < R1.THRESHOLD) & (scores <= thresholds.max()) & gate[:, :, 1, :]
    shape = (len(rows), 13)
    arrays = {name: np.zeros(shape, bool) for name in
              ('contact', 'graze', 'boundary', 'unchecked_contact', 'near_contact',
               'edge_contact', 'side_contact', 'unchecked_unresolved')}
    arrays.update({name: np.zeros((*shape, 2), bool) for name in
                   ('fov_in', 'fov_angular_in', 'fov_unresolved', 'core_truth_gap', 'core_gap_evaluated')})
    arrays['core_missing_points'] = np.zeros((*shape, 2), np.int32)
    hashes = {str(calibration_path.relative_to(R1.ROOT)): R1.sha(calibration_path),
              str((OUT / 'online.npz').relative_to(R1.ROOT)): R1.sha(OUT / 'online.npz'),
              str((OUT / 'rows.json').relative_to(R1.ROOT)): R1.sha(OUT / 'rows.json'),
              str(Path(__file__).relative_to(R1.ROOT)): R1.sha(__file__),
              str(Path(R1.__file__).relative_to(R1.ROOT)): R1.sha(R1.__file__),
              str(Path(R2.__file__).relative_to(R1.ROOT)): R1.sha(R2.__file__),
              str(Path(G.__file__).relative_to(R1.ROOT)): R1.sha(G.__file__)}
    manifest = {}
    for path in (R1.MARGIN / 'scene_manifest.json', R1.FUSION / 'natural97000/scene_manifest.json'):
        manifest.update({(int(s['unit']), int(s['config'])): s for s in R1.read(path)})
        hashes[str(path.relative_to(R1.ROOT))] = R1.sha(path)
    current_key = None
    geometry_seconds = 0.
    for index, row in enumerate(rows):
        deadline()
        unit, config, batch = row['unit'], row['config'], row['batch']
        if batch >= 98000:
            key = (batch, unit)
            if key != current_key:
                path = (R1.AUG if batch == 98000 else R1.CONT) / f'truth/{"calibration" if batch == 98000 else "evaluation"}/unit{unit}.json'
                truth = R1.read(path)
                hashes[str(path.relative_to(R1.ROOT))] = R1.sha(path)
                travel, sensor = np.asarray(truth['travel']), np.asarray(truth['sensor_center'])
                scenes = {s['config']: s for s in truth['scenes']}
                current_key = key
            boxes = scenes[config]['boxes']
        else:
            sensor, travel, _ = R1.original_motion(unit, config)
            boxes = manifest[unit, config]['boxes']
        estimated = poses[index]
        for time_index, frame in enumerate(FRAMES):
            frame = int(frame)
            core = R1.contact_boxes(boxes, travel[frame], zmax=2.1)
            arrays['contact'][index, time_index] = bool(core)
            arrays['graze'][index, time_index] = bool(not core and R1.contact_boxes(boxes, travel[frame], width=.4, zmax=2.1))
            arrays['boundary'][index, time_index] = bool(core and any(r['boundary'] for r in core))
            if core:
                witnesses = np.concatenate([np.asarray(r['witness_world']) for r in core])
                attribution = unchecked_witnesses(witnesses, estimated, frame, margin)
                for name, source in [('unchecked_contact', 'unchecked'), ('near_contact', 'near'),
                                     ('edge_contact', 'edge'), ('side_contact', 'side'),
                                     ('unchecked_unresolved', 'unresolved')]:
                    arrays[name][index, time_index] = attribution[source]
                for arm_index, angles in enumerate(ARMS):
                    angular_hit = any(R1.fov(witnesses, sensor[frame] @ R1.extrinsic(a), None).any() for a in angles)
                    range_hit = any(R1.fov(witnesses, sensor[frame] @ R1.extrinsic(a), RANGE).any() for a in angles)
                    arrays['fov_in'][index, time_index, arm_index] = range_hit
                    arrays['fov_angular_in'][index, time_index, arm_index] = angular_hit
                    arrays['fov_unresolved'][index, time_index, arm_index] = not range_hit
            if candidate[index, time_index].any():
                geometry_tick = time.monotonic()
                direction = R2.direction(estimated, frame)
                if direction is None:
                    raise ValueError('Online clear had unavailable direction')
                points = R2.points(estimated, frame, direction)
                mask = nominal_mask(points, estimated, frame, margin)
                if helper is not None:
                    # The optional online helper is also checked, never silently substituted.
                    if not np.array_equal(mask, helper.core_mask(points, estimated, frame, margin)):
                        raise ValueError('Evaluator mask differs from online core mask')
                if not mask.any():
                    raise ValueError('Online clear had empty core mask')
                for arm_index, angles in enumerate(ARMS):
                    if not candidate[index, time_index, arm_index]:
                        continue
                    fresh = fresh_true(points[mask], sensor, frame, angles)
                    missing = int((~fresh).sum())
                    arrays['core_gap_evaluated'][index, time_index, arm_index] = True
                    arrays['core_truth_gap'][index, time_index, arm_index] = missing > 0
                    arrays['core_missing_points'][index, time_index, arm_index] = missing
                geometry_seconds += time.monotonic() - geometry_tick
        if (index + 1) % 480 == 0:
            print('r3 truth', batch, unit, index + 1, 'sequences', round(time.monotonic() - tick, 1), 's', flush=True)
    source_path = OUT / 'source' / Path(__file__).name
    source_path.parent.mkdir(exist_ok=True)
    with source_path.open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    np.savez_compressed(OUT / 'truth.npz', **arrays)
    R1.save(OUT / 'truth_receipt.json', dict(
        source_sha256=hashes, truth_sha256=R1.sha(OUT / 'truth.npz'),
        calibration_sha256=R1.sha(calibration_path), online_sha256=R1.sha(OUT / 'online.npz'),
        selected_model=calibration['selected_model'], selected_m=margin,
        rows=len(rows), frames=len(rows) * 13, seconds=time.monotonic() - tick,
        core_geometry_seconds=geometry_seconds,
        core_evaluated_frames=arrays['core_gap_evaluated'].sum((0, 1)).tolist(),
        core_missing_frames=arrays['core_truth_gap'].sum((0, 1)).tolist(),
        scope='Evaluator-only labels after saved online outputs; full current 0..2.1m, +/-.30m HEAD/BODY contact counts including unchecked region',
        core_gap='Estimated core points evaluated under actual true sensor-history angular/range union. Geometry proxy only, no occlusion/first-hit proof. Evaluated only maximum-threshold A clear frames per arm; all lower thresholds and B are subsets.',
        fov_in='Contact-intersection witness has current true angular/range FOV overlap; no unoccluded-return or causal readout-fault proof.',
        unchecked='Any finite contact witness outside estimated check volume or contracted nominal-history FOV. Overlap proxy, not exclusive partition; absent witness does not prove no unchecked intrusion.',
        boundary='R1 exact contact intersection boundary flag, within1cm; proxy overlaps other sources.',
        person_front_offset='NOT_AVAILABLE; saved current travel origin proxy',
        preceding_online_receipt_sha256=R1.sha(OUT / 'online_receipt.json')))
    print('r3 truth complete', round(time.monotonic() - tick, 1), 's', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--helper-module')
    args = parser.parse_args()
    evaluate(args.helper_module)
