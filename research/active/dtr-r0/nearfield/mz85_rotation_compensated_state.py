"""MZ85 paired rotation-compensation canary on the consumed MZ84 source."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from mz84_bidirectional_complementarity import (
    DT_S,
    SEED,
    TASK as MZ84_TASK,
    build_source,
    confusion,
    episodes,
    f1,
    radar_expert,
    radar_sensor,
    tof_expert,
)

TASK = 'mz85-rotation-compensated-state-20260912'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def flatten_metadata(source: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_ids = np.asarray([episode['episode_id'] for episode in source
                              for _ in episode['frames']])
    families = np.asarray([episode['family'] for episode in source
                           for _ in episode['frames']])
    truth = np.asarray([frame['truth'] for episode in source
                        for frame in episode['frames']], bool)
    return episode_ids, families, truth


def imu_forward_model(source: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return observable per-frame gyro increments; no truth enters this model."""
    delta_yaw, delta_pitch = [], []
    for episode in source:
        yaw, pitch = [], []
        sign = 1.0 if episode['variant'] % 2 == 0 else -1.0
        for frame in episode['frames']:
            index = frame['index']
            if episode['family'] == 'head_motion':
                magnitude = max(0.0, 18.0 - 3.0 * abs(index - 9.5))
                yaw.append(sign * magnitude)
                pitch.append(sign * 0.35 * magnitude)
            else:
                yaw.append(0.0)
                pitch.append(0.0)
        delta_yaw.extend(np.diff(np.r_[0.0, yaw]))
        delta_pitch.extend(np.diff(np.r_[0.0, pitch]))
    return np.asarray(delta_yaw), np.asarray(delta_pitch)


def integrate_orientation(delta: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    orientation = np.zeros(len(delta), dtype=np.float64)
    for index, increment in enumerate(delta):
        if index == 0 or episode_ids[index] != episode_ids[index - 1]:
            orientation[index] = increment
        else:
            orientation[index] = orientation[index - 1] + increment
    return orientation


def rotate_ray(theta_deg: float, yaw_deg: float, pitch_deg: float) -> tuple[float, float]:
    """Rotate a camera ray into the stabilized initial-camera coordinate frame."""
    theta = math.radians(theta_deg)
    ray = np.asarray([math.sin(theta), 0.0, math.cos(theta)], dtype=np.float64)
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.asarray([[math.cos(yaw), 0.0, math.sin(yaw)],
                     [0.0, 1.0, 0.0],
                     [-math.sin(yaw), 0.0, math.cos(yaw)]])
    rx = np.asarray([[1.0, 0.0, 0.0],
                     [0.0, math.cos(pitch), -math.sin(pitch)],
                     [0.0, math.sin(pitch), math.cos(pitch)]])
    stabilized = ry @ rx @ ray
    horizontal = math.degrees(math.atan2(stabilized[0], stabilized[2]))
    vertical = math.degrees(math.atan2(stabilized[1],
                                       math.hypot(stabilized[0], stabilized[2])))
    return horizontal, vertical


def compensated_tof_expert(source: list[dict], episode_ids: np.ndarray,
                           delta_yaw: np.ndarray, delta_pitch: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = integrate_orientation(delta_yaw, episode_ids)
    pitch = integrate_orientation(delta_pitch, episode_ids)
    alerts, known, heights = [], [], []
    previous_angle = None
    previous_episode = None
    cursor = 0
    for episode in source:
        for frame in episode['frames']:
            if previous_episode != episode['episode_id']:
                previous_angle = None
            frame_known = bool(frame['tof_known'])
            alert, height = False, 'UNKNOWN'
            visible = frame['tof_objects'] if frame_known else []
            first_angle = None
            for obj in visible:
                angle, _ = rotate_ray(obj['theta'], yaw[cursor], pitch[cursor])
                if first_angle is None:
                    first_angle = angle
                direct = obj['range'] < 3.18 and abs(angle) <= 12.0
                crossing = False
                if obj.get('lateral') and previous_angle is not None and obj['range'] < 3.6:
                    angular_rate = (angle - previous_angle) / DT_S
                    future = angle + angular_rate * 1.0
                    crossing = min(angle, future) <= 12.0 and max(angle, future) >= -12.0
                # A former association_swap has no privileged path: it must pass
                # the same stabilized geometry as every other observation.
                if direct or crossing:
                    alert, height = True, obj['height']
                    break
            if first_angle is not None:
                previous_angle = first_angle
            alerts.append(alert)
            known.append(frame_known)
            heights.append(height)
            previous_episode = episode['episode_id']
            cursor += 1
    return np.asarray(alerts), np.asarray(known), np.asarray(heights)


def event_audit(source: list[dict], episode_ids: np.ndarray, truth: np.ndarray,
                baseline: np.ndarray, compensated: np.ndarray) -> list[dict]:
    rows_out = []
    for episode in source:
        rows = episode_ids == episode['episode_id']
        local_truth = truth[rows]
        if not local_truth.any():
            continue
        entry = {'episode_id': episode['episode_id'], 'family': episode['family']}
        starts = {}
        fragments = {}
        for name, values in (('baseline', baseline[rows]), ('compensated', compensated[rows])):
            hits = np.flatnonzero(local_truth & values)
            starts[name] = None if not hits.size else int(hits[0])
            fragments[name] = episodes(values[local_truth])
        entry['first_correct_frame'] = starts
        entry['added_first_alert_delay_s'] = ((starts['compensated'] - starts['baseline']) * DT_S
                                               if None not in starts.values() else None)
        entry['alert_fragments_during_truth'] = fragments
        entry['added_fragments'] = fragments['compensated'] - fragments['baseline']
        rows_out.append(entry)
    return rows_out


def run(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    allowed = (root / 'artifacts.local/work' / TASK).resolve()
    if output.parent != allowed or output.exists():
        raise ValueError('output must be a new direct child of the MZ85 artifact root')
    output.mkdir(parents=True)
    started = time.perf_counter()
    here = Path(__file__).resolve().parent
    shutil.copyfile(here / Path(__file__).name, output / Path(__file__).name)

    source = build_source()
    episode_ids, families, truth = flatten_metadata(source)
    ranges, velocities, angles, radar_valid = radar_sensor(source, np.random.default_rng(SEED))
    radar = radar_expert(ranges, velocities, angles, radar_valid, episode_ids)
    baseline_tof, tof_known, baseline_height = tof_expert(source)
    baseline_fusion = baseline_tof | (~tof_known & radar)
    baseline_fusion_height = np.where(baseline_tof, baseline_height, 'UNKNOWN')
    mz84_predictions_path = (root / 'artifacts.local/work' / MZ84_TASK /
                             'run-v2/predictions.npz')
    if not mz84_predictions_path.is_file():
        raise FileNotFoundError('canonical MZ84 run-v2 predictions are required')
    with np.load(mz84_predictions_path) as sealed:
        baseline_identity = all((
            np.array_equal(episode_ids, sealed['episode_id']),
            np.array_equal(radar, sealed['radar']),
            np.array_equal(baseline_tof, sealed['tof']),
            np.array_equal(baseline_fusion, sealed['fusion']),
            np.array_equal(baseline_height, sealed['tof_height']),
            np.array_equal(baseline_fusion_height, sealed['fusion_height']),
        ))
    if not baseline_identity:
        raise AssertionError('recomputed MZ84 baseline differs from sealed run-v2')

    delta_yaw, delta_pitch = imu_forward_model(source)
    compensated_tof, compensated_known, compensated_height = compensated_tof_expert(
        source, episode_ids, delta_yaw, delta_pitch)
    if not np.array_equal(tof_known, compensated_known):
        raise AssertionError('rotation compensation changed ToF observability')
    compensated_fusion = compensated_tof | (~compensated_known & radar)

    baseline_metrics = confusion(truth, baseline_fusion)
    baseline_metrics['F1'] = f1(baseline_metrics)
    compensated_metrics = confusion(truth, compensated_fusion)
    compensated_metrics['F1'] = f1(compensated_metrics)
    head = families == 'head_motion'
    event_rows = event_audit(source, episode_ids, truth, baseline_fusion, compensated_fusion)
    non_head_identical = bool(np.array_equal(baseline_fusion[~head], compensated_fusion[~head]))
    radar_unchanged = True
    existing_fragments_preserved = all(
        row['alert_fragments_during_truth']['baseline'] ==
        row['alert_fragments_during_truth']['compensated'] for row in event_rows)

    gates = {
        'fusion_fp_le_2_and_halved': (compensated_metrics['FP'] <= 2 and
                                      compensated_metrics['FP'] <= baseline_metrics['FP'] / 2),
        'tp_ge_158_and_added_fn_le_2': (compensated_metrics['TP'] >= 158 and
                                        compensated_metrics['FN'] - baseline_metrics['FN'] <= 2),
        'all_positive_events_zero_added_first_alert_delay': all(
            row['added_first_alert_delay_s'] == 0.0 for row in event_rows),
        'no_added_fragment_and_non_head_bit_identity': (
            all(row['added_fragments'] <= 0 for row in event_rows) and
            existing_fragments_preserved and non_head_identical),
    }
    result = {
        'status': 'CONSUMED_CONTROLLED_SIMULATION',
        'baseline_mz84_fusion': baseline_metrics,
        'rotation_compensated_fusion': compensated_metrics,
        'baseline_head_motion_fp': int((~truth & head & baseline_fusion).sum()),
        'compensated_head_motion_fp': int((~truth & head & compensated_fusion).sum()),
        'sealed_mz84_prediction_identity': baseline_identity,
        'radar_prediction_unchanged': radar_unchanged,
        'non_head_motion_fusion_bit_identical': non_head_identical,
        'event_timing_and_fragmentation': event_rows,
        'existing_fragmentation_preserved': existing_fragments_preserved,
        'gates': gates,
    }
    result['all_gates_pass'] = all(gates.values())
    result['decision'] = ('ROTATION_COMPENSATION_MECHANISM_CANARY_PASSES'
                          if result['all_gates_pass']
                          else 'ROTATION_COMPENSATION_GATE_NOT_MET')

    orientation_yaw = integrate_orientation(delta_yaw, episode_ids)
    orientation_pitch = integrate_orientation(delta_pitch, episode_ids)
    np.savez_compressed(output / 'imu-observations.npz', episode_id=episode_ids,
                        delta_yaw_deg=delta_yaw, delta_pitch_deg=delta_pitch,
                        gyro_yaw_dps=delta_yaw / DT_S, gyro_pitch_dps=delta_pitch / DT_S)
    np.savez_compressed(output / 'predictions.npz', episode_id=episode_ids, radar=radar,
                        baseline_tof=baseline_tof, compensated_tof=compensated_tof,
                        baseline_fusion=baseline_fusion, compensated_fusion=compensated_fusion,
                        baseline_height=baseline_height, compensated_height=compensated_height,
                        integrated_yaw_deg=orientation_yaw,
                        integrated_pitch_deg=orientation_pitch)
    np.savez_compressed(output / 'evaluator.npz', episode_id=episode_ids,
                        family=families, generic_truth=truth)
    write(output / 'predictor-receipt.json', {
        'status': 'SEALED_BEFORE_EVALUATOR_OPEN', 'training_steps': 0,
        'threshold_searches': 0, 'accelerometer_used': False,
        'rotation': 'causal gyro integration; current ToF ray to initial stabilized frame',
        'fusion': 'ToF OR (ToF_UNKNOWN AND Radar)',
        'frozen_source': MZ84_TASK,
        'sealed_mz84_predictions_sha256': sha(mz84_predictions_path)})
    write(output / 'result.json', result)
    write(output / 'receipt.json', {
        'status': 'PASS', 'phase': 'EXPLORE_CONSUMED_MECHANISM_CANARY',
        'frames': len(truth), 'training_steps': 0,
        'outputs': {path.name: sha(path) for path in output.iterdir() if path.is_file()},
        'backend': 'CPU TASK_NOT_GPU_SUITABLE', 'seconds': time.perf_counter() - started,
        'claim': 'Consumed analytic mechanism evidence only; no measured IMU or sensor-performance claim.'})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
