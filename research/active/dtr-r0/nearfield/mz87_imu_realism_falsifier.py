"""MZ87 frozen single-factor IMU transform stress grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from mz84_bidirectional_complementarity import DT_S, confusion, episodes, f1
from mz85_rotation_compensated_state import (
    build_source,
    compensated_tof_expert,
    event_audit,
    flatten_metadata,
    imu_forward_model,
    integrate_orientation,
    rotate_ray,
)

TASK = 'mz87-imu-realism-falsifier-20260912'
NOISE_SEEDS = (8701, 8702, 8703, 8704)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def configs() -> list[dict]:
    rows = []
    for value in (-0.5, 0.5, -2.0, 2.0, -5.0, 5.0):
        rows.append({'name': f'bias_{value:+g}dps', 'factor': 'bias',
                     'level': abs(value), 'bias_dps': value})
    for value in (-5.0, 5.0, -20.0, 20.0, -50.0, 50.0):
        rows.append({'name': f'sync_{value:+g}ms', 'factor': 'sync',
                     'level': abs(value), 'sync_ms': value})
    for value in (-0.5, 0.5, -2.0, 2.0, -5.0, 5.0):
        rows.append({'name': f'extrinsic_{value:+g}deg', 'factor': 'extrinsic',
                     'level': abs(value), 'extrinsic_deg': value})
    for value in (0.5, 2.0, 10.0):
        for seed in NOISE_SEEDS:
            rows.append({'name': f'noise_{value:g}dps_seed{seed}', 'factor': 'noise',
                         'level': value, 'noise_dps': value, 'seed': seed})
    for value in (1, 2, 3):
        rows.append({'name': f'dropout_{value}f', 'factor': 'dropout',
                     'level': value, 'dropout_frames': value})
    rows.extend((
        {'name': 'combo_medium_positive', 'factor': 'combination', 'level': 2,
         'bias_dps': 2.0, 'sync_ms': 20.0, 'extrinsic_deg': 2.0,
         'noise_dps': 2.0, 'dropout_frames': 1, 'seed': 8701},
        {'name': 'combo_medium_negative', 'factor': 'combination', 'level': 2,
         'bias_dps': -2.0, 'sync_ms': -20.0, 'extrinsic_deg': -2.0,
         'noise_dps': 2.0, 'dropout_frames': 1, 'seed': 8702},
        {'name': 'combo_high_positive', 'factor': 'combination', 'level': 5,
         'bias_dps': 5.0, 'sync_ms': 50.0, 'extrinsic_deg': 5.0,
         'noise_dps': 10.0, 'dropout_frames': 2, 'seed': 8703},
    ))
    return rows


def sample_with_offset(orientation: np.ndarray, episode_ids: np.ndarray,
                       offset_ms: float) -> np.ndarray:
    shifted = np.empty_like(orientation)
    for episode_id in np.unique(episode_ids):
        rows = np.flatnonzero(episode_ids == episode_id)
        times = np.arange(len(rows), dtype=np.float64) * DT_S
        query = np.clip(times + offset_ms / 1000.0, times[0], times[-1])
        shifted[rows] = np.interp(query, times, orientation[rows])
    return shifted


def orientation_to_delta(orientation: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    delta = np.empty_like(orientation)
    for index, value in enumerate(orientation):
        if index == 0 or episode_ids[index] != episode_ids[index - 1]:
            delta[index] = value
        else:
            delta[index] = value - orientation[index - 1]
    return delta


def perturb_deltas(ideal_yaw_delta: np.ndarray, ideal_pitch_delta: np.ndarray,
                   episode_ids: np.ndarray, config: dict
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = int(config.get('seed', 8799))
    rng = np.random.default_rng(seed)
    bias = float(config.get('bias_dps', 0.0))
    noise = float(config.get('noise_dps', 0.0))
    yaw_delta = ideal_yaw_delta + bias * DT_S + rng.normal(0.0, noise * DT_S,
                                                           len(ideal_yaw_delta))
    pitch_delta = ideal_pitch_delta + 0.35 * bias * DT_S + rng.normal(
        0.0, 0.35 * noise * DT_S, len(ideal_pitch_delta))
    dropout = int(config.get('dropout_frames', 0))
    if dropout:
        for episode_id in np.unique(episode_ids):
            rows = np.flatnonzero(episode_ids == episode_id)
            local = np.arange(10 - dropout, 10)
            yaw_delta[rows[local]] = 0.0
            pitch_delta[rows[local]] = 0.0
    yaw = integrate_orientation(yaw_delta, episode_ids)
    pitch = integrate_orientation(pitch_delta, episode_ids)
    sync_ms = float(config.get('sync_ms', 0.0))
    yaw = sample_with_offset(yaw, episode_ids, sync_ms)
    pitch = sample_with_offset(pitch, episode_ids, sync_ms)
    extrinsic = float(config.get('extrinsic_deg', 0.0))
    yaw = yaw + extrinsic
    pitch = pitch + 0.35 * extrinsic
    return orientation_to_delta(yaw, episode_ids), orientation_to_delta(pitch, episode_ids), yaw, pitch


def ray_mismatch(source: list[dict], ideal_yaw: np.ndarray, ideal_pitch: np.ndarray,
                 stressed_yaw: np.ndarray, stressed_pitch: np.ndarray) -> dict:
    errors, head_errors = [], []
    cursor = 0
    for episode in source:
        for frame in episode['frames']:
            if frame['tof_known']:
                for obj in frame['tof_objects']:
                    ideal, _ = rotate_ray(obj['theta'], ideal_yaw[cursor], ideal_pitch[cursor])
                    stressed, _ = rotate_ray(obj['theta'], stressed_yaw[cursor], stressed_pitch[cursor])
                    error = abs(stressed - ideal)
                    errors.append(error)
                    if episode['family'] == 'head_motion':
                        head_errors.append(error)
            cursor += 1
    return {'mean_deg': float(np.mean(errors)), 'max_deg': float(np.max(errors)),
            'head_motion_max_deg': float(max(head_errors, default=0.0))}


def scenario_stable(row: dict) -> bool:
    return (row['head_motion_fp'] <= 2 and row['metrics']['TP'] >= 158 and
            row['new_non_head_motion_fp'] == 0 and
            row['max_added_first_alert_delay_s'] <= 0.2 + 1e-9 and
            row['added_fragments'] == 0)


def family_counts(mask: np.ndarray, families: np.ndarray) -> dict:
    return {str(family): int((mask & (families == family)).sum())
            for family in np.unique(families) if (mask & (families == family)).any()}


def run(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    allowed = (root / 'artifacts.local/work' / TASK).resolve()
    if output.parent != allowed or output.exists():
        raise ValueError('output must be a new direct child of the MZ87 artifact root')
    output.mkdir(parents=True)
    started = time.perf_counter()
    shutil.copyfile(Path(__file__).resolve(), output / Path(__file__).name)

    m84 = root / 'artifacts.local/work/mz84-bidirectional-complementarity-20260912/run-v2'
    m85 = root / 'artifacts.local/work/mz85-rotation-compensated-state-20260912/run-v2'
    source = build_source()
    episode_ids, families, truth = flatten_metadata(source)
    with np.load(m84 / 'observations.npz') as observations, \
            np.load(m85 / 'predictions.npz') as sealed:
        radar = sealed['radar']
        ideal_tof = sealed['compensated_tof']
        ideal_fusion = sealed['compensated_fusion']
        tof_known = observations['tof_known']
    ideal_yaw_delta, ideal_pitch_delta = imu_forward_model(source)
    reproduced_tof, reproduced_known, _ = compensated_tof_expert(
        source, episode_ids, ideal_yaw_delta, ideal_pitch_delta)
    if not np.array_equal(reproduced_tof, ideal_tof) or not np.array_equal(reproduced_known, tof_known):
        raise AssertionError('ideal MZ85 baseline identity failed')
    ideal_yaw = integrate_orientation(ideal_yaw_delta, episode_ids)
    ideal_pitch = integrate_orientation(ideal_pitch_delta, episode_ids)
    head = families == 'head_motion'

    scenario_rows = []
    for config in configs():
        yaw_delta, pitch_delta, yaw, pitch = perturb_deltas(
            ideal_yaw_delta, ideal_pitch_delta, episode_ids, config)
        stressed_tof, stressed_known, _ = compensated_tof_expert(
            source, episode_ids, yaw_delta, pitch_delta)
        if not np.array_equal(stressed_known, tof_known):
            raise AssertionError('stress changed ToF observability')
        fusion = stressed_tof | (~tof_known & radar)
        metrics = confusion(truth, fusion)
        metrics['F1'] = f1(metrics)
        events = event_audit(source, episode_ids, truth, ideal_fusion, fusion)
        delays = [row['added_first_alert_delay_s'] for row in events
                  if row['added_first_alert_delay_s'] is not None]
        new_non_head = ~truth & ~head & ~ideal_fusion & fusion
        lost_true = truth & ideal_fusion & ~fusion
        row = dict(config)
        row.update({
            'metrics': metrics,
            'head_motion_fp': int((~truth & head & fusion).sum()),
            'new_non_head_motion_fp': int(new_non_head.sum()),
            'new_non_head_motion_fp_by_family': family_counts(new_non_head, families),
            'lost_true_positive_frames': int(lost_true.sum()),
            'lost_true_positive_frames_by_family': family_counts(lost_true, families),
            'max_added_first_alert_delay_s': float(max(delays, default=math.inf)),
            'added_fragments': int(sum(max(0, event['added_fragments']) for event in events)),
            'rotation_association_mismatch': ray_mismatch(
                source, ideal_yaw, ideal_pitch, yaw, pitch),
        })
        row['stable'] = scenario_stable(row)
        scenario_rows.append(row)

    medium_singles = [row for row in scenario_rows if row['factor'] != 'combination' and (
        (row['factor'] == 'bias' and row['level'] <= 2) or
        (row['factor'] == 'sync' and row['level'] <= 20) or
        (row['factor'] == 'extrinsic' and row['level'] <= 2) or
        (row['factor'] == 'noise' and row['level'] <= 2) or
        (row['factor'] == 'dropout' and row['level'] <= 1))]
    medium_combos = [row for row in scenario_rows if row['name'].startswith('combo_medium')]
    smallest = {'bias': 0.5, 'sync': 5.0, 'extrinsic': 0.5, 'noise': 0.5, 'dropout': 1}
    smallest_rows = [row for row in scenario_rows if row['factor'] in smallest and
                     row['level'] == smallest[row['factor']]]
    gates = {
        'ideal_mz85_identity': True,
        'all_small_medium_single_factors_stable': all(row['stable'] for row in medium_singles),
        'both_medium_combinations_stable': all(row['stable'] for row in medium_combos),
    }
    all_bounded = all(gates.values())
    smallest_fragile = not all(row['stable'] for row in smallest_rows)
    decision = ('ROTATION_COMPENSATION_HAS_BOUNDED_SYNTHETIC_TOLERANCE' if all_bounded else
                'ROTATION_COMPENSATION_FRAGILE_AT_SMALLEST_PRESSURE' if smallest_fragile else
                'ROTATION_COMPENSATION_TOLERANCE_NOT_ESTABLISHED')
    result = {
        'status': 'CONSUMED_ANALYTIC_STRESS_GRID',
        'ideal_mz85_metrics': {**confusion(truth, ideal_fusion),
                               'F1': f1(confusion(truth, ideal_fusion))},
        'scenario_count': len(scenario_rows), 'scenarios': scenario_rows,
        'gates': gates, 'decision': decision,
        'pressure_values_are_not_device_distributions': True,
    }
    write(output / 'grid.json', configs())
    write(output / 'result.json', result)
    write(output / 'receipt.json', {
        'status': 'PASS', 'training_steps': 0, 'threshold_searches': 0,
        'scenario_count': len(scenario_rows),
        'mz84_observations_sha256': sha(m84 / 'observations.npz'),
        'mz85_predictions_sha256': sha(m85 / 'predictions.npz'),
        'outputs': {path.name: sha(path) for path in output.iterdir() if path.is_file()},
        'backend': 'CPU TASK_NOT_GPU_SUITABLE', 'seconds': time.perf_counter() - started,
        'claim': 'Analytic transform-pressure evidence only; no device error distribution or hardware claim.'})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
