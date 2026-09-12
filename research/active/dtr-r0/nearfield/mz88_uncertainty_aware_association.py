"""MZ88 zero-fit uncertainty-aware geometric association canary."""
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
    confusion,
    episodes,
    f1,
    radar_expert,
)
from mz85_rotation_compensated_state import rotate_ray

TASK = 'mz88-uncertainty-aware-association-20260912'
SEED = 8801
FRAMES = 30
VARIANTS = 6
FAMILIES = (
    'boundary_inside',
    'boundary_outside',
    'lateral_crossing',
    'head_motion',
    'multi_target_gap',
    'weak_radar',
)
MAX_TOF_OBJECTS = 2
CORRIDOR_DEG = 12.0
K_SIGMA = 1.0
BIAS_SIGMA_DPS = 2.0
GYRO_NOISE_SIGMA_DPS = 2.0
EXTRINSIC_SIGMA_DEG = 2.0
SYNC_SIGMA_S = 0.020


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def orientation_waveform(variant: int) -> tuple[np.ndarray, np.ndarray]:
    """Six fresh, deterministic yaw/pitch paths, each anchored at frame zero."""
    amplitude = (16.0, 18.0, 20.0, 22.0, 17.0, 24.0)[variant]
    frequency = (0.17, 0.21, 0.14, 0.24, 0.19, 0.12)[variant]
    phase = (0.15, 0.75, 1.35, 2.0, 2.55, 0.45)[variant]
    index = np.arange(FRAMES, dtype=np.float64)
    yaw = amplitude * np.sin(frequency * index + phase)
    yaw += 0.18 * amplitude * np.sin(0.47 * frequency * index + 0.6 * phase)
    pitch = 0.28 * amplitude * np.sin(0.73 * frequency * index + phase + 0.9)
    return yaw - yaw[0], pitch - pitch[0]


def stress_profile(variant: int) -> dict:
    profiles = (
        {'name': 'nominal', 'bias_dps': 0.0, 'sync_ms': 0.0,
         'extrinsic_deg': 0.0, 'noise_dps': 0.0, 'dropout': ()},
        {'name': 'medium_positive', 'bias_dps': 2.0, 'sync_ms': 20.0,
         'extrinsic_deg': 2.0, 'noise_dps': 2.0, 'dropout': ()},
        {'name': 'medium_negative', 'bias_dps': -2.0, 'sync_ms': -20.0,
         'extrinsic_deg': -2.0, 'noise_dps': 2.0, 'dropout': ()},
        {'name': 'medium_positive_gap2', 'bias_dps': 2.0, 'sync_ms': -20.0,
         'extrinsic_deg': 2.0, 'noise_dps': 2.0, 'dropout': (10, 11)},
        {'name': 'medium_negative_gap2', 'bias_dps': -2.0, 'sync_ms': 20.0,
         'extrinsic_deg': -2.0, 'noise_dps': 2.0, 'dropout': (17, 18)},
        {'name': 'mixed_medium_gap1', 'bias_dps': 2.0, 'sync_ms': 20.0,
         'extrinsic_deg': -2.0, 'noise_dps': 2.0, 'dropout': (14,)},
    )
    return dict(profiles[variant])


def segment_intersects_corridor(angle_now: float, angle_future: float) -> bool:
    return min(angle_now, angle_future) <= CORRIDOR_DEG and \
        max(angle_now, angle_future) >= -CORRIDOR_DEG


def episode_source(family: str, variant: int) -> dict:
    yaw, pitch = orientation_waveform(variant)
    sign = 1.0 if variant % 2 == 0 else -1.0
    inside_offsets = (0.8, 1.3, 2.1, 0.5, 1.7, 3.0)
    outside_offsets = (0.5, 1.1, 1.9, 0.7, 2.6, 4.0)
    gap_starts = (9, 12, 15, 8, 17, 13)
    peak_rows = [int(row) for row in np.argsort(np.abs(yaw))[::-1]
                 if 5 <= row <= 24][:3]
    swap_rows = tuple(sorted(peak_rows))
    frames = []
    for index in range(FRAMES):
        tof_objects, radar_objects = [], []
        tof_known = True
        truth = False
        truth_height = None
        height = 'HEAD' if variant % 2 else 'BODY'

        if family == 'boundary_inside':
            world_angle = sign * (CORRIDOR_DEG - inside_offsets[variant])
            distance = 4.35 + 0.04 * variant - (0.085 + 0.003 * variant) * index
            truth = distance < 3.18
            truth_height = height if truth else None
            camera_angle = world_angle - yaw[index]
            obj = {'range': distance, 'theta': camera_angle, 'height': height}
            tof_objects.append(obj)
            radar_objects.append({'range': distance, 'theta': world_angle,
                                  'vr': -(0.85 + 0.03 * variant), 'rcs': 0.9})
        elif family == 'boundary_outside':
            world_angle = sign * (CORRIDOR_DEG + outside_offsets[variant])
            distance = 2.75 + 0.03 * variant - 0.01 * index
            camera_angle = world_angle - yaw[index]
            obj = {'range': distance, 'theta': camera_angle, 'height': height}
            tof_objects.append(obj)
            radar_objects.append({'range': distance, 'theta': world_angle,
                                  'vr': -0.55, 'rcs': 0.95})
        elif family == 'lateral_crossing':
            z = 2.45 + 0.11 * variant
            speed = 0.14 + 0.018 * variant
            x = sign * (2.65 - speed * index)
            future_x = x - sign * speed * 10.0
            world_angle = math.degrees(math.atan2(x, z))
            future_angle = math.degrees(math.atan2(future_x, z))
            distance = math.hypot(x, z)
            truth = distance < 3.6 and segment_intersects_corridor(world_angle, future_angle)
            truth_height = height if truth else None
            camera_angle = world_angle - yaw[index]
            obj = {'range': distance, 'theta': camera_angle, 'height': height, 'lateral': True}
            tof_objects.append(obj)
            previous_x = x + sign * speed
            previous_range = math.hypot(previous_x, z)
            radar_objects.append({'range': distance, 'theta': world_angle,
                                  'vr': (distance - previous_range) / DT_S, 'rcs': 0.8})
        elif family == 'head_motion':
            if index in swap_rows:
                tof_objects.append({'range': 2.62 + 0.03 * variant,
                                    'theta': -0.15 * yaw[index], 'height': height})
            if index in range(max(0, swap_rows[0] - 2), min(FRAMES, swap_rows[-1] + 3)):
                radar_objects.append({'range': 2.5 + 0.02 * abs(index - swap_rows[1]),
                                      'theta': 4.0 * math.sin(index * 0.63),
                                      'vr': -0.6 if index <= swap_rows[1] else 0.35,
                                      'rcs': 0.95})
        elif family == 'multi_target_gap':
            distance = 4.2 + 0.03 * variant - (0.095 + 0.004 * variant) * index
            truth = distance < 3.18
            truth_height = height if truth else None
            primary_world = sign * (3.0 + 0.35 * variant)
            secondary_world = -sign * (6.0 - 0.25 * variant)
            primary = {'range': distance, 'theta': primary_world - yaw[index],
                       'height': height}
            secondary = {'range': distance + 0.24,
                         'theta': secondary_world - yaw[index],
                         'height': 'BODY' if height == 'HEAD' else 'HEAD'}
            tof_objects.extend((primary, secondary))
            radar_objects.extend((
                {'range': distance, 'theta': primary_world, 'vr': -1.0, 'rcs': 0.92},
                {'range': distance + 0.24, 'theta': secondary_world, 'vr': -0.9, 'rcs': 0.68},
            ))
            gap_length = 2 + variant % 3
            tof_known = not (gap_starts[variant] <= index < gap_starts[variant] + gap_length)
        elif family == 'weak_radar':
            world_angle = sign * (1.0 + 0.25 * variant)
            distance = 4.15 + 0.02 * variant - (0.09 + 0.002 * variant) * index
            truth = distance < 3.18
            truth_height = height if truth else None
            obj = {'range': distance, 'theta': world_angle - yaw[index], 'height': height}
            tof_objects.append(obj)
            radar_objects.append({'range': distance, 'theta': world_angle,
                                  'vr': -0.9, 'rcs': 0.035})
        else:
            raise ValueError(f'unknown family: {family}')

        frames.append({
            'index': index,
            'truth': truth,
            'truth_height': truth_height,
            'tof_known': tof_known,
            'tof_objects': tof_objects,
            'radar_objects': radar_objects,
        })
    profile = stress_profile(variant)
    return {
        'episode_id': f'{family}_{variant}',
        'family': family,
        'variant': variant,
        'stress': profile['name'],
        'true_yaw_deg': yaw.tolist(),
        'true_pitch_deg': pitch.tolist(),
        'frames': frames,
    }


def build_source() -> list[dict]:
    return [episode_source(family, variant)
            for family in FAMILIES for variant in range(VARIANTS)]


def flatten_source(source: list[dict]) -> dict[str, np.ndarray]:
    count = len(source) * FRAMES
    result = {
        'episode_id': np.empty(count, dtype='<U40'),
        'family': np.empty(count, dtype='<U32'),
        'variant': np.empty(count, dtype=np.int16),
        'stress': np.empty(count, dtype='<U32'),
        'truth': np.zeros(count, dtype=bool),
        'truth_height': np.empty(count, dtype='<U8'),
        'tof_known': np.zeros(count, dtype=bool),
        'tof_range_m': np.full((count, MAX_TOF_OBJECTS), np.nan, dtype=np.float64),
        'tof_theta_deg': np.full((count, MAX_TOF_OBJECTS), np.nan, dtype=np.float64),
        'tof_lateral': np.zeros((count, MAX_TOF_OBJECTS), dtype=bool),
        'tof_height': np.full((count, MAX_TOF_OBJECTS), 'UNKNOWN', dtype='<U8'),
    }
    cursor = 0
    for episode in source:
        for frame in episode['frames']:
            result['episode_id'][cursor] = episode['episode_id']
            result['family'][cursor] = episode['family']
            result['variant'][cursor] = episode['variant']
            result['stress'][cursor] = episode['stress']
            result['truth'][cursor] = frame['truth']
            result['truth_height'][cursor] = frame['truth_height'] or 'NONE'
            result['tof_known'][cursor] = frame['tof_known']
            for slot, obj in enumerate(frame['tof_objects'][:MAX_TOF_OBJECTS]):
                result['tof_range_m'][cursor, slot] = obj['range']
                result['tof_theta_deg'][cursor, slot] = obj['theta']
                result['tof_lateral'][cursor, slot] = bool(obj.get('lateral', False))
                result['tof_height'][cursor, slot] = obj['height']
            cursor += 1
    return result


def radar_materializer(source: list[dict], rng: np.random.Generator
                       ) -> tuple[np.ndarray, ...]:
    """Apply the frozen MZ84 radar proxy with the MZ88 frame count."""
    total = sum(len(episode['frames']) for episode in source)
    ranges = np.zeros((total, 4), np.float32)
    velocities = np.zeros_like(ranges)
    angles = np.zeros_like(ranges)
    valid = np.zeros_like(ranges, bool)
    cursor = 0
    for episode in source:
        for frame in episode['frames']:
            returns = []
            for obj in frame['radar_objects']:
                probability = float(np.clip(
                    obj['rcs'] * math.exp(-obj['range'] / 18.0), 0.01, 0.97))
                if rng.random() > probability:
                    continue
                returns.append((
                    round((obj['range'] + rng.normal(0, 0.06)) / 0.05) * 0.05,
                    round((obj['vr'] + rng.normal(0, 0.08)) / 0.1) * 0.1,
                    round((obj['theta'] + rng.normal(0, 2.0)) / 10.0) * 10.0,
                ))
            merged = []
            for row in sorted(returns, key=lambda value: (value[2], value[0])):
                if (merged and row[2] == merged[-1][2] and
                        abs(row[0] - merged[-1][0]) <= 0.35):
                    prior = merged[-1]
                    merged[-1] = (min(prior[0], row[0]), min(prior[1], row[1]), row[2])
                else:
                    merged.append(row)
            for slot, row in enumerate(merged[:4]):
                ranges[cursor, slot], velocities[cursor, slot], angles[cursor, slot] = row
                valid[cursor, slot] = True
            cursor += 1
    return ranges, velocities, angles, valid


def sample_with_offset(values: np.ndarray, offset_ms: float) -> np.ndarray:
    times = np.arange(FRAMES, dtype=np.float64) * DT_S
    query = np.clip(times + offset_ms / 1000.0, times[0], times[-1])
    return np.interp(query, times, values)


def materialize_imu(source: list[dict]) -> dict[str, np.ndarray]:
    count = len(source) * FRAMES
    delta_yaw = np.zeros(count, dtype=np.float64)
    delta_pitch = np.zeros(count, dtype=np.float64)
    valid = np.ones(count, dtype=bool)
    cursor = 0
    for episode in source:
        variant = int(episode['variant'])
        profile = stress_profile(variant)
        yaw = sample_with_offset(np.asarray(episode['true_yaw_deg']), profile['sync_ms'])
        pitch = sample_with_offset(np.asarray(episode['true_pitch_deg']), profile['sync_ms'])
        dyaw = np.diff(np.r_[0.0, yaw])
        dpitch = np.diff(np.r_[0.0, pitch])
        rng = np.random.default_rng(SEED + variant * 101 + FAMILIES.index(episode['family']))
        dyaw += profile['bias_dps'] * DT_S
        dpitch += 0.3 * profile['bias_dps'] * DT_S
        dyaw += rng.normal(0.0, profile['noise_dps'] * DT_S, FRAMES)
        dpitch += rng.normal(0.0, 0.3 * profile['noise_dps'] * DT_S, FRAMES)
        dyaw[0] += profile['extrinsic_deg']
        dpitch[0] += 0.3 * profile['extrinsic_deg']
        local_valid = np.ones(FRAMES, dtype=bool)
        if profile['dropout']:
            local_valid[list(profile['dropout'])] = False
            dyaw[~local_valid] = 0.0
            dpitch[~local_valid] = 0.0
        rows = slice(cursor, cursor + FRAMES)
        delta_yaw[rows], delta_pitch[rows], valid[rows] = dyaw, dpitch, local_valid
        cursor += FRAMES
    return {'delta_yaw_deg': delta_yaw, 'delta_pitch_deg': delta_pitch, 'valid': valid}


def integrate_with_uncertainty(delta: np.ndarray, valid: np.ndarray,
                               episode_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    orientation = np.zeros(len(delta), dtype=np.float64)
    sigma = np.zeros(len(delta), dtype=np.float64)
    value = 0.0
    last_rate = 0.0
    valid_steps = 0
    gap_steps = 0
    elapsed = 0.0
    for index in range(len(delta)):
        reset = index == 0 or episode_ids[index] != episode_ids[index - 1]
        if reset:
            value, last_rate, valid_steps, gap_steps, elapsed = 0.0, 0.0, 0, 0, 0.0
        if valid[index]:
            value += delta[index]
            last_rate = delta[index] / DT_S
            valid_steps += 1
            gap_steps = 0
        else:
            gap_steps += 1
        orientation[index] = value
        sigma_gyro = GYRO_NOISE_SIGMA_DPS * DT_S * math.sqrt(valid_steps)
        sigma_bias = BIAS_SIGMA_DPS * elapsed
        sigma_sync = abs(last_rate) * SYNC_SIGMA_S
        sigma_dropout = gap_steps * DT_S * (abs(last_rate) + BIAS_SIGMA_DPS)
        sigma[index] = math.sqrt(sigma_gyro ** 2 + sigma_bias ** 2 +
                                 EXTRINSIC_SIGMA_DEG ** 2 + sigma_sync ** 2 +
                                 sigma_dropout ** 2)
        elapsed += DT_S
    return orientation, sigma


def point_state(angle_deg: float, range_m: float, sigma_theta_deg: float) -> str:
    theta = math.radians(angle_deg)
    y_hat = range_m * math.sin(theta)
    width = range_m * math.sin(math.radians(CORRIDOR_DEG))
    sigma_y = range_m * abs(math.cos(theta)) * math.radians(sigma_theta_deg)
    z = (width - abs(y_hat)) / max(sigma_y, 1e-6)
    if z >= K_SIGMA:
        return 'CERTAIN_IN'
    if z <= -K_SIGMA:
        return 'CERTAIN_OUT'
    return 'UNCERTAIN'


def crossing_state(angle_now: float, angle_future: float, sigma_now: float,
                   sigma_future: float) -> str:
    if angle_now * angle_future <= 0.0:
        closest = 0.0
    else:
        closest = min(abs(angle_now), abs(angle_future))
    uncertainty = max(sigma_now, sigma_future)
    if closest + K_SIGMA * uncertainty <= CORRIDOR_DEG:
        return 'CERTAIN_IN'
    same_outside_side = ((angle_now > CORRIDOR_DEG and angle_future > CORRIDOR_DEG) or
                         (angle_now < -CORRIDOR_DEG and angle_future < -CORRIDOR_DEG))
    if same_outside_side and closest - K_SIGMA * uncertainty > CORRIDOR_DEG:
        return 'CERTAIN_OUT'
    return 'UNCERTAIN'


def stronger_state(left: str, right: str) -> str:
    order = {'CERTAIN_OUT': 0, 'UNCERTAIN': 1, 'CERTAIN_IN': 2}
    return left if order[left] >= order[right] else right


def tof_association(flat: dict[str, np.ndarray], yaw: np.ndarray, pitch: np.ndarray,
                    sigma_yaw: np.ndarray) -> dict[str, np.ndarray]:
    count = len(flat['episode_id'])
    hard = np.zeros(count, dtype=bool)
    state = np.full(count, 'CERTAIN_OUT', dtype='<U12')
    height = np.full(count, 'UNKNOWN', dtype='<U8')
    previous_angle = None
    previous_sigma = None
    previous_episode = None
    for index in range(count):
        episode_id = flat['episode_id'][index]
        if episode_id != previous_episode:
            previous_angle, previous_sigma = None, None
        if not flat['tof_known'][index]:
            state[index] = 'MISSING'
            previous_episode = episode_id
            continue
        first_angle, first_sigma = None, None
        frame_state = 'CERTAIN_OUT'
        frame_height = 'UNKNOWN'
        for slot in range(MAX_TOF_OBJECTS):
            distance = flat['tof_range_m'][index, slot]
            if not np.isfinite(distance):
                continue
            angle, _ = rotate_ray(flat['tof_theta_deg'][index, slot],
                                  yaw[index], pitch[index])
            if first_angle is None:
                first_angle, first_sigma = angle, sigma_yaw[index]
            direct_hard = distance < 3.18 and abs(angle) <= CORRIDOR_DEG
            object_state = ('CERTAIN_OUT' if distance >= 3.18 else
                            point_state(angle, distance, sigma_yaw[index]))
            crossing_hard = False
            if (flat['tof_lateral'][index, slot] and previous_angle is not None and
                    distance < 3.6):
                angular_rate = (angle - previous_angle) / DT_S
                future = angle + angular_rate
                crossing_hard = segment_intersects_corridor(angle, future)
                rate_sigma = math.sqrt(sigma_yaw[index] ** 2 + previous_sigma ** 2) / DT_S
                future_sigma = math.sqrt(sigma_yaw[index] ** 2 +
                                         (DT_S * 10.0 * rate_sigma) ** 2)
                object_state = stronger_state(
                    object_state,
                    crossing_state(angle, future, sigma_yaw[index], future_sigma),
                )
            if direct_hard or crossing_hard:
                hard[index] = True
            frame_state = stronger_state(frame_state, object_state)
            if object_state == 'CERTAIN_IN' and frame_height == 'UNKNOWN':
                frame_height = flat['tof_height'][index, slot]
        if first_angle is not None:
            previous_angle, previous_sigma = first_angle, first_sigma
        state[index], height[index] = frame_state, frame_height
        previous_episode = episode_id
    return {'hard': hard, 'state': state, 'height': height}


def apply_authority(state: np.ndarray, tof_known: np.ndarray, radar: np.ndarray,
                    episode_ids: np.ndarray) -> dict[str, np.ndarray]:
    certain = state == 'CERTAIN_IN'
    uncertain = state == 'UNCERTAIN'
    missing = ~tof_known
    tri_unknown = certain | (missing & radar)
    radar_confirm = certain | ((uncertain | missing) & radar)
    full = radar_confirm.copy()
    held = np.zeros(len(state), dtype=bool)
    credential = radar_confirm.copy()
    for index in range(len(state)):
        if index == 0 or episode_ids[index] != episode_ids[index - 1]:
            continue
        if uncertain[index] and not radar_confirm[index] and full[index - 1] and credential[index - 1]:
            full[index] = True
            held[index] = True
            credential[index] = False
    return {
        'tri_unknown': tri_unknown,
        'radar_confirm': radar_confirm,
        'full': full,
        'held': held,
        'credential': credential,
    }


def event_audit(episode_ids: np.ndarray, truth: np.ndarray, baseline: np.ndarray,
                candidate: np.ndarray) -> list[dict]:
    rows_out = []
    for episode_id in np.unique(episode_ids):
        rows = episode_ids == episode_id
        local_truth = truth[rows]
        if not local_truth.any():
            continue
        starts, fragments = {}, {}
        for name, values in (('baseline', baseline[rows]), ('candidate', candidate[rows])):
            hits = np.flatnonzero(local_truth & values)
            starts[name] = None if not hits.size else int(hits[0])
            fragments[name] = episodes(values[local_truth])
        if starts['baseline'] is None:
            delay = 0.0
        elif starts['candidate'] is None:
            delay = None
        else:
            delay = (starts['candidate'] - starts['baseline']) * DT_S
        rows_out.append({
            'episode_id': str(episode_id),
            'first_correct_frame': starts,
            'added_first_alert_delay_s': delay,
            'fragments': fragments,
            'added_fragments': fragments['candidate'] - fragments['baseline'],
        })
    return rows_out


def metrics(truth: np.ndarray, alert: np.ndarray) -> dict:
    row = confusion(truth, alert)
    row['F1'] = f1(row)
    return row


def performance_checks(baseline_metrics: dict, candidate_metrics: dict,
                       truth: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
                       nominal: np.ndarray, boundary_head_stress: np.ndarray,
                       audit: list[dict]) -> dict:
    baseline_target_fp = int((~truth & baseline & boundary_head_stress).sum())
    removed_target_fp = int((~truth & baseline & ~candidate & boundary_head_stress).sum())
    finite_delays = [row['added_first_alert_delay_s'] for row in audit
                     if row['added_first_alert_delay_s'] is not None]
    missing_after_baseline = any(row['added_first_alert_delay_s'] is None for row in audit)
    max_delay = max(finite_delays, default=0.0)
    max_added_fragments = max(row['added_fragments'] for row in audit)
    return {
        'overall_f1_higher_fp_halved_tp_loss_le2': (
            candidate_metrics['F1'] > baseline_metrics['F1'] and
            candidate_metrics['FP'] * 2 <= baseline_metrics['FP'] and
            candidate_metrics['TP'] >= baseline_metrics['TP'] - 2),
        'nominal_zero_tp_loss_zero_new_fp': (
            int((truth & baseline & ~candidate & nominal).sum()) == 0 and
            int((~truth & ~baseline & candidate & nominal).sum()) == 0),
        'half_stressed_boundary_head_fp_removed': (
            baseline_target_fp > 0 and removed_target_fp * 2 >= baseline_target_fp),
        'timing_le_0p2_and_added_fragments_le1': (
            not missing_after_baseline and max_delay <= 0.2 + 1e-9 and
            max_added_fragments <= 1),
        'baseline_target_fp': baseline_target_fp,
        'removed_target_fp': removed_target_fp,
        'max_added_first_alert_delay_s': float(max_delay),
        'positive_events_missed_after_baseline_alert': int(missing_after_baseline),
        'max_added_fragments': int(max_added_fragments),
    }


def run(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    allowed = (root / 'artifacts.local/work' / TASK).resolve()
    if output.parent != allowed or output.exists():
        raise ValueError('output must be a new direct child of the MZ88 artifact root')
    output.mkdir(parents=True)
    started = time.perf_counter()
    shutil.copyfile(Path(__file__).resolve(), output / Path(__file__).name)

    source = build_source()
    flat = flatten_source(source)
    write(output / 'source.json', {
        'schema': 'MZ88_FRESH_ANALYTIC_SOURCE_V1',
        'seed': SEED,
        'families': list(FAMILIES),
        'episodes': source,
    })
    imu = materialize_imu(source)
    ranges, velocities, angles, radar_valid = radar_materializer(
        source, np.random.default_rng(SEED))
    observations_path = output / 'observations.npz'
    np.savez_compressed(
        observations_path,
        episode_id=flat['episode_id'], variant=flat['variant'],
        tof_known=flat['tof_known'], tof_range_m=flat['tof_range_m'],
        tof_theta_deg=flat['tof_theta_deg'], tof_lateral=flat['tof_lateral'],
        tof_height=flat['tof_height'], radar_range_m=ranges,
        radar_radial_velocity_mps=velocities, radar_azimuth_deg=angles,
        radar_valid=radar_valid,
    )
    imu_path = output / 'imu-observations.npz'
    np.savez_compressed(imu_path, episode_id=flat['episode_id'], **imu)
    write(output / 'source-receipt.json', {
        'status': 'SEALED_BEFORE_PREDICTION',
        'episodes': len(source), 'frames': len(flat['episode_id']),
        'source_sha256': sha(output / 'source.json'),
        'observations_sha256': sha(observations_path),
        'imu_observations_sha256': sha(imu_path),
        'reuses_mz84_or_mz87_frames': False,
    })

    yaw, sigma_yaw = integrate_with_uncertainty(
        imu['delta_yaw_deg'], imu['valid'], flat['episode_id'])
    pitch, sigma_pitch = integrate_with_uncertainty(
        imu['delta_pitch_deg'], imu['valid'], flat['episode_id'])
    radar = radar_expert(ranges, velocities, angles, radar_valid, flat['episode_id'])
    association = tof_association(flat, yaw, pitch, sigma_yaw)
    authority = apply_authority(
        association['state'], flat['tof_known'], radar, flat['episode_id'])
    baseline = association['hard'] | (~flat['tof_known'] & radar)
    baseline_height = np.where(association['hard'], association['height'], 'UNKNOWN')
    full_height = np.where(association['state'] == 'CERTAIN_IN', association['height'], 'UNKNOWN')

    predictions_path = output / 'predictions.npz'
    np.savez_compressed(
        predictions_path,
        episode_id=flat['episode_id'], hard_baseline=baseline,
        tri_unknown=authority['tri_unknown'], radar_confirm=authority['radar_confirm'],
        full=authority['full'], association_state=association['state'],
        radar=radar, held=authority['held'], credential=authority['credential'],
        baseline_height=baseline_height, full_height=full_height,
        estimated_yaw_deg=yaw, estimated_pitch_deg=pitch,
        sigma_yaw_deg=sigma_yaw, sigma_pitch_deg=sigma_pitch,
    )
    write(output / 'predictor-receipt.json', {
        'status': 'SEALED_BEFORE_EVALUATOR_OPEN',
        'training_steps': 0, 'threshold_searches': 0, 'k_sigma': K_SIGMA,
        'uncertainty_envelope': {
            'bias_sigma_dps': BIAS_SIGMA_DPS,
            'gyro_noise_sigma_dps': GYRO_NOISE_SIGMA_DPS,
            'extrinsic_sigma_deg': EXTRINSIC_SIGMA_DEG,
            'sync_sigma_ms': SYNC_SIGMA_S * 1000.0,
        },
        'predictions_sha256': sha(predictions_path),
        'mz87_results_read_by_predictor': False,
    })

    evaluator_path = output / 'evaluator.npz'
    np.savez_compressed(
        evaluator_path,
        episode_id=flat['episode_id'], family=flat['family'],
        variant=flat['variant'], stress=flat['stress'],
        generic_truth=flat['truth'], truth_height=flat['truth_height'],
    )
    truth = flat['truth']
    nominal = flat['variant'] == 0
    boundary_head_stress = ((flat['family'] == 'boundary_outside') |
                            (flat['family'] == 'head_motion')) & ~nominal
    arm_values = {
        'hard_mz85': baseline,
        'tri_unknown': authority['tri_unknown'],
        'radar_confirm': authority['radar_confirm'],
        'full': authority['full'],
    }
    metric_rows = {name: metrics(truth, values) for name, values in arm_values.items()}
    audits = {name: event_audit(flat['episode_id'], truth, baseline, values)
              for name, values in arm_values.items() if name != 'hard_mz85'}
    checks = {
        name: performance_checks(
            metric_rows['hard_mz85'], metric_rows[name], truth, baseline,
            arm_values[name], nominal, boundary_head_stress, audits[name])
        for name in ('tri_unknown', 'radar_confirm', 'full')
    }
    uncertain_only_violation = int((
        (association['state'] == 'UNCERTAIN') & ~radar & ~authority['held'] &
        authority['full']).sum())
    bad_holds = 0
    for index in np.flatnonzero(authority['held']):
        if (index == 0 or flat['episode_id'][index] != flat['episode_id'][index - 1] or
                not authority['credential'][index - 1] or not authority['full'][index - 1]):
            bad_holds += 1
    fabricated_height = int(((authority['held'] |
                              ((association['state'] == 'UNCERTAIN') & radar)) &
                             (full_height != 'UNKNOWN')).sum())
    integrity = {
        'uncertain_tof_only_new_hazards': uncertain_only_violation,
        'holds_without_immediately_preceding_credential': bad_holds,
        'radar_or_hold_fabricated_height_frames': fabricated_height,
    }
    checks['full']['authority_integrity'] = all(value == 0 for value in integrity.values())
    full_gate_keys = (
        'overall_f1_higher_fp_halved_tp_loss_le2',
        'nominal_zero_tp_loss_zero_new_fp',
        'half_stressed_boundary_head_fp_removed',
        'timing_le_0p2_and_added_fragments_le1',
        'authority_integrity',
    )
    full_pass = all(bool(checks['full'][key]) for key in full_gate_keys)
    attribution_pass = []
    for name in ('tri_unknown', 'radar_confirm'):
        if all(bool(checks[name][key]) for key in full_gate_keys[:-1]):
            attribution_pass.append(name)
    if full_pass:
        decision = 'UNCERTAINTY_AWARE_ASSOCIATION_CANARY_PASSES'
    elif attribution_pass:
        decision = 'UNCERTAINTY_SUBMECHANISM_ONLY'
    else:
        decision = 'SCALAR_UNCERTAINTY_ASSOCIATION_NOT_RETAINED'

    family_metrics = {}
    for family in FAMILIES:
        rows = flat['family'] == family
        family_metrics[family] = {
            name: metrics(truth[rows], values[rows]) for name, values in arm_values.items()
        }
    state_counts = {value: int((association['state'] == value).sum())
                    for value in ('CERTAIN_IN', 'UNCERTAIN', 'CERTAIN_OUT', 'MISSING')}
    result = {
        'status': 'FRESH_CONTROLLED_ANALYTIC_SOURCE',
        'metrics': metric_rows,
        'family_metrics': family_metrics,
        'association_state_counts': state_counts,
        'held_frames': int(authority['held'].sum()),
        'checks': checks,
        'authority_integrity': integrity,
        'attribution_arms_passing_performance_checks': attribution_pass,
        'decision': decision,
        'all_full_gates_pass': full_pass,
        'event_audit': audits,
    }
    write(output / 'result.json', result)
    write(output / 'receipt.json', {
        'status': 'PASS', 'phase': 'EXPLORE_FRESH_CONTROLLED_SIMULATION',
        'frames': len(truth), 'training_steps': 0, 'threshold_searches': 0,
        'outputs': {path.name: sha(path) for path in output.iterdir() if path.is_file()},
        'backend': 'CPU TASK_NOT_GPU_SUITABLE',
        'seconds': time.perf_counter() - started,
        'claim': ('Fresh constructed analytic association evidence only; no real covariance, '
                  'hardware, alert, deployment, user-benefit or safety claim.'),
    })
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
