"""MZ84 analytic zero-fit bidirectional ToF/radar complementarity falsifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

TASK = 'mz84-bidirectional-complementarity-20260912'
SEED = 8401
DT_S = 0.1
FRAMES = 20
FAMILIES = ('weak_reflector', 'offcorridor_clutter', 'wall_multipath',
            'lateral_crossing', 'head_motion', 'multi_target')
HEIGHTS = ('BODY', 'HEAD')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def episode_source(family: str, variant: int) -> dict:
    frames = []
    height = HEIGHTS[variant % 2]
    for index in range(FRAMES):
        radar_objects, tof_objects = [], []
        tof_known = True
        truth = False
        if family == 'weak_reflector':
            distance = 4.25 + 0.05 * variant - 0.15 * index
            truth = distance < 3.18
            obj = {'range': distance, 'theta': (-1) ** variant * 2.0, 'height': height}
            tof_objects.append(obj)
            radar_objects.append({**obj, 'vr': -1.5, 'rcs': 0.06})
        elif family == 'offcorridor_clutter':
            distance = 3.0 - 0.035 * index
            angle = (22.0 + variant) * ((-1) ** variant)
            tof_objects.append({'range': distance, 'theta': angle, 'height': height})
            radar_objects.append({'range': distance, 'theta': angle, 'vr': -0.5, 'rcs': 0.9})
        elif family == 'wall_multipath':
            if variant < 2:
                distance = 4.2 - 0.15 * index
                truth = distance < 3.18
                radar_objects.append({'range': distance, 'theta': 0.0, 'vr': -1.5, 'rcs': 1.0})
                tof_objects.append({'range': distance, 'theta': 0.0, 'height': 'BODY'})
                if index >= 7:
                    tof_known = False  # strong-light/low-return proxy
            else:
                ghost_range = 2.55 - 0.025 * max(0, index - 4)
                radar_objects.append({'range': ghost_range, 'theta': (-1) ** variant * 5.0,
                                      'vr': -0.65, 'rcs': 1.0, 'ghost': True})
        elif family == 'lateral_crossing':
            x = 2.2 - (0.23 + 0.01 * variant) * index
            z = 2.8
            distance = math.hypot(x, z)
            angle = math.degrees(math.atan2(x, z))
            next_x = x - (0.23 + 0.01 * variant) * 10
            truth = min(abs(x), abs(next_x), 0.0 if x * next_x <= 0 else min(abs(x), abs(next_x))) <= 0.6
            tof_objects.append({'range': distance, 'theta': angle, 'height': height, 'lateral': True})
            previous_x = x + (0.23 + 0.01 * variant)
            previous_range = math.hypot(previous_x, z)
            radar_objects.append({'range': distance, 'theta': angle,
                                  'vr': (distance - previous_range) / DT_S, 'rcs': 0.9})
        elif family == 'head_motion':
            phase = index - 6
            apparent = abs(phase) <= 5
            if apparent:
                radar_objects.append({'range': 2.45 + 0.03 * abs(phase),
                                      'theta': 3.0 * math.sin(index * 0.8),
                                      'vr': -0.7 if phase <= 1 else 0.4, 'rcs': 0.95,
                                      'ego_apparent': True})
            # Two transient ToF association swaps remain visible but incorrect.
            if index in (9, 10):
                tof_objects.append({'range': 2.7, 'theta': 0.0, 'height': height,
                                    'association_swap': True})
        elif family == 'multi_target':
            distance = 4.1 + 0.04 * variant - 0.14 * index
            truth = distance < 3.18
            primary = {'range': distance, 'theta': -4.0, 'height': height}
            secondary = {'range': distance + 0.22, 'theta': 4.0, 'height': HEIGHTS[(variant + 1) % 2]}
            tof_objects.extend((primary, secondary))
            radar_objects.extend(({**primary, 'vr': -1.4, 'rcs': 0.9},
                                  {**secondary, 'vr': -1.3, 'rcs': 0.65}))
            if 8 <= index <= 12:
                tof_known = False  # fixed short packet gap
        frames.append({'index': index, 'truth': truth, 'truth_height': height if truth else None,
                       'tof_known': tof_known, 'tof_objects': tof_objects,
                       'radar_objects': radar_objects})
    return {'episode_id': f'{family}_{variant}', 'family': family, 'variant': variant,
            'frames': frames}


def build_source() -> list[dict]:
    return [episode_source(family, variant) for family in FAMILIES for variant in range(4)]


def radar_sensor(source: list[dict], rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    total = len(source) * FRAMES
    ranges = np.zeros((total, 4), np.float32)
    velocities = np.zeros_like(ranges)
    angles = np.zeros_like(ranges)
    valid = np.zeros_like(ranges, bool)
    cursor = 0
    for episode in source:
        for frame in episode['frames']:
            returns = []
            for obj in frame['radar_objects']:
                probability = float(np.clip(obj['rcs'] * math.exp(-obj['range'] / 18.0), 0.01, 0.97))
                if rng.random() > probability:
                    continue
                measured = (round((obj['range'] + rng.normal(0, 0.06)) / 0.05) * 0.05,
                            round((obj['vr'] + rng.normal(0, 0.08)) / 0.1) * 0.1,
                            round((obj['theta'] + rng.normal(0, 2.0)) / 10.0) * 10.0)
                returns.append(measured)
            merged = []
            for row in sorted(returns, key=lambda value: (value[2], value[0])):
                if merged and row[2] == merged[-1][2] and abs(row[0] - merged[-1][0]) <= 0.35:
                    prior = merged[-1]
                    merged[-1] = (min(prior[0], row[0]), min(prior[1], row[1]), row[2])
                else:
                    merged.append(row)
            for slot, row in enumerate(merged[:4]):
                ranges[cursor, slot], velocities[cursor, slot], angles[cursor, slot] = row
                valid[cursor, slot] = True
            cursor += 1
    return ranges, velocities, angles, valid


def hysteresis(raw: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    alert = np.zeros(len(raw), bool)
    active, empty = False, 0
    for index in range(len(raw)):
        if index == 0 or episode_ids[index] != episode_ids[index - 1]:
            active, empty = False, 0
        start = max(0, index - 2)
        while start < index and episode_ids[start] != episode_ids[index]:
            start += 1
        if not active and raw[start:index + 1].sum() >= 2:
            active = True
        empty = 0 if raw[index] else empty + 1
        if active and empty >= 2:
            active = False
        alert[index] = active
    return alert


def radar_expert(ranges: np.ndarray, velocities: np.ndarray, angles: np.ndarray,
                 valid: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    candidate = valid & (ranges < 3.18) & (velocities <= -0.35) & (np.abs(angles) <= 20.0)
    return hysteresis(candidate.any(axis=1), episode_ids)


def tof_expert(source: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alerts, known, heights = [], [], []
    previous_angle = None
    previous_episode = None
    for episode in source:
        for frame in episode['frames']:
            if previous_episode != episode['episode_id']:
                previous_angle = None
            frame_known = bool(frame['tof_known'])
            alert, height = False, 'UNKNOWN'
            visible = frame['tof_objects'] if frame_known else []
            for obj in visible:
                direct = obj['range'] < 3.18 and abs(obj['theta']) <= 12.0
                crossing = False
                if obj.get('lateral') and previous_angle is not None and obj['range'] < 3.6:
                    angular_rate = (obj['theta'] - previous_angle) / DT_S
                    future = obj['theta'] + angular_rate * 1.0
                    crossing = min(obj['theta'], future) <= 12.0 and max(obj['theta'], future) >= -12.0
                if direct or crossing or obj.get('association_swap', False):
                    alert, height = True, obj['height']
                    break
            if visible:
                previous_angle = visible[0]['theta']
            alerts.append(alert)
            known.append(frame_known)
            heights.append(height)
            previous_episode = episode['episode_id']
    return np.asarray(alerts), np.asarray(known), np.asarray(heights)


def confusion(truth: np.ndarray, alert: np.ndarray) -> dict:
    return {'TP': int((truth & alert).sum()), 'FP': int((~truth & alert).sum()),
            'FN': int((truth & ~alert).sum()), 'TN': int((~truth & ~alert).sum())}


def f1(metrics: dict) -> float:
    denominator = 2 * metrics['TP'] + metrics['FP'] + metrics['FN']
    return 0.0 if denominator == 0 else 2 * metrics['TP'] / denominator


def episodes(values: np.ndarray) -> int:
    values = np.asarray(values, bool)
    return int((values & ~np.r_[False, values[:-1]]).sum())


def run(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    allowed = (root / 'artifacts.local/work' / TASK).resolve()
    if output.parent != allowed or output.exists():
        raise ValueError('output must be a new direct child of the MZ84 artifact root')
    output.mkdir(parents=True)
    started = time.perf_counter()
    here = Path(__file__).resolve().parent
    shutil.copyfile(here / Path(__file__).name, output / Path(__file__).name)
    source = build_source()
    write(output / 'source.json', {'schema': 'MZ84_ANALYTIC_SOURCE_V1', 'seed': SEED,
                                   'families': list(FAMILIES), 'episodes': source})
    write(output / 'source-receipt.json', {'status': 'SEALED_BEFORE_PREDICTION',
                                           'episodes': 24, 'frames': 480,
                                           'source_sha256': sha(output / 'source.json')})
    episode_ids = np.asarray([episode['episode_id'] for episode in source for _ in episode['frames']])
    families = np.asarray([episode['family'] for episode in source for _ in episode['frames']])
    truth = np.asarray([frame['truth'] for episode in source for frame in episode['frames']], bool)
    truth_height = np.asarray([frame['truth_height'] or 'NONE' for episode in source for frame in episode['frames']])
    ranges, velocities, angles, radar_valid = radar_sensor(source, np.random.default_rng(SEED))
    radar = radar_expert(ranges, velocities, angles, radar_valid, episode_ids)
    tof, tof_known, tof_height = tof_expert(source)
    fusion = tof | (~tof_known & radar)
    fusion_height = np.where(tof, tof_height, 'UNKNOWN')
    observation_path = output / 'observations.npz'
    np.savez_compressed(observation_path, episode_id=episode_ids, family=families,
                        radar_range_m=ranges, radar_radial_velocity_mps=velocities,
                        radar_azimuth_deg=angles, radar_valid=radar_valid,
                        tof_known=tof_known)
    prediction_path = output / 'predictions.npz'
    np.savez_compressed(prediction_path, episode_id=episode_ids, radar=radar, tof=tof,
                        fusion=fusion, tof_height=tof_height, fusion_height=fusion_height)
    write(output / 'predictor-receipt.json', {
        'status': 'SEALED_BEFORE_EVALUATOR_OPEN', 'training_steps': 0,
        'future_sensor_frames': 0, 'threshold_searches': 0,
        'observations_sha256': sha(observation_path), 'predictions_sha256': sha(prediction_path),
        'fusion': 'ToF OR (ToF_UNKNOWN AND Radar)',
        'radar_expert': 'r<3.18 and vr<=-0.35 and abs(theta)<=20; causal hysteresis',
        'tof_expert': 'r<3.18 and abs(theta)<=12; one-second angular forecast for lateral motion'})
    evaluator_path = output / 'evaluator.npz'
    np.savez_compressed(evaluator_path, episode_id=episode_ids, family=families,
                        generic_truth=truth, truth_height=truth_height)

    metrics = {name: confusion(truth, values) for name, values in
               (('ToF', tof), ('Radar', radar), ('Fusion', fusion))}
    for row in metrics.values():
        row['F1'] = f1(row)
    family_rows = {}
    tof_win_families, radar_win_families = [], []
    for family in FAMILIES:
        rows = families == family
        tof_unique = int((truth & rows & tof & ~radar).sum())
        radar_unique = int((truth & rows & radar & ~tof).sum())
        if tof_unique:
            tof_win_families.append(family)
        if radar_unique:
            radar_win_families.append(family)
        family_rows[family] = {
            'truth_positive_frames': int((truth & rows).sum()),
            'ToF': confusion(truth[rows], tof[rows]),
            'Radar': confusion(truth[rows], radar[rows]),
            'Fusion': confusion(truth[rows], fusion[rows]),
            'tof_unique_true_frames': tof_unique, 'radar_unique_true_frames': radar_unique}
    timing = []
    for episode in source:
        rows = episode_ids == episode['episode_id']
        local_truth = truth[rows]
        if not local_truth.any():
            continue
        starts = {}
        fragments = {}
        for name, values in (('ToF', tof[rows]), ('Radar', radar[rows]), ('Fusion', fusion[rows])):
            hits = np.flatnonzero(local_truth & values)
            starts[name] = None if not hits.size else int(hits[0])
            fragments[name] = episodes(values[local_truth])
        available = [value for name, value in starts.items() if name != 'Fusion' and value is not None]
        best = min(available) if available else None
        delay = None if best is None or starts['Fusion'] is None else (starts['Fusion'] - best) * DT_S
        timing.append({'episode_id': episode['episode_id'], 'family': episode['family'],
                       'first_correct_frame': starts, 'fusion_delay_vs_best_s': delay})
        timing[-1]['alert_episodes_during_truth'] = fragments
    best_single = max(('ToF', 'Radar'), key=lambda name: metrics[name]['F1'])
    f1_gate = metrics['Fusion']['F1'] > max(metrics['ToF']['F1'], metrics['Radar']['F1'])
    complement_gate = bool(tof_win_families and radar_win_families and
                           set(tof_win_families) != set(radar_win_families))
    fp_gate = metrics['Fusion']['FP'] <= metrics[best_single]['FP'] + 2
    timing_gate = all(row['fusion_delay_vs_best_s'] is not None and
                      row['fusion_delay_vs_best_s'] <= 0.2 + 1e-9 for row in timing)
    attribution = {
        'tof_attributed_true_frames': int((truth & tof & (tof_height != 'UNKNOWN')).sum()),
        'radar_only_height_unknown_true_frames': int((truth & ~tof & radar & (fusion_height == 'UNKNOWN')).sum()),
        'fabricated_radar_height_frames': int((~tof & radar & (fusion_height != 'UNKNOWN')).sum())}
    result = {'status': 'CONTROLLED_SIMULATION', 'metrics': metrics,
              'families': family_rows, 'tof_win_families': tof_win_families,
              'radar_win_families': radar_win_families, 'event_timing': timing,
              'attribution': attribution,
              'gates': {'fusion_f1_strictly_best': f1_gate,
                        'bidirectional_distinct_failure_sources': complement_gate,
                        'fusion_fp_within_2_of_best_single': fp_gate,
                        'all_positive_event_first_alert_delay_le_0_2s': timing_gate}}
    result['all_gates_pass'] = all(result['gates'].values())
    result['decision'] = ('BIDIRECTIONAL_TOF_RADAR_COMPLEMENTARITY_CANARY_PASSES'
                          if result['all_gates_pass'] else 'BIDIRECTIONAL_COMPLEMENTARITY_GATE_NOT_MET')
    write(output / 'result.json', result)
    write(output / 'receipt.json', {
        'status': 'PASS', 'phase': 'EXPLORE_CONTROLLED_SIMULATION',
        'episodes': 24, 'frames': 480, 'training_steps': 0,
        'outputs': {path.name: sha(path) for path in output.iterdir() if path.is_file()},
        'backend': 'CPU TASK_NOT_GPU_SUITABLE', 'seconds': time.perf_counter() - started,
        'claim': 'Constructed failure-direction and policy-logic evidence only; no calibrated sensor, natural-scene, alert, user-benefit or safety claim.'})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
