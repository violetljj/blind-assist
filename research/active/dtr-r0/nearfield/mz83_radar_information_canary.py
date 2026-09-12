"""MZ83 deterministic radar-like information canary on consumed MZ77."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

TASK = 'mz83-radar-information-canary-20260912'
PROFILES = ('5KLUX_WHITE88_TYP', '5KLUX_LIGHTGRAY54_TYP', '5KLUX_GRAY17_TYP')
SEED = 8301
DT_S = 0.1
AZIMUTH_CENTERS = np.arange(-40.0, 41.0, 10.0)
MAX_RETURNS = 18
RANGE_BINS = np.arange(0.4, 8.0001, 0.25)
VERTICAL_APERTURE = (0.12, 0.76)
MIN_CLUSTER_PIXELS = 12


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def raw_clusters(depth: np.ndarray, horizontal_fov_deg: float) -> list[list[tuple[float, int]]]:
    height, width = depth.shape
    y0, y1 = int(height * VERTICAL_APERTURE[0]), int(height * VERTICAL_APERTURE[1])
    x_angles = (np.arange(width) + 0.5 - width * 0.5) * horizontal_fov_deg / width
    result = []
    for center in AZIMUTH_CENTERS:
        columns = np.abs(x_angles - center) < 5.0
        values = depth[y0:y1, columns]
        values = values[np.isfinite(values) & (values >= RANGE_BINS[0]) & (values <= RANGE_BINS[-1])]
        if not values.size:
            result.append([])
            continue
        counts, edges = np.histogram(values, bins=np.r_[RANGE_BINS, RANGE_BINS[-1] + 0.25])
        candidates = []
        for index in np.flatnonzero(counts >= MIN_CLUSTER_PIXELS):
            selected = values[(values >= edges[index]) & (values < edges[index + 1])]
            candidates.append((float(np.median(selected)), int(counts[index])))
        candidates.sort(key=lambda row: (-row[1], row[0]))
        result.append(sorted(candidates[:2], key=lambda row: row[0]))
    return result


def merge_cell(returns: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    merged = []
    for current in sorted(returns, key=lambda row: row[0]):
        if merged and current[0] - merged[-1][0] <= 0.35:
            prior = merged[-1]
            merged[-1] = (min(prior[0], current[0]), min(prior[1], current[1]),
                          0.5 * (prior[2] + current[2]))
        else:
            merged.append(current)
    return merged[:2]


def simulate(depths: list[np.ndarray], clips: np.ndarray, horizontal_fov_deg: float,
             rng: np.random.Generator) -> dict[str, np.ndarray]:
    count = len(depths)
    ranges = np.zeros((count, MAX_RETURNS), np.float32)
    velocities = np.zeros_like(ranges)
    azimuths = np.zeros_like(ranges)
    valid = np.zeros_like(ranges, bool)
    physical_counts = np.zeros(count, np.int16)
    clutter_counts = np.zeros(count, np.int16)
    previous_clusters = None
    previous_clip = None
    for frame, depth in enumerate(depths):
        clusters = raw_clusters(depth, horizontal_fov_deg)
        if previous_clip != clips[frame]:
            previous_clusters = None
        cells: list[list[tuple[float, float, float]]] = [[] for _ in AZIMUTH_CENTERS]
        for cell, rows in enumerate(clusters):
            for source_range, support in rows:
                probability = float(np.clip(0.96 - 0.045 * source_range + 0.02 * np.log1p(support), 0.55, 0.96))
                if rng.random() > probability:
                    continue
                radial_velocity = math.nan
                if previous_clusters is not None and previous_clusters[cell]:
                    prior_range = min(previous_clusters[cell], key=lambda row: abs(row[0] - source_range))[0]
                    if abs(prior_range - source_range) <= 0.55:
                        radial_velocity = (source_range - prior_range) / DT_S
                measured_range = round((source_range + rng.normal(0.0, 0.06)) / 0.05) * 0.05
                measured_angle = round((AZIMUTH_CENTERS[cell] + rng.normal(0.0, 2.0)) / 5.0) * 5.0
                measured_velocity = (math.nan if not math.isfinite(radial_velocity) else
                                     round((radial_velocity + rng.normal(0.0, 0.08)) / 0.1) * 0.1)
                cells[cell].append((max(0.4, measured_range), measured_velocity, measured_angle))
                physical_counts[frame] += 1
        for _ in range(int(rng.poisson(0.16))):
            angle = rng.uniform(-45.0, 45.0)
            cell = int(np.argmin(np.abs(AZIMUTH_CENTERS - angle)))
            clutter_range = round(rng.uniform(0.6, 6.0) / 0.05) * 0.05
            clutter_velocity = rng.normal(0.0, 0.18) if rng.random() < 0.75 else rng.uniform(-0.9, -0.2)
            cells[cell].append((clutter_range, round(clutter_velocity / 0.1) * 0.1,
                                round(angle / 5.0) * 5.0))
            clutter_counts[frame] += 1
        cursor = 0
        for rows in cells:
            for measured_range, velocity, angle in merge_cell(rows):
                if cursor >= MAX_RETURNS:
                    break
                ranges[frame, cursor] = measured_range
                velocities[frame, cursor] = 0.0 if not math.isfinite(velocity) else velocity
                azimuths[frame, cursor] = angle
                valid[frame, cursor] = math.isfinite(velocity)
                cursor += 1
        previous_clusters, previous_clip = clusters, clips[frame]
    return {'range_m': ranges, 'radial_velocity_mps': velocities,
            'azimuth_deg': azimuths, 'valid': valid,
            'physical_candidates': physical_counts, 'injected_clutter': clutter_counts}


def expert(observations: dict[str, np.ndarray], clips: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = (observations['valid'] & (observations['range_m'] < 3.18) &
                 (observations['radial_velocity_mps'] <= -0.35) &
                 (np.abs(observations['azimuth_deg']) <= 20.0))
    raw = candidate.any(axis=1)
    alert = np.zeros(len(clips), bool)
    ttc = np.full(len(clips), np.nan, np.float64)
    active = False
    empty_run = 0
    for index in range(len(clips)):
        if index == 0 or clips[index] != clips[index - 1]:
            active, empty_run = False, 0
        start = index
        while start > 0 and clips[start - 1] == clips[index] and index - start + 1 < 3:
            start -= 1
        if not active and raw[start:index + 1].sum() >= 2:
            active = True
        empty_run = 0 if raw[index] else empty_run + 1
        if active and empty_run >= 2:
            active = False
        alert[index] = active
        rows = np.flatnonzero(candidate[index])
        if rows.size:
            values = observations['range_m'][index, rows] / -observations['radial_velocity_mps'][index, rows]
            ttc[index] = float(np.min(values))
    return alert, ttc, raw


def confusion(truth: np.ndarray, known: np.ndarray, alert: np.ndarray) -> dict:
    return {'TP': int((truth & known & alert).sum()), 'FP': int((~truth & known & alert).sum()),
            'FN': int((truth & known & ~alert).sum()), 'TN': int((~truth & known & ~alert).sum())}


def episodes(values: np.ndarray) -> int:
    values = np.asarray(values, bool)
    return int((values & ~np.r_[False, values[:-1]]).sum())


def run(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    allowed = (root / 'artifacts.local/work' / TASK).resolve()
    if output.parent != allowed or output.exists():
        raise ValueError('output must be a new direct child of the MZ83 artifact root')
    output.mkdir(parents=True)
    started = time.perf_counter()
    inputs = {}

    def bind(path: Path, expected: str | None = None) -> Path:
        path = path.resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'hash mismatch: {path}')
        inputs[str(path)] = digest
        return path

    here = Path(__file__).resolve().parent
    shutil.copyfile(bind(here / Path(__file__).name), output / Path(__file__).name)
    prior = root / 'artifacts.local/work/mz77-approach-sequence-20260911'
    sensor_receipt = json.loads(bind(prior / 'sensor-v1/receipt.json').read_text(encoding='utf-8'))
    manifest_path = bind(prior / 'sensor-v1/predictor.json', sensor_receipt['outputs']['predictor.json'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    clips = np.asarray([row['clip_id'] for row in manifest['frames']])
    capture = prior / 'returned-v1/capture-v1/evaluator/native'
    depths = [np.load(bind(capture / f'{index:04d}.npy'), allow_pickle=False) for index in range(160)]
    observations = simulate(depths, clips, manifest['calibration']['horizontal_fov_degrees'],
                            np.random.default_rng(SEED))
    packet_path = output / 'radar_observations.npz'
    np.savez_compressed(packet_path, clip=clips, **observations)
    write(output / 'sensor-receipt.json', {
        'status': 'SEALED_BEFORE_EVALUATOR_OPEN', 'frames': 160,
        'native_depth_reads': 160, 'source_spec_reads': 0, 'target_id_reads': 0,
        'evaluator_label_reads': 0, 'seed': SEED, 'packet_sha256': sha(packet_path),
        'forward_model': {'azimuth_cells_deg': AZIMUTH_CENTERS.tolist(),
                          'vertical_aperture_fraction': VERTICAL_APERTURE,
                          'range_source_bin_m': 0.25, 'range_quantization_m': 0.05,
                          'range_noise_sigma_m': 0.06, 'velocity_quantization_mps': 0.1,
                          'velocity_noise_sigma_mps': 0.08, 'merge_range_m': 0.35,
                          'clutter_poisson_lambda': 0.16, 'max_returns': MAX_RETURNS},
        'claim': 'Uncalibrated radar-like sensitivity model; not radar physics or hardware evidence.'})
    del depths

    with np.load(packet_path, allow_pickle=False) as values:
        sealed = {name: np.asarray(values[name]) for name in values.files if name != 'clip'}
    radar_alert, radar_ttc, raw = expert(sealed, clips)
    predictor_path = output / 'radar_predictions.npz'
    np.savez_compressed(predictor_path, clip=clips, generic_hazard=radar_alert,
                        raw_candidate=raw, ttc_s=radar_ttc)
    write(output / 'predictor-receipt.json', {
        'status': 'SEALED_BEFORE_EVALUATOR_OPEN', 'training_steps': 0,
        'future_frames': 0, 'threshold_searches': 0,
        'expert': {'range_max_m': 3.18, 'closing_velocity_max_mps': -0.35,
                   'corridor_abs_azimuth_max_deg': 20.0,
                   'activation': '2 candidate frames in causal last 3',
                   'release': '2 consecutive empty frames'},
        'predictions_sha256': sha(predictor_path), 'evaluator_reads': 0})

    mz79 = root / 'artifacts.local/work/mz79-tof-range-cap-20260912/run-v1'
    receipt79 = json.loads(bind(mz79 / 'receipt.json').read_text(encoding='utf-8'))
    predictions79_path = bind(mz79 / 'predictions.npz', receipt79['outputs']['predictions.npz'])
    evaluator_path = bind(prior / 'score-v1/evaluator.npz')
    with np.load(evaluator_path, allow_pickle=False) as values:
        query_truth, query_known = np.asarray(values['truth'], bool), np.asarray(values['known'], bool)
        front_distance = np.asarray(values['front_distance_m'], float)
    generic_truth, generic_known = query_truth.any(axis=1), query_known.all(axis=1)
    standalone = confusion(generic_truth, generic_known, radar_alert)
    result = {'status': 'CONSUMED_DEVELOPMENT', 'standalone_radar_generic': standalone,
              'profiles': {}, 'attribution_policy': 'Radar-only rescue is HEIGHT_UNKNOWN; ToF query attribution unchanged.'}
    with np.load(predictions79_path, allow_pickle=False) as values:
        for profile in PROFILES:
            tof = (np.asarray(values[profile + '/GEOMETRY_TEMPORAL']) >= 0).any(axis=1)
            tof_metrics = confusion(generic_truth, generic_known, tof)
            combined = tof | radar_alert
            rescue = generic_truth & generic_known & ~tof & radar_alert
            added = ~generic_truth & generic_known & ~tof & radar_alert
            required = math.ceil(0.25 * tof_metrics['FN'])
            result['profiles'][profile] = {
                'tof_generic': tof_metrics, 'rescued_tp_frames': int(rescue.sum()),
                'added_fp_frames': int(added.sum()), 'required_25pct_rescue': required,
                'fixed_tof_or_radar_generic': confusion(generic_truth, generic_known, combined),
                'passes': int(rescue.sum()) >= required and int(added.sum()) <= 5,
                'radar_only_height_unknown_frames': int(rescue.sum())}
    clip_metrics = {}
    for clip in ('head_bar', 'thin_pole', 'wall', 'control'):
        rows = clips == clip
        positive = rows & generic_truth & generic_known
        true_alerts = positive & radar_alert
        indices = np.flatnonzero(true_alerts)
        clip_ttc = radar_ttc[true_alerts & np.isfinite(radar_ttc)]
        reference = front_distance[true_alerts & np.isfinite(radar_ttc)]
        clip_metrics[clip] = {
            'TP': int(true_alerts.sum()), 'FP': int((rows & ~generic_truth & generic_known & radar_alert).sum()),
            'FN': int((positive & ~radar_alert).sum()),
            'first_alert_front_distance_m': None if not indices.size else float(front_distance[indices[0]]),
            'ttc_mae_s': None if not clip_ttc.size else float(np.mean(np.abs(clip_ttc - reference))),
            'alert_episodes_during_positive': episodes(radar_alert[positive]),
            'truth_positive_frames': int(positive.sum())}
    result['clips'] = clip_metrics
    result['packet_diagnostics'] = {
        'valid_returns': int(sealed['valid'].sum()),
        'frames_with_valid_return': int(sealed['valid'].any(axis=1).sum()),
        'injected_clutter': int(sealed['injected_clutter'].sum()),
        'raw_hazard_frames': int(raw.sum()), 'hysteresis_hazard_frames': int(radar_alert.sum())}
    result['all_profiles_pass'] = all(row['passes'] for row in result['profiles'].values())
    result['decision'] = ('RADAR_INFORMATION_CANARY_PASSES_LOW_FP_RESCUE' if result['all_profiles_pass']
                          else 'COARSE_RADAR_INFORMATION_CANARY_GATE_NOT_MET')
    write(output / 'result.json', result)
    for path, digest in inputs.items():
        if sha(Path(path)) != digest:
            raise ValueError(f'input changed during run: {path}')
    write(output / 'receipt.json', {
        'status': 'PASS', 'phase': 'EXPLORE_CONSUMED_DEVELOPMENT', 'frames': 160,
        'training_steps': 0, 'new_deployable_cutoffs': 0, 'input_hashes': inputs,
        'outputs': {path.name: sha(path) for path in output.iterdir() if path.is_file()},
        'backend': 'CPU TASK_NOT_GPU_SUITABLE', 'seconds': time.perf_counter() - started,
        'claim': 'Uncalibrated radar-like information sensitivity on consumed posed simulation; no hardware, alert, user-benefit or safety claim.'})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
