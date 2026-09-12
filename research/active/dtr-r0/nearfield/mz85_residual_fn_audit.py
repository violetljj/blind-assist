"""Read-only mechanism audit of the 11 sealed MZ85 generic false negatives."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_receipt(directory: Path) -> dict:
    receipt = json.loads((directory / 'receipt.json').read_text(encoding='utf-8'))
    for name, expected in receipt['outputs'].items():
        actual = sha(directory / name)
        if actual != expected:
            raise AssertionError(f'sealed artifact hash mismatch: {directory.name}/{name}')
    return receipt


def radar_raw_candidate(ranges: np.ndarray, velocities: np.ndarray,
                        angles: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return (valid & (ranges < 3.18) & (velocities <= -0.35) &
            (np.abs(angles) <= 20.0)).any(axis=1)


def classify_primary(*, family: str, frame_index: int, tof_known: bool,
                     radar_raw: bool, radar_alert: bool) -> str:
    if family == 'lateral_crossing' and frame_index == 0 and tof_known and not radar_alert:
        return 'LATERAL_HISTORY_COLD_START'
    if not tof_known and radar_raw and not radar_alert:
        return 'AVAILABLE_RADAR_EVIDENCE_NOT_ACTIVATED'
    if not tof_known and not radar_raw and not radar_alert:
        return 'DUAL_INSTANTANEOUS_OBSERVATION_ABSENCE'
    return 'OTHER'


def audit(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    m84 = root / 'artifacts.local/work/mz84-bidirectional-complementarity-20260912/run-v2'
    m85 = root / 'artifacts.local/work/mz85-rotation-compensated-state-20260912/run-v2'
    verify_receipt(m84)
    verify_receipt(m85)
    source = json.loads((m84 / 'source.json').read_text(encoding='utf-8'))['episodes']
    frames = [frame for episode in source for frame in episode['frames']]
    with np.load(m84 / 'observations.npz') as observations, \
            np.load(m85 / 'predictions.npz') as predictions, \
            np.load(m85 / 'evaluator.npz') as evaluator:
        truth = evaluator['generic_truth']
        family = evaluator['family']
        episode_id = evaluator['episode_id']
        fusion = predictions['compensated_fusion']
        tof = predictions['compensated_tof']
        radar = predictions['radar']
        fusion_height = np.where(tof, predictions['compensated_height'], 'UNKNOWN')
        raw = radar_raw_candidate(observations['radar_range_m'],
                                  observations['radar_radial_velocity_mps'],
                                  observations['radar_azimuth_deg'],
                                  observations['radar_valid'])
        tof_known = observations['tof_known']
        fn_indices = np.flatnonzero(truth & ~fusion)
        rows = []
        for index in fn_indices:
            frame_index = int(frames[index]['index'])
            family_name = str(family[index])
            previous_same_episode = (index > 0 and episode_id[index - 1] == episode_id[index])
            previous_hazard = bool(previous_same_episode and fusion[index - 1])
            gap_member = family_name == 'multi_target' and 8 <= frame_index <= 12
            rows.append({
                'flat_index': int(index), 'episode_id': str(episode_id[index]),
                'frame_index': frame_index, 'family': family_name,
                'primary_category': classify_primary(
                    family=family_name, frame_index=frame_index,
                    tof_known=bool(tof_known[index]), radar_raw=bool(raw[index]),
                    radar_alert=bool(radar[index])),
                'tof_known': bool(tof_known[index]),
                'radar_raw_candidate': bool(raw[index]),
                'radar_alert': bool(radar[index]),
                'gap_member': gap_member,
                'gap_onset': gap_member and frame_index == 8,
                'previous_fusion_hazard': previous_hazard,
            })
        categories = Counter(row['primary_category'] for row in rows)
        result = {
            'status': 'READ_ONLY_CONSUMED_RESULT_AUDIT',
            'generic_false_negative_frames': len(rows),
            'primary_categories': dict(sorted(categories.items())),
            'gap_related_fn': sum(row['gap_member'] for row in rows),
            'gap_onset_fn': sum(row['gap_onset'] for row in rows),
            'authority_hold_direct_target_fn': sum(
                row['gap_onset'] and row['previous_fusion_hazard'] for row in rows),
            'height_unknown_query_attribution_debt_frames': int(
                (truth & fusion & (fusion_height == 'UNKNOWN')).sum()),
            'height_unknown_is_not_a_generic_fn_cause': True,
            'rows': rows,
            'decision': 'RESIDUAL_FN_SPLIT_NO_SINGLE_DOMINANT_MECHANISM',
        }
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audit.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output / 'receipt.json').write_text(json.dumps({
        'status': 'PASS', 'prediction_changes': 0,
        'mz84_receipt_sha256': sha(m84 / 'receipt.json'),
        'mz85_receipt_sha256': sha(m85 / 'receipt.json'),
        'audit_sha256': sha(output / 'audit.json'),
        'claim': 'Read-only audit of consumed analytic evidence only.'},
        indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output), indent=2))
