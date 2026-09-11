"""MZ80 observability-conditioned rescue and posthoc score-ceiling diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from tof_range_cap_sensitivity import PROFILES

TASK = 'mz80-observability-rescue-20260912'
QUERY_END_M = np.array((1.68, 3.18, 1.63, 3.13), dtype=np.float64)
FP_BUDGETS = (0, 5, 10, 20, 40, 64)
FIVE_KLUX_TYPICAL = ('5KLUX_WHITE88_TYP', '5KLUX_LIGHTGRAY54_TYP', '5KLUX_GRAY17_TYP')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def observability_unknown(inner_m: float, corner_m: float) -> np.ndarray:
    """Whether the conservative cap cannot cover an entire fixed query range."""
    if not (0 < corner_m <= inner_m <= 4.0):
        raise ValueError('expected 0 < corner <= inner <= 4 m')
    return min(inner_m, corner_m) < QUERY_END_M


def conditioned_rescue(tof_margin: np.ndarray, rgb_margin: np.ndarray,
                       unknown_query: np.ndarray, threshold: float) -> np.ndarray:
    tof_margin, rgb_margin = np.asarray(tof_margin), np.asarray(rgb_margin)
    unknown_query = np.asarray(unknown_query, dtype=bool)
    if tof_margin.shape != rgb_margin.shape or tof_margin.ndim != 2 or tof_margin.shape[1] != 4:
        raise ValueError('ToF/RGB margins must have matching [N,4] shapes')
    if unknown_query.shape != (4,):
        raise ValueError('unknown_query must be [4]')
    tof_alert = tof_margin >= 0
    return tof_alert | ((~tof_alert) & unknown_query[None, :] & (rgb_margin >= threshold))


def confusion(truth: np.ndarray, known: np.ndarray, alert: np.ndarray) -> dict:
    truth, known, alert = (np.asarray(x, dtype=bool) for x in (truth, known, alert))
    if truth.shape != known.shape or truth.shape != alert.shape:
        raise ValueError('truth/known/alert shapes must match')
    return {
        'TP': int((truth & known & alert).sum()),
        'FP': int((~truth & known & alert).sum()),
        'FN': int((truth & known & ~alert).sum()),
        'TN': int((~truth & known & ~alert).sum()),
    }


def oracle_curve(tof_margin: np.ndarray, rgb_margin: np.ndarray, unknown_query: np.ndarray,
                 truth: np.ndarray, known: np.ndarray, budgets=FP_BUDGETS) -> dict:
    """Posthoc ceiling only; returned thresholds are forbidden deployment cuts."""
    tof_alert = np.asarray(tof_margin) >= 0
    rgb_margin = np.asarray(rgb_margin)
    unknown = np.broadcast_to(np.asarray(unknown_query, bool), tof_alert.shape)
    opportunity = ~tof_alert & unknown
    candidates = np.r_[np.unique(rgb_margin[opportunity]), np.inf]
    rows = []
    for threshold in candidates:
        added = opportunity & (rgb_margin >= threshold)
        rows.append({
            'threshold': float(threshold) if np.isfinite(threshold) else None,
            'rescued_tp': int((truth & known & added).sum()),
            'added_fp': int((~truth & known & added).sum()),
        })
    result = {}
    for budget in budgets:
        eligible = [row for row in rows if row['added_fp'] <= budget]
        best = max(eligible, key=lambda row: (row['rescued_tp'], -row['added_fp'],
                                              row['threshold'] if row['threshold'] is not None else math.inf))
        result[str(budget)] = best
    return result


def run(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    allowed = (root / 'artifacts.local' / 'work' / TASK).resolve()
    if output.parent != allowed or output.exists():
        raise ValueError('output must be a new direct child of the MZ80 artifact root')
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
    for name in ('mz80_observability_rescue.py', 'tof_range_cap_sensitivity.py'):
        shutil.copyfile(bind(here / name), output / name)
    mz79 = root / 'artifacts.local' / 'work' / 'mz79-tof-range-cap-20260912' / 'run-v1'
    mz79_receipt_path = bind(mz79 / 'receipt.json')
    mz79_receipt = json.loads(mz79_receipt_path.read_text(encoding='utf-8'))
    predictions_path = bind(mz79 / 'predictions.npz', mz79_receipt['outputs']['predictions.npz'])
    evaluator_path = root / 'artifacts.local' / 'work' / 'mz77-approach-sequence-20260911' / 'score-v1' / 'evaluator.npz'
    with np.load(predictions_path, allow_pickle=False) as z:
        predictions = dict(z)
    # This experiment is explicitly posthoc: labels are used to estimate the
    # best possible threshold ceiling, never to select a deployable cutoff.
    bind(evaluator_path)
    with np.load(evaluator_path, allow_pickle=False) as z:
        truth, known = np.asarray(z['truth'], bool), np.asarray(z['known'], bool)
    result = {
        'status': 'POSTHOC_DIAGNOSTIC_ONLY',
        'formula': 'ToF OR (U_ToF AND RGB_margin>=threshold)',
        'unknown_rule': 'minimum profile range cap is shorter than full fixed query interval',
        'query_end_m': QUERY_END_M.tolist(),
        'fixed_threshold': 0.0,
        'profiles': {},
    }
    for profile, (inner_m, corner_m) in PROFILES.items():
        tof = predictions[profile + '/GEOMETRY_TEMPORAL']
        rgb = predictions[profile + '/RGB']
        unknown = observability_unknown(inner_m, corner_m)
        tof_alert = tof >= 0
        fixed = conditioned_rescue(tof, rgb, unknown, 0.0)
        base = confusion(truth, known, tof_alert)
        combined = confusion(truth, known, fixed)
        fn = base['FN']
        required = math.ceil(0.25 * fn)
        curve = oracle_curve(tof, rgb, unknown, truth, known)
        result['profiles'][profile] = {
            'unknown_queries': unknown.tolist(),
            'tof': base,
            'fixed_conditioned_rescue': combined,
            'fixed_increment': {'rescued_tp': combined['TP'] - base['TP'],
                                'added_fp': combined['FP'] - base['FP']},
            'oracle_ceiling_by_added_fp_budget': curve,
            'required_rescue_at_25pct_of_tof_fn': required,
            'meets_25pct_rescue_at_added_fp_le_5': curve['5']['rescued_tp'] >= required,
        }
    result['clean_exact_parity'] = (
        result['profiles']['IDEAL_4M']['fixed_conditioned_rescue'] ==
        result['profiles']['IDEAL_4M']['tof']
    )
    result['five_klux_typical_all_pass'] = all(
        result['profiles'][profile]['meets_25pct_rescue_at_added_fp_le_5']
        for profile in FIVE_KLUX_TYPICAL
    )
    result['decision'] = 'CURRENT_RGB_SCORE_NO_SELECTIVE_RESCUE' if not result['five_klux_typical_all_pass'] else 'CURRENT_RGB_SCORE_HAS_RESCUE_CEILING'
    write(output / 'result.json', result)
    write(output / 'receipt.json', {
        'status': 'PASS', 'phase': 'EXPLORE_POSTHOC_DIAGNOSTIC',
        'frames': 160, 'training_steps': 0, 'new_deployable_cutoffs': 0,
        'input_hashes': inputs,
        'outputs': {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        'backend': 'CPU TASK_NOT_GPU_SUITABLE', 'seconds': time.perf_counter() - started,
        'claim': 'Consumed-simulation score ceiling only; oracle thresholds are not deployable results.',
    })
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
