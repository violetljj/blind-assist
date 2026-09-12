"""MZ82 causal explicit-RGB-motion canary on consumed MZ77 Development."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

TASK = 'mz82-temporal-visual-canary-20260912'
PROFILES = ('5KLUX_WHITE88_TYP', '5KLUX_LIGHTGRAY54_TYP', '5KLUX_GRAY17_TYP')
QUERY_NAMES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
QUERY_END_M = np.array((1.68, 3.18, 1.63, 3.13), dtype=np.float64)
FP_BUDGETS = (0, 5, 10, 20, 40)
FLOW_SIZE = (320, 180)
DT_S = 0.1
WINDOW = 5
BANDS = {
    'BODY': (0.25, 0.75, 0.40, 0.95),
    'HEAD': (0.25, 0.75, 0.15, 0.55),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def band_features(gray: np.ndarray, flow: np.ndarray, band: tuple[float, ...]) -> tuple[float, float, float]:
    height, width = gray.shape
    x0, x1 = int(width * band[0]), int(width * band[1])
    y0, y1 = int(height * band[2]), int(height * band[3])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - width * 0.5, yy - height * 0.5
    radius = np.hypot(dx, dy)
    local = flow[y0:y1, x0:x1]
    outward = (local[..., 0] * dx + local[..., 1] * dy) / np.maximum(radius, 4.0)
    expansion = outward / np.maximum(radius, 8.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)[y0:y1, x0:x1]
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)[y0:y1, x0:x1]
    edge = np.hypot(gx, gy)
    eligible = (edge >= 12.0) & (radius >= 8.0) & np.isfinite(expansion)
    positive = eligible & (outward > 0.0)
    if positive.sum() < 32:
        return 0.0, 0.0, math.inf
    flow_p90 = float(np.quantile(outward[positive], 0.90))
    expansion_p75 = float(np.quantile(expansion[positive], 0.75))
    rates = expansion[positive]
    rates = rates[rates > 1e-4]
    ttc = math.inf if rates.size < 16 else float(DT_S / np.quantile(rates, 0.75))
    return flow_p90, expansion_p75, ttc


def temporal_scores(flow: np.ndarray, expansion: np.ndarray, ttc: np.ndarray,
                    clips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(clips)
    stability = np.zeros((count, 2), np.float64)
    smooth_flow = np.zeros((count, 2), np.float64)
    smooth_expansion = np.zeros((count, 2), np.float64)
    smooth_ttc = np.full((count, 2), np.inf, np.float64)
    for index in range(count):
        start = index
        while start > 0 and clips[start - 1] == clips[index] and index - start + 1 < WINDOW:
            start -= 1
        rows = slice(start, index + 1)
        smooth_flow[index] = np.median(flow[rows], axis=0)
        smooth_expansion[index] = np.median(expansion[rows], axis=0)
        stability[index] = np.mean(expansion[rows] > 0.002, axis=0)
        for band in range(2):
            finite = ttc[rows, band][np.isfinite(ttc[rows, band])]
            if finite.size:
                smooth_ttc[index, band] = float(np.median(finite))
    evidence = np.log1p(np.maximum(smooth_flow, 0.0)) + np.log1p(
        20.0 * np.maximum(smooth_expansion, 0.0)) + 0.5 * stability
    scores = np.full((count, 4), -1e9, np.float64)
    for query, (band, lower, upper) in enumerate(((0, 0.0, 1.68), (0, 1.68, 3.18),
                                                  (1, 0.0, 1.63), (1, 1.63, 3.13))):
        values = smooth_ttc[:, band]
        distance = np.where(values < lower, lower - values, np.where(values > upper, values - upper, 0.0))
        distance[~np.isfinite(values)] = 20.0
        midpoint_penalty = np.zeros(count) if lower == 0.0 else 0.20 * np.abs(values - (lower + upper) * 0.5)
        midpoint_penalty[~np.isfinite(midpoint_penalty)] = 20.0
        scores[:, query] = evidence[:, band] - 1.5 * distance - midpoint_penalty
    return scores, np.stack((smooth_flow, smooth_expansion, smooth_ttc, stability), axis=1)


def oracle(scores: np.ndarray, opportunity: np.ndarray, truth: np.ndarray,
           known: np.ndarray) -> dict:
    mask = opportunity & known
    candidates = np.r_[np.unique(scores[mask]), np.inf]
    rows = []
    for threshold in candidates:
        selected = opportunity & (scores >= threshold)
        rows.append({'threshold': None if not np.isfinite(threshold) else float(threshold),
                     'rescued_tp': int((truth & known & selected).sum()),
                     'added_fp': int((~truth & known & selected).sum())})
    answer = {}
    for budget in FP_BUDGETS:
        eligible = [row for row in rows if row['added_fp'] <= budget]
        answer[str(budget)] = max(eligible, key=lambda row: (row['rescued_tp'], -row['added_fp']))
    return answer


def selection_taxonomy(scores: np.ndarray, opportunity: np.ndarray, truth: np.ndarray,
                       known: np.ndarray, clips: np.ndarray, threshold: float | None) -> dict:
    selected = np.zeros_like(opportunity) if threshold is None else opportunity & (scores >= threshold)
    by_query = {}
    for query, name in enumerate(QUERY_NAMES):
        by_query[name] = {
            'rescued_tp': int((selected[:, query] & truth[:, query] & known[:, query]).sum()),
            'added_fp': int((selected[:, query] & ~truth[:, query] & known[:, query]).sum()),
        }
    by_clip = {}
    for clip in np.unique(clips):
        rows = clips == clip
        by_clip[str(clip)] = {
            'rescued_tp': int((selected[rows] & truth[rows] & known[rows]).sum()),
            'added_fp': int((selected[rows] & ~truth[rows] & known[rows]).sum()),
        }
    return {'by_query': by_query, 'by_clip': by_clip}


def run(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    allowed = (root / 'artifacts.local' / 'work' / TASK).resolve()
    if output.parent != allowed or output.exists():
        raise ValueError('output must be a new direct child of the MZ82 artifact root')
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
    prior = root / 'artifacts.local' / 'work' / 'mz77-approach-sequence-20260911'
    sensor_receipt = json.loads(bind(prior / 'sensor-v1/receipt.json').read_text(encoding='utf-8'))
    predictor_path = bind(prior / 'sensor-v1/predictor.json', sensor_receipt['outputs']['predictor.json'])
    predictor = json.loads(predictor_path.read_text(encoding='utf-8'))
    clips = np.asarray([row['clip_id'] for row in predictor['frames']])
    flow_values = np.zeros((160, 2), np.float64)
    expansion = np.zeros((160, 2), np.float64)
    ttc = np.full((160, 2), np.inf, np.float64)
    previous = None
    previous_clip = None
    for index, row in enumerate(predictor['frames']):
        image_path = bind(Path(row['rgb_path']), row['rgb_sha256'])
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'failed to read {image_path}')
        gray = cv2.resize(image, FLOW_SIZE, interpolation=cv2.INTER_AREA)
        if previous is not None and previous_clip == row['clip_id']:
            dense = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            for band_index, name in enumerate(('BODY', 'HEAD')):
                flow_values[index, band_index], expansion[index, band_index], ttc[index, band_index] = band_features(
                    gray, dense, BANDS[name])
        previous, previous_clip = gray, row['clip_id']
    scores, smoothed = temporal_scores(flow_values, expansion, ttc, clips)
    feature_path = output / 'features.npz'
    np.savez_compressed(feature_path, clip=clips, raw_flow_p90=flow_values,
                        raw_expansion_p75=expansion, raw_ttc_p25_s=ttc,
                        smoothed=smoothed, scores=scores)
    write(output / 'predictor-receipt.json', {
        'status': 'SEALED_BEFORE_EVALUATOR_OPEN', 'frames': 160,
        'history_frames': WINDOW, 'future_frames': 0, 'training_steps': 0,
        'fitted_parameters': 0, 'feature_constants': {
            'flow_size': FLOW_SIZE, 'dt_s': DT_S, 'bands': BANDS,
            'query_end_m': QUERY_END_M.tolist(), 'farneback': [0.5, 3, 21, 3, 5, 1.2, 0]},
        'features_sha256': sha(feature_path), 'evaluator_reads': 0,
        'claim': 'Causal RGB-only motion scores on consumed posed simulation; no deployable cutoff.'})

    mz79 = root / 'artifacts.local/work/mz79-tof-range-cap-20260912/run-v1'
    receipt79 = json.loads(bind(mz79 / 'receipt.json').read_text(encoding='utf-8'))
    predictions_path = bind(mz79 / 'predictions.npz', receipt79['outputs']['predictions.npz'])
    evaluator_path = bind(prior / 'score-v1/evaluator.npz')
    with np.load(evaluator_path, allow_pickle=False) as values:
        truth, known = np.asarray(values['truth'], bool), np.asarray(values['known'], bool)
    result = {'status': 'POSTHOC_DIAGNOSTIC_ONLY', 'profiles': {},
              'primary': 'rescued TP at added FP <= 5', 'scores_selected_on_labels': False}
    with np.load(predictions_path, allow_pickle=False) as values:
        for profile in PROFILES:
            baseline = np.asarray(values[profile + '/GEOMETRY_TEMPORAL']) >= 0
            opportunity = ~baseline
            curve = oracle(scores, opportunity, truth, known)
            tof_fn = int((truth & known & opportunity).sum())
            at_five = curve['5']
            result['profiles'][profile] = {
                'tof_fn': tof_fn, 'required_25pct_rescue': math.ceil(0.25 * tof_fn),
                'oracle_ceiling_by_added_fp_budget': curve,
                'selection_taxonomy_at_5fp': selection_taxonomy(
                    scores, opportunity, truth, known, clips, at_five['threshold']),
                'meets_canary_10tp_at_5fp': curve['5']['rescued_tp'] >= 10,
                'meets_25pct_at_5fp': curve['5']['rescued_tp'] >= math.ceil(0.25 * tof_fn)}
    best = max(row['oracle_ceiling_by_added_fp_budget']['5']['rescued_tp'] for row in result['profiles'].values())
    result['decision'] = ('TEMPORAL_VISUAL_CANARY_HAS_LOW_FP_TAIL' if best >= 10
                          else 'EXPLICIT_TEMPORAL_VISUAL_CANARY_NO_LOW_FP_TAIL')
    write(output / 'result.json', result)
    for path, digest in inputs.items():
        if sha(Path(path)) != digest:
            raise ValueError(f'input changed during run: {path}')
    write(output / 'receipt.json', {
        'status': 'PASS', 'phase': 'EXPLORE_POSTHOC_DIAGNOSTIC', 'frames': 160,
        'training_steps': 0, 'new_deployable_cutoffs': 0, 'input_hashes': inputs,
        'outputs': {path.name: sha(path) for path in output.iterdir() if path.is_file()},
        'backend': f'CPU TASK_NOT_GPU_SUITABLE; OpenCV {cv2.__version__}',
        'seconds': time.perf_counter() - started,
        'claim': 'Consumed-simulation explicit-motion score ceiling only; no source-separated, hardware, alert, user-benefit or safety claim.'})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
