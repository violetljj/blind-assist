"""MZ81 frozen-RGB conditional collision probe with source-disjoint testing.

MZ61 trains a 4.7k-parameter readout, MZ67 selects one epoch and one global
threshold, and MZ77 is opened only for final scoring. The RGB backbone, ToF
baseline, MZ80 observability gate, and MZ79 packets remain frozen.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from mz5_ensemble_readout import read, sha
from mz54_full_rgb import parameters_sha
from mz54_full_rgb_source import RGBStore
from mz74_return_survival import SOURCES, setup, load_models, DesignBindings, DESIGN_SHA
from mz76_observed_reliability import open_images
from mz76_replay_diagnostic import nchw_features
from mz80_observability_rescue import observability_unknown, QUERY_END_M
from tof_range_cap_sensitivity import PROFILES

TASK = 'mz81-conditional-feature-probe-20260912'
QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
TEST_PROFILES = ('5KLUX_WHITE88_TYP', '5KLUX_LIGHTGRAY54_TYP', '5KLUX_GRAY17_TYP')
POOL_SHAPE = (3, 3)
HIDDEN = 8
EPOCHS = 400
SEED = 81
TEST_FP_BUDGET = 5


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def average_precision(scores: np.ndarray, truth: np.ndarray) -> float:
    scores = np.asarray(scores, np.float64).reshape(-1)
    truth = np.asarray(truth, bool).reshape(-1)
    positives = int(truth.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind='stable')
    ranked = truth[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def select_threshold(scores: np.ndarray, truth: np.ndarray, fp_budget: int) -> dict:
    """Best global score cutoff under a fixed bit-level false-positive budget."""
    scores = np.asarray(scores, np.float64).reshape(-1)
    truth = np.asarray(truth, bool).reshape(-1)
    rows = []
    for threshold in np.r_[np.unique(scores), np.inf]:
        selected = scores >= threshold
        rows.append((int((selected & truth).sum()), int((selected & ~truth).sum()), float(threshold)))
    tp, fp, threshold = max((r for r in rows if r[1] <= fp_budget),
                            key=lambda r: (r[0], -r[1], r[2]))
    return {'threshold': None if math.isinf(threshold) else threshold, 'TP': tp, 'FP': fp}


class TinyProbe(nn.Module):
    def __init__(self, inputs: int = 64 * 3 * 3, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, value):
        return self.net(value)


def pooled_features(images, models) -> torch.Tensor:
    dense = nchw_features(images, models)
    return F.adaptive_avg_pool2d(dense, POOL_SHAPE).flatten(1)


def source_features(data, models, store, progress_name: str) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for begin in range(0, len(data['rgb_refs']), 16):
            ids = np.arange(begin, min(begin + 16, len(data['rgb_refs'])))
            images = open_images([data['rgb_refs'][int(i)] for i in ids], store)
            try:
                chunks.append(pooled_features(images, models).cpu().numpy())
            finally:
                for image in images:
                    image.close()
            if begin % 512 == 0 or begin + len(ids) == len(data['rgb_refs']):
                print('FEATURES', progress_name, begin + len(ids), flush=True)
    return np.concatenate(chunks).astype(np.float32)


def test_features(manifest: dict, models, bind) -> np.ndarray:
    chunks = []
    rows = manifest['frames']
    with torch.inference_mode():
        for begin in range(0, len(rows), 16):
            images = []
            try:
                for row in rows[begin:begin + 16]:
                    path = bind(Path(row['rgb_path']), row['rgb_sha256'])
                    with Image.open(path) as image:
                        images.append(image.convert('RGB'))
                chunks.append(pooled_features(images, models).cpu().numpy())
            finally:
                for image in images:
                    image.close()
            print('FEATURES mz77', begin + len(images), flush=True)
    return np.concatenate(chunks).astype(np.float32)


def fit_probe(train_x, train_y, validation_x, validation_y, validation_fp_budget):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mean = train_x.mean(0, dtype=np.float64).astype(np.float32)
    std = train_x.std(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-4)
    tx = torch.from_numpy((train_x - mean) / std).cuda()
    ty = torch.from_numpy(train_y.astype(np.float32)).cuda()
    vx = torch.from_numpy((validation_x - mean) / std).cuda()
    model = TinyProbe(train_x.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    positive_weight = torch.from_numpy((~train_y).sum(0) / train_y.sum(0)).float().cuda()
    best = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(tx), ty, pos_weight=positive_weight)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0:
            model.eval()
            with torch.inference_mode():
                scores = model(vx).cpu().numpy()
            cut = select_threshold(scores, validation_y, validation_fp_budget)
            row = (cut['TP'], -cut['FP'], average_precision(scores, validation_y), -epoch)
            if best is None or row > best['key']:
                best = {'key': row, 'epoch': epoch, 'loss': float(loss.item()),
                        'cut': cut, 'average_precision': row[2],
                        'state': copy.deepcopy(model.state_dict())}
    model.load_state_dict(best.pop('state'))
    best.pop('key')
    return model.eval(), mean, std, best


def rescue_metrics(scores, threshold, tof_margin, truth, known, unknown_query):
    tof_alert = np.asarray(tof_margin) >= 0
    opportunity = (~tof_alert) & np.broadcast_to(np.asarray(unknown_query, bool), tof_alert.shape)
    selected = opportunity & (scores >= threshold)
    oracle = select_threshold(scores[opportunity & known], truth[opportunity & known], TEST_FP_BUDGET)
    tof_fn = int((truth & known & ~tof_alert).sum())
    return {
        'tof_TP': int((truth & known & tof_alert).sum()),
        'tof_FP': int((~truth & known & tof_alert).sum()),
        'tof_FN': tof_fn,
        'rescued_TP': int((truth & known & selected).sum()),
        'added_FP': int((~truth & known & selected).sum()),
        'rescue_fraction_of_tof_FN': float((truth & known & selected).sum() / max(1, (truth & known & ~tof_alert).sum())),
        'conditional_PR_AUC': average_precision(scores[opportunity & known], truth[opportunity & known]),
        'opportunity_bits': int((opportunity & known).sum()),
        'posthoc_oracle_ceiling_at_added_fp_5': oracle,
        'oracle_meets_20pct_rescue': oracle['TP'] >= math.ceil(0.20 * tof_fn),
    }


def run(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    allowed = (root / 'artifacts.local' / 'work' / TASK).resolve()
    if output.parent != allowed or output.exists():
        raise ValueError('output must be a new direct child of the MZ81 artifact root')
    output.mkdir(parents=True)
    started = time.perf_counter()
    work = root / 'artifacts.local' / 'work'
    inputs = {}

    def bind(path: Path, expected: str | None = None) -> Path:
        path = Path(path).resolve(strict=True)
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'hash mismatch: {path}')
        inputs[str(path)] = digest
        return path

    here = Path(__file__).resolve().parent
    for name in ('mz81_conditional_feature_probe.py', 'mz80_observability_rescue.py',
                 'tof_range_cap_sensitivity.py'):
        shutil.copyfile(bind(here / name), output / name)
    design_path = bind(work / 'mz69-topology-transfer-20260911' / 'design-inputs.json', DESIGN_SHA)
    design = read(design_path)
    model_bind = DesignBindings(design)
    model_bind(design_path, DESIGN_SHA)
    store = models = spatial = heads = None
    try:
        setup()
        models, spatial, heads, cuts, checkpoints = load_models(root, model_bind, design, output)
        backbone_before = parameters_sha(models.context.base)
        store = RGBStore()
        datasets = {}
        features = {}
        labels = {}
        metadata = {}
        for source in ('mz61', 'mz67'):
            name, digest, loader = SOURCES[source]
            source_root = work / name
            datasets[source] = loader(source_root, model_bind, digest)[0]
            index = read(bind(source_root / 'source-index.json', digest))
            evaluator_ref = index['combined']['evaluator.npz']
            evaluator_path = bind(source_root / evaluator_ref['path'], evaluator_ref['sha256'])
            metadata_ref = index['combined']['metadata.json']
            metadata_path = bind(source_root / metadata_ref['path'], metadata_ref['sha256'])
            with np.load(evaluator_path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['frame_ids'], datasets[source]['frame_ids'])
                labels[source] = np.asarray(z['truth'], bool)
                assert np.asarray(z['known'], bool).all()
            metadata[source] = read(metadata_path)
            features[source] = source_features(datasets[source], models, store, source)
        manifest_path = work / 'mz77-approach-sequence-20260911' / 'sensor-v1' / 'predictor.json'
        manifest = read(bind(manifest_path))
        features['mz77'] = test_features(manifest, models, bind)
        assert parameters_sha(models.context.base) == backbone_before
        np.savez_compressed(output / 'frozen-features.npz', **features)

        test_evaluator = bind(work / 'mz77-approach-sequence-20260911' / 'score-v1' / 'evaluator.npz')
        with np.load(test_evaluator, allow_pickle=False) as z:
            test_truth = np.asarray(z['truth'], bool)
            test_known = np.asarray(z['known'], bool)
        train_y, validation_y = labels['mz61'], labels['mz67']
        test_negatives = int((~test_truth & test_known).sum())
        validation_negatives = int((~validation_y).sum())
        validation_fp_budget = math.floor(TEST_FP_BUDGET * validation_negatives / test_negatives)
        probe, mean, std, selection = fit_probe(features['mz61'], train_y,
                                                 features['mz67'], validation_y,
                                                 validation_fp_budget)
        torch.save({'state_dict': probe.state_dict(), 'mean': mean, 'std': std,
                    'pool_shape': POOL_SHAPE, 'hidden': HIDDEN, 'queries': QUERIES},
                   output / 'probe.pt')
        with torch.inference_mode():
            test_x = torch.from_numpy((features['mz77'] - mean) / std).cuda()
            test_scores = probe(test_x).cpu().numpy()
        np.savez_compressed(output / 'test-scores.npz', scores=test_scores)

        mz79 = work / 'mz79-tof-range-cap-20260912' / 'run-v1'
        receipt79 = read(bind(mz79 / 'receipt.json'))
        predictions79 = bind(mz79 / 'predictions.npz', receipt79['outputs']['predictions.npz'])
        threshold = selection['cut']['threshold']
        if threshold is None:
            threshold = math.inf
        profiles = {}
        with np.load(predictions79, allow_pickle=False) as z:
            for profile in TEST_PROFILES:
                unknown = observability_unknown(*PROFILES[profile])
                profiles[profile] = rescue_metrics(test_scores, threshold,
                    z[profile + '/GEOMETRY_TEMPORAL'], test_truth, test_known, unknown)
        clean_unknown = observability_unknown(*PROFILES['IDEAL_4M'])
        assert not clean_unknown.any()

        def negative_taxonomy(records, truth):
            visible_nonintruding = sum(row['relation'] == 'VISIBLE_NONINTRUDING' for row in records)
            wrong_band_bits = int(((~truth) & truth.any(1, keepdims=True)).sum())
            return {'visible_nonintruding_frames': visible_nonintruding,
                    'all_collision_negative_frames': int((~truth.any(1)).sum()),
                    'wrong_query_band_bits_lower_bound': wrong_band_bits}

        parameters = sum(p.numel() for p in probe.parameters())
        result = {
            'status': 'SOURCE_DISJOINT_DEVELOPMENT_PROBE',
            'hypothesis': 'Frozen RGB representation contains selectively readable collision-query evidence under U_ToF.',
            'roles': {'train': 'MZ61 geometry families', 'validation_and_cutoff': 'MZ67 disjoint topology/material families',
                      'test_once': 'MZ77 posed approach clips'},
            'probe': {'pool': 'adaptive_average_3x3', 'architecture': '576-8-4 MLP',
                      'parameters': parameters, 'backbone_frozen': True, 'epochs_run': EPOCHS,
                      'selected_epoch': selection['epoch']},
            'hard_negatives': {
                'mz61': negative_taxonomy(metadata['mz61']['records'], train_y),
                'mz67': negative_taxonomy(metadata['mz67']['records'], validation_y),
                'note': 'Far/near and BODY/HEAD negatives are query-specific; MZ77 pre-onset frames are transfer test only.'},
            'threshold_selection': {'test_fp_budget': TEST_FP_BUDGET,
                                    'validation_fp_budget_scaled_by_negative_bits': validation_fp_budget,
                                    'validation_negative_bits': validation_negatives,
                                    'test_negative_bits': test_negatives,
                                    **selection},
            'test_all_bits_PR_AUC': average_precision(test_scores, test_truth),
            'profiles': profiles,
            'clean_exact_parity_by_gate': True,
        }
        passes = [row['added_FP'] <= TEST_FP_BUDGET and
                  row['rescued_TP'] >= math.ceil(0.20 * row['tof_FN'])
                  for row in profiles.values()]
        near_zero = max(row['posthoc_oracle_ceiling_at_added_fp_5']['TP']
                        for row in profiles.values()) <= 2
        result['decision'] = ('REPRESENTATION_READABLE_CONTINUE_RESIDUAL_EXPERT' if all(passes)
                              else 'PAUSE_SINGLE_FRAME_RGB_RESIDUAL' if near_zero
                              else 'WEAK_OR_UNSTABLE_TRANSFER_DO_NOT_SCALE_MODEL')
        write(output / 'result.json', result)
        outputs = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        write(output / 'receipt.json', {
            'status': 'PASS', 'phase': 'EXPLORE_CONSUMED_DEVELOPMENT',
            'frames': {'train': 4096, 'validation': 4096, 'test': 160},
            'feature_dimensions': int(features['mz61'].shape[1]),
            'probe_parameters': parameters, 'training_steps': EPOCHS,
            'new_backbone_parameters': 0, 'new_tof_parameters': 0,
            'source_disjoint_test': True, 'test_labels_opened_after_selection': True,
            'device': torch.cuda.get_device_name(), 'backend': 'CUDA',
            'inputs': {**inputs, **model_bind.inputs}, 'outputs': outputs,
            'seconds': time.perf_counter() - started,
            'claim': 'Consumed controlled-simulation separability probe only; not natural-scene, hardware, or safety evidence.'})
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if store is not None:
            store.close()
        models = spatial = heads = None
        gc.collect()
        write(output / 'release.json', {'status': 'PASS', 'rgb_handles_closed': True,
                                        'persistent_processes': False, 'process_exit_required': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
