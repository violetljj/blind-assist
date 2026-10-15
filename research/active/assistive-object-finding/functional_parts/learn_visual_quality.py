#!/usr/bin/env python3
"""Matched frozen-feature quality learning; prediction never accepts fresh truth."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import pickle
import time
import learn_candidate_quality as old

np, torch, ev = old.np, old.torch, old.ev
VARIANTS = ('scores', 'dino_point', 'dino_gap', 'head_point', 'head_gap')
SEEDS = old.SEEDS
FEATURES = old.FEATURES + ['action_' + a for a in old.ACTIONS] + ['visual_' + str(i) for i in range(64)]


def projection():
    return np.random.default_rng(20261015).standard_normal((384, 64)) / math.sqrt(64)


def visual_vector(dense, point, bbox, mode, matrix):
    """Point coordinates follow the inherited half-pixel bilinear convention."""
    if dense.shape != (384, 14, 14) or not np.isfinite(dense).all():
        raise ValueError('Dense feature shape or finiteness')
    x0, y0, x1, y1 = bbox
    if mode == 'gap':
        vector = dense.mean((1, 2))
    else:
        grid = [(point[0] - x0) * 14 / (x1 - x0), (point[1] - y0) * 14 / (y1 - y0)]
        vector = np.array([ev.bilinear_value(channel, grid) for channel in dense])
    return (vector / max(float(np.linalg.norm(vector)), 1e-12)) @ matrix


def extract(obs, family, matrix):
    shared, points = old.extract(obs, family)
    action = np.zeros((6, 36)); action[:, old.ACTIONS.index(obs['action'])] = 1
    common = np.concatenate((shared, action), axis=1)
    result = {'scores': np.concatenate((common, np.zeros((6, 64))), axis=1)}
    for source in ('dino', 'head'):
        collected = {mode: [] for mode in ('point', 'gap')}
        for i, spec in enumerate(obs[family]):
            view = obs['views'][spec['name']]
            dense = np.load(view[source + '_dense_path'], allow_pickle=False).astype(np.float64)
            for point in points[i * 2:i * 2 + 2]:
                for mode in collected:
                    collected[mode].append(visual_vector(dense, point, view['bbox_xyxy'], mode, matrix))
        for mode in collected:
            result[source + '_' + mode] = np.concatenate((common, np.asarray(collected[mode])), axis=1)
    assert all(x.shape == (6, 193) and np.isfinite(x).all() for x in result.values())
    return result, points


def assemble(observations, family, matrix, manifests=None):
    features = {v: [] for v in VARIANTS}; points = []; targets = []
    for obs in observations:
        x, p = extract(obs, family, matrix)
        for v in VARIANTS: features[v].append(x[v])
        points.append(p)
        if manifests is not None:
            row = manifests[obs['sample_id']]
            if row['image_sha256'] != obs['image_sha256'] or row['action'] != obs['action']:
                raise ValueError('Identity mismatch')
            if old.digest(row['gt_path']) != row['gt_sha256']: raise ValueError('GT integrity')
            gt = old.cv2.imread(row['gt_path'], old.cv2.IMREAD_GRAYSCALE)
            if gt is None or gt.shape != (obs['height'], obs['width']) or gt.max() <= 0:
                raise ValueError('Invalid native GT')
            targets.append([ev.bilinear_value(gt, q) / float(gt.max()) for q in p])
    return {v: np.asarray(x) for v, x in features.items()}, points, np.asarray(targets) if manifests is not None else None


def train_variant(X, y, directory, budget_start):
    directory.mkdir(parents=True, exist_ok=False)
    mean = X.reshape(-1, 193).mean(0); std = X.reshape(-1, 193).std(0)
    std = np.where(std < 1e-6, 1., std)
    Z = ((X - mean) / std).astype(np.float32)
    np.savez(directory / 'standardizer.npz', mean=mean, std=std)
    x = torch.from_numpy(Z); target = torch.from_numpy(y.astype(np.float32)); receipts = []
    for seed in SEEDS:
        tick = time.perf_counter(); cpu = time.process_time()
        torch.manual_seed(seed); initial = old.model(193).state_dict()
        batches = np.random.default_rng(seed).integers(0, len(X), size=(600, 32))
        torch.save(initial, directory / f'initial_{seed}.pt'); np.save(directory / f'batches_{seed}.npy', batches)
        net = old.model(193); net.load_state_dict(copy.deepcopy(initial))
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
        history = []
        for step, indices in enumerate(batches):
            if time.process_time() - budget_start > 600: raise TimeoutError('Training CPU budget 600s')
            logits = net(x[indices]).squeeze(-1)
            loss = torch.nn.functional.mse_loss(torch.sigmoid(logits), target[indices])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step in (0, 99, 299, 599): history.append(dict(step=step + 1, mse=float(loss.detach())))
        torch.save(net.state_dict(), directory / f'mse_{seed}.pt')
        receipts.append(dict(method='mse', seed=seed, steps=600, cpu_s=time.process_time()-cpu, wall_s=time.perf_counter()-tick, history=history))
    for name, estimator in (
        ('ridge', old.Ridge(alpha=1.)),
        ('hgb', old.HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=7, learning_rate=.05,
             l2_regularization=1., min_samples_leaf=20, early_stopping=False, random_state=20261015))):
        if time.process_time() - budget_start > 600: raise TimeoutError('Training CPU budget 600s')
        tick = time.perf_counter(); cpu = time.process_time()
        estimator.fit(Z.reshape(-1, 193), y.reshape(-1))
        with (directory / f'{name}.pkl').open('wb') as stream: pickle.dump(estimator, stream)
        receipts.append(dict(method=name, cpu_s=time.process_time()-cpu, wall_s=time.perf_counter()-tick))
    old.write_json(directory / 'training_receipt.json', receipts)
    return receipts


def predict_variant(X, directory):
    scaler = np.load(directory / 'standardizer.npz', allow_pickle=False)
    Z = ((X - scaler['mean']) / scaler['std']).astype(np.float32); scores = []
    for seed in SEEDS:
        net = old.model(193)
        net.load_state_dict(torch.load(directory / f'mse_{seed}.pt', map_location='cpu', weights_only=True)); net.eval()
        with torch.inference_mode(): scores.append(torch.sigmoid(net(torch.from_numpy(Z))).squeeze(-1).numpy())
    values = {'mse': np.mean(scores, axis=0)}
    for name in ('ridge', 'hgb'):
        with (directory / f'{name}.pkl').open('rb') as stream: estimator = pickle.load(stream)
        values[name] = np.clip(estimator.predict(Z.reshape(-1, 193)).reshape(Z.shape[:2]), 0., 1.)
    return values, {'mse': np.asarray(scores)}


def train(args):
    start = time.perf_counter(); cpu = time.process_time(); out = args.root / 'training'
    out.mkdir(parents=True, exist_ok=False)
    plan = Path(__file__).with_name('VISUAL_RANK_PLAN.md')
    matrix = projection(); np.save(out / 'projection.npy', matrix)
    old.write_json(out / 'configuration.json', dict(plan_sha256=old.digest(plan), script_sha256=old.digest(__file__),
        inherited_script_sha256=old.digest(old.__file__), features=FEATURES, variants=VARIANTS, seeds=SEEDS,
        steps=600, batch_images=32, adamw_lr=.001, weight_decay=.001, fit_only_standardization=True,
        ridge_alpha=1, projection_seed=20261015, projection_scale='1/sqrt(64)', fresh_GT_read=False,
        observation_sha256=old.digest(args.predictions), fit_manifest_sha256=old.digest(args.root/'data/fit_samples.jsonl'),
        cal_manifest_sha256=old.digest(args.root/'data/calibration_samples.jsonl')))
    Path(out / 'learn_visual_quality_at_training.py').write_bytes(Path(__file__).read_bytes())
    Path(out / 'learn_candidate_quality_dependency.py').write_bytes(Path(old.__file__).read_bytes())
    old.write_json(out / 'feature_schema.json', dict(names=FEATURES, dimension=193, visual='L2 normalized 384D point or GAP, fixed 384x64 Gaussian projection', scores_padding=64))
    observations = {r['sample_id']: r for r in old.read_rows(args.predictions)}
    fit = {r['sample_id']: r for r in old.read_rows(args.root/'data/fit_samples.jsonl')}
    cal = {r['sample_id']: r for r in old.read_rows(args.root/'data/calibration_samples.jsonl')}
    assert not set(fit) & set(cal)
    fit_obs = [observations[k] for k in sorted(fit)]; cal_obs = [observations[k] for k in sorted(cal)]
    prepared = {}; extraction_start = time.process_time()
    for family in ('candidates', 'fixed'):
        X, p, y = assemble(fit_obs, family, matrix, fit); C, cp, cy = assemble(cal_obs, family, matrix, cal)
        prepared[family] = X, y, C, cp
        for variant in VARIANTS:
            np.savez(out/f'{family}_{variant}_fit_features_targets.npz', X=X[variant], y=y, points=np.asarray(p), sample_ids=np.array([o['sample_id'] for o in fit_obs]))
            np.savez(out/f'{family}_{variant}_cal_features_targets.npz', X=C[variant], y=cy, points=np.asarray(cp), sample_ids=np.array([o['sample_id'] for o in cal_obs]))
        if time.process_time()-extraction_start > 600: raise TimeoutError('Feature preparation CPU budget 600s')
    feature_cpu = time.process_time()-extraction_start
    training_start = time.process_time(); predictions = []; gates = {}; receipts = {}
    for family, (X, y, C, cp) in prepared.items():
        scores = {}; seeds = {}; receipts[family] = {}
        for variant in VARIANTS:
            directory = out/family/variant
            receipts[family][variant] = train_variant(X[variant], y, directory, training_start)
            values, per_seed = predict_variant(C[variant], directory)
            if time.process_time()-training_start > 600: raise TimeoutError('Training and calibration CPU budget 600s')
            scores.update({variant+'_'+k: v for k, v in values.items()})
            seeds.update({variant+'_'+k: v for k, v in per_seed.items()})
        predictions.extend(old.prediction_rows(cal_obs, family, cp, scores, seeds))
        gates[family] = {name: ev.calibrate_coverage(values.max(1).tolist(), len(cal_obs)) for name, values in scores.items()}
        for name, values in seeds.items():
            for i, seed in enumerate(SEEDS): gates[family][f'{name}_seed{seed}'] = ev.calibrate_coverage(values[i].max(1).tolist(), len(cal_obs))
    old.write_rows(out/'calibration_predictions.jsonl', predictions)
    old.write_json(out/'sealed_gates.json', dict(settings=gates, cal_images=len(cal), fit_images=len(fit), fresh_GT_read=False))
    files = {str(p.relative_to(out)): old.digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    receipt = dict(status='SUCCESS', files=files, fit_images=len(fit), cal_images=len(cal), fresh_GT_read=False,
        wall_s=time.perf_counter()-start, cpu_s=time.process_time()-cpu, feature_cpu_s=feature_cpu,
        training_cpu_s=time.process_time()-training_start, process_cpu_s=time.process_time(), backend='CPU', threads=2, receipts=receipts)
    old.write_json(out/'training_seal.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('files', 'receipts')}), flush=True)


def predict(args):
    start = time.perf_counter(); cpu = time.process_time(); training = args.root/'training'
    seal = json.loads((training/'training_seal.json').read_text())
    for name, expected in seal['files'].items():
        if old.digest(training/name) != expected: raise ValueError('Training seal mismatch: '+name)
    matrix = np.load(training/'projection.npy', allow_pickle=False)
    observations = old.read_rows(args.predictions)
    if len({r['sample_id'] for r in observations}) != len(observations): raise ValueError('Duplicate IDs')
    out = args.root/'learned_predictions'; out.mkdir(parents=True, exist_ok=False); records = []
    for family in ('candidates', 'fixed'):
        X, points, _ = assemble(observations, family, matrix)
        scores = {}; seeds = {}
        for variant in VARIANTS:
            np.savez(out/f'{family}_{variant}_features.npz', X=X[variant], points=np.asarray(points), sample_ids=np.array([o['sample_id'] for o in observations]))
            values, per_seed = predict_variant(X[variant], training/family/variant)
            scores.update({variant+'_'+k: v for k, v in values.items()}); seeds.update({variant+'_'+k: v for k, v in per_seed.items()})
        records.extend(old.prediction_rows(observations, family, points, scores, seeds))
        if seal['feature_cpu_s']+time.process_time()-cpu > 600:
            raise TimeoutError('Combined development/fresh feature preparation CPU budget 600s')
    old.write_rows(out/'fresh_predictions.jsonl', records)
    receipt = dict(status='SUCCESS', images=len(observations), records=len(records), fresh_GT_read=False,
        observation_predictions_sha256=old.digest(args.predictions), training_seal_sha256=old.digest(training/'training_seal.json'),
        script_sha256=old.digest(__file__), prediction_sha256=old.digest(out/'fresh_predictions.jsonl'),
        wall_s=time.perf_counter()-start, cpu_s=time.process_time()-cpu, process_cpu_s=time.process_time())
    old.write_json(out/'prediction_seal.json', receipt); print(json.dumps(receipt), flush=True)


def selftest():
    matrix = projection(); assert matrix.shape == (384, 64) and np.array_equal(matrix, projection())
    dense = np.arange(384*14*14, dtype=np.float64).reshape(384, 14, 14)
    vector = dense[:, 0, 0]; expected = vector/np.linalg.norm(vector) @ matrix
    assert np.allclose(visual_vector(dense, [.5, .5], [0, 0, 14, 14], 'point', matrix), expected)
    vector = dense.mean((1, 2)); expected = vector/np.linalg.norm(vector) @ matrix
    assert np.allclose(visual_vector(dense, [5, 9], [0, 0, 14, 14], 'gap', matrix), expected)
    assert np.array_equal(visual_vector(np.zeros_like(dense), [.5, .5], [0, 0, 14, 14], 'point', matrix), np.zeros(64))
    assert len(FEATURES) == 193 and len(set(FEATURES)) == 193
    assert sum(p.numel() for p in old.model(193).parameters()) == 6753
    print('SELFTEST PASS: projection, half-pixel point, GAP, zero norm, 193D and 6753-parameter schema')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'predict', 'selftest'))
    parser.add_argument('--root', type=Path); parser.add_argument('--predictions', type=Path)
    args = parser.parse_args()
    if args.mode == 'selftest': selftest()
    elif args.mode == 'train': train(args)
    else: predict(args)


if __name__ == '__main__': main()
