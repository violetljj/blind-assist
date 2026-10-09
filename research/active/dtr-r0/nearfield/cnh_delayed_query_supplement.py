"""Paired query-yaw and inherited held-unit descriptions for delayed-query arms.

Final checkpoints only. The parent run's cumulative 1200-second CUDA wall budget
also covers this supplement; existing failures remain charged and preserved.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import cnh_delayed_query_dev as D
import cnh_delayed_query_model as M
import cnh_bar_fusion_probe_dev as F
import cnh_aligned_boundary_dev as B


OUT = D.OUT / 'supplement'
POSE = D.ROOT / 'artifacts.local/work/cnh-bar-transfer-dev-20261009/pose/query_3_+1.npz'


def threshold_record(value):
    return dict(threshold=float(value) if np.isfinite(value) else None,
                threshold_is_positive_infinity=bool(np.isposinf(value)))


def infer(histories, transforms, lengths, destination, cfg, check, models):
    """Both representations see the same observations; no truth is read here."""
    destination.mkdir(parents=True, exist_ok=False)
    raw, receipts, checkpoint_epochs = {}, {}, set()
    permitted_plans = {D.sha(D.OUT/'PLAN.json')}
    if (models.parent/'PLAN.json').exists():
        permitted_plans.add(D.sha(models.parent/'PLAN.json'))
    for mode in ('center', 'extent'):
        path = destination / (mode + '.npy')
        receipts[mode] = D.prepare_features(histories, transforms, lengths, mode,
                                            path, check)
        features = torch.as_tensor(np.load(path, mmap_mode='r'), device='cuda')
        le = torch.as_tensor(np.asarray(lengths), device='cuda')
        try:
            for supervision in ('bce', 'pair'):
                check()
                arm = mode + '_' + supervision
                checkpoint = torch.load(models/f'{arm}.pt', map_location='cuda', weights_only=True)
                if (checkpoint['arm'] != arm or checkpoint['epochs'] < cfg['epochs']
                        or checkpoint['seed'] != cfg['seed']):
                    raise ValueError('Final checkpoint identity mismatch: ' + arm)
                if 'plan_sha256' in checkpoint:
                    if checkpoint['plan_sha256'] not in permitted_plans:
                        raise ValueError('Checkpoint plan identity mismatch: ' + arm)
                else:
                    original_path = D.OUT/'models'/f'{arm}.pt'
                    original = torch.load(original_path, map_location='cpu', weights_only=True)
                    continuation = json.loads((models.parent/'PLAN.json').read_text(encoding='utf8'))
                    if (checkpoint.get('parent_checkpoint_sha256') != D.sha(original_path)
                            or original.get('plan_sha256') != D.sha(D.OUT/'PLAN.json')
                            or continuation.get('parent_plan_sha256') != D.sha(D.OUT/'PLAN.json')):
                        raise ValueError('Continuation checkpoint lineage mismatch: ' + arm)
                    del original
                checkpoint_epochs.add(int(checkpoint['epochs']))
                if len(checkpoint_epochs) != 1:
                    raise ValueError('Supplement requires the same final epoch for all four arms')
                model = M.DelayedQueryReadout().cuda().eval()
                model.load_state_dict(checkpoint['state_dict'])
                del checkpoint
                try:
                    answer = np.empty((len(lengths), 2), np.float32)
                    with torch.inference_mode():
                        for start in range(0, len(lengths), cfg['batch']):
                            check()
                            ix = slice(start, min(start+cfg['batch'], len(lengths)))
                            answer[ix] = model(features[ix].float(), le[ix]).cpu().numpy()
                    if not np.isfinite(answer).all():
                        raise ValueError('Nonfinite supplementary raw logits')
                    raw[arm] = answer
                finally:
                    del model
        finally:
            del features, le
            torch.cuda.empty_cache()
    receipts['checkpoint_epochs'] = next(iter(checkpoint_epochs))
    return raw, receipts


def yaw_metrics(raw, ideal):
    with np.load(D.ALIGNED/'geometry.npz', allow_pickle=False) as z:
        category = z['category']
    with np.load(POSE, allow_pickle=False) as z:
        m3 = F.smooth(z['m3_raw'])
    baseline, timely = F.metrics(m3 >= B.THETA, category)
    if (baseline['counts'] != [467, 375] or baseline['clear_slots'] != 176
            or baseline['clear_denominator'] != 4576):
        raise ValueError('Frozen +3deg M3 parity mismatch')
    clear = (category == 'clear').all(1)
    result = dict(status='COMPLETE', query_yaw_degrees=3,
        baseline=dict(threshold=B.THETA, metrics=baseline), arms={},
        input_contract='Same original492 scene photons and physical truth; only public query yaw changes. '
            'Original query categories remain the evaluator reference.',
        evidence_scope='Consumed simulated Development query-error diagnostic; '
            'fixed ideal thresholds primary, yaw same-cost thresholds descriptive only.')
    for arm in D.ARMS:
        score = F.smooth(raw[arm].reshape(492, 4, 13, 2))
        record = ideal['arms'][arm]
        fixed = np.inf if record.get('threshold_is_positive_infinity') else record['threshold']
        if fixed is None:
            raise ValueError('Missing ideal threshold: ' + arm)
        summary, candidate = F.metrics(score >= fixed, category)
        matched, cost = F.nearest(score[clear].max(-1), 176)
        match_summary, match_timely = F.metrics(score >= matched, category)
        if match_summary['clear_slots'] != cost:
            raise ValueError('Same-cost diagnostic cost mismatch')
        result['arms'][arm] = dict(
            fixed_ideal_threshold=dict(**threshold_record(fixed), metrics=summary,
                paired_minus_M3=F.compare(timely, candidate, category)),
            yaw_cost_matched_diagnostic=dict(**threshold_record(matched), metrics=match_summary,
                target_clear_slots=176, clear_cost_residual=cost-176,
                exactly_matched_clear_cost=cost == 176,
                paired_minus_M3=F.compare(timely, match_timely, category)))
    return result


def auc(positive, negative):
    """Equivalent pair-count AUC, including half credit for exact ties."""
    pos, neg = np.asarray(positive).ravel(), np.asarray(negative).ravel()
    if not len(pos) or not len(neg):
        return None
    order = np.sort(neg)
    left = np.searchsorted(order, pos, side='left')
    right = np.searchsorted(order, pos, side='right')
    return float((left + .5*(right-left)).sum()/(len(pos)*len(neg)))


def fresh_metrics(raw):
    folder = D.OLD/'inputs/fresh_evaluation'
    with np.load(folder/'rows.npz', allow_pickle=False) as z:
        rows = {k: z[k] for k in ('unit', 'variant', 'replica', 'frame', 'group', 'front')}
    units = sorted(set(rows['unit'].tolist()))
    expected_units = [u for u in range(221001, 221073) if u % 3 in (0, 1)][:48]
    keys = list(zip(*(rows[k].tolist() for k in ('unit', 'variant', 'replica', 'frame'))))
    expected = [(u, v, k, f) for u in expected_units for v in range(7)
                for k in range(4) for f in range(3, 16)]
    if units != expected_units or keys != expected:
        raise ValueError('Inherited fresh rows must have every original episode and frame')
    with np.load(D.OLD/'predictions/M3_fresh_evaluation.npz', allow_pickle=False) as z:
        baseline = z['raw']
    if baseline.shape != (17472, 2):
        raise ValueError('Frozen fresh M3 raw shape mismatch')
    values = dict(M3=baseline, **raw)
    per_scene = []
    groups = rows['group'].reshape(48, 7, 4, 13)
    front = rows['front'].reshape(48, 7, 4, 13)
    for i in range(48):
        if not np.all(groups[i] == groups[i, 0, 0, 0]):
            raise ValueError('Target query must be constant across an inherited scene')
        np.testing.assert_array_equal(front[i], np.broadcast_to(front[i, 0, 0], front[i].shape))
    scores = {arm: F.smooth(value.reshape(48, 7, 4, 13, 2)) for arm, value in values.items()}
    for i, unit in enumerate(units):
        q = int(groups[i, 0, 0, 0])
        keep = (front[i, 0, 0] >= 1.2) & (front[i, 0, 0] < 2.1)
        row = dict(unit=unit, group=q, frames=int(keep.sum()), positive_samples=2*4*int(keep.sum()),
                   negative_samples=3*4*int(keep.sum()), auc={})
        for arm, score in scores.items():
            selected = score[i, ..., q]
            row['auc'][arm] = auc(selected[[0, 1]][..., keep], selected[[4, 5, 6]][..., keep])
        per_scene.append(row)
    result = dict(status='COMPLETE', primary_range_m=[1.2, 2.1], scene_equal_weight=True,
        scenes=48, replicas=4, positive_variants=[0, 1], negative_variants=[4, 5, 6],
        statistic='Mean of per-scene target-query AUC after original five-score causal smoothing.',
        evidence_scope='Training-excluded unit IDs within the same simulator family; already consumed Development. '
            'No checkpoint or threshold selection on these scores; no independent confirmation or hardware claim.',
        arms={}, per_scene=per_scene)
    for arm in values:
        a = [r['auc'][arm] for r in per_scene]
        if any(v is None for v in a):
            raise ValueError('Missing primary AUC must not change the48scene denominator')
        result['arms'][arm] = dict(macro_auc=float(np.mean(a)),
            paired_mean_minus_M3=float(np.mean([r['auc'][arm]-r['auc']['M3'] for r in per_scene])))
    return result


def execute(metrics_path, include_fresh=True, models=None, output=None, base_seconds=None):
    models = Path(models or D.OUT/'models').resolve()
    output = Path(output or OUT).resolve()
    artifact = (D.ROOT/'artifacts.local').resolve()
    if not output.is_relative_to(artifact) or not models.is_relative_to(artifact):
        raise ValueError('Models and supplementary output must stay in canonical artifacts.local')
    cfg = json.loads((D.OUT/'PLAN.json').read_text(encoding='utf8'))
    main = json.loads((D.OUT/'run_receipt.json').read_text(encoding='utf8'))
    if main['status'] != 'COMPLETE':
        raise ValueError('Parent training must complete before supplement')
    if (output/'run_receipt.json').exists() or (output/'yaw_scores.npz').exists():
        raise FileExistsError('Preserve previous supplementary outputs')
    ideal = json.loads(Path(metrics_path).read_text(encoding='utf8'))
    if ideal['status'] != 'COMPLETE' or set(ideal['arms']) != set(D.ARMS):
        raise ValueError('Complete matching ideal evaluation required')
    output.mkdir(parents=True, exist_ok=True)
    previous = float(main['seconds'])
    previous += sum(float(json.loads(p.read_text())['seconds']) for p in D.OUT.glob('failure_*.json'))
    previous += sum(float(json.loads(p.read_text())['seconds']) for p in output.glob('failure_*.json'))
    if base_seconds is not None:
        if base_seconds < 0:
            raise ValueError('Additional parent GPU wall seconds must be nonnegative')
        previous += float(base_seconds)
    elif models.parent != D.OUT.resolve():
        extra = json.loads((models.parent/'run_receipt.json').read_text(encoding='utf8'))
        if extra['status'] != 'COMPLETE':
            raise ValueError('Selected continuation must complete before supplement')
        previous += float(extra['seconds'])
        previous += sum(float(json.loads(p.read_text())['seconds']) for p in models.parent.glob('failure_*.json'))
    limit = float(cfg['budgets_seconds']['GPU_stage_cumulative_wall'])
    began = time.monotonic()
    def check():
        if previous + time.monotonic()-began >= limit:
            raise TimeoutError('Parent plus supplement cumulative GPU wall budget reached')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    identity = {str(models/f'{a}.pt'): D.sha(models/f'{a}.pt') for a in D.ARMS}
    identity[str(D.OUT/'PLAN.json')] = D.sha(D.OUT/'PLAN.json')
    receipts, completed = {}, []
    stamp = str(time.time_ns())
    try:
        check()
        h, t, le = D.aligned_observations(query_yaw=3)
        raw, features = infer(h, t, le, output/('yaw_features_'+stamp), cfg, check, models)
        del h, t, le
        np.savez_compressed(output/'yaw_scores.npz', arm_names=np.array(D.ARMS),
                            raw=np.stack([raw[a].reshape(492, 4, 13, 2) for a in D.ARMS]))
        D.save(output/'yaw_metrics.json', yaw_metrics(raw, ideal))
        receipts['yaw'] = features
        completed.append('query_yaw_+3')
        del raw
        # A representative complete mode from yaw supplies a measured admission
        # estimate for the smaller inherited held-unit feature queue. Budget checks
        # remain authoritative even if this estimate is optimistic.
        elapsed = time.monotonic()-began
        fresh_estimate = elapsed * (17472 / (492*4*13)) * 1.3 + 5
        if include_fresh and limit-previous-elapsed > fresh_estimate:
            folder = D.OLD/'inputs/fresh_evaluation'
            h, t, le = (np.load(folder/f'{k}.npy', mmap_mode='r')
                        for k in ('histories', 'transforms', 'length'))
            if len(le) != 17472:
                raise ValueError('Inherited fresh input row count changed')
            raw, features = infer(h, t, le, output/('fresh_features_'+stamp), cfg, check, models)
            np.savez_compressed(output/'fresh_scores.npz', arm_names=np.array(D.ARMS),
                                raw=np.stack([raw[a] for a in D.ARMS]))
            D.save(output/'fresh_metrics.json', fresh_metrics(raw))
            receipts['fresh'] = features
            completed.append('inherited_training_excluded_units')
            del h, t, le, raw
        else:
            receipts['fresh'] = dict(status='NOT_RUN', reason='remaining cumulative wall budget'
                                     if include_fresh else 'yaw-only requested',
                                     measured_estimate_seconds=fresh_estimate)
        for path, digest in identity.items():
            if D.sha(path) != digest:
                raise ValueError('Parent checkpoint/plan changed during supplement')
        check()
        result = dict(status='COMPLETE', seconds=time.monotonic()-began,
            prior_charged_seconds=previous, cumulative_seconds=previous+time.monotonic()-began,
            cumulative_limit_seconds=limit, completed=completed, features=receipts,
            backend='CUDA', device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
            model_and_plan_sha256=identity, source_sha256=D.sha(__file__),
            ideal_metrics_sha256=D.sha(metrics_path), checkpoint_selection='Final epoch only; no evaluation selection')
        D.save(output/'run_receipt.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    except Exception as e:
        D.save(output/f'failure_{time.time_ns()}.json', dict(status='FAILED', seconds=time.monotonic()-began,
            prior_charged_seconds=previous, completed=completed, error=repr(e)))
        raise
    finally:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=D.OUT/'evaluation/metrics.json')
    parser.add_argument('--models', type=Path, default=D.OUT/'models')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--base-seconds', type=float,
                        help='Additional parent GPU wall seconds beyond v1; default reads selected model parent receipt')
    parser.add_argument('--yaw-only', action='store_true')
    args = parser.parse_args()
    execute(args.metrics, not args.yaw_only, args.models, args.output, args.base_seconds)
