"""CPU-only fit stratification and paired cluster AUC analysis.

Evaluator supervision stays outside student inputs. This module does not train,
render, inspect deployment thresholds, or invoke an earlier decision function.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-t2-fit-diagnostic-20261004'
ARMS = ('M3_seed0', 'V_seed0', 'VD_seed0', 'T2_seed0', 'S_seed0')
SEED = 2026100601
REPLICATES = 1000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def bce(raw, labels):
    raw = np.asarray(raw, np.float64)
    return np.maximum(raw, 0) - raw * labels + np.log1p(np.exp(-np.abs(raw)))


class RankedAUC:
    """Exact tie-aware AUC, including repeated whole-unit bootstrap draws."""
    def __init__(self, labels, scores, cluster=None):
        labels, scores = np.asarray(labels, bool), np.asarray(scores)
        self.order = np.argsort(scores, kind='stable')
        ordered = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        self.positive = labels[self.order]
        self.cluster = None if cluster is None else np.asarray(cluster)[self.order]

    def value(self, multiplicity=None):
        if not len(self.positive):
            return None
        weight = np.ones(len(self.positive), np.float64) if multiplicity is None else multiplicity[self.cluster]
        pos = np.add.reduceat(weight * self.positive, self.starts)
        neg = np.add.reduceat(weight * ~self.positive, self.starts)
        total_pos, total_neg = float(pos.sum()), float(neg.sum())
        if not total_pos or not total_neg:
            return None
        before = np.cumsum(neg) - neg
        return float(np.sum(pos * (before + .5 * neg)) / (total_pos * total_neg))


def auc(labels, raw):
    if not len(labels):
        return None
    return RankedAUC(labels, raw).value()


def load_split(out, split):
    folder = out / 'supervision' / split
    supervision_receipt = read(folder / 'receipt.json')
    if supervision_receipt['status'] != 'COMPLETE' or supervision_receipt['plan_sha256'] != sha(out / 'PLAN.json'):
        raise ValueError('Supervision receipt is not bound to the frozen diagnostic PLAN')
    paths = [folder / f'{name}.npy' for name in ('labels', 'mask', 'weights')]
    paths += [folder / 'metadata.npz', folder / 'receipt.json']
    for path_string, expected in supervision_receipt['output_sha256'].items():
        if sha(Path(path_string)) != expected:
            raise ValueError(f'Sealed supervision changed: {path_string}')
    labels, mask, weights = (np.load(path, allow_pickle=False) for path in paths[:3])
    with np.load(folder / 'metadata.npz', allow_pickle=False) as stored:
        metadata = {key: stored[key].copy() for key in stored.files}
    n = len(labels)
    if any(array.shape != (n, 2) for array in (labels, mask, weights)) or len(metadata['unit']) != n:
        raise ValueError('Supervision axes must be row x HEAD/BODY')
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Supervision weights must be nonnegative finite')
    predictions = {}
    for arm in ARMS:
        path = out / 'predictions' / split / f'{arm}.npz'
        with np.load(path, allow_pickle=False) as stored:
            raw = np.asarray(stored['raw']).copy()
            # If predictions retain row identities, require exact agreement.
            for name in ('unit', 'config', 'variant', 'replica', 'frame', 'domain'):
                if name in stored.files and not np.array_equal(stored[name], metadata[name]):
                    raise ValueError(f'Prediction row identity mismatch: {arm}/{name}')
        if raw.shape != (n, 2) or not np.isfinite(raw).all():
            raise ValueError(f'Invalid frozen raw logits: {arm}')
        predictions[arm] = raw
        paths.append(path)
        receipt_path = path.with_suffix('.json')
        if not receipt_path.exists():
            raise FileNotFoundError(f'Frozen raw predictions need a receipt: {receipt_path}')
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt.get('status') != 'COMPLETE' or receipt.get('plan_sha256') != sha(out / 'PLAN.json'):
                raise ValueError(f'Prediction is not complete under this PLAN: {path}')
            row_hashes = {value for key, value in supervision_receipt['input_sha256'].items()
                          if Path(key).name == 'rows.npz'}
            if receipt.get('row_sha256') not in row_hashes:
                raise ValueError(f'Prediction/supervision row provenance differs: {path}')
            for key in ('raw_sha256', 'output_sha256'):
                expected = receipt.get(key)
                if isinstance(expected, str) and sha(path) != expected:
                    raise ValueError(f'Prediction receipt mismatch: {path}')
            paths.append(receipt_path)
    return labels, mask > 0, weights, metadata, predictions, {str(path): sha(path) for path in paths}


def strata(metadata, labels):
    n = len(labels)
    all_queries = np.ones((n, 2), bool)
    yield 'all', 'all', all_queries
    for value, name in ((0, 'natural'), (1, 'displacement')):
        yield 'domain', name, np.broadcast_to((metadata['domain'] == value)[:, None], (n, 2))
    front = metadata['front']
    for lo, hi, name in ((-np.inf, 1.2, '<1.2m'), (1.2, 2.1, '1.2-2.1m'),
                         (2.1, 2.6, '2.1-2.6m'), (2.6, np.inf, '>=2.6m')):
        yield 'distance', name, np.broadcast_to(((front >= lo) & (front < hi))[:, None], (n, 2))
    if not np.isfinite(front).all():
        yield 'distance', 'NOT_AVAILABLE', np.broadcast_to(~np.isfinite(front[:, None]), (n, 2))
    yield 'label', 'positive', labels == 1
    yield 'label', 'negative', labels == 0
    for value, name in ((0, 'zero'), (1, 'low'), (2, 'high')):
        yield 'query_support', name, metadata['query_support_bin'] == value
    for value, name in ((-1, 'NOT_AVAILABLE'), (0, 'FOV_OUT'), (1, 'FOV_IN')):
        yield 'scene_fov', name, np.broadcast_to((metadata['scene_fov_in'] == value)[:, None], (n, 2))
    for q, name in enumerate(('HEAD', 'BODY')):
        selected = np.zeros((n, 2), bool)
        selected[:, q] = True
        yield 'query', name, selected


def table_split(split, labels, valid, weights, metadata, predictions):
    cells = []
    for axis, stratum, selected in strata(metadata, labels):
        retained = selected & valid
        nrows = int(selected.any(axis=1).sum())
        nvalid, positives = int(retained.sum()), int(labels[retained].sum())
        for arm, raw in predictions.items():
            effective = weights * retained
            mass = float(effective.sum(dtype=np.float64))
            numerator = float((bce(raw, labels) * effective).sum(dtype=np.float64))
            query_auc = [auc(labels[:, q][retained[:, q]], raw[:, q][retained[:, q]]) for q in range(2)]
            cells.append(dict(split=split, arm=arm, axis=axis, stratum=stratum,
                rows=nrows, valid_queries=nvalid, positives=positives, negatives=nvalid-positives,
                weight_mass=mass, weighted_bce_sum=numerator,
                hard_bce_per_row=numerator/nrows if nrows else None,
                weight_normalized_bce=numerator/mass if mass else None,
                pooled_raw_auc=auc(labels[retained], raw[retained]),
                head_raw_auc=query_auc[0], body_raw_auc=query_auc[1]))
    return cells


def paired_bootstrap(split, labels, valid, metadata, predictions, selected=None):
    if selected is not None:
        valid = valid & selected[:, None]
    units = np.unique(metadata['unit'][valid.any(axis=1)])
    if not len(units):
        return []
    # All retained queries belonging to a unit receive the same multiplicity.
    cluster = np.searchsorted(units, metadata['unit'])[:, None].repeat(2, axis=1)[valid]
    ranked = {arm: RankedAUC(labels[valid], predictions[arm][valid], cluster)
              for arm in ('S_seed0', 'T2_seed0', 'V_seed0')}
    pairs = (('S_seed0', 'T2_seed0'), ('T2_seed0', 'V_seed0'))
    observed = {arm: scorer.value() for arm, scorer in ranked.items()}
    draws = {pair: [] for pair in pairs}
    rng = np.random.default_rng(SEED)
    for _ in range(REPLICATES):
        multiplicity = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
        values = {arm: scorer.value(multiplicity) for arm, scorer in ranked.items()}
        for pair in pairs:
            left, right = pair
            if values[left] is not None and values[right] is not None:
                draws[pair].append(values[left] - values[right])
    cells = []
    for left, right in pairs:
        values = draws[(left, right)]
        difference = None if observed[left] is None or observed[right] is None else observed[left]-observed[right]
        cells.append(dict(split=split, cohort='all' if selected is None else 'displacement',
            pair=f'{left}-{right}', left_auc=observed[left], right_auc=observed[right], difference=difference,
            ci95=np.quantile(values, [.025, .975]).tolist() if values else None,
            units=len(units), valid_queries=int(valid.sum()), bootstrap_replicates=REPLICATES,
            valid_replicates=len(values), seed=SEED, cluster='unit, preserving all config/variant/K/frame/query rows'))
    return cells


def curve_summary(paths):
    cells = []
    for path in paths:
        receipt = read(path)
        history = receipt.get('history', receipt.get('curve', []))
        if isinstance(history, dict):
            history = [history]
        cell = dict(path=str(path), sha256=sha(path), status=receipt.get('status'),
                    arm=receipt.get('arm'), seed=receipt.get('seed'),
                    rows=receipt.get('rows'), epochs=receipt.get('epochs'),
                    fit=receipt.get('fit'), seconds=receipt.get('seconds'), history=history)
        cells.append(cell)
    return cells


def format_number(value, decimals=4):
    return 'NA' if value is None else f'{value:.{decimals}f}'


def branch_flags(cells, bootstrap, plan, out):
    tolerance = plan['tolerance']
    main = {(cell['split'], cell['arm']): cell for cell in cells if cell['axis'] == 'all'}
    def close(split, left, right):
        a, b = main[split, left], main[split, right]
        loss_difference = abs(a['hard_bce_per_row'] - b['hard_bce_per_row'])
        loss_tolerance = max(tolerance['loss_close_absolute'], tolerance['loss_close_relative']*b['hard_bce_per_row'])
        auc_difference = abs(a['pooled_raw_auc'] - b['pooled_raw_auc'])
        return dict(passed=bool(auc_difference <= tolerance['auc_close'] and loss_difference <= loss_tolerance),
                    absolute_auc_difference=auc_difference, absolute_loss_difference=loss_difference,
                    loss_tolerance=loss_tolerance)
    matches = {split: close(split, 'S_seed0', 'T2_seed0') for split in ('train', 'fresh_evaluation')}
    heldout = close('fresh_evaluation', 'T2_seed0', 'V_seed0')
    tiny = []
    for size in plan['tiny']['sizes']:
        path = out / f'tiny/T2_n{size}/receipt.json'
        # Root subset keys are n256/n2048; accept its explicit stable size key too.
        if not path.exists():
            path = out / f'tiny/T2_{size}/receipt.json'
        if not path.exists():
            tiny.append(dict(rows=size, status='PENDING', passed=None))
            continue
        receipt = read(path)
        if receipt['status'] != 'COMPLETE' or receipt['plan_sha256'] != sha(out / 'PLAN.json'):
            raise ValueError('Tiny receipt must belong to frozen diagnostic PLAN')
        metric = receipt['curve'][-1]
        passed = metric['hard_weighted_loss'] <= plan['tiny']['near_zero_hard_loss'] and metric['pooled_auc'] >= plan['tiny']['near_zero_auc']
        tiny.append(dict(rows=size, status=receipt['stop'], passed=bool(passed), last_metric=metric,
                         path=str(path), sha256=sha(path)))
    tiny_passed = all(cell['passed'] for cell in tiny) if all(cell['passed'] is not None for cell in tiny) else None
    paired = next(cell for cell in bootstrap if cell['split'] == 'fresh_evaluation' and cell['pair'] == 'S_seed0-T2_seed0')
    s, t2 = main['fresh_evaluation', 'S_seed0'], main['fresh_evaluation', 'T2_seed0']
    loss_reduction = t2['hard_bce_per_row']-s['hard_bce_per_row']
    score_conditions = (paired['difference'] >= tolerance['S_better_auc'] and paired['ci95'][0] > 0 and loss_reduction >= tolerance['S_better_loss'])
    return dict(S_MATCHES_T2=all(cell['passed'] for cell in matches.values()),
                S_MATCHES_T2_details=matches,
                S_BETTER_T2=bool(score_conditions and tiny_passed) if tiny_passed is not None else None,
                S_BETTER_T2_score_conditions=bool(score_conditions), S_BETTER_T2_loss_reduction=loss_reduction,
                T2_TINY_UNFIT=None if tiny_passed is None else not tiny_passed, tiny=tiny,
                T2_HOLDOUT_MATCHES_V=heldout['passed'], T2_HOLDOUT_MATCHES_V_details=heldout,
                flags_nonexclusive=True, original_gate_rescued=False)


def report(cells, bootstrap, curves, flags):
    lines = ['# T2拟合诊断 A/C：固定分数的损失、排序和分层', '',
        '本报告只核对固定单种子模型的硬标签拟合。所有数值为已消费模拟Development；未运行训练、硬件或部署阈值评价。'
        '训练整体的自然/位移损失质量各占一半，而fresh仅位移域，因此跨域泛化应比较下表训练位移行与fresh行的权重归一化BCE。', '',
        '|数据/域|模型|行数/有效查询|正/负|硬BCE/行|权重归一BCE|pooled raw AUC|HEAD/BODY raw AUC|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for cell in cells:
        if cell['axis'] == 'all' or (cell['split'] == 'train' and cell['axis'] == 'domain' and cell['stratum'] == 'displacement'):
            lines.append('|'+ '|'.join((f"{cell['split']}/{cell['stratum']}", cell['arm'],
                f"{cell['rows']}/{cell['valid_queries']}", f"{cell['positives']}/{cell['negatives']}",
                format_number(cell['hard_bce_per_row']), format_number(cell['weight_normalized_bce']),
                format_number(cell['pooled_raw_auc']),
                f"{format_number(cell['head_raw_auc'])}/{format_number(cell['body_raw_auc'])}"))+'|')
    lines += ['', '硬BCE/行使用Σ(BCEWithLogits×mask×原冻结weights)/Nrows；归一BCE使用同一分子除以有效权重质量。'
              'AUC为有效查询未加权的raw logits排序，含并列分数半计；单类格AUC为NA。'
              '全量单轴表见stratified_metrics.csv/json：距离四档包含<1.2m及≥2.6m溢出，另保留标签、几何支持、视场和高度带；没有交叉筛选。', '',
              '|数据/域|配对|AUC差|95%区间|整单位/有效查询|', '|---|---|---:|---|---:|']
    for cell in bootstrap:
        ci = cell['ci95']
        interval = 'NA' if ci is None else f'[{ci[0]:.4f},{ci[1]:.4f}]'
        lines.append(f"|{cell['split']}/{cell['cohort']}|{cell['pair']}|{format_number(cell['difference'])}|{interval}|{cell['units']}/{cell['valid_queries']}|")
    lines += ['', '区间为1000次整unit配对bootstrap，seed=2026100601，同一场景的位移、光子种子、帧及HEAD/BODY一起重采样。'
              '它描述固定模型的不确定性，不包含重新训练的种子波动；pooled AUC并非先前场景macroAUC，不可直接与旧主AUC替换。', '',
              '拟合曲线只用于核对优化过程，完整原始receipt摘要保存在training_curves.json；最终预测不按中间epoch择优。']
    for cell in curves:
        history = cell['history']
        losses = [item.get('loss', item.get('hard_loss', item.get('hard_weighted_loss')))
                  for item in history if isinstance(item, dict)]
        losses = [value for value in losses if value is not None]
        if losses:
            lines.append(f"- {cell['arm'] or Path(cell['path']).parent.name}：{len(history)}条曲线，首/末记录损失{losses[0]:.5f}/{losses[-1]:.5f}；该记录的损失口径以原receipt为准。")
    lines += ['', '预设描述分支（可同时命中，不替换原gate）：' + '；'.join(
        f'{name}={flags[name]}' for name in ('S_MATCHES_T2', 'S_BETTER_T2', 'T2_TINY_UNFIT', 'T2_HOLDOUT_MATCHES_V')) + '。'
        'None表示tiny结果尚未交付。近似判据为两split AUC差≤.02且硬损失差≤max(.03,对照的20%)；不是统计等价证明。', '',
        '解读边界：T2_UNFIT仅说明本配方没有达到拟合门槛，不能推出逐帧表示无用。'
        'S与T2即使接近，也不能唯一确认预训练是原因：两者架构及优化适配同时不同。'
        '支持切点来自训练几何正支持中位数，fresh固定复用；训练没有可复核的目标可见模板，视场记为NA。'
        '训练已接触过这些样本，M3预训练还可能与自然域重合；fresh只是本轮未用于训练的模拟批。'
        '共享模拟器、已知背景教师及位姿噪声假设仍限制外推。', '']
    return '\n'.join(lines)


def run(out=OUT, curve_paths=()):
    out = Path(out)
    plan_path = out / 'PLAN.json'
    if not plan_path.exists():
        raise FileNotFoundError('Freeze PLAN before formal CPU analysis')
    started = time.monotonic()
    plan_sha = sha(plan_path)
    cells, bootstrap, inputs = [], [], {str(plan_path): plan_sha}
    for split in ('train', 'fresh_evaluation'):
        labels, valid, weights, metadata, predictions, hashes = load_split(out, split)
        inputs.update(hashes)
        cells += table_split(split, labels, valid, weights, metadata, predictions)
        bootstrap += paired_bootstrap(split, labels, valid, metadata, predictions)
        if split == 'train':
            bootstrap += paired_bootstrap(split, labels, valid, metadata, predictions, metadata['domain'] == 1)
    curves = curve_summary(curve_paths)
    inputs.update({cell['path']: cell['sha256'] for cell in curves})
    flags = branch_flags(cells, bootstrap, read(plan_path), out)
    inputs.update({cell['path']: cell['sha256'] for cell in flags['tiny'] if 'path' in cell})
    if sha(plan_path) != plan_sha:
        raise ValueError('Frozen PLAN changed during analysis')
    destination = out / 'analysis'
    destination.mkdir(parents=True, exist_ok=True)
    save(destination / 'stratified_metrics.json', cells)
    with (destination / 'stratified_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    save(destination / 'paired_bootstrap.json', bootstrap)
    save(destination / 'training_curves.json', curves)
    save(destination / 'branch_flags.json', flags)
    (destination / 'REPORT_A_C.md').write_text(report(cells, bootstrap, curves, flags), encoding='utf-8')
    outputs = {str(path): sha(path) for path in destination.iterdir() if path.is_file() and path.name != 'receipt.json'}
    result = dict(status='COMPLETE', plan_sha256=plan_sha, source_sha256=sha(__file__),
                  input_sha256=inputs, output_sha256=outputs, elapsed_s=time.monotonic()-started,
                  gpu_used=False, training=False, tables=len(cells), bootstrap_cells=len(bootstrap))
    save(destination / 'receipt.json', result)
    return result


def refresh(out=OUT, curve_paths=()):
    """Append completed tiny evidence without repeating fixed-score bootstrap."""
    out = Path(out)
    destination = out / 'analysis'
    prior = read(destination / 'receipt.json')
    if prior['status'] != 'COMPLETE' or prior['plan_sha256'] != sha(out / 'PLAN.json'):
        raise ValueError('The fixed-score analysis must be complete under this PLAN')
    if prior['source_sha256'] != sha(__file__):
        raise ValueError('Analysis code changed; cannot reuse frozen metric computations')
    for path, expected in prior['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Analysis source evidence changed: {path}')
    for path, expected in prior['output_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Analysis output changed: {path}')
    cells = read(destination / 'stratified_metrics.json')
    bootstrap = read(destination / 'paired_bootstrap.json')
    prior_curves = read(destination / 'training_curves.json')
    paths = {cell['path']: Path(cell['path']) for cell in prior_curves}
    paths.update({str(path): Path(path) for path in curve_paths})
    curves = curve_summary(paths.values())
    flags = branch_flags(cells, bootstrap, read(out / 'PLAN.json'), out)
    save(destination / 'training_curves.json', curves)
    save(destination / 'branch_flags.json', flags)
    (destination / 'REPORT_A_C.md').write_text(report(cells, bootstrap, curves, flags), encoding='utf-8')
    prior['input_sha256'].update({cell['path']: cell['sha256'] for cell in curves})
    prior['input_sha256'].update({cell['path']: cell['sha256'] for cell in flags['tiny'] if 'path' in cell})
    prior['output_sha256'] = {str(path): sha(path) for path in destination.iterdir()
                            if path.is_file() and path.name != 'receipt.json'}
    prior['tiny_refreshes'] = prior.get('tiny_refreshes', 0) + 1
    prior['bootstrap_recomputed_on_refresh'] = False
    save(destination / 'receipt.json', prior)
    return prior


def selfcheck():
    # Ties plus replicated clusters must equal explicit resampling, not a macro AUC.
    labels = np.array([0, 1, 1, 0, 1, 0])
    scores = np.array([1., 1., 2., 3., 3., 3.])
    cluster = np.array([0, 0, 1, 1, 2, 2])
    weights = np.array([2, 0, 1])
    repeated = np.repeat(np.arange(len(labels)), weights[cluster])
    expected = auc(labels[repeated], scores[repeated])
    actual = RankedAUC(labels, scores, cluster).value(weights)
    if actual != expected or auc(np.ones(3), np.arange(3)) is not None:
        raise AssertionError('Exact cluster/tie AUC parity failed')
    if not np.isfinite(bce(np.array([-1000., 1000.]), np.array([1., 0.]))).all():
        raise AssertionError('Extreme raw BCE must stay finite')
    fixture_labels = np.array([[1., 0.], [0., 1.]])
    fixture_valid = np.array([[True, False], [True, True]])
    fixture_weights = np.array([[.5, 0.], [.5, 1.]])
    metadata = dict(unit=np.array([1, 2]), domain=np.array([0, 1]), front=np.array([.8, 2.7]),
                    query_support_bin=np.array([[0, 1], [1, 2]]), scene_fov_in=np.array([-1, 0]))
    cells = table_split('fixture', fixture_labels, fixture_valid, fixture_weights, metadata,
                        {'fixture': np.zeros((2, 2))})
    overall = cells[0]
    if overall['rows'] != 2 or overall['valid_queries'] != 3 or overall['positives'] != 2:
        raise AssertionError('Row/query denominators changed')
    if abs(overall['hard_bce_per_row'] - np.log(2)) > 1e-14:
        raise AssertionError('Fixed global weights were renormalized per batch/query')
    if not any(cell['stratum'] == '>=2.6m' and cell['rows'] == 1 for cell in cells):
        raise AssertionError('Overflow distance rows must remain represented')
    return dict(status='PASS', tied_cluster_auc=actual, gpu_used=False, scientific_data_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('selfcheck', 'run', 'refresh'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--curve', type=Path, action='append', default=[])
    args = parser.parse_args()
    operation = {'selfcheck': lambda: selfcheck(), 'run': lambda: run(args.out, args.curve),
                 'refresh': lambda: refresh(args.out, args.curve)}
    print(json.dumps(operation[args.command](), ensure_ascii=False), flush=True)
