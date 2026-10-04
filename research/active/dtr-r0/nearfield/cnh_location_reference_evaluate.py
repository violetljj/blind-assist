"""Evaluation-only location-marginal reference; no templates, scores or training.

The primary R_any is the user's ANY-HEIGHT location-marginal discrimination
reference. Rquery is auxiliary and is computed for both public queries before
evaluator-only target-band selection. Every cost threshold is global over the
selected cohort, describing an evaluation frontier rather than calibration.
"""
import argparse
from pathlib import Path
import time

import numpy as np

import cnh_readout_pilot2_evaluate as P2
import cnh_temporal_readout_evaluate as T
import cnh_unknown_target_reference_evaluate as U
import cnh_sequence_observed_geometry as G
import cnh_proposal_attribution_scenes as S

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT/'artifacts.local/work/cnh-readout-pilot2-20261005'
FRAMES = np.arange(3, 16)
DELTAS = [1., 2., 5., -5., -10., -15., -20.]
PRIMARY = 'primary1.2-2.1m'
BASE_THRESHOLD = .8557642486787612
BASE_ARMS = ('M3', 'V_retest', 'VD', 'D')
VALID_CATEGORIES = ('clear', 'pass0-10cm', 'contact0-2cm', 'contact2-5cm', 'contact>5cm')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    import json
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def groups_for(scenes):
    groups = {'all': np.ones(len(scenes), bool)}
    for q, name in ((0, 'HEAD'), (1, 'BODY')):
        groups[name] = np.array([s['group'] == q for s in scenes])
        for context in ('none', 'panel'):
            groups[name+'/'+context] = np.array([s['group'] == q and s['context'] == context for s in scenes])
    for context in ('none', 'panel'):
        groups[context] = np.array([s['context'] == context for s in scenes])
    groups['FOV_IN'] = np.array([s['fov_in'] for s in scenes])
    groups['FOV_OUT'] = ~groups['FOV_IN']
    return groups


def auc_metrics(scores, scenes, boot):
    ranges = np.array([s['front_range_m'] for s in scenes])
    domains = {f'{lo:g}-{hi:g}m': (ranges >= lo)&(ranges < hi) for lo, hi in U.BINS}
    domains.update({PRIMARY: (ranges >= 1.2)&(ranges < 2.1), '1.6-2.6m': (ranges >= 1.6)&(ranges < 2.6)})
    groups = groups_for(scenes)
    result, per_scene = {}, {domain: {} for domain in domains}
    for domain, masks in domains.items():
        values = {}
        for arm, score in scores.items():
            value = np.array([U.binary_auc(score[i, [0, 1]][..., mask], score[i, [4, 5, 6]][..., mask])
                              for i, mask in enumerate(masks)], float)
            values[arm] = value
            per_scene[domain][arm] = [None if not np.isfinite(x) else float(x) for x in value]
        result[domain] = {}
        for group, keep in groups.items():
            cell = dict(scenes=int(keep.sum()), arms={}, paired_minus_M3={}, paired_minus_VD={})
            K = next(iter(scores.values())).shape[2]
            cell.update(positive_n=int(masks[keep].sum()*2*K), negative_n=int(masks[keep].sum()*3*K))
            for arm, val in values.items():
                cell['arms'][arm] = U.macro_summary(val, keep, boot)[0]
                if arm != 'M3':
                    cell['paired_minus_M3'][arm] = U.macro_summary(val-values['M3'], keep, boot)[0]
                if arm != 'VD':
                    cell['paired_minus_VD'][arm] = U.macro_summary(val-values['VD'], keep, boot)[0]
            result[domain][group] = cell
    return result, per_scene


def load_baselines():
    """Reconstruct old sealed scores and prove full48 parity before reading R."""
    plan_path, result_path = OLD/'PLAN.json', OLD/'result.json'
    plan, old = U.read(plan_path), U.read(result_path)
    units = plan['eval_units']
    if len(units) != 48 or units != sorted(set(units)):
        raise ValueError('Original pilot2 must retain48 unique sorted scenes')
    expected_hashes = old['provenance']['input_sha256']
    hashes = {str(plan_path): U.sha(plan_path), str(result_path): U.sha(result_path)}

    def identity(path):
        path = Path(path)
        digest = U.sha(path)
        if str(path) not in expected_hashes or expected_hashes[str(path)] != digest:
            raise ValueError('Input differs from sealed original pilot2 evaluation: '+str(path))
        hashes[str(path)] = digest

    scenes, visible = P2.fresh_geometry(OLD, units, hashes)
    rows, row_path = T.rows_for(OLD, 'fresh_evaluation')
    identity(row_path)
    scores = {}
    for arm in ('V_retest', 'VD'):
        raw = []
        for seed in range(3):
            value, path = T.prediction(OLD, 'fresh_evaluation', arm, seed)
            identity(path)
            raw.append(value)
        scores[arm] = P2.fresh_scores(np.mean(raw, axis=0), rows, units, scenes)
    raw, path = T.prediction(OLD, 'fresh_evaluation', 'M3')
    identity(path)
    scores['M3'] = P2.fresh_scores(raw, rows, units, scenes)
    path, rp = OLD/'reference/all_scores.npz', OLD/'reference/scores_receipt.json'
    identity(path); identity(rp)
    receipt = U.read(rp)
    if receipt['status'] != 'COMPLETE' or receipt['score_sha256'] != hashes[str(path)] or receipt['plan_sha256'] != hashes[str(plan_path)]:
        raise ValueError('Original D reference receipt mismatch')
    with np.load(path, allow_pickle=False) as z:
        if z['units'].tolist() != units or not np.array_equal(z['frames'], FRAMES):
            raise ValueError('Original D axes differ')
        scores['D'] = z['D'].copy()
    scores = {arm: scores[arm] for arm in BASE_ARMS}
    boot = P2.boot_weights(48, int(plan['bootstrap']['seed']))
    auc, _ = auc_metrics(scores, scenes, boot)
    checks = {}
    for arm in BASE_ARMS:
        value = auc[PRIMARY]['all']['arms'][arm]['value']
        expected = old['auc'][PRIMARY]['all']['arms'][arm]['value']
        checks[arm] = dict(value=value, original=expected, max_abs=abs(value-expected))
        if abs(value-expected) > 1e-12:
            raise ValueError('Full pilot2 baseline macroAUC parity failed: '+arm)
    # Exact all-object labels at the interpolated .9m reference remain evaluator-only.
    categories, refs, covered = [], [], []
    for scene in scenes:
        unit = scene['unit']
        tp, op = OLD/'truth/evaluation'/f'unit{unit}.json', OLD/'observations/evaluation'/f'unit{unit}.npz'
        identity(tp)
        truth = U.read(tp)
        label = np.asarray(truth['categories'])[:, FRAMES, scene['group']]
        if label.shape != (7, 13) or not np.isin(label, VALID_CATEGORIES).all():
            raise ValueError('Actual all-object categories incomplete; UNKNOWN is not clear')
        with np.load(op, allow_pickle=False) as z:
            travel = z['travel'][FRAMES].copy()
        hashes[str(op)] = U.sha(op)
        current_refs, current_covered = [], []
        for boxes in truth['boxes']:
            ranges, ref = G.deadline_reference(G.corners(boxes[0]), travel)
            np.testing.assert_allclose(ranges, scene['front_range_m'], atol=1e-12, rtol=0)
            mesh = np.concatenate([S.box_mesh(b['lo'], b['hi']) for b in boxes])
            current_refs.append(G.surface_category(G.transform(mesh, ref['reference_pose']), scene['group']) if ref['covered'] else 'censored')
            current_covered.append(ref['covered'])
        categories.append(label); refs.append(current_refs); covered.append(current_covered)
    for path, digest in hashes.items():
        if path in expected_hashes and digest != expected_hashes[path]:
            raise ValueError('Geometry provenance mismatch: '+path)
    return scores, scenes, np.stack(categories), visible, np.asarray(refs), np.asarray(covered), dict(
        status='PASS', full48_primary_auc=checks, source_input_sha256=hashes,
        all_object_truth='Original allphysical surface categories plus independently interpolated .9m reference',
        original_fov_in=int(sum(s['fov_in'] for s in scenes)), original_fov_out=int(sum(not s['fov_in'] for s in scenes)))


def events(score, spec, ranges, covered):
    alarm = score >= spec['threshold'] if spec['operator'] == '>=' else score > spec['threshold']
    stopped, first = alarm.any(-1), alarm.argmax(-1)
    first_range = ranges[np.arange(len(score))[:, None, None], first]
    timely = stopped & (first_range >= .9) & covered[..., None]
    return dict(alarm=alarm, stopped=stopped, first_index=first, first_range=first_range,
                timely=timely, covered=covered, early=stopped & (first_range >= 2.1))


def episode_thresholds(scores, categories):
    K = scores['M3'].shape[2]
    clear = np.all(categories[:, [5, 6]] == 'clear', -1)
    keep = np.broadcast_to(clear[..., None], (len(clear), 2, K))
    base = scores['M3'][:, [5, 6]].max(-1)[keep]
    target = int((base >= BASE_THRESHOLD).sum())
    if not len(base):
        raise ValueError('No actualclear outside15/20 episode available for matching')
    result = {'M3': dict(threshold=BASE_THRESHOLD, operator='>=', n=len(base), allowed_stops=target,
        actual_stops=target, unattained_stops=0, rule='Frozen original M3 threshold')}
    for arm, score in scores.items():
        if arm != 'M3':
            result[arm] = U.matched_threshold(score[:, [5, 6]].max(-1)[keep], target)
    return result


def pooled(events_array, denominator, keep, boot):
    numerator = events_array.sum((1, 2)).astype(int)
    count = denominator.sum((1, 2)).astype(int)
    return U.pooled_rate(numerator, count, keep, boot)


def sequence_metrics(scores, scenes, categories, visible, refs, covered, thresholds, boot):
    groups, ranges = groups_for(scenes), np.array([s['front_range_m'] for s in scenes])
    N, _, K, _ = scores['M3'].shape
    ledgers = {arm: events(score, thresholds[arm], ranges, covered) for arm, score in scores.items()}
    cells, differences = {group: {} for group in groups}, {group: {} for group in groups}
    for group, keep in groups.items():
        for arm, e in ledgers.items():
            cell = dict(threshold=thresholds[arm], branches={}, labels={})
            for name, ids in [('inside1_2cm', [0, 1]), ('inside5cm', [2]), ('outside5_10_pass', [3, 4]), ('outside15_20cm', [5, 6])]:
                den = np.ones((N, len(ids), K), bool)
                stop, _ = pooled(e['stopped'][:, ids], den, keep, boot)
                valid = np.broadcast_to(covered[:, ids, None], den.shape)
                timely, _ = pooled(e['timely'][:, ids] & valid, valid, keep, boot)
                censored = ~valid
                vf = visible[np.arange(N)[:, None, None], np.asarray(ids)[None, :, None], e['first_index'][:, ids]]
                early = keep[:, None, None] & e['early'][:, ids]
                selected = keep[:, None, None] & e['timely'][:, ids]
                leads = (e['first_range'][:, ids]-.5)/.8
                cell['branches'][name] = dict(first_stops=stop, timely=timely,
                    right_or_left_censored_n=int(censored[keep].sum()),
                    unalarmed_censored_n=int((censored & ~e['stopped'][:, ids])[keep].sum()),
                    early_first_front_ge2p1=dict(n=int(early.sum()), target_visible=int((early & vf).sum()), no_target_visible_ray=int((early & ~vf).sum())),
                    median_lead_conditional_timely_s=float(np.median(leads[selected])) if selected.any() else None)
            masks = {'actual_clear_all13': np.all(categories == 'clear', -1),
                     'actual_pass_all13': np.all(categories == 'pass0-10cm', -1),
                     'contact0-2cm_at_deadline': covered & (refs == 'contact0-2cm'),
                     'contact2-5cm_at_deadline': covered & (refs == 'contact2-5cm'),
                     'contact>5cm_at_deadline': covered & (refs == 'contact>5cm')}
            for label, mask in masks.items():
                den = np.broadcast_to(mask[..., None], (N, 7, K))
                stops, _ = pooled(e['stopped'] & den, den, keep, boot)
                timely, _ = pooled(e['timely'] & den, den, keep, boot)
                minutes = stops['n']*2.6/60
                cell['labels'][label] = dict(first_stops=stops, timely=timely, proxy_minutes=minutes,
                    first_stops_per_proxy_min=stops['stops']/minutes if minutes else None)
            cells[group][arm] = cell
        for arm, e in ledgers.items():
            differences[group][arm] = {}
            for comparator in ('M3', 'VD'):
                if arm == comparator:
                    continue
                a, b = e['timely'][:, :2], ledgers[comparator]['timely'][:, :2]
                den = np.broadcast_to(covered[:, :2, None], a.shape)
                am, ad = pooled(a & den, den, keep, boot)
                bm, bd = pooled(b & den, den, keep, boot)
                differences[group][arm][comparator] = dict(delta_stops=am['stops']-bm['stops'],
                    delta_rate=am['rate']-bm['rate'] if am['rate'] is not None and bm['rate'] is not None else None,
                    paired_scene_ci95=U.interval(ad-bd), rescues=int((a & ~b & den)[keep].sum()),
                    losses=int((b & ~a & den)[keep].sum()), n=am['n'])
    return dict(cells=cells, paired_shallow_changes=differences), ledgers


def decide(auc_delta, timely_delta, n):
    if auc_delta is None or timely_delta is None or not n:
        return dict(branch='NOT_EVALUABLE', reason='Primary macroAUC or FOVout covered denominator unavailable')
    scaled = timely_delta/n*88
    low = auc_delta <= .02 and scaled <= 5
    high = auc_delta >= .05 or scaled >= 10
    return dict(branch='READOUT_NEAR_LOCATION_FREE_LIMIT' if low else 'LOCATION_FREE_HEADROOM' if high else 'INTERMEDIATE',
        R_any_minus_VD_macro_auc=auc_delta, FOV_OUT_timely_delta=timely_delta, FOV_OUT_n=n,
        timely_delta_normalized_to88=scaled, interpretation='Signed point differences, descriptive branches; no equivalence test or strict sensor upper bound',
        low_rule='AUC delta<=.02 AND FOVout timely rate delta<=5/88',
        high_rule='AUC delta>=.05 OR FOVout timely rate delta>=10/88', automatic_training=False)


def evaluate(out, score_path=None):
    out = Path(out)
    started = time.monotonic()
    pp = out/'PLAN.json'
    if not pp.exists():
        raise FileNotFoundError('Scientific PLAN must be frozen before any R scores are opened')
    plan = U.read(pp)
    if (out/'result.json').exists():
        raise FileExistsError('Completed location-reference result is immutable')
    full_scores, full_scenes, full_cats, full_visible, full_refs, full_covered, parity = load_baselines()
    units = plan.get('units', plan.get('eval_units'))
    if not units or len(set(units)) != len(units):
        raise ValueError('PLAN requires explicit unique selectedunits')
    K = int(plan.get('K', plan.get('eval_K', 4)))
    replicas = plan.get('replica_indices', list(range(K)))
    if K not in (2, 4) or len(replicas) != K or len(set(replicas)) != K or any(k not in range(4) for k in replicas):
        raise ValueError('PLAN supports immutable K4 or K2 originalreplica selection only')
    full_units = [s['unit'] for s in full_scenes]
    ids = np.array([full_units.index(u) for u in units])
    scenes = [full_scenes[i] for i in ids]
    categories, visible, refs, covered = (a[ids] for a in (full_cats, full_visible, full_refs, full_covered))
    scores = {arm: a[ids][:, :, replicas] for arm, a in full_scores.items()}
    path = Path(score_path) if score_path else out/'scores/all_scores.npz'
    before = U.sha(path)
    with np.load(path, allow_pickle=False) as z:
        if z['units'].tolist() != units:
            raise ValueError('Location scores unit order differs from frozenPLAN')
        if 'frames' in z and not np.array_equal(z['frames'], FRAMES):
            raise ValueError('Location score frameidentity mismatch')
        any_score, query_score = z['R_any'].copy(), z['Rquery'].copy()
    expected = (len(units), 7, K, 13)
    if any_score.shape != expected or query_score.shape != expected+(2,):
        raise ValueError('R_any/Rquery must retain all7branch/Kreplica/13frame axes')
    scores['R_any'] = any_score
    # Only now may evaluator-only actualqueryband select the sealedauxiliaryscore.
    scores['R_query'] = np.stack([query_score[i, ..., scene['group']] for i, scene in enumerate(scenes)])
    if any(not np.isfinite(x).all() for x in scores.values()):
        raise ValueError('Nonfinite location score retained as error, never dropped')
    bcfg = plan.get('bootstrap', {})
    if int(bcfg.get('n', 1000)) != 1000:
        raise ValueError('Frozen location assay requires1000 whole-scene paired draws')
    boot = P2.boot_weights(len(units), int(bcfg.get('seed', 2026100601)))
    auc, per_scene = auc_metrics(scores, scenes, boot)
    thresholds = episode_thresholds(scores, categories)
    sequences, ledgers = sequence_metrics(scores, scenes, categories, visible, refs, covered, thresholds, boot)
    change = sequences['paired_shallow_changes']['FOV_OUT']['R_any']['VD']
    decision = decide(auc[PRIMARY]['all']['paired_minus_VD']['R_any']['value'], change['delta_stops'], change['n'])
    result = dict(status='COMPLETE', decision=decision, auc=auc, per_scene_auc=per_scene,
        scenes=scenes, baseline_full48_parity=parity, thresholds=thresholds, sequences=sequences,
        n=dict(scenes=len(units), K=K, replicas=replicas, FOV_IN=int(sum(s['fov_in'] for s in scenes)),
            FOV_OUT=int(sum(not s['fov_in'] for s in scenes)), branch_episodes=len(units)*7*K,
            retained_frame_scores=len(units)*7*K*13, UNKNOWN_categories=0,
            censored_target_query_episodes=int((~covered).sum()*K)),
        bootstrap=dict(n=1000, seed=int(bcfg.get('seed', 2026100601)), unit='wholepairedscene', thresholds_fixed=True),
        scope=dict(primary='R_any removes side/band/z using generatorprior, but ANYHEIGHT target-class reference differs from a deployable queryreader',
            auxiliary='R_query includes oppositeheight intrusions as negatives; bothqueryscores sealed before evaluator selects truthband',
            cost='Global actualclear outside15/20 firststop frontier matched to selectedoriginalM3 count; no subgroupcalibration or naturalcandidateevaluation',
            privileges='Exact background geometry/rho including FLOOR and BACK even in none; truecurrentanchor; exactphoton law and assumedtarget/positionprior',
            limitations='Finitegrid and deliberateprior mismatch (.65frozen observation vs Uniform(.6,2.6) templateprior): lowR is not informationimpossibility or stricttheoreticallimit; reusedsimulationDevelopment only; no hardwareclaim or M3replacement'),
        elapsed_s=time.monotonic()-started, provenance=dict(plan_sha256=U.sha(pp), score_path=str(path), score_sha256=before,
            evaluator_sha256=U.sha(__file__), baseline_input_sha256=parity['source_input_sha256']))
    if U.sha(path) != before:
        raise ValueError('Location score changed duringevaluation')
    save(out/'result.json', result)
    np.savez(out/'episode_ledger.npz', units=units, frames=FRAMES, categories=categories, reference_categories=refs,
             covered=covered, **{arm+'_'+field: value for arm, ledger in ledgers.items() for field, value in ledger.items()})
    report(result, out)
    return result


def report(result, out):
    main = result['auc'][PRIMARY]['all']
    fmt = lambda x: '—' if x is None else f'{x:.4f}'
    lines = [result['decision']['branch'], '', '# 位置边缘化参照 R', '',
        f"{result['n']['scenes']}场景，K{result['n']['K']}；视场内/外{result['n']['FOV_IN']}/{result['n']['FOV_OUT']}。全部7位移、13帧保留。", '',
        '| 臂 | 主macroAUC | 对VD差及场景95%区间 |', '|---|---:|---:|']
    for arm in main['arms']:
        d = main['paired_minus_VD'].get(arm)
        delta = '—' if d is None else f"{d['value']:+.4f} [{fmt(d['ci95'][0])},{fmt(d['ci95'][1])}]"
        lines.append(f"| {arm} | {fmt(main['arms'][arm]['value'])} | {delta} |")
    lines += ['', '| 层 | 臂 | 浅及时/覆盖分母 | 清晰外15/20首停/分母 |', '|---|---|---:|---:|']
    for group in ('all', 'FOV_IN', 'FOV_OUT'):
        for arm, cell in result['sequences']['cells'][group].items():
            a = cell['branches']['inside1_2cm']['timely']; b = cell['branches']['outside15_20cm']['first_stops']
            lines.append(f"| {group} | {arm} | {a['stops']}/{a['n']} | {b['stops']}/{b['n']} |")
    lines += ['', '| 层 | 主R_any AUC | R_query AUC | R_any−VD |', '|---|---:|---:|---:|']
    for group, cell in result['auc'][PRIMARY].items():
        lines.append(f"| {group} | {fmt(cell['arms']['R_any']['value'])} | {fmt(cell['arms']['R_query']['value'])} | {fmt(cell['paired_minus_VD']['R_any']['value'])} |")
    delta = result['sequences']['paired_shallow_changes']['FOV_OUT']['R_any']['VD']
    lines += ['', f"视场外R_any对VD：补回{delta['rescues']}、丢失{delta['losses']}，净{delta['delta_stops']:+d}/{delta['n']}；归一化88分母差{result['decision'].get('timely_delta_normalized_to88',0):+.2f}。",
        '每臂只用一条全体阈值，以M3冻结阈值在所选批的实际清晰首停整数成本匹配；属于评价前沿，不能部署。',
        'R_any是未知高度的任意高度类判别；R_query为两个公开查询的补充诊断。none仍知道地面和后墙，真实当前锚点及精确光子模型仍是特权。',
        '有限网格、额外tiny先验以及位置先验与固定.65评价目标的错配，使低R不能证明传感器信息极限。不自动训练，不替换M3；全部为已消费模拟Development。',
        f"CPU评价耗时{result['elapsed_s']:.2f}秒；完整计算耗时另见生命周期收据。", '']
    (Path(out)/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def check():
    assert decide(.02, 5, 88)['branch'] == 'READOUT_NEAR_LOCATION_FREE_LIMIT'
    assert decide(.0200001, 5, 88)['branch'] == 'INTERMEDIATE'
    assert decide(.05, 0, 88)['branch'] == 'LOCATION_FREE_HEADROOM'
    assert decide(0., 5, 44)['branch'] == 'LOCATION_FREE_HEADROOM'
    assert decide(0., 2, 44)['branch'] == 'READOUT_NEAR_LOCATION_FREE_LIMIT'
    # Firstalarm, censoring and whole-tie thresholds matter for the scientificresult.
    score = np.array([[[[0., 1., 2.], [0., 0., 3.]]]])
    ranges = np.array([[1.3, 1., .7]])
    e = events(score, dict(threshold=1., operator='>='), ranges, np.array([[True]]))
    assert e['first_index'].tolist() == [[[1, 2]]]
    assert e['timely'].tolist() == [[[True, False]]]
    censored = events(score, dict(threshold=1., operator='>='), ranges, np.array([[False]]))
    assert not censored['timely'].any() and censored['stopped'].all()
    t = U.matched_threshold(np.array([2., 2., 1., 0.]), 1)
    assert t['actual_stops'] == 0 and t['unattained_stops'] == 1
    print('PASS branchboundaries, K2-normalizeddenominators, earliestalarm, censoring and wholeties; no scientificscoresread', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=('check', 'baseline-parity', 'evaluate'))
    p.add_argument('--out', type=Path)
    p.add_argument('--scores', type=Path)
    a = p.parse_args()
    if a.stage == 'check':
        check()
    elif a.stage == 'baseline-parity':
        *_, receipt = load_baselines()
        if a.out:
            save(a.out/'baseline_parity.json', receipt)
        print(receipt['status'], receipt['full48_primary_auc'], flush=True)
    else:
        if a.out is None:
            p.error('--out required for frozen scientificevaluation')
        value = evaluate(a.out, a.scores)
        print(value['decision'], flush=True)
