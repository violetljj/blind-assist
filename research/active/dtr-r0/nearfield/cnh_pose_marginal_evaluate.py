"""Read-only cost/sequence/AUC evaluation of sealed pose-marginal references.

Evaluation-batch matched thresholds describe frontiers, never deployment
calibration. K4, deltas and frames remain nested within48 whole scenes.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import cnh_unknown_target_reference_evaluate as U
import cnh_pose_factorial_evaluate as F

ROOT = Path(__file__).resolve().parents[4]
RUN_OUT = ROOT / 'artifacts.local/work/cnh-pose-marginal-reference-20261004'
OUT = RUN_OUT / 'evaluation'
OLD = ROOT / 'artifacts.local/work/cnh-displacement-ceiling-20261003'
REFERENCE = ROOT / 'artifacts.local/work/cnh-unknown-target-reference-20261004'
ARMS = ('M3_smooth', 'marginal8', 'marginal12', 'plugin8', 'plugin12', 'true8', 'true12')
BASE = 'M3_smooth'
FRAMES = np.arange(3, 16)
DELTA = np.array([1, 2, 5, -5, -10, -15, -20])
SEED = 2026100421
N_BOOT = 1000
DOMAINS = {'main1.2-2.1m': (1.2, 2.1), 'near0.9-1.2m': (.9, 1.2), 'combined0.9-2.1m': (.9, 2.1)}
BASE_THRESHOLD = .8557642486787612


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def input_hashes(paths):
    return {str(Path(p)): U.sha(p) for p in paths}


def bootstrap():
    rng = np.random.default_rng(SEED)
    return np.array([np.bincount(rng.integers(48, size=48), minlength=48) for _ in range(N_BOOT)])


def geometry():
    paths = [REFERENCE / 'scores/all_scores.npz']
    with np.load(paths[0], allow_pickle=False) as cache:
        units = cache['units'].tolist()
        baseline = cache[BASE].copy()
    if len(units) != 48 or len(set(units)) != 48 or baseline.shape != (48, 7, 4, 13):
        raise ValueError('Original48 score axes or identities differ')
    scenes, categories, visible = [], [], []
    for unit in units:
        truth_path = OLD / 'truth' / f'unit{unit}.json'
        template_path = OLD / 'templates' / f'unit{unit}.npz'
        paths.extend([truth_path, template_path])
        truth = U.read(truth_path)
        group = int(truth['group'])
        if truth['unit'] != unit or truth['intrusion_cm'] != DELTA.tolist():
            raise ValueError('Truth identity/delta order mismatch')
        labels = np.asarray(truth['categories'])
        if labels.shape != (7, 16, 2):
            raise ValueError('Stored actualcategory axes differ')
        category = labels[:, FRAMES, group]
        if not np.isin(category, ['clear', 'pass0-10cm', 'contact0-2cm', 'contact2-5cm', 'contact>5cm']).all():
            raise ValueError('Unknown actualcategory must not become clear')
        with np.load(template_path, allow_pickle=False) as cache:
            target_visible = np.any(cache['object_id'][:, FRAMES] == 0, axis=(-3, -2, -1))
        if target_visible.shape != (7, 13):
            raise ValueError('Target visibility axes differ')
        vf = float(target_visible[:2].mean())
        scenes.append(dict(unit=unit, group=group, context=truth['context'],
            front_range_m=np.asarray(truth['front_range_m'])[FRAMES].tolist(),
            visibility=dict(vf=vf, fov_in=vf>=.5)))
        categories.append(category)
        visible.append(target_visible)
    return baseline, scenes, np.stack(categories), np.stack(visible), paths


def strata(scenes):
    high = np.array([s['visibility']['fov_in'] for s in scenes])
    if int(high.sum()) != 36:
        raise ValueError('Frozen all13frame visibility strata must remain36high/12low')
    return {'all': np.ones(48, bool), 'high36': high, 'low12': ~high}


def event_arrays(score, threshold, operator, ranges):
    alarm = score >= threshold if operator == '>=' else score > threshold
    stopped = alarm.any(-1)
    first = alarm.argmax(-1)
    first_range = ranges[np.arange(48)[:, None, None], first]
    covered = (ranges[:, 0] >= .9) & (ranges[:, -1] <= .9)
    timely = stopped & (first_range >= .9) & covered[:, None, None]
    return dict(alarm=alarm, stopped=stopped, first_index=first, first_range=first_range,
                timely=timely, covered=covered, early=stopped & (first_range >= 2.1))


def baseline_check_data(baseline, scenes, categories, visible):
    ranges = np.array([s['front_range_m'] for s in scenes])
    groups = strata(scenes)
    events = event_arrays(baseline, BASE_THRESHOLD, '>=', ranges)
    clear = np.all(categories == 'clear', -1)
    actual_clear_n = int(clear[:, [5, 6]].sum()*4)
    clear_stops = int((events['stopped'][:, [5, 6]] & clear[:, [5, 6], None]).sum())
    timely = {g: dict(stops=int(events['timely'][keep, :2].sum()), n=int(keep.sum()*8)) for g, keep in groups.items()}
    main = (ranges>=1.2) & (ranges<2.1)
    frame_n = int(main.sum()*4)
    frame_stops = [int((events['alarm'][:, j] & main[:, None]).sum()) for j in (5, 6)]
    expected = {'all': dict(stops=309, n=384), 'high36': dict(stops=277, n=288), 'low12': dict(stops=32, n=96)}
    if timely != expected or actual_clear_n != 384 or clear_stops != 18 or frame_n != 1152 or frame_stops != [5, 11]:
        raise ValueError('Frozen M3 costs/denominators/sequences did not reproduce')
    # Actualoutside5/10 are pass, not clear negative evidence.
    if not np.all(categories[:, [3, 4]] == 'pass0-10cm'):
        raise ValueError('Original pass-only branch geometry changed')
    return dict(status='PASS', threshold=BASE_THRESHOLD, operator='>=', timely=timely,
        actual_clear_outside15_20=dict(stops=18, n=384, proxy_minutes=384*2.6/60),
        main_frame_outside15_20=dict(stops=frame_stops, n=frame_n),
        outside5_10_actual_pass_episodes=384, unknown_categories=0)


def protocol(units):
    return dict(run='CNH_POSE_MARGINAL_EVALUATION_20261004', units=units, arms=list(ARMS),
        score_axes=['scene48', 'delta7', 'replica4', 'frame13'], frames=FRAMES.tolist(), delta_cm=DELTA.tolist(),
        primary_cost=dict(merged_indices=[5, 6], labels='all13frames actualclear only',
            target='Frozen M3>=0.8557642486787612 actual18/384 firststops',
            new_threshold='strict > largest whole-score tie groups under18; one global48 threshold perarm',
            exposure='13 x .2 seconds per episode including after alarm; proxy only'),
        secondary_frame_cost=dict(domain=DOMAINS['main1.2-2.1m'], outside_indices=[5, 6],
            budgets=[5, 11], denominators=[1152, 1152], inheritance='global48 threshold unchanged in high36/low12 and whole13frame sequence'),
        strata='all13frame inside1/2 target-visible branch fraction>=.5: high36/low12; evaluator only',
        sequence='firstalarm over all13frames; timely firstfront>=.9m; early firstfront>=2.1m; no future scoring',
        auc=dict(positives=[0, 1], negatives=[4, 5, 6], domains=DOMAINS, equal_scene_macro=True, pooled=True),
        bootstrap=dict(seed=SEED, replicates=N_BOOT, unit='48 whole scenes; retain deltas/K4/frames together',
            pairing='one common scene multiplicity matrix across arms/domains/strata; thresholds held fixed'),
        comparison='Each arm minus frozen M3; report both marginal8/12 and every reference, no post-result arm selection',
        extra_branches='inside5/outside5/outside10 retained descriptively; no primary gate contribution',
        limits=['known geometry/target/background/current-anchor privileged; pose prior assumed',
            'evaluation-batch thresholds are descriptive cost matching, not deployment calibration',
            'conditional score bootstrap excludes pose-prior, model, threshold and real-sensor uncertainty',
            'K4 and12 low scenes do not become48 independent low units; no training or hardware claims'])


def freeze_plan():
    path = OUT / 'PLAN.json'
    if path.exists():
        frozen = U.read(path)
        expected_protocol = json.loads(json.dumps(protocol(frozen['units'])))
        if any(frozen.get(key) != value for key, value in expected_protocol.items()):
            raise ValueError('Existing frozen statistical protocol differs')
        for input_path, digest in frozen['fixed_input_sha256'].items():
            if U.sha(input_path) != digest:
                raise ValueError('Frozen baseline input identity changed')
        current_sources = input_hashes([Path(__file__), Path(U.__file__), Path(F.__file__)])
        if current_sources == frozen['source_sha256']:
            return frozen
        amendment_path = OUT / 'IMPLEMENTATION_AMENDMENT.json'
        amendment = U.read(amendment_path)
        if (amendment.get('frozen_evaluation_plan_sha256') != U.sha(path) or
                amendment.get('previous_source_sha256') != frozen['source_sha256'] or
                amendment.get('effective_source_sha256') != current_sources or
                amendment.get('statistical_methods_changed') is not False):
            raise ValueError('Implementation identity change lacks compatible amendment')
        # Return an effective view, preserving every byte of the frozen PLAN.
        return dict(frozen, source_sha256=current_sources,
                    frozen_source_sha256=frozen['source_sha256'],
                    implementation_amendment_sha256=U.sha(amendment_path))
    baseline, scenes, categories, visible, paths = geometry()
    baseline_check = baseline_check_data(baseline, scenes, categories, visible)
    plan = protocol([s['unit'] for s in scenes])
    plan['fixed_input_sha256'] = input_hashes(paths)
    plan['source_sha256'] = input_hashes([Path(__file__), Path(U.__file__), Path(F.__file__)])
    save(path, plan)
    save(OUT / 'baseline_check.json', dict(**baseline_check, plan_sha256=U.sha(path), pid=os.getpid()))
    return plan


def weighted_pooled_auc(score, masks, keep, boot):
    """Exact weighted pair AUC; common whole-scene multiplicities, ties0.5."""
    values, owners, positive = [], [], []
    for scene in np.flatnonzero(keep):
        for delta, label in ((0, True), (1, True), (4, False), (5, False), (6, False)):
            value = score[scene, delta][:, masks[scene]].ravel()
            values.extend(value.tolist())
            owners.extend([scene]*len(value))
            positive.extend([label]*len(value))
    if not values:
        return dict(value=None, ci95=[None, None], positive_n=0, negative_n=0), np.full(len(boot), np.nan)
    order = np.argsort(values, kind='stable')
    values = np.asarray(values)[order]
    owners = np.asarray(owners)[order]
    positive = np.asarray(positive, bool)[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1]

    def weighted(weights):
        point = weights[:, owners]
        pos = np.add.reduceat(point*positive, starts, axis=1)
        neg = np.add.reduceat(point*~positive, starts, axis=1)
        numerator = (pos*(np.cumsum(neg, axis=1)-.5*neg)).sum(1)
        denominator = pos.sum(1)*neg.sum(1)
        return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator>0)

    point = weighted(np.ones((1, 48)))[0]
    draws = np.concatenate([weighted(boot[i:i+64]) for i in range(0, len(boot), 64)])
    return dict(value=float(point) if np.isfinite(point) else None, ci95=U.interval(draws),
        positive_n=int(positive.sum()), negative_n=int((~positive).sum())), draws


def auc_summary(scores, ranges, groups, boot):
    cells = {g: {} for g in groups}
    changes = {g: {} for g in groups}
    per_scene = {}
    for domain, (lo, hi) in DOMAINS.items():
        masks = (ranges>=lo) & (ranges<hi)
        scene_values, sampled = {}, {g: {} for g in groups}
        for arm, score in scores.items():
            values = []
            for i, mask in enumerate(masks):
                a = U.binary_auc(score[i, [0, 1]][..., mask], score[i, [4, 5, 6]][..., mask])
                values.append(np.nan if a is None else a)
            scene_values[arm] = np.asarray(values)
        per_scene[domain] = {arm: [None if not np.isfinite(x) else float(x) for x in value]
                             for arm, value in scene_values.items()}
        for group, keep in groups.items():
            cells[group][domain] = {}
            changes[group][domain] = {}
            for arm, score in scores.items():
                macro, md = U.macro_summary(scene_values[arm], keep, boot)
                pooled, pd = weighted_pooled_auc(score, masks, keep, boot)
                cells[group][domain][arm] = dict(macro=macro, pooled=pooled)
                sampled[group][arm] = dict(macro=md, pooled=pd)
            for arm in scores:
                if arm == BASE:
                    continue
                changes[group][domain][arm] = {}
                for kind in ('macro', 'pooled'):
                    a, b = cells[group][domain][arm][kind]['value'], cells[group][domain][BASE][kind]['value']
                    changes[group][domain][arm][kind] = dict(delta=a-b if a is not None and b is not None else None,
                        paired_scene_ci95=U.interval(sampled[group][arm][kind]-sampled[group][BASE][kind]))
    return dict(cells=cells, changes_minus_M3=changes, per_scene=per_scene)


def count_metric(events, denominator, keep, boot):
    numerator = events.sum((1, 2)).astype(int)
    n = denominator.sum((1, 2)).astype(int)
    return U.pooled_rate(numerator, n, keep, boot)


def sequence_summary(scores, thresholds, scenes, categories, visible, groups, boot):
    ranges = np.array([s['front_range_m'] for s in scenes])
    cells = {g: {} for g in groups}
    changes = {g: {} for g in groups}
    ledgers = {}
    for arm, score in scores.items():
        spec = thresholds[arm]
        events = event_arrays(score, spec['threshold'], spec['operator'], ranges)
        ledgers[arm] = events
        for group, keep in groups.items():
            cells[group][arm] = {}
            for j, delta in enumerate(DELTA):
                cells[group][arm][f'delta{delta:+g}cm'] = F.sequence_cell(score, spec['threshold'], spec['operator'],
                    ranges, visible, [j], keep, boot)
            for name, indices in {'inside1_2cm': [0, 1], 'outside15_20cm': [5, 6], 'outside5_10_pass': [3, 4]}.items():
                cells[group][arm][name] = F.sequence_cell(score, spec['threshold'], spec['operator'],
                    ranges, visible, indices, keep, boot)
            label_cells = {}
            label_masks = {'actual_clear_all13': np.all(categories=='clear', -1),
                'actual_pass_all13': np.all(categories=='pass0-10cm', -1),
                'actual_contact_anyframe': np.any(np.char.startswith(categories, 'contact'), -1)}
            for name, mask in label_masks.items():
                den = np.broadcast_to(mask[..., None], (48, 7, 4))
                stop_metric, _ = count_metric(events['stopped'] & den, den, keep, boot)
                n = stop_metric['n']
                minutes = n*2.6/60
                first_range = events['first_range'][keep][(events['stopped'] & den)[keep]]
                label_cells[name] = dict(first_stops=stop_metric, n=n, proxy_minutes=minutes,
                    first_stops_per_proxy_min=stop_metric['stops']/minutes if minutes else None,
                    early_first_ge2p1=int((events['early'] & den)[keep].sum()),
                    median_first_front_m=float(np.median(first_range)) if len(first_range) else None)
            cells[group][arm]['label_aware'] = label_cells
    for group, keep in groups.items():
        base = ledgers[BASE]
        for arm in scores:
            if arm == BASE:
                continue
            current = ledgers[arm]
            changes[group][arm] = {}
            for kind in ('timely', 'stopped'):
                a, b = current[kind][:, :2], base[kind][:, :2]
                n = np.broadcast_to(base['covered'][:, None, None] if kind=='timely' else True, a.shape)
                am, ad = count_metric(a & n, n, keep, boot)
                bm, bd = count_metric(b & n, n, keep, boot)
                changes[group][arm][kind] = dict(delta_rate=am['rate']-bm['rate'],
                    paired_scene_ci95=U.interval(ad-bd), rescues=int((a & ~b)[keep].sum()),
                    losses=int((b & ~a)[keep].sum()), n=am['n'])
    return dict(cells=cells, changes_minus_M3=changes), ledgers


def episode_thresholds(scores, categories):
    clear = np.all(categories[:, [5, 6]] == 'clear', -1)
    keep = np.broadcast_to(clear[..., None], (48, 2, 4))
    base_values = scores[BASE][:, [5, 6]].max(-1)[keep]
    target = int((base_values >= BASE_THRESHOLD).sum())
    if target != 18 or len(base_values) != 384:
        raise ValueError('Frozen mergedactualclear cost differs from18/384')
    result = {BASE: dict(threshold=BASE_THRESHOLD, operator='>=', n=384, allowed_stops=18, actual_stops=18,
                        unattained_stops=0, rule='Frozen M3 threshold; no calibration')}
    for arm in scores:
        if arm != BASE:
            result[arm] = U.matched_threshold(scores[arm][:, [5, 6]].max(-1)[keep], target)
    return result


def frame_workpoints(scores, ranges, scenes, categories, visible, groups, boot):
    masks = (ranges>=1.2) & (ranges<2.1)
    take = np.broadcast_to(masks[:, None], (48, 4, 13))
    output = {}
    for index in (5, 6):
        target = int((scores[BASE][:, index][take] >= BASE_THRESHOLD).sum())
        if target != (5 if index==5 else 11) or int(take.sum()) != 1152:
            raise ValueError('Frozen frame cost changed')
        thresholds = {BASE: dict(threshold=BASE_THRESHOLD, operator='>=', n=1152, allowed_stops=target,
                                actual_stops=target, unattained_stops=0, rule='Frozen M3 threshold')}
        for arm in scores:
            if arm != BASE:
                thresholds[arm] = U.matched_threshold(scores[arm][:, index][take], target)
        cells, changes, draws = {g: {} for g in groups}, {g: {} for g in groups}, {g: {} for g in groups}
        for arm, score in scores.items():
            alarm = score>=thresholds[arm]['threshold'] if arm==BASE else score>thresholds[arm]['threshold']
            for group, keep in groups.items():
                cells[group][arm], draws[group][arm] = {}, {}
                for j, delta in enumerate(DELTA):
                    numerator = (alarm[:, j] & masks[:, None]).sum((1, 2))
                    n = masks.sum(1)*4
                    metric, sampled = U.pooled_rate(numerator, n, keep, boot)
                    cells[group][arm][f'delta{delta:+g}cm'] = metric
                    draws[group][arm][f'delta{delta:+g}cm'] = sampled
        for group in groups:
            for arm in scores:
                if arm != BASE:
                    changes[group][arm] = {name: dict(delta_rate=cell['rate']-cells[group][BASE][name]['rate'],
                        paired_scene_ci95=U.interval(draws[group][arm][name]-draws[group][BASE][name]))
                        for name, cell in cells[group][arm].items()}
        sequence, _ = sequence_summary(scores, thresholds, scenes, categories, visible, groups, boot)
        output[f'outside{-DELTA[index]:g}cm'] = dict(thresholds=thresholds, frames=cells,
            changes_minus_M3=changes, whole13frame_sequences=sequence,
            role='Descriptive primaryframe cost frontier; global48 threshold inherited, not deployment calibration')
    return output


def check_pooled():
    # Compare exact scene-multiplicity implementation with explicit duplicate scenes.
    rng = np.random.default_rng(421)
    score = rng.integers(-3, 4, size=(48, 7, 4, 13)).astype(float)
    masks = np.zeros((48, 13), bool)
    masks[:, 3:6] = True
    keep = np.arange(48)<4
    weights = np.zeros((3, 48), int)
    weights[0, :4] = [1, 2, 1, 3]
    weights[1, :4] = [0, 0, 2, 1]
    weights[2, :4] = [3, 1, 0, 0]
    _, values = weighted_pooled_auc(score, masks, keep, weights)
    for i, row in enumerate(weights):
        pos, neg = [], []
        for scene in np.flatnonzero(keep):
            pos.extend([score[scene, [0, 1]][..., masks[scene]].ravel()]*int(row[scene]))
            neg.extend([score[scene, [4, 5, 6]][..., masks[scene]].ravel()]*int(row[scene]))
        expected = U.binary_auc(np.concatenate(pos), np.concatenate(neg))
        if abs(values[i]-expected)>1e-12:
            raise ValueError('Weighted pooled AUC differs from explicit scene duplication')
    tied = U.matched_threshold(np.array([3., 3., 2., 1.]), 1)
    if tied['actual_stops'] != 0 or tied['threshold'] != 3:
        raise ValueError('Cost matcher split a forbidden tied group')
    return dict(status='PASS', weighted_pooled_auc_explicit_scene_replication=True, no_tie_splitting=True)


def check():
    plan = freeze_plan()
    result = check_pooled()
    save(OUT / 'cpu_check.json', dict(**result, plan_sha256=U.sha(OUT/'PLAN.json'), pid=os.getpid(),
         scenes=len(plan['units']), likelihood_scoring=False, model_inference=False, training=False))
    print('PASS frozen M3 sequence/frame costs, actualclear/pass labels, weighted whole-scene pooled AUC and ties')


def read_scores(units):
    plan_path = RUN_OUT/'PLAN.json'
    amendment_path = RUN_OUT/'PLAN_AMENDMENT.json'
    paths = [plan_path, amendment_path]
    plan_sha, amendment_sha = U.sha(plan_path), U.sha(amendment_path)
    performance_path = RUN_OUT/'PERFORMANCE_AMENDMENT.json'
    performance_sha = None
    if performance_path.exists():
        performance_sha = U.sha(performance_path)
        paths.append(performance_path)
    scores = {arm: [] for arm in ARMS}
    with np.load(REFERENCE/'scores/all_scores.npz', allow_pickle=False) as old:
        reference = {BASE: old[BASE].copy(), 'true8': old['oracle8'].copy(), 'true12': old['oracle12'].copy()}
    for i, unit in enumerate(units):
        path = RUN_OUT/'scores'/f'unit{unit}.npz'
        receipt_path = RUN_OUT/'units'/f'unit{unit}.json'
        receipt = U.read(receipt_path)
        if receipt.get('status') != 'COMPLETE':
            raise ValueError('New score unit receipt incomplete')
        if receipt.get('plan_sha256') != plan_sha or receipt.get('amendment_sha256') != amendment_sha:
            raise ValueError('Score does not belong to frozen24pose plan/amendment')
        if ('performance_amendment_sha256' in receipt and
                receipt['performance_amendment_sha256'] != performance_sha):
            raise ValueError('Score performance implementation identity mismatch')
        digest = U.sha(path)
        recorded = receipt.get('score_sha256')
        if recorded is None:
            candidates = receipt.get('output_sha256', receipt.get('outputs_sha256', {}))
            recorded = candidates.get(str(path), candidates.get(f'scores/unit{unit}.npz'))
        if recorded != digest:
            raise ValueError('Sealed unit score hash mismatch')
        with np.load(path, allow_pickle=False) as cache:
            if not np.array_equal(cache['frames'], FRAMES):
                raise ValueError('Score frame order differs')
            for arm in ARMS:
                value = cache[arm].copy()
                if value.shape != (7, 4, 13) or not np.isfinite(value).all():
                    raise ValueError('Score axes/nonfinite:'+arm)
                if arm in reference:
                    np.testing.assert_allclose(value, reference[arm][i], atol=1e-8, rtol=0)
                scores[arm].append(value)
        paths.extend([path, receipt_path])
    return {arm: np.stack(value) for arm, value in scores.items()}, paths


def report(result):
    lines = ['# 已知场景姿态边缘化：成本与首次报警', '',
        '这是48场景已消费Development上的固定24姿态先验参考；没有训练。阈值在评价批成本匹配，只用于描述前沿，不能作为部署校准。', '',
        '主轴为全13帧实际清晰外15/20cm的首停最大值，匹配冻结M3的18/384；每臂一个全48场景strict >阈值，整ties不拆，high36/low12直接继承。', '',
        '|臂|全组浅及时/n|high36浅及时/n|low12浅及时/n|外15/20实际清晰首停/n|low补回/丢失|',
        '|---|---:|---:|---:|---:|---:|']
    cells = result['episode_cost']['sequences']['cells']
    changes = result['episode_cost']['sequences']['changes_minus_M3']
    for arm in ARMS:
        def count(group, field, metric):
            cell = cells[group][arm][field]
            n = cell['covered_n'] if metric=='timely_stops' else cell['n']
            return f"{cell[metric]['stops']}/{n}"
        delta = changes['low12'].get(arm, {}).get('timely', {})
        rescue = f"{delta.get('rescues',0)}/{delta.get('losses',0)}"
        lines.append('|'+arm+'|'+'|'.join(count(g, 'inside1_2cm', 'timely_stops') for g in ('all','high36','low12'))+
            '|'+count('all', 'outside15_20cm', 'first_stops')+'|'+rescue+'|')
    lines += ['', '所有内5/外5/外10cm分支、实际pass、首次≥2.1m报警、首报位置与target可见性、代理分钟及补回/丢失区间均保存在result.json。主/近/combined的macro与pooled AUC采用同一1000整场景bootstrap；K4不是独立单位，低组仍只有12个场景。', '',
        '另保留主1.2–2.1m外15和外20帧预算5/1152、11/1152两个成本点；其全48阈值继承到分层和全13帧序列，不能拼接不同工作点指标。', '',
        '及时指全窗首次报警前距≥0.9m；首次≥2.1m单列。2.6秒代理暴露包含报警后时间，提前量条件于报警；不代表真实提醒负担、安全停步或实机效果。已知目标、背景和当前anchor仍有特权，姿态先验是假设；误差范围内的表现不能证明真实回波信息上界。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf8')


def evaluate():
    start = time.monotonic()
    plan = freeze_plan()
    result_path = OUT/'result.json'
    if result_path.exists():
        old = U.read(result_path)
        if old.get('status') != 'COMPLETE':
            raise ValueError('Preserve incomplete scientific result for inspection')
        for path, digest in old['provenance']['input_sha256'].items():
            if U.sha(path) != digest:
                raise ValueError('Saved result input identity changed')
        return old
    baseline, scenes, categories, visible, paths = geometry()
    scores, score_paths = read_scores(plan['units'])
    np.testing.assert_array_equal(scores[BASE], baseline)
    identity_paths = [OUT/'PLAN.json', Path(__file__), Path(U.__file__), Path(F.__file__)]
    if 'implementation_amendment_sha256' in plan:
        identity_paths.append(OUT/'IMPLEMENTATION_AMENDMENT.json')
    before = input_hashes(paths+score_paths+identity_paths)
    groups = strata(scenes)
    boot = bootstrap()
    ranges = np.array([s['front_range_m'] for s in scenes])
    thresholds = episode_thresholds(scores, categories)
    sequence, ledgers = sequence_summary(scores, thresholds, scenes, categories, visible, groups, boot)
    frame_cost = frame_workpoints(scores, ranges, scenes, categories, visible, groups, boot)
    auc = auc_summary(scores, ranges, groups, boot)
    ledger_path = OUT/'episode_ledger.npz'
    entries = {'units': np.array(plan['units']), 'frames': FRAMES, 'delta_cm': DELTA,
               'ranges': ranges, 'categories': categories, 'target_visible': visible, 'high36': groups['high36']}
    for arm, ledger in ledgers.items():
        entries.update({arm+'__'+field: value for field, value in ledger.items()})
    np.savez_compressed(ledger_path, **entries)
    result = dict(status='COMPLETE', units=plan['units'], episode_cost=dict(thresholds=thresholds, sequences=sequence),
        frame_cost=frame_cost, auc=auc, baseline_parity=baseline_check_data(baseline, scenes, categories, visible),
        n=dict(scenes=48, high_scenes=36, low_scenes=12, shallow_episodes=384, actualclear_episodes=384),
        elapsed_s=time.monotonic()-start, pid=os.getpid(), plan=plan,
        provenance=dict(input_sha256=before, ledger_sha256=U.sha(ledger_path),
            operations=dict(training=False, model_inference=False, rendering=False, likelihood_scoring=False)))
    for path, digest in before.items():
        if U.sha(path) != digest:
            raise ValueError('Input changed during evaluation')
    save(result_path, result)
    report(result)
    save(OUT/'evaluation_receipt.json', dict(status='COMPLETE', elapsed_s=result['elapsed_s'], pid=os.getpid(),
        plan_sha256=U.sha(OUT/'PLAN.json'), output_sha256=input_hashes([result_path, ledger_path, OUT/'REPORT.md'])))
    print('COMPLETE pose-marginal evaluation seconds', round(result['elapsed_s'], 2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['check', 'plan', 'evaluate'])
    args = parser.parse_args()
    {'check': check, 'plan': freeze_plan, 'evaluate': evaluate}[args.stage]()
