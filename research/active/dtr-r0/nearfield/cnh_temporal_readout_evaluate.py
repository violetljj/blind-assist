"""Frozen three-seed pilot evaluation; no training, fitting, or score selection.

Fresh scene macroAUC measures target-position discrimination. Whole-scene
natural sequence truth independently constrains clear first-stop burden.
"""
import argparse
from pathlib import Path
import time

import numpy as np

import cnh_unknown_target_reference_evaluate as U
import cnh_sequence_observed_evaluate as OE
import cnh_three_level_sequence as SE

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-temporal-readout-20261004'
UNITS = [u for u in range(221001, 221073) if u % 3 in (0, 1)][:48]
FRAMES = np.arange(3, 16)
ARMS = ('M3', 'V', 'T', 'D')
PRIMARY = 'primary1.2-2.1m'
SEED = 2026100403


def boot_weights(n, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.asarray([np.bincount(rng.integers(n, size=n), minlength=n) for _ in range(1000)])


def rows_for(out, split):
    path = out / 'inputs' / split / 'rows.npz'
    with np.load(path, allow_pickle=False) as z:
        rows = {key: z[key] for key in z.files}
    for key in ('unit', 'config', 'variant', 'replica', 'frame'):
        if key not in rows or rows[key].ndim != 1 or len(rows[key]) != len(rows['unit']):
            raise ValueError('Missing or unaligned input row identity '+key)
    return rows, path


def prediction(out, split, arm, seed=None):
    if seed is None:
        path = out / 'predictions' / f'{arm}_{split}.npz'
    else:
        path = out / 'scores' / split / f'{arm}_seed{seed}.npy'
        if not path.exists():
            path = out / 'predictions' / f'{arm}_seed{seed}_{split}.npz'
    if path.suffix == '.npy':
        score = np.asarray(np.load(path, allow_pickle=False), np.float64)
    else:
        with np.load(path, allow_pickle=False) as z:
            score = np.asarray(z['raw'], np.float64)
    if score.ndim != 2 or score.shape[1] != 2 or not np.isfinite(score).all():
        raise ValueError('Finite raw logits [N,2] required: '+str(path))
    return score, path


def ordered_episodes(raw, rows, *, fresh=False):
    """Validate complete causal episode identities and apply frozen five-score smoothing."""
    if len(raw) != len(rows['unit']):
        raise ValueError('Prediction and input rows differ')
    keys = ('unit', 'variant', 'replica') if fresh else ('unit', 'config')
    tuples = list(zip(*(rows[k].astype(int).tolist() for k in keys)))
    unique = sorted(set(tuples))
    lookup = {key: [] for key in unique}
    for i, key in enumerate(tuples):
        lookup[key].append(i)
    result = []
    for key in unique:
        idx = np.asarray(lookup[key])
        idx = idx[np.argsort(rows['frame'][idx])]
        if not np.array_equal(rows['frame'][idx], FRAMES):
            raise ValueError('Exactly one prediction per retained frame3..15 required: '+str(key))
        result.append(raw[idx])
    ordered = np.stack(result)
    smoothed = SE.smooth(ordered[:, None])[:, 0]
    return smoothed, unique


def fresh_scores(raw, rows, scenes):
    ordered, keys = ordered_episodes(raw, rows, fresh=True)
    expected = [(u, v, k) for u in UNITS for v in range(7) for k in range(4)]
    if keys != expected:
        raise ValueError('Fresh cohort must contain every48 scene x7variant xK4 episode')
    full = ordered.reshape(48, 7, 4, 13, 2)
    return np.stack([full[i, ..., s['group']] for i, s in enumerate(scenes)])


def fresh_geometry(out):
    scenes, visibility, paths = [], [], []
    for unit in UNITS:
        tp = out / 'truth/evaluation' / f'unit{unit}.json'
        ep = out / 'templates/evaluation' / f'unit{unit}.npz'
        truth = U.read(tp)
        if truth['unit'] != unit or truth['intrusion_cm'] != [1., 2., 5., -5., -10., -15., -20.]:
            raise ValueError('Fresh scenario branch identity mismatch')
        with np.load(ep, allow_pickle=False) as z:
            ids = z['object_id']
            visible = np.any(ids.reshape(7, 16, -1) == 0, axis=-1)[:, FRAMES]
        vf = float(visible[:2].mean())
        scenes.append(dict(unit=unit, group=int(truth['group']), context=truth['context'],
            front_range_m=np.asarray(truth['front_range_m'])[FRAMES].tolist(),
            visibility_fraction=vf, fov_in=vf >= .5))
        visibility.append(visible)
        paths.extend([tp, ep])
    return scenes, np.stack(visibility), paths


def strata(scenes):
    result = dict(all=np.ones(48, bool))
    for group, title in ((0, 'HEAD'), (1, 'BODY')):
        result[title] = np.array([s['group'] == group for s in scenes])
        for context in ('none', 'panel'):
            result[title+'/'+context] = np.array([s['group'] == group and s['context'] == context for s in scenes])
    for context in ('none', 'panel'):
        result[context] = np.array([s['context'] == context for s in scenes])
    result['FOV_IN'] = np.array([s['fov_in'] for s in scenes])
    result['FOV_OUT'] = ~result['FOV_IN']
    return result


def gap_fraction(numerator, denominator, keep, boot):
    num, num_draw = U.macro_summary(numerator, keep, boot)
    den, den_draw = U.macro_summary(denominator, keep, boot)
    stable = den['value'] is not None and den['value'] >= .01 and den['ci95'][0] is not None and den['ci95'][0] > 0
    ratio = None if den['value'] in (None, 0) else num['value']/den['value']
    with np.errstate(divide='ignore', invalid='ignore'):
        draws = num_draw/den_draw
    return dict(value=ratio, ci95=U.interval(draws) if stable else [None, None],
        status='STABLE_CONDITIONAL_RATIO' if stable else 'UNSTABLE_DENOMINATOR',
        numerator=num, denominator=den,
        rule='Same-cohort difference of scene means; ratios outside[0,1] retained; denominator<.01 or paired bootstrap crossing0 marks unstable')


def auc_metrics(scores, scenes, boot):
    ranges = np.asarray([s['front_range_m'] for s in scenes])
    domains = {f'{lo:g}-{hi:g}m': (ranges >= lo) & (ranges < hi) for lo, hi in U.BINS}
    domains[PRIMARY] = (ranges >= 1.2) & (ranges < 2.1)
    groups = strata(scenes)
    cells, per_scene = {}, []
    for i, s in enumerate(scenes):
        per_scene.append(dict(**s, auc={}))
    for domain, masks in domains.items():
        values = {}
        for arm, score in scores.items():
            value = []
            for i, mask in enumerate(masks):
                positive = score[i, [0, 1]][..., mask].ravel()
                negative = score[i, [4, 5, 6]][..., mask].ravel()
                auc = U.binary_auc(positive, negative)
                value.append(np.nan if auc is None else auc)
                per_scene[i]['auc'].setdefault(domain, {})[arm] = dict(value=auc, positive_n=len(positive), negative_n=len(negative))
            values[arm] = np.asarray(value)
        cells[domain] = {}
        for group, keep in groups.items():
            cell = dict(scenes=int(keep.sum()), arms={}, paired_minus_M3={}, gap_closed={})
            for arm, val in values.items():
                cell['arms'][arm] = U.macro_summary(val, keep, boot)[0]
                if arm != 'M3':
                    cell['paired_minus_M3'][arm] = U.macro_summary(val-values['M3'], keep, boot)[0]
            for arm in ('V', 'T'):
                cell['gap_closed'][arm] = gap_fraction(values[arm]-values['M3'], values['D']-values['M3'], keep, boot)
            cell['T_minus_V'] = U.macro_summary(values['T']-values['V'], keep, boot)[0]
            cells[domain][group] = cell
    return cells, per_scene


def natural(out, input_hashes):
    geometry, old_scores, baseline_receipt = OE.load_inputs()
    baseline = OE.read(OE.OUT / 'result.json')
    input_hashes[str(OE.OUT / 'result.json')] = U.sha(OE.OUT / 'result.json')
    candidate = {}
    for split, old_split in [('calibration', 'calib'), ('evaluation', 'evaluation')]:
        rows, path = rows_for(out, split)
        input_hashes[str(path)] = U.sha(path)
        selected = geometry['split'] == old_split
        expected = sorted(set(zip(geometry['unit'][selected].astype(int), geometry['config'][selected].astype(int))))
        for arm in ('V', 'T'):
            raw = []
            for seed in range(3):
                score, path = prediction(out, split, arm, seed)
                raw.append(score)
                input_hashes[str(path)] = U.sha(path)
            ordered, keys = ordered_episodes(np.mean(raw, axis=0), rows)
            if keys != expected:
                raise ValueError('Natural prediction cohort differs from frozen geometry')
            candidate.setdefault(arm, {})[split] = ordered.transpose(0, 2, 1).reshape(-1, 13)
    cal = geometry['split'] == 'calib'
    ev = geometry['split'] == 'evaluation'
    fields = ('unit', 'query', 'frame_ranges', 'frame_category', 'ref_category', 'clear_all', 'covered', 'last_category')
    g = {key: geometry[key][ev] for key in fields}
    unique, ui = np.unique(g['unit'], return_inverse=True)
    boot = boot_weights(len(unique))
    weights = {category: (g['covered'] & (g['ref_category'] == category)).astype(float)
               for category in OE.CONTACTS+('pass0-10cm',)}
    weights['clear'] = g['clear_all'].astype(float)
    rows_geom = OE.read(OE.OUT / 'rows.json')
    target_groups = np.asarray([r['target_group'] for r, keep in zip(rows_geom, ev) if keep])
    scores = {'M3': old_scores['M3'][ev]}
    thresholds = {'M3': float(baseline['cells']['M3']['threshold'])}
    calibration = {'M3': baseline['cells']['M3']['calibration']}
    for arm in ('V', 'T'):
        thresholds[arm], calibration[arm] = SE.calibrate(candidate[arm]['calibration'], geometry['clear_all'][cal], 1.)
        scores[arm] = candidate[arm]['evaluation']
    if thresholds['M3'] != .8557642486787612:
        raise ValueError('Frozen natural M3 threshold changed')
    cells, draws, flags = {}, {}, {}
    for arm, score in scores.items():
        stopped, timely, lead = SE.first_stops(score, thresholds[arm], g['frame_ranges'])
        metrics, sampled = SE.summarize(weights, g['covered'], stopped, timely, lead, ui, boot)
        for category, metric in metrics.items():
            metric.pop('episode_draw_pairs', None)
        clear_num = SE.unit_totals(weights['clear']*stopped, ui, len(unique))
        clear_den = SE.unit_totals(weights['clear'], ui, len(unique))*13*.2/60
        _, sampled['clear'] = SE.rates(clear_num, clear_den, boot)
        other = g['clear_all'] & (g['query'] != target_groups)
        other_num = SE.unit_totals(other*stopped, ui, len(unique))
        other_den = SE.unit_totals(other, ui, len(unique))*13*.2/60
        other_rate, _ = SE.rates(other_num, other_den, boot)
        cells[arm] = dict(threshold=thresholds[arm], calibration=calibration[arm], metrics=metrics,
            other_height_clear=dict(stops=int((other & stopped).sum()), n=int(other.sum()), false_stops_per_min=other_rate),
            censored=dict(n=int((~g['covered']).sum()), already_alarm=int((~g['covered'] & stopped).sum()),
                          unalarmed=int((~g['covered'] & ~stopped).sum())))
        draws[arm], flags[arm] = sampled, dict(stopped=stopped, timely=timely)
    comparisons = {}
    for arm in ('V', 'T'):
        shallow = cells[arm]['metrics']['contact0-2cm']['timely_rate']['value']-cells['M3']['metrics']['contact0-2cm']['timely_rate']['value']
        deep = cells[arm]['metrics']['contact>5cm']['timely_rate']['value']-cells['M3']['metrics']['contact>5cm']['timely_rate']['value']
        clear = cells[arm]['metrics']['clear']['false_stops_per_min']['value']-cells['M3']['metrics']['clear']['false_stops_per_min']['value']
        comparisons[arm] = dict(shallow_delta=shallow, shallow_ci95=SE.interval(draws[arm]['contact0-2cm']-draws['M3']['contact0-2cm']),
            deep_delta=deep, deep_ci95=SE.interval(draws[arm]['contact>5cm']-draws['M3']['contact>5cm']),
            clear_delta_per_min=clear, clear_ci95=SE.interval(draws[arm]['clear']-draws['M3']['clear']),
            guardrail_pass=clear <= .1+1e-12 and deep >= -.02-1e-12)
    for category, expected in [('contact0-2cm', (26., 31.)), ('contact>5cm', (162., 164.))]:
        metric = cells['M3']['metrics'][category]
        if (metric['expected_timely_stops'], metric['expected_episodes']) != expected:
            raise ValueError('Frozen natural baseline failed parity: '+category)
    if cells['M3']['metrics']['clear']['expected_stops'] != 205 or not np.isclose(cells['M3']['metrics']['clear']['clear_minutes'], 194.35):
        raise ValueError('Frozen natural clear baseline failed parity')
    return dict(cells=cells, comparisons=comparisons, threshold_origin='95000-95047 all-object full13frame clear, <=1 first stop/proxy minute',
        scope='96000-96095; covered contacts only; unalarmed right-censored queries are not missed deadlines',
        clear_definition='All physical surfaces clear at all13 saved query poses, one first alarm/query; 2.6sec exposure includes after-alarm time',
        source_provenance=baseline_receipt), thresholds


def fresh_sequences(scores, scenes, visible, thresholds, boot):
    ranges = np.asarray([s['front_range_m'] for s in scenes])
    groups = strata(scenes)
    result = {}
    for arm in ('M3', 'V', 'T'):
        hit = scores[arm] >= thresholds[arm]
        stopped = hit.any(-1)
        first = hit.argmax(-1)
        stop_range = ranges[np.arange(48)[:, None, None], first]
        timely = stopped & (stop_range >= .9)
        result[arm] = {}
        for group, keep in groups.items():
            cell = dict(threshold=thresholds[arm], shallow={}, outside={})
            for name, ids in [('inside1/2cm', [0, 1]), ('inside5cm', [2])]:
                n = np.full(48, len(ids)*4)
                counts = timely[:, ids].sum(axis=(1, 2))
                stops = stopped[:, ids].sum(axis=(1, 2))
                metric, _ = U.pooled_rate(counts, n, keep, boot)
                metric['all_first_stops'] = int(stops[keep].sum())
                lead = (stop_range[:, ids]-.5)/.8
                chosen = keep[:, None, None] & timely[:, ids]
                metric['median_lead_conditional_timely_s'] = float(np.median(lead[chosen])) if chosen.any() else None
                vf = visible[np.arange(48)[:, None, None], np.asarray(ids)[None, :, None], first[:, ids]]
                metric['first_at_least2p1m_and_target_visible'] = int((chosen & (stop_range[:, ids] >= 2.1) & vf).sum())
                cell['shallow'][name] = metric
            for name, ids in [('outside15cm', [5]), ('outside20cm', [6]), ('outside15/20cm', [5, 6])]:
                cell['outside'][name] = U.pooled_rate(stopped[:, ids].sum(axis=(1, 2)), np.full(48, len(ids)*4), keep, boot)[0]
            result[arm][group] = cell
    return result


def decide(main, guards):
    arms = {}
    for arm in ('V', 'T'):
        change = main['paired_minus_M3'][arm]
        candidate = change['value'] >= .03 and change['ci95'][0] is not None and change['ci95'][0] > 0 and guards[arm]['guardrail_pass']
        arms[arm] = dict(candidate=candidate, auc_delta=change['value'], paired_ci95=change['ci95'], guardrail_pass=guards[arm]['guardrail_pass'])
    if any(v['candidate'] for v in arms.values()):
        branch = 'CANDIDATE'
    elif all(v['auc_delta'] < .015 for v in arms.values()):
        branch = 'NOT_LEARNED'
    else:
        branch = 'INTERMEDIATE'
    return dict(branch=branch, arms=arms,
        representation_effect_descriptive=main['T_minus_V']['value'] >= .02,
        data_or_objective_sufficient_descriptive=main['paired_minus_M3']['V']['value'] >= .03,
        mapping='CANDIDATE proposes fresh confirmation, never promotes M3; NOT_LEARNED stops architecture search and turns to background privilege; INTERMEDIATE awaits discussion')


def report(result, out):
    main = result['auc'][PRIMARY]['all']
    lines = [result['decision']['branch'], '', '# 远距时序读出训练试点', '',
        '新48场景主域1.2–2.1m；V/T三个种子先平均raw logits，再沿用M3五分数因果平滑。M3五种子冻结。场景等权AUC，1000次整场景配对bootstrap。', '',
        '| 臂 | macroAUC | 对M3差及95%区间 | 条件缺口补上比例 |', '|---|---:|---:|---:|']
    for arm in ARMS:
        auc = main['arms'][arm]['value']
        delta = main['paired_minus_M3'].get(arm)
        d = '—' if delta is None else f"{delta['value']:+.4f} [{delta['ci95'][0]:+.4f},{delta['ci95'][1]:+.4f}]"
        fraction = main['gap_closed'].get(arm)
        ratio = '—' if fraction is None else f"{fraction['value']:.1%} ({fraction['status']})" if fraction['value'] is not None else '不可计算'
        lines.append(f'| {arm} | {auc:.4f} | {d} | {ratio} |')
    lines += ['', '| 臂/seed | 主AUC | 对M3差及95%区间 |', '|---|---:|---:|']
    for arm in ('V', 'T'):
        for seed in range(3):
            key = f'{arm}_seed{seed}'
            delta = main['paired_minus_M3'][key]
            lines.append(f"| {key} | {main['arms'][key]['value']:.4f} | {delta['value']:+.4f} [{delta['ci95'][0]:+.4f},{delta['ci95'][1]:+.4f}] |")
    lines += ['', '| 自然96000护栏 | M3 | V | T |', '|---|---:|---:|---:|']
    for category in ('contact0-2cm', 'contact>5cm', 'clear'):
        values = []
        for arm in ('M3', 'V', 'T'):
            m = result['natural']['cells'][arm]['metrics'][category]
            if category == 'clear':
                values.append(f"{m['expected_stops']:g}/{m['clear_minutes']:.2f}={m['false_stops_per_min']['value']:.4f}/min")
            else:
                values.append(f"{m['expected_timely_stops']:g}/{m['expected_episodes']:g}={m['timely_rate']['value']:.2%}")
        lines.append('| '+category+' | '+' | '.join(values)+' |')
    lines += ['', '| 分层 | M3 | D | V | T | V补上比例 | T补上比例 |', '|---|---:|---:|---:|---:|---:|---:|']
    for domain in (PRIMARY, '1.6-2.1m', '2.1-2.6m'):
        for group in ('all', 'HEAD/none', 'HEAD/panel', 'BODY/none', 'BODY/panel', 'FOV_IN', 'FOV_OUT'):
            cell = result['auc'][domain][group]
            fmt = lambda value: '—' if value is None else f'{value:.4f}'
            frac = lambda arm: '—' if cell['gap_closed'][arm]['value'] is None else f"{cell['gap_closed'][arm]['value']:.1%}" + ('*' if cell['gap_closed'][arm]['status'].startswith('UNSTABLE') else '')
            lines.append('| '+' | '.join([domain+'/'+group]+[fmt(cell['arms'][a]['value']) for a in ('M3', 'D', 'V', 'T')]+[frac('V'), frac('T')])+' |')
    lines += ['', '*表示分母不稳定：D−M3<.01或bootstrap区间跨零；数值仅描述，不能据此判机制。', '',
        '| 新位移序列/臂 | 浅及时 | 外15cm首停 | 外20cm首停 |', '|---|---:|---:|---:|']
    for group in ('all', 'FOV_IN', 'FOV_OUT'):
        for arm in ('M3', 'V', 'T'):
            cell = result['fresh_sequences'][arm][group]
            shallow = cell['shallow']['inside1/2cm']
            outside = cell['outside']
            lines.append(f"| {group}/{arm} | {shallow['stops']}/{shallow['n']} | {outside['outside15cm']['stops']}/{outside['outside15cm']['n']} | {outside['outside20cm']['stops']}/{outside['outside20cm']['n']} |")
    lines += ['', f"评价CPU耗时 {result['elapsed_s']:.2f} 秒；完整训练/渲染总耗时由父运行收据报告。", '',
        '边界：全部为模拟Development训练试点；D已知目标、背景和当前真实锚位姿，补上比例分母偏乐观。自然阈值按整段清晰首停预算冻结，新批不重新选阈值。右删失不计漏报；代理分钟不是人的实际连续步行误停率。区间条件于已训练分数和阈值，不含模拟器、训练过程或硬件不确定性。', '',
        '结果只触发预设主分支；分层、逐seed与T−V均描述，不择优、不自动替换M3或启动新实验。完整分母、区间、输入哈希见result.json。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def evaluate(out=OUT):
    import cnh_temporal_readout as C
    C.check_budget()
    target = out / 'result.json'
    if target.exists():
        raise FileExistsError('Completed pilot result must remain immutable')
    started = time.monotonic()
    plan = U.read(out / 'PLAN.json')
    if plan['eval_units'] != UNITS or plan['seeds'] != [0, 1, 2]:
        raise ValueError('Frozen cohort/seeds differ from evaluator')
    if plan['bootstrap']['n'] != 1000 or plan['bootstrap']['seed'] != SEED:
        raise ValueError('Bootstrap differs from frozen PLAN')
    input_hashes = {str(out / 'PLAN.json'): U.sha(out / 'PLAN.json')}
    scenes, visible, geom_paths = fresh_geometry(out)
    input_hashes.update({str(p): U.sha(p) for p in geom_paths})
    rows, path = rows_for(out, 'fresh_evaluation')
    input_hashes[str(path)] = U.sha(path)
    scores = {}
    for arm in ('V', 'T'):
        raw_seeds = []
        for seed in range(3):
            raw, path = prediction(out, 'fresh_evaluation', arm, seed)
            raw_seeds.append(raw)
            scores[f'{arm}_seed{seed}'] = fresh_scores(raw, rows, scenes)
            input_hashes[str(path)] = U.sha(path)
        scores[arm] = fresh_scores(np.mean(raw_seeds, axis=0), rows, scenes)
    raw, path = prediction(out, 'fresh_evaluation', 'M3')
    scores['M3'] = fresh_scores(raw, rows, scenes)
    input_hashes[str(path)] = U.sha(path)
    ref_path = out / 'reference/all_scores.npz'
    receipt_path = out / 'reference/scores_receipt.json'
    receipt = U.read(receipt_path)
    if receipt['status'] != 'COMPLETE' or receipt['score_sha256'] != U.sha(ref_path) or receipt['plan_sha256'] != U.sha(out / 'PLAN.json'):
        raise ValueError('D reference receipt mismatch')
    with np.load(ref_path, allow_pickle=False) as z:
        if not np.array_equal(z['units'], UNITS) or not np.array_equal(z['frames'], FRAMES):
            raise ValueError('D reference axis identities differ')
        scores['D'] = z['D']
    input_hashes.update({str(ref_path): U.sha(ref_path), str(receipt_path): U.sha(receipt_path)})
    if any(s.shape != (48, 7, 4, 13) or not np.isfinite(s).all() for s in scores.values()):
        raise ValueError('All fresh score arms must be finite48x7x4x13')
    boot = boot_weights(48)
    auc, per_scene = auc_metrics(scores, scenes, boot)
    C.check_budget()
    natural_result, thresholds = natural(out, input_hashes)
    result = dict(status='COMPLETE', auc=auc, per_scene=per_scene, natural=natural_result,
        fresh_sequences=fresh_sequences(scores, scenes, visible, thresholds, boot),
        decision=decide(auc[PRIMARY]['all'], natural_result['comparisons']),
        n=dict(fresh_scenes=48, fov_in=int(strata(scenes)['FOV_IN'].sum()), fov_out=int(strata(scenes)['FOV_OUT'].sum())),
        bootstrap=dict(n=1000, seed=SEED, cluster='whole fresh scenes or whole96natural units, paired across arms',
                       condition='fixed trained models and calibrated thresholds; three individual seeds reported, no seed selection'),
        elapsed_s=time.monotonic()-started,
        provenance=dict(input_sha256=input_hashes, evaluator_sha256=U.sha(__file__)))
    if any(U.sha(path) != digest for path, digest in input_hashes.items()):
        raise ValueError('Frozen evaluation input changed during evaluation')
    C.check_budget()
    report(result, out)
    U.save(target, result)
    print('PILOT', result['decision']['branch'], result['elapsed_s'], flush=True)
    return result


def check():
    # Focused checks for causal grouping, ensemble order, ratios and branch boundaries.
    rows = dict(unit=np.repeat([1, 2], 13), config=np.zeros(26, int), variant=np.zeros(26, int),
                replica=np.zeros(26, int), frame=np.tile(FRAMES, 2))
    raw = np.arange(52.).reshape(26, 2)
    value, keys = ordered_episodes(raw, rows)
    assert keys == [(1, 0), (2, 0)] and value.shape == (2, 13, 2)
    assert np.array_equal(value[:, 0], raw.reshape(2, 13, 2)[:, 0])
    expected = np.tensordot(raw.reshape(2, 13, 2)[:, -5:], np.array([1, 2, 4, 8, 16])/31, axes=([1], [0]))
    np.testing.assert_allclose(value[:, -1], expected)
    main = dict(paired_minus_M3={a: dict(value=.03, ci95=[.001, .05]) for a in ('V', 'T')}, T_minus_V=dict(value=.02))
    guards = {a: dict(guardrail_pass=True) for a in ('V', 'T')}
    assert decide(main, guards)['branch'] == 'CANDIDATE'
    guards['V']['guardrail_pass'] = guards['T']['guardrail_pass'] = False
    assert decide(main, guards)['branch'] == 'INTERMEDIATE'
    for a in ('V', 'T'):
        main['paired_minus_M3'][a]['value'] = .014
    assert decide(main, guards)['branch'] == 'NOT_LEARNED'
    print('PILOT EVALUATOR focused checks PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['check', 'evaluate'], required=True)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == 'check':
        check()
    else:
        evaluate(args.out)
