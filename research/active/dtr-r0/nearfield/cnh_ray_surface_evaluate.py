"""Single-seed native ray-surface readout evaluation.

Inference accepts only native observations, public sensor-to-travel transforms and
retained M3 logits. All-object truth is loaded only after scores are complete.
No history-cache receipt or voxel feature dependency is inherited.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME
import cnh_sequence_extrinsic_evaluate as XE
import cnh_sequence_observed_evaluate as OE
import cnh_three_level_sequence as SE
import cnh_yaw_envelope_evaluate as YE
import cnh_yaw_source_transfer_evaluate as ST

OUT = MC.SS.WORK/'cnh-ray-surface-20261002'
RUN = 'CNH_RAY_SURFACE_20261002'
NEW_ARMS = ('BCEO', 'RAY')
ARMS = ('M3', 'BCEO', 'RAY')
UNITS = MC.SPLITS['calib']+MC.SPLITS['evaluation']
FRAMES = np.arange(3, 16)
M3_THRESHOLD = .8557642486787612
BOOT_SEED = 2026100234
N_BOOT = 1000
CATEGORIES = OE.CONTACTS+('pass0-10cm', 'clear')
PRIMARY_GROUP = 'same_height_nominal_outside10_20cm'


def file_snapshot(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def stable_hash(path):
    before = file_snapshot(path)
    digest = ME.sha(path)
    if before != file_snapshot(path):
        raise ValueError('Input changed while hashing: '+str(path))
    return digest, before


def verify_inputs(receipt):
    """SHA small evidence; bulk bytes were hashed once and are stat-guarded."""
    YE.verify_hashes(receipt['input_sha256'])
    for path, expected in receipt['bulk_stat'].items():
        if file_snapshot(path) != expected:
            raise ValueError('Bulk input changed since full hash: '+path)


def comparison_checks(comparison):
    shallow = comparison['contact0-2cm']['paired_unit_ci95'][0]
    deep = comparison['contact>5cm']['delta_timely']
    outside = comparison['primary_outside_clear']['delta_per_min']
    clear = comparison['clear']['delta_per_min']
    return dict(shallow_timely_ci_lower_gt_zero=shallow is not None and shallow > 0,
        deep_timely_decline_le_2pp=deep is not None and deep >= -.02,
        outside_clear_point_increase_le_zero=outside is not None and outside <= 0,
        overall_clear_increase_le_0p2_per_min=clear is not None and clear <= .2)


def analyze(g, scores):
    cal, ev = g['split'] == 'calib', g['split'] == 'evaluation'
    fields = ('unit', 'query', 'target_group', 'target_off', 'frame_ranges', 'covered', 'clear_all', 'ref_category')
    if any(g[k].shape[0] != len(ev) for k in fields):
        raise ValueError('Required geometry rows do not align with split')
    e = {key: g[key][ev] for key in fields}
    units, ui = np.unique(e['unit'], return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units)) for _ in range(N_BOOT)])
    thresholds, calibration = {'M3': M3_THRESHOLD}, {'M3': dict(role='Retained old threshold; no new calibration')}
    for arm in NEW_ARMS:
        thresholds[arm], calibration[arm] = SE.calibrate(scores[arm], cal & g['clear_all'], 1.)
    weights = {c: (e['covered'] & (e['ref_category'] == c)).astype(float) for c in CATEGORIES[:4]}
    weights['clear'] = e['clear_all'].astype(float)
    groups = ST.clear_groups(e)
    cells, sampled, flags, subgroup_samples = {}, {}, {}, {}
    for arm in ARMS:
        score = scores[arm][ev]
        if score.shape != e['frame_ranges'].shape or not np.isfinite(score).all():
            raise ValueError('Score/geometry mismatch for '+arm)
        cells[arm], sampled[arm], flags[arm] = XE.summarize_cell(e, score, thresholds[arm], weights, ui, boot)
        cells[arm]['calibration'] = calibration[arm]
        cells[arm]['clear_subgroups'], subgroup_samples[arm] = ST.summarize_clear_groups(groups, flags[arm]['stopped'], ui, boot)
    comparisons, decisions = {}, {}
    for control in ('BCEO', 'M3'):
        comparison = XE.paired_changes(cells, sampled, flags, weights, 'RAY', control)
        group_changes = {}
        for name, keep in groups.items():
            a, b = (cells[arm]['clear_subgroups'][name]['false_stops_per_min']['value'] for arm in ('RAY', control))
            na, ca = flags['RAY']['stopped'], flags[control]['stopped']
            group_changes[name] = dict(delta_per_min=a-b if a is not None and b is not None else None,
                paired_unit_ci95=SE.interval(subgroup_samples['RAY'][name]-subgroup_samples[control][name]),
                added_first_stops=int((keep & na & ~ca).sum()), removed_first_stops=int((keep & ca & ~na).sum()))
        comparison['clear_subgroups'] = group_changes
        comparison['primary_outside_clear'] = group_changes[PRIMARY_GROUP]
        comparisons[control] = comparison
        decisions[control] = comparison_checks(comparison)
    verdict = ('RAY_SURFACE_PROMISING_SINGLE_SEED_DEV' if all(all(checks.values()) for checks in decisions.values())
               else 'NOT_ESTABLISHED_SINGLE_SEED_DEV')
    return dict(status='COMPLETE', verdict=verdict, comparisons_RAY_minus_control=comparisons,
        per_control_decision_checks=decisions, cells=cells, thresholds=thresholds, calibration=calibration,
        evaluation_units=units.tolist(), clear_subgroup_counts={k: int(m.sum()) for k, m in groups.items()},
        n=dict(calibration_units=len(np.unique(g['unit'][cal])), evaluation_units=len(units), query_episodes=int(ev.sum()),
            covered=int(e['covered'].sum()), right_censored=int((~e['covered']).sum()), all_clear=int(e['clear_all'].sum()),
            shallow_episodes=int(weights['contact0-2cm'].sum()),
            shallow_contributing_units=len(np.unique(e['unit'][weights['contact0-2cm'] > 0]))),
        bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEED, cluster='whole evaluation units',
            pairing='one common bootstrap matrix for every arm/subgroup', conditional_on='trained checkpoints and calibrated BCEO/RAY thresholds'),
        primary_group_definition='query==target_group AND -0.20<=nominal target_off<-0.10 AND all13frame clear_all; nominal metadata, not dynamic gap',
        rule='RAY versus BOTH BCEO and frozen M3: shallow timely CI lower>0, deep timely point>=-0.02, same-height nominal outside10-20cm clear point increase<=0/min, overall clear point increase<=0.2/min')


def validate_plan(plan):
    expected = dict(run=RUN, calibration_units=MC.SPLITS['calib'],
        evaluation_units=MC.SPLITS['evaluation'], decision_frames=FRAMES.tolist(),
        arms=list(ARMS), train_arms=list(NEW_ARMS), seeds=[0],
        frozen_M3_threshold=M3_THRESHOLD)
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError('Frozen PLAN differs from evaluator: '+key)
    if plan['bootstrap']['replicates'] != N_BOOT or plan['bootstrap']['seed'] != BOOT_SEED:
        raise ValueError('Frozen bootstrap differs from evaluator')
    if RUN not in (Path(__file__).resolve().parents[1]/'RUNS.md').read_text(encoding='utf8'):
        raise ValueError('RUNS pre-registration is required')
    YE.verify_hashes(plan['prior_sha256'])


def retained_m3():
    """Load raw frame3..15 logits only; no geometry, rows or task labels."""
    prior_path = MC.OUT/'sequence_result.json'
    prior = OE.read(prior_path)
    if prior.get('status') != 'COMPLETE' or prior['units'] != MC.SPLITS:
        raise ValueError('Retained M3 source sequence identity differs')
    hashes = {str(prior_path): ME.sha(prior_path)}
    names = ('frame_scores_M3_early.npz', 'frame_scores_M3.npz', 'scores_M3.npz')
    for name in names:
        path = MC.OUT/name
        hashes[str(path)] = ME.sha(path)
        if hashes[str(path)] != prior['provenance']['input_sha256'][name]:
            raise ValueError('Retained M3 score hash differs: '+name)
    expected = set(map(str, UNITS))
    with np.load(MC.OUT/names[0], allow_pickle=False) as early, np.load(MC.OUT/names[1], allow_pickle=False) as late, np.load(MC.OUT/names[2], allow_pickle=False) as final:
        if any(set(cache.files) != expected for cache in (early, late, final)):
            raise ValueError('Retained M3 unit keys differ')
        values = []
        for unit in UNITS:
            a, b = early[str(unit)], late[str(unit)]
            if a.shape != (40, 8, 2) or b.shape != (40, 5, 2):
                raise ValueError('Retained M3 early/late shape differs')
            values.append(np.concatenate((a, b), axis=1))
        raw = np.stack(values)
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite retained M3 logits')
        error = float(np.max(np.abs(SE.smooth(raw)[:, :, -1]-np.stack([final[str(u)] for u in UNITS]))))
        if error >= 1e-4:
            raise ValueError('Retained M3 final score parity failed')
    return raw, hashes, dict(final_score_max_abs_error=error, seeds=5, frames=FRAMES.tolist())


def load_evaluation_inputs():
    plan_path, receipt_path = OUT/'PLAN.json', OUT/'scores_receipt.json'
    plan, receipt = OE.read(plan_path), OE.read(receipt_path)
    validate_plan(plan)
    if receipt.get('status') != 'COMPLETE' or receipt.get('plan_sha256') != ME.sha(plan_path):
        raise ValueError('Scores incomplete or belong to another PLAN')
    verify_inputs(receipt)
    # This is deliberately the first entry point that may load truth geometry.
    g, retained, old_provenance = OE.load_inputs()
    rows_path = OE.OUT/'rows.json'
    rows = OE.read(rows_path)
    keys = [(r['split'], r['unit'], r['config'], r['query']) for r in rows]
    if keys != list(zip(g['split'].tolist(), g['unit'].tolist(), g['config'].tolist(), g['query'].tolist())):
        raise ValueError('All-object nominal row identity differs')
    g['target_group'] = np.asarray([r['target_group'] for r in rows])
    g['target_off'] = np.asarray([r['target_off'] for r in rows])
    hashes = dict(receipt['input_sha256']); hashes.update(old_provenance['input_sha256'])
    hashes.update({str(p): ME.sha(p) for p in (plan_path, receipt_path, rows_path)})
    scores = {'M3': retained['M3']}
    for arm in NEW_ARMS:
        path = OUT/f'frame_scores_{arm}.npz'
        digest = ME.sha(path)
        if receipt['output_sha256'].get(path.name) != digest:
            raise ValueError('New score cache changed: '+arm)
        with np.load(path, allow_pickle=False) as cache:
            if set(cache.files) != set(map(str, UNITS)):
                raise ValueError('New score unit keys differ: '+arm)
            raw = np.stack([cache[str(u)] for u in UNITS])
        if raw.shape != (144, 40, 13, 2) or not np.isfinite(raw).all():
            raise ValueError('New score shape/values differ: '+arm)
        scores[arm] = SE.smooth(raw).transpose(0, 1, 3, 2).reshape(-1, 13)
        hashes[str(path)] = digest
    prior_path = OE.OUT/'result.json'
    prior = OE.read(prior_path)
    if prior.get('status') != 'COMPLETE' or prior['cells']['M3']['threshold'] != M3_THRESHOLD:
        raise ValueError('Old M3 threshold/result differs')
    hashes[str(prior_path)] = ME.sha(prior_path)
    return g, scores, prior, dict(input_sha256=hashes, bulk_sha256=receipt['bulk_sha256'],
        bulk_stat=receipt['bulk_stat'], plan=plan, scores_receipt=receipt,
        training_summary=OE.read(OUT/'training_receipt.json'), evaluator_sha256=ME.sha(__file__),
        helper_sha256={p: ME.sha(p) for p in (SE.__file__, OE.__file__, ST.__file__, XE.__file__)})


def number(value, scale=1.):
    return '--' if value is None else f'{value*scale:.2f}'


def write_report(result):
    lines = [result['verdict'], '', '# 逐射线表面表示的单种子序列检验', '',
        'BCEO/RAY均仅seed0新分支，原M3五seed平均logit作为共同冻结残差起点。RAY增加训练期逐射线深度与可见性监督，预测射线点经公共变换和1cm软边界查询进入报警路径；BCEO同结构仅报警损失。推理不读取表面几何标签。此结果不是五seed验证。', '',
        '部署沿用五时刻指数因果平滑；新臂分别在48校准单位以整体清晰1次首停/代理分钟定一个HEAD/BODY共用阈值；M3阈值冻结。', '',
        '| 比较 | 浅及时差及95%区间，pp | 深及时差，pp | 外10–20cm清晰首停差及95%区间，次/分钟 | 整体清晰差，次/分钟 |',
        '| --- | --- | --- | --- | --- |']
    for control, comparison in result['comparisons_RAY_minus_control'].items():
        p, s, d, c = (comparison[k] for k in ('primary_outside_clear', 'contact0-2cm', 'contact>5cm', 'clear'))
        lines.append(f"| RAY−{control} | {number(s['delta_timely'],100)} [{number(s['paired_unit_ci95'][0],100)}, {number(s['paired_unit_ci95'][1],100)}] | {number(d['delta_timely'],100)} | {number(p['delta_per_min'])} [{number(p['paired_unit_ci95'][0])}, {number(p['paired_unit_ci95'][1])}] | {number(c['delta_per_min'])} |")
    lines += ['', '| 指标 | M3 | BCEO | RAY |', '| --- | --- | --- | --- |']
    for category in CATEGORIES:
        lines.append('| '+category+' | '+' | '.join(OE.rate_text(result['cells'][arm]['metrics'][category], category == 'clear', category == 'pass0-10cm') for arm in ARMS)+' |')
    lines += ['', '| 清晰子组 | M3 | BCEO | RAY |', '| --- | --- | --- | --- |']
    for group in result['clear_subgroup_counts']:
        texts = []
        for arm in ARMS:
            metric = result['cells'][arm]['clear_subgroups'][group]
            rate = metric['false_stops_per_min']
            texts.append(f"{metric['first_stops']}/{metric['clear_minutes']:.2f}={number(rate['value'])} [{number(rate['ci95'][0])}, {number(rate['ci95'][1])}]（n={metric['n']}）")
        lines.append('| '+group+' | '+' | '.join(texts)+' |')
    lines += ['', '| 模型/擦碰档 | 已报警条件提前量中位数及95%区间，秒 |', '| --- | --- |']
    for arm in ARMS:
        for category in OE.CONTACTS:
            metric = result['cells'][arm]['metrics'][category]['median_lead_to_0p5m_s']
            lines.append(f"| {arm}/{category} | {number(metric['value'])} [{number(metric['ci95'][0])}, {number(metric['ci95'][1])}] |")
    lines += ['', '阈值与校准：', '']
    for arm in ARMS:
        lines.append(f"- {arm}: {result['thresholds'][arm]!r}; {result['calibration'][arm]}")
    lines += ['', '配对补回/丢失、差值区间及逐比较判读：', '', '```json',
        json.dumps(result['comparisons_RAY_minus_control'], ensure_ascii=False, indent=2), '```', '',
        json.dumps(result['per_control_decision_checks'], ensure_ascii=False), '',
        '分母与删失：'+json.dumps(result['n'], ensure_ascii=False), '',
        '1000次整单位共同bootstrap，seed2026100234；区间条件于seed0检查点及校准阈值，不含重训练、校准集或模拟器不确定性。两个比较均须通过；失败不选臂、不扩成五种子。', '',
        '逐射线深度与可见性表示保留角度和距离对应，但128×128输出依赖仿真监督先验，没有增加8×8物理传感器分辨率。1cm软边界是计算近似，不是厘米精度保证；预测几何查询不是经过标定的碰撞概率。训练valid仅表示几何域内命中，不等于实际信号可检出性，也不是产品UNKNOWN语义。', '',
        '已消费Development、同名义仿真源，非盲测、跨yaw或硬件安全证据。几何监督不证明厘米级可观测性；负z1不是自由空间。清晰2.6秒/查询包含首停后暴露，13采样点清晰不保证帧间连续清晰，不是现场步行分钟。右删失不作漏停；提前量条件于已报警，按0.8m/s相对0.5m换算，不是人已安全停止。', '',
        '旧M3点复现：'+json.dumps(result['nominal_M3_point_reproduction'], ensure_ascii=False), '',
        '输入、模型、推理运行时和耗时见result.json provenance。启动时全SHA核验大输入；后续使用size/mtime_ns快照，不声称快照等同重新哈希。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def evaluate():
    path = OUT/'result.json'
    if path.exists():
        previous = OE.read(path)
        if previous.get('status') != 'COMPLETE':
            raise ValueError('Existing incomplete result requires inspection')
        verify_inputs(previous['provenance'])
        print('Existing COMPLETE result verified; no repeated evaluation', flush=True)
        return previous
    g, scores, prior, provenance = load_evaluation_inputs()
    result = analyze(g, scores)
    result['nominal_M3_point_reproduction'] = ST.nominal_parity({'0|BASE': result['cells']['M3']}, prior)
    result['old_M3_vs_NEAR_verdict_retained'] = prior['verdict']
    result['provenance'] = provenance
    result['new_branch_seeds'] = [0]
    result['frozen_M3_seeds'] = list(range(5))
    result['heading_sigma_deg'] = 0
    result['representation_limits'] = [
        'Per-ray depth and visibility preserve the angular-range association but 128x128 output depends on simulated supervision priors; native8x8 physical resolution is unchanged',
        'The1cm soft query boundary is a computational approximation, not centimetre accuracy or calibrated collision probability',
        'Training valid means an in-domain geometric hit, not actual signal detectability or product UNKNOWN',
        'Single new branch seed; unit-bootstrap intervals exclude retraining and calibration uncertainty']
    verify_inputs(provenance)
    write_report(result)
    OE.save(path, result)
    print(result['verdict'], json.dumps(result['per_control_decision_checks']), flush=True)
    return result


def validate_prepared(prepared, n):
    expected = ((n, 8, 8, 8, 16), (n, 104), (n, 4, 4))
    if len(prepared) != 3 or any(value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all() for value, shape in zip(prepared, expected)):
        raise ValueError('Label-free native ray preparation contract differs')


def validate_outputs(logits, surface, n):
    import torch
    if logits.shape != (n, 2) or not torch.isfinite(logits).all():
        raise ValueError('Model alarm logits shape/finite check failed')
    if not isinstance(surface, (tuple, list)) or len(surface) != 2 or any(value.shape != (n, 128, 128) or not torch.isfinite(value).all() for value in surface):
        raise ValueError('Ray depth/visibility shape or finite values differ')


def check():
    import torch
    prepared = (np.zeros((1, 8, 8, 8, 16), np.float32), np.zeros((1, 104), np.float32), np.eye(4, dtype=np.float32)[None])
    validate_prepared(prepared, 1)
    validate_outputs(torch.zeros(1, 2), (torch.zeros(1, 128, 128), torch.zeros(1, 128, 128)), 1)
    try:
        validate_prepared((*prepared[:2], np.zeros((1, 2, 16, 16, 129), np.float32)), 1)
    except ValueError:
        pass
    else:
        raise AssertionError('Previous distribution weights were accepted as a ray transform')
    try:
        validate_outputs(torch.zeros(1, 2), torch.zeros(1, 16, 16, 129), 1)
    except ValueError:
        pass
    else:
        raise AssertionError('Previous distribution output was accepted as a depth/visibility pair')
    comparison = {'primary_outside_clear': {'delta_per_min': 0.},
        'contact0-2cm': {'paired_unit_ci95': [.001, .1]},
        'contact>5cm': {'delta_timely': -.02}, 'clear': {'delta_per_min': .2}}
    assert all(comparison_checks(comparison).values())
    comparison['contact0-2cm']['paired_unit_ci95'][0] = 0.
    assert not comparison_checks(comparison)['shallow_timely_ci_lower_gt_zero']
    comparison['contact0-2cm']['paired_unit_ci95'][0] = .001
    comparison['primary_outside_clear']['delta_per_min'] = 1e-12
    assert not comparison_checks(comparison)['outside_clear_point_increase_le_zero']
    ref = np.array(['clear', 'clear', 'contact0-2cm', 'contact>5cm', 'pass0-10cm', 'clear', 'clear', 'censored'])
    g = dict(split=np.array(['calib']*2+['evaluation']*6), unit=np.array([95000, 95001, 96000, 96000, 96001, 96001, 96001, 96001]),
        query=np.array([0, 0, 0, 0, 0, 0, 1, 0]), target_group=np.zeros(8, int),
        target_off=np.array([-.15, -.15, .01, .1, -.05, -.15, .1, .01]), frames=FRAMES,
        covered=np.array([True]*7+[False]), clear_all=ref == 'clear', ref_category=ref,
        frame_ranges=np.tile(np.linspace(1.2, .7, 13), (8, 1)))
    g['frame_ranges'][-1] += .3
    scores = {a: np.zeros((8, 13)) for a in ARMS}
    for arm in ARMS:
        scores[arm][3:5, 1] = 2.
    scores['RAY'][2, 1] = 2.
    scores['M3'][5, 1] = scores['BCEO'][5, 1] = 2.
    result = analyze(g, scores)
    assert result['thresholds']['M3'] == M3_THRESHOLD
    assert result['comparisons_RAY_minus_control']['M3']['contact0-2cm']['rescues'] == 1
    assert result['verdict'] == 'RAY_SURFACE_PROMISING_SINGLE_SEED_DEV'
    equal = {a: s.copy() for a, s in scores.items()}
    equal['BCEO'] = equal['RAY'].copy()
    assert analyze(g, equal)['verdict'] == 'NOT_ESTABLISHED_SINGLE_SEED_DEV'
    changed = {a: s.copy() for a, s in scores.items()}
    for value in changed.values():
        value[2:] += 100
    assert analyze(g, changed)['thresholds'] == result['thresholds']
    assert result['n']['right_censored'] == 1
    assert result['cells']['RAY']['clear_subgroups']['remaining_same_height']['false_stops_per_min']['value'] is None
    json.dumps(result, allow_nan=False)
    # Causal 1/2/4/8/16 smoothing: future logits cannot affect a past decision.
    raw = np.arange(26, dtype=float).reshape(1, 1, 13, 2)
    expected = []
    for t in range(13):
        window = raw[..., max(0, t-4):t+1, :]
        weights = np.array([1., 2., 4., 8., 16.])[-window.shape[-2]:]
        expected.append(np.sum(window*weights[:, None], axis=-2)/weights.sum())
    assert np.allclose(SE.smooth(raw), np.stack(expected, axis=-2))
    print('PASS ray input/output contract, strict shallow benefit, guard boundaries, both comparisons, calib-only thresholds, censoring, and causal smoothing; no cohort inference')


def inference_inputs():
    """Bind observation/model bytes without opening any surface label arrays."""
    import cnh_ray_surface_train as T
    import cnh_ray_surface_data as D
    plan_path, receipt_path, request_path = OUT/'PLAN.json', OUT/'training_receipt.json', OUT/'training_request.json'
    plan, training = OE.read(plan_path), OE.read(receipt_path)
    validate_plan(plan)
    if training.get('status') != 'COMPLETE' or training.get('plan_sha256') != ME.sha(plan_path) or training.get('request_sha256') != ME.sha(request_path):
        raise ValueError('Matching COMPLETE training receipt is required')
    models = {key.replace('\\', '/'): value for key, value in training['models_sha256'].items()}
    expected = {f'models/{arm}/model_seed0.pt' for arm in NEW_ARMS}
    if set(models) != expected:
        raise ValueError('Exactly one seed0 model per new arm is required')
    hashes = {str(path): ME.sha(path) for path in (plan_path, receipt_path, request_path)}
    request = OE.read(request_path)
    if training['source_sha256'] != request['source_sha256']:
        raise ValueError('Training source/request identity differs')
    YE.verify_hashes(training['source_sha256'])
    hashes.update(training['source_sha256'])
    for name, digest in models.items():
        path = OUT/name
        if ME.sha(path) != digest:
            raise ValueError('Trained model SHA differs: '+name)
        hashes[str(path)] = digest
    for path in (__file__, T.__file__, D.__file__, D.D.__file__,
                 T.CP.SOURCE/'cnh_track_a_readout.py', T.CP.SOURCE/'cnh_route_sensor.py',
                 SE.__file__, OE.__file__, ST.__file__, XE.__file__, MC.__file__, ME.__file__, YE.__file__):
        hashes[str(Path(path).resolve())] = ME.sha(path)
    bulk, snapshots = {}, {}
    for split in ('calib', 'evaluation'):
        for unit in MC.SPLITS[split]:
            path = MC.OUT/'features'/split/f'unit{unit}.npz'
            bulk[str(path)], snapshots[str(path)] = stable_hash(path)
            # Inspect public observation/identity fields only, not source labels.
            with np.load(path, allow_pickle=False) as observation:
                z, scene, frame = (observation[k] for k in ('z1', 'scene', 'frame'))
            if z.shape != (40*16, 8, 8, 16) or not np.isfinite(z).all():
                raise ValueError('Native observation shape or finite values differ')
            if scene.shape != (640,) or frame.shape != (640,):
                raise ValueError('Native identity columns differ')
            actual = sorted(zip(scene.tolist(), frame.tolist()))
            if actual != [(c, f) for c in range(40) for f in range(16)]:
                raise ValueError('Native observation identity must cover every config/frame once')
    m3, old_hashes, m3_receipt = retained_m3()
    hashes.update(old_hashes)
    # Raw caches were fully hashed above. Avoid rereading them after every unit.
    for path in old_hashes:
        if path.endswith('.npz'):
            bulk[path] = hashes.pop(path)
            snapshots[path] = file_snapshot(path)
    verify_inputs(dict(input_sha256=hashes, bulk_stat=snapshots))
    return training, hashes, bulk, snapshots, m3, m3_receipt


def infer():
    receipt_path = OUT/'scores_receipt.json'
    if receipt_path.exists():
        prior = OE.read(receipt_path)
        if prior.get('status') != 'COMPLETE':
            raise ValueError('Existing incomplete inference receipt requires inspection')
        verify_inputs(prior)
        for name, digest in prior['output_sha256'].items():
            if ME.sha(OUT/name) != digest:
                raise ValueError('Completed score cache changed: '+name)
        print('Existing COMPLETE scores verified; no repeated inference', flush=True)
        return prior
    training, hashes, bulk, snapshots, m3, m3_receipt = inference_inputs()
    import torch
    from cnh_ray_surface_train import RaySurface, NativeRayInputs
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Validated CUDA runtime is required')
    started = time.monotonic()
    models, stores, outputs = {}, {}, {}
    values = {arm: np.full((144, 40, 13, 2), np.nan, np.float32) for arm in NEW_ARMS}
    identity = dict(input_sha256=hashes, bulk_sha256=bulk, bulk_stat=snapshots)
    checkpoint_dir = OUT/'inference_units'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    batch = logits = distribution = model = state = None
    try:
        for arm in NEW_ARMS:
            model = RaySurface().cuda()
            state = torch.load(OUT/'models'/arm/'model_seed0.pt', weights_only=True, map_location='cpu')
            model.load_state_dict(state, strict=True)
            models[arm] = model.eval()
        for split in ('calib', 'evaluation'):
            stores[split] = NativeRayInputs(split, units=MC.SPLITS[split])
            declared = {str(Path(path).resolve()): digest for path, digest in stores[split].input_sha256.items()}
            bound = {str(Path(path).resolve()): digest for path, digest in {**hashes, **bulk}.items()}
            if any(bound.get(path) != digest for path, digest in declared.items()):
                raise ValueError('Native input preparation uses bytes outside the bound inference identity')
        config = np.repeat(np.arange(40), 13)
        frame = np.tile(FRAMES, 40)
        with torch.no_grad():
            for ui, unit in enumerate(UNITS):
                split = 'calib' if unit in MC.SPLITS['calib'] else 'evaluation'
                checkpoint, record_path = checkpoint_dir/f'unit{unit}.npz', checkpoint_dir/f'unit{unit}.json'
                if checkpoint.exists() or record_path.exists():
                    record = OE.read(record_path)
                    if record.get('status') != 'COMPLETE' or record.get('identity') != identity or record.get('unit') != unit or record.get('split') != split or record.get('output_sha256') != ME.sha(checkpoint):
                        raise ValueError('Incomplete or changed inference unit checkpoint')
                    with np.load(checkpoint, allow_pickle=False) as cache:
                        if set(cache.files) != set(NEW_ARMS):
                            raise ValueError('Checkpoint arm keys differ')
                        for arm in NEW_ARMS:
                            if cache[arm].shape != (40, 13, 2) or not np.isfinite(cache[arm]).all():
                                raise ValueError('Checkpoint score shape/values differ')
                            values[arm][ui] = cache[arm]
                    print('reuse inferred unit', unit, flush=True)
                    continue
                for begin in range(0, 520, 64):
                    end = min(begin+64, 520)
                    cc, ff = config[begin:end], frame[begin:end]
                    prepared = stores[split].batch(np.full(end-begin, unit), cc, ff)
                    validate_prepared(prepared, end-begin)
                    batch = [torch.as_tensor(value, device='cuda') for value in prepared]
                    base = torch.as_tensor(m3[ui, cc, ff-3], dtype=torch.float32, device='cuda')
                    for arm, model in models.items():
                        logits, distribution = model(*batch, base)
                        validate_outputs(logits, distribution, end-begin)
                        values[arm][ui, cc, ff-3] = logits.cpu().numpy()
                verify_inputs(identity)
                with checkpoint.open('xb') as stream:
                    np.savez_compressed(stream, **{arm: values[arm][ui] for arm in NEW_ARMS})
                OE.save(record_path, dict(status='COMPLETE', unit=unit, split=split,
                    identity=identity, output_sha256=ME.sha(checkpoint)))
                print('infer completed unit', unit, 'elapsed_s', round(time.monotonic()-started, 1), flush=True)
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError('Incomplete inference score array')
        verify_inputs(identity)
        for arm in NEW_ARMS:
            path = OUT/f'frame_scores_{arm}.npz'
            if path.exists():
                # Recover only exactly matching fully written outputs after an interrupted assembly.
                with np.load(path, allow_pickle=False) as cache:
                    if set(cache.files) != set(map(str, UNITS)) or any(not np.array_equal(cache[str(u)], values[arm][i]) for i, u in enumerate(UNITS)):
                        raise ValueError('Existing assembled output differs; preserve evidence')
            else:
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **{str(u): values[arm][i] for i, u in enumerate(UNITS)})
            outputs[path.name] = ME.sha(path)
    finally:
        for store in stores.values():
            store.close()
        models.clear(); stores.clear()
        batch = logits = distribution = model = state = base = prepared = None
        gc.collect()
        torch.cuda.empty_cache()
    receipt = dict(status='COMPLETE', **identity, output_sha256=outputs,
        plan_sha256=ME.sha(OUT/'PLAN.json'), training_receipt_sha256=ME.sha(OUT/'training_receipt.json'),
        units=UNITS, frames=FRAMES.tolist(), shape_per_unit=[40, 13, 2], arms=list(NEW_ARMS),
        new_branch_seeds=[0], retained_M3=m3_receipt, elapsed_s=time.monotonic()-started,
        schema='NPZ string unit keys; raw seed0 alarm logits before causal smoothing',
        operations=dict(native_observations=True, public_sensor_to_travel=True, frozen_M3_raw=True,
            surface_labels=False, geometry_truth=False, rendering=False, voxel_history=False, training=False),
        bulk_validation='One full observation SHA pass at startup; later source/model SHA plus bulk size/mtime_ns checks',
        runtime=dict(torch=torch.__version__, device=torch.cuda.get_device_name(), tf32=False,
            threads=2, batch=64, allocated_after_release=torch.cuda.memory_allocated()))
    OE.save(receipt_path, receipt)
    print('COMPLETE native ray surface seed0 inference', flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'infer', 'evaluate'))
    args = parser.parse_args()
    {'check': check, 'infer': infer, 'evaluate': evaluate}[args.stage]()
