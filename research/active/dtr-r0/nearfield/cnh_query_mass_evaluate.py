"""Query-local visible surface support: sealed-score Development evaluation.

Only evaluate/finalize opens all-object truth. The new scores are QBCE/QMASS
raw seed0 logits, before the inherited five-frame causal smoothing. This
adapter preserves the existing all-object statistics, without changing or
rerunning the stopped RAY experiment.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import cnh_margin_confirm as MC
import cnh_margin_confirm_evaluate as ME
import cnh_ray_surface_evaluate as RE
import cnh_sequence_extrinsic_evaluate as XE
import cnh_sequence_observed_evaluate as OE
import cnh_three_level_sequence as SE
import cnh_yaw_envelope_evaluate as YE
import cnh_yaw_source_transfer_evaluate as ST

OUT = MC.SS.WORK/'cnh-query-mass-20261003'
RUN = 'CNH_QUERY_MASS_20261003'
NEW_ARMS = ('QBCE', 'QMASS')
ARMS = ('M3',)+NEW_ARMS
UNITS = MC.SPLITS['calib']+MC.SPLITS['evaluation']
FRAMES = np.arange(3, 16)
M3_THRESHOLD = .8557642486787612
BOOT_SEED = 2026100303
N_BOOT = 1000
CATEGORIES = OE.CONTACTS+('pass0-10cm', 'clear')
PRIMARY_GROUP = 'same_height_nominal_outside10_20cm'


def verify_inputs(receipt):
    RE.verify_inputs(receipt)


def analyze(g, scores):
    cal, ev = g['split'] == 'calib', g['split'] == 'evaluation'
    fields = ('unit', 'query', 'target_group', 'target_off', 'frame_ranges', 'covered', 'clear_all', 'ref_category')
    if any(g[k].shape[0] != len(ev) for k in fields):
        raise ValueError('Required geometry rows do not align with split')
    e = {key: g[key][ev] for key in fields}
    units, ui = np.unique(e['unit'], return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(rng.integers(len(units), size=len(units)), minlength=len(units)) for _ in range(N_BOOT)])
    thresholds = {'M3': M3_THRESHOLD}
    calibration = {'M3': dict(role='Retained old threshold; no new calibration')}
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
    comparisons, checks = {}, {}
    for control in ('QBCE', 'M3'):
        comparison = XE.paired_changes(cells, sampled, flags, weights, 'QMASS', control)
        group_changes = {}
        for name, keep in groups.items():
            a, b = (cells[arm]['clear_subgroups'][name]['false_stops_per_min']['value'] for arm in ('QMASS', control))
            na, ca = flags['QMASS']['stopped'], flags[control]['stopped']
            group_changes[name] = dict(delta_per_min=a-b if a is not None and b is not None else None,
                paired_unit_ci95=SE.interval(subgroup_samples['QMASS'][name]-subgroup_samples[control][name]),
                added_first_stops=int((keep & na & ~ca).sum()), removed_first_stops=int((keep & ca & ~na).sum()))
        comparison['clear_subgroups'] = group_changes
        comparison['primary_outside_clear'] = group_changes[PRIMARY_GROUP]
        comparisons[control] = comparison
        # The previous experiment's yardstick is descriptive here, not its stop rule.
        checks[control] = RE.comparison_checks(comparison)
    verdict = ('QUERY_MASS_PROMISING_SINGLE_SEED_DEV' if all(all(c.values()) for c in checks.values())
               else 'QUERY_MASS_NOT_ESTABLISHED_SINGLE_SEED_DEV')
    return dict(status='COMPLETE', verdict=verdict,
        comparisons_QMASS_minus_control=comparisons, descriptive_reference_checks=checks,
        cells=cells, thresholds=thresholds, calibration=calibration,
        evaluation_units=units.tolist(), clear_subgroup_counts={k: int(m.sum()) for k, m in groups.items()},
        n=dict(calibration_units=len(np.unique(g['unit'][cal])), evaluation_units=len(units), query_episodes=int(ev.sum()),
            covered=int(e['covered'].sum()), right_censored=int((~e['covered']).sum()), all_clear=int(e['clear_all'].sum()),
            shallow_episodes=int(weights['contact0-2cm'].sum()),
            shallow_contributing_units=len(np.unique(e['unit'][weights['contact0-2cm'] > 0]))),
        bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEED, cluster='whole evaluation units',
            pairing='one common bootstrap matrix for every arm/subgroup',
            conditional_on='trained seed0 checkpoints and calibrated QBCE/QMASS thresholds'),
        primary_group_definition='query==target_group AND -0.20<=nominal target_off<-0.10 AND all13frame clear_all; nominal metadata, not dynamic gap',
        readout='QMASS versus same-structure QBCE and frozen M3; timely benefit, deep retention and clear false-stop cost reported jointly; reference checks do not impose the stopped RAY run rule')


def validate_plan(plan):
    expected = dict(run=RUN, calibration_units=MC.SPLITS['calib'], evaluation_units=MC.SPLITS['evaluation'],
        decision_frames=FRAMES.tolist(), arms=list(ARMS), train_arms=list(NEW_ARMS), seeds=[0],
        frozen_M3_threshold=M3_THRESHOLD)
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError('PLAN differs from query-mass evaluator: '+key)
    if plan['bootstrap']['replicates'] != N_BOOT or plan['bootstrap']['seed'] != BOOT_SEED:
        raise ValueError('PLAN bootstrap differs from evaluator')
    if RUN not in (Path(__file__).resolve().parents[1]/'RUNS.md').read_text(encoding='utf8'):
        raise ValueError('RUNS registration is required')
    YE.verify_hashes(plan['prior_sha256'])


def load_evaluation_inputs():
    plan_path, receipt_path, training_path = OUT/'PLAN.json', OUT/'scores_receipt.json', OUT/'training_receipt.json'
    plan, receipt, training = OE.read(plan_path), OE.read(receipt_path), OE.read(training_path)
    validate_plan(plan)
    if receipt.get('status') != 'COMPLETE' or receipt.get('plan_sha256') != ME.sha(plan_path):
        raise ValueError('Scores incomplete or belong to another PLAN')
    if training.get('status') != 'COMPLETE' or training.get('plan_sha256') != ME.sha(plan_path):
        raise ValueError('Training incomplete or belongs to another PLAN')
    if receipt.get('training_receipt_sha256') != ME.sha(training_path):
        raise ValueError('Score/training receipt identity differs')
    expected = dict(units=UNITS, frames=FRAMES.tolist(), shape_per_unit=[40, 13, 2], arms=list(NEW_ARMS))
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError('Score receipt identity differs: '+key)
    if set(receipt['output_sha256']) != {f'frame_scores_{a}.npz' for a in NEW_ARMS}:
        raise ValueError('Score outputs must bind exactly QBCE and QMASS')
    verify_inputs(receipt)
    new_scores, score_hashes = {}, {}
    for arm in NEW_ARMS:
        path = OUT/f'frame_scores_{arm}.npz'
        digest = ME.sha(path)
        if receipt['output_sha256'][path.name] != digest:
            raise ValueError('New score cache changed: '+arm)
        with np.load(path, allow_pickle=False) as cache:
            if set(cache.files) != set(map(str, UNITS)):
                raise ValueError('New score unit keys differ: '+arm)
            raw = np.stack([cache[str(u)] for u in UNITS])
        if raw.shape != (144, 40, 13, 2) or not np.isfinite(raw).all():
            raise ValueError('New score shape/values differ: '+arm)
        new_scores[arm] = SE.smooth(raw).transpose(0, 1, 3, 2).reshape(-1, 13)
        score_hashes[str(path)] = digest
    # This is the first entry point permitted to open evaluation truth.
    g, retained, old_provenance = OE.load_inputs()
    rows_path = OE.OUT/'rows.json'
    rows = OE.read(rows_path)
    keys = [(r['split'], r['unit'], r['config'], r['query']) for r in rows]
    if keys != list(zip(g['split'].tolist(), g['unit'].tolist(), g['config'].tolist(), g['query'].tolist())):
        raise ValueError('All-object nominal row identity differs')
    g['target_group'] = np.asarray([r['target_group'] for r in rows])
    g['target_off'] = np.asarray([r['target_off'] for r in rows])
    hashes = dict(receipt['input_sha256']); hashes.update(old_provenance['input_sha256'])
    hashes.update({str(p): ME.sha(p) for p in (plan_path, receipt_path, training_path, rows_path)})
    hashes.update(score_hashes)
    scores = {'M3': retained['M3'], **new_scores}
    prior_path = OE.OUT/'result.json'
    prior = OE.read(prior_path)
    if prior.get('status') != 'COMPLETE' or prior['cells']['M3']['threshold'] != M3_THRESHOLD:
        raise ValueError('Old M3 threshold/result differs')
    hashes[str(prior_path)] = ME.sha(prior_path)
    return g, scores, prior, dict(input_sha256=hashes, bulk_sha256=receipt['bulk_sha256'], bulk_stat=receipt['bulk_stat'],
        plan=plan, scores_receipt=receipt, training_summary=training, evaluator_sha256=ME.sha(__file__),
        helper_sha256={p: ME.sha(p) for p in (RE.__file__, SE.__file__, OE.__file__, ST.__file__, XE.__file__)})


def old_ray_diagnostic():
    path = RE.OUT/'result.json'
    if not path.exists():
        return dict(status='NOT_AVAILABLE', role='Optional prior diagnostic; no RAY model executed')
    previous = OE.read(path)
    if previous.get('status') != 'COMPLETE':
        return dict(status='NOT_COMPLETE', role='Optional prior diagnostic; no RAY model executed')
    return dict(status='COMPLETE', role='Historical point values only; different bootstrap seed, no paired rerun',
        result_path=str(path), result_sha256=ME.sha(path), verdict=previous['verdict'],
        thresholds=previous['thresholds'], n=previous['n'], cells=previous['cells'])


def number(value, scale=1.):
    return '--' if value is None else f'{value*scale:.2f}'


def write_report(result):
    lines = [result['verdict'], '', '# 身体查询局部可见表面支持量：单种子Development小试', '',
        'QBCE/QMASS是同结构seed0新分支，共用冻结M3五seed平均logit残差起点；QMASS增加训练期局部可见表面支持量监督，QBCE仅报警监督。每个公开查询条件预测2×16×16的归一化log支持量；报警头联合训练。推理不读取几何真值。', '',
        '沿用五时刻指数因果平滑；新臂分别在48校准单位，以整体清晰1次首停/代理分钟确定HEAD/BODY共用阈值；M3阈值冻结。96评估单位使用三级全物体真值。', '',
        '| 比较 | 浅及时差及95%区间，pp | 深及时差，pp | 外10–20cm清晰首停差及95%区间，次/分钟 | 整体清晰差，次/分钟 |',
        '| --- | --- | --- | --- | --- |']
    for control, comparison in result['comparisons_QMASS_minus_control'].items():
        p, s, d, c = (comparison[k] for k in ('primary_outside_clear', 'contact0-2cm', 'contact>5cm', 'clear'))
        lines.append(f"| QMASS−{control} | {number(s['delta_timely'],100)} [{number(s['paired_unit_ci95'][0],100)}, {number(s['paired_unit_ci95'][1],100)}] | {number(d['delta_timely'],100)} | {number(p['delta_per_min'])} [{number(p['paired_unit_ci95'][0])}, {number(p['paired_unit_ci95'][1])}] | {number(c['delta_per_min'])} |")
    lines += ['', '| 指标 | M3 | QBCE | QMASS |', '| --- | --- | --- | --- |']
    for category in CATEGORIES:
        lines.append('| '+category+' | '+' | '.join(OE.rate_text(result['cells'][a]['metrics'][category], category == 'clear', category == 'pass0-10cm') for a in ARMS)+' |')
    lines += ['', '| 清晰子组 | M3 | QBCE | QMASS |', '| --- | --- | --- | --- |']
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
    lines += ['', '分母与删失：'+json.dumps(result['n'], ensure_ascii=False), '',
        '1000次整单位共同bootstrap，seed2026100303；区间条件于seed0模型与校准阈值，不含重训练、校准集或模拟器不确定性。', '',
        '沿用旧报警实验的四项参照判读（浅及时区间下界>0、深差≥−2pp、外侧误停差≤0、整体误停差≤0.2次/分钟），仅作描述，不继承旧RAY配方的硬停规则：', '',
        json.dumps(result['descriptive_reference_checks'], ensure_ascii=False), '',
        'PROMISING/NOT_ESTABLISHED仅为本小试参照判读，不是新主线晋升或实机能力声明。', '',
        '配对补回/丢失与差值区间：', '', '```json', json.dumps(result['comparisons_QMASS_minus_control'], ensure_ascii=False, indent=2), '```', '',
        '旧RAY仅引用保留结果点值，未运行旧模型、未用它选择当前阈值：', '']
    old = result['historical_RAY_diagnostic']
    if old['status'] == 'COMPLETE':
        lines += ['| 指标 | 旧RAY |', '| --- | --- |']
        for category in CATEGORIES:
            lines.append('| '+category+' | '+OE.rate_text(old['cells']['RAY']['metrics'][category], category == 'clear', category == 'pass0-10cm')+' |')
        lines += ['', '旧结果判读：'+old['verdict']+'；来源：'+old['result_path'], '']
    else:
        lines += [json.dumps(old, ensure_ascii=False), '']
    lines += [
        '本轮是已消费Development、同名义仿真源、单种子探索，不代表盲测、跨源或实机效果。局部可见表面支持量不是碰撞概率，也不增加8×8物理分辨率；零可见支持不证明畅通，valid不是实际SNR或产品UNKNOWN。', '',
        '清晰2.6秒/查询包含首停后暴露，13采样点清晰不保证帧间连续清晰，代理分钟不是现场步行分钟。右删失不作漏停；提前量条件于已报警，按0.8m/s相对0.5m换算，不代表人已安全停止。', '',
        '旧M3点复现：'+json.dumps(result['nominal_M3_point_reproduction'], ensure_ascii=False), '',
        '输入、模型与运行时来源见result.json provenance。大输入全SHA启动绑定，后续size/mtime_ns检查不等同重新哈希。', '']
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
    result['historical_RAY_diagnostic'] = old_ray_diagnostic()
    result['provenance'] = provenance
    result['new_branch_seeds'] = [0]
    result['frozen_M3_seeds'] = list(range(5))
    result['heading_sigma_deg'] = 0
    verify_inputs(provenance)
    write_report(result)
    OE.save(path, result)
    print(result['verdict'], json.dumps(result['descriptive_reference_checks']), flush=True)
    return result


def check():
    # Meaningful adapter check: exact statistical parity with the old evaluator
    # after only arm renaming and common bootstrap-seed substitution.
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
    scores['QMASS'][2, 1] = 2.
    scores['M3'][5, 1] = scores['QBCE'][5, 1] = 2.
    result = analyze(g, scores)
    old_seed = RE.BOOT_SEED
    try:
        RE.BOOT_SEED = BOOT_SEED
        old = RE.analyze(g, dict(M3=scores['M3'], BCEO=scores['QBCE'], RAY=scores['QMASS']))
    finally:
        RE.BOOT_SEED = old_seed
    for new, former in (('M3', 'M3'), ('QBCE', 'BCEO'), ('QMASS', 'RAY')):
        assert result['cells'][new] == old['cells'][former]
        assert result['thresholds'][new] == old['thresholds'][former]
    assert result['n'] == old['n']
    for new, former in (('QBCE', 'BCEO'), ('M3', 'M3')):
        assert result['comparisons_QMASS_minus_control'][new] == old['comparisons_RAY_minus_control'][former]
    assert result['n']['right_censored'] == 1
    changed = {a: s.copy() for a, s in scores.items()}
    for value in changed.values():
        value[2:] += 100
    assert analyze(g, changed)['thresholds'] == result['thresholds']
    json.dumps(result, allow_nan=False)
    print('PASS renamed-arm statistical parity, paired bootstrap, calibration-only thresholds and censoring; no cohort truth loaded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'evaluate', 'finalize'))
    args = parser.parse_args()
    {'check': check, 'evaluate': evaluate, 'finalize': evaluate}[args.stage]()
