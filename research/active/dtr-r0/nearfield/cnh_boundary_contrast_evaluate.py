"""Thin sealed-score adapter for the boundary-contrast Development pilot.

Reuses the existing query-mass statistics and raw-seal checks with temporary
module settings restored in finally. No old checkpoint or experiment is run.
"""
import argparse
from contextlib import contextmanager
import json

import numpy as np

import cnh_query_mass_evaluate as Q

MC, ME, OE, SE, ST = Q.MC, Q.ME, Q.OE, Q.SE, Q.ST
OUT = MC.SS.WORK/'cnh-boundary-contrast-20261003'
RUN = 'CNH_BOUNDARY_CONTRAST_20261003'
NEW_ARMS = ('CBASE', 'CCON')
ARMS = ('M3',)+NEW_ARMS
UNITS, FRAMES, M3_THRESHOLD = Q.UNITS, Q.FRAMES, Q.M3_THRESHOLD
BOOT_SEED = 2026100304
N_BOOT = 1000
CATEGORIES = Q.CATEGORIES
number = Q.number
verify_inputs = Q.verify_inputs


@contextmanager
def settings(**updates):
    """Synchronous reuse only; preserve imported helper state even on failure."""
    previous = {key: getattr(Q, key) for key in updates}
    try:
        for key, value in updates.items():
            setattr(Q, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(Q, key, value)


def rename_result(value):
    if isinstance(value, dict):
        return {rename_result(k): rename_result(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rename_result(v) for v in value]
    if isinstance(value, tuple):
        return tuple(rename_result(v) for v in value)
    if isinstance(value, str):
        return value.replace('QBCE', 'CBASE').replace('QMASS', 'CCON').replace('QUERY_MASS', 'BOUNDARY_CONTRAST')
    return value


def analyze(g, scores):
    # Internal statistical aliases are never used to select input files.
    aliases = dict(M3=scores['M3'], QBCE=scores['CBASE'], QMASS=scores['CCON'])
    with settings(BOOT_SEED=BOOT_SEED, N_BOOT=N_BOOT):
        result = rename_result(Q.analyze(g, aliases))
    result['readout'] = ('CCON versus same-structure CBASE and frozen M3; boundary-contrast auxiliary loss; '
        'timely benefit, deep retention and clear false-stop cost jointly reported; reference checks are descriptive')
    return result


def load_evaluation_inputs():
    # This binds real CBASE/CCON files/receipts and verifies both raw NPZ caches
    # before Q's first all-object truth read. Restoring settings keeps Q frozen.
    with settings(OUT=OUT, RUN=RUN, NEW_ARMS=NEW_ARMS, ARMS=ARMS, BOOT_SEED=BOOT_SEED, N_BOOT=N_BOOT):
        g, scores, prior, provenance = Q.load_evaluation_inputs()
    provenance['evaluator_sha256'] = ME.sha(__file__)
    provenance['helper_sha256'][Q.__file__] = ME.sha(Q.__file__)
    provenance['input_sha256'][__file__] = ME.sha(__file__)
    return g, scores, prior, provenance


def write_report(result):
    lines = [result['verdict'], '', '# 分界对比：单种子Development小试', '',
        'CBASE/CCON从同一QueryMass结构随机初始化，共用冻结M3五seed平均logit残差起点，seed0各训练10epochs；两臂初始化、base样本顺序与pair抽样完全相同，maps仅为latent，不做可见支持量回归。推理不读取几何真值。', '',
        '每步两臂均额外看到同样8个边界pair，并在base BCE上加入0.5倍pair query BCE；CCON在完全相同base/pair样本BCE之外，再加入权重1的mean relu(1−posLogit+negLogit)。因此CBASE控制额外pair样本暴露，CCON检验分界对比损失的增量。', '',
        '沿用五时刻指数因果平滑；新臂分别在48校准单位，以整体清晰1次首停/代理分钟确定HEAD/BODY共用阈值；M3阈值冻结。96评估单位使用三级全物体真值。', '',
        '| 比较 | 浅及时差及95%区间，pp | 深及时差，pp | 外10–20cm清晰首停差及95%区间，次/分钟 | 整体清晰差，次/分钟 |',
        '| --- | --- | --- | --- | --- |']
    for control, comparison in result['comparisons_CCON_minus_control'].items():
        p, s, d, c = (comparison[k] for k in ('primary_outside_clear', 'contact0-2cm', 'contact>5cm', 'clear'))
        lines.append(f"| CCON−{control} | {number(s['delta_timely'],100)} [{number(s['paired_unit_ci95'][0],100)}, {number(s['paired_unit_ci95'][1],100)}] | {number(d['delta_timely'],100)} | {number(p['delta_per_min'])} [{number(p['paired_unit_ci95'][0])}, {number(p['paired_unit_ci95'][1])}] | {number(c['delta_per_min'])} |")
    lines += ['', '| 指标 | M3 | CBASE | CCON |', '| --- | --- | --- | --- |']
    for category in CATEGORIES:
        lines.append('| '+category+' | '+' | '.join(OE.rate_text(result['cells'][a]['metrics'][category], category == 'clear', category == 'pass0-10cm') for a in ARMS)+' |')
    lines += ['', '| 清晰子组 | M3 | CBASE | CCON |', '| --- | --- | --- | --- |']
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
        '1000次整单位共同bootstrap，seed2026100304；区间条件于seed0模型与校准阈值，不含重训练、校准集或模拟器不确定性。', '',
        '四项参照联合判读（浅及时区间下界>0、深差≥−2pp、外侧误停差≤0、整体误停差≤0.2次/分钟）仅作描述，不是硬停规则：', '',
        json.dumps(result['descriptive_reference_checks'], ensure_ascii=False), '',
        'PROMISING/NOT_ESTABLISHED仅为本小试参照判读，不是新主线晋升或实机能力声明。旧RAY和QMASS均保留，本轮没有重新运行。', '',
        '配对补回/丢失与差值区间：', '', '```json', json.dumps(result['comparisons_CCON_minus_control'], ensure_ascii=False, indent=2), '```', '',
        '已消费Development、同名义仿真源、单种子探索，不代表盲测、跨源或实机效果。训练边界对比标签不是新的传感器信息，不增加8×8物理分辨率，也不是经过标定的碰撞概率。', '',
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
    scores['CCON'][2, 1] = 2.
    scores['M3'][5, 1] = scores['CBASE'][5, 1] = 2.
    before = (Q.OUT, Q.RUN, Q.NEW_ARMS, Q.ARMS, Q.BOOT_SEED)
    result = analyze(g, scores)
    with settings(BOOT_SEED=BOOT_SEED):
        expected = rename_result(Q.analyze(g, dict(M3=scores['M3'], QBCE=scores['CBASE'], QMASS=scores['CCON'])))
    for key in ('cells', 'thresholds', 'comparisons_CCON_minus_control', 'bootstrap', 'n'):
        assert result[key] == expected[key]
    assert result['bootstrap']['seed'] == BOOT_SEED
    assert set(result['cells']) == set(ARMS)
    assert result['comparisons_CCON_minus_control']['M3']['contact0-2cm']['rescues'] == 1
    assert result['n']['right_censored'] == 1
    assert 'QMASS' not in json.dumps(result) and 'QBCE' not in json.dumps(result)
    try:
        with settings(OUT=OUT, RUN=RUN, NEW_ARMS=NEW_ARMS, ARMS=ARMS, BOOT_SEED=BOOT_SEED):
            raise RuntimeError('Synthetic adapter failure')
    except RuntimeError:
        pass
    assert (Q.OUT, Q.RUN, Q.NEW_ARMS, Q.ARMS, Q.BOOT_SEED) == before
    changed = {a: s.copy() for a, s in scores.items()}
    for value in changed.values():
        value[2:] += 100
    assert analyze(g, changed)['thresholds'] == result['thresholds']
    json.dumps(result, allow_nan=False)
    print('PASS CBASE/CCON adapter parity, bootstrap seed, calib-only thresholds, censoring and exception-safe helper restoration; no cohort truth loaded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check', 'evaluate', 'finalize'))
    args = parser.parse_args()
    {'check': check, 'evaluate': evaluate, 'finalize': evaluate}[args.stage]()
