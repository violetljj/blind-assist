"""Compact Chinese report from candidate-quality analysis; no metric retuning."""
import argparse
import json
from pathlib import Path


def pct(value):
    return '—' if value is None else f'{value*100:.1f}%'


def plot_curves(data, output):
    """Static descriptive dose curves; marker crosses flag FA budget mismatch."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    panels = (
        ('Added false candidates per frame', [0, 1, 3, 10], ['G1', 'FA1', 'FA3', 'FA10']),
        ('Angular offset (zones)', [0, .25, .5, 1], ['G1', 'SHIFT025', 'SHIFT05', 'SHIFT1']),
        ('Footprint dilation (zones)', [0, .5, 1], ['G1', 'DILATE05', 'DILATE1']),
        ('Candidate drop probability', [0, .2, .4], ['G1', 'DROP20', 'DROP40']))
    colors = {'HEAD': '#1764a0', 'BODY': '#b45309'}
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, layout='constrained')
        for index, (ax, (label, xs, arms)) in enumerate(zip(axes.flat, panels)):
            for group, color in colors.items():
                rows = data['quality_at_10pct'][group]['policies']
                series = [(arms, '-', 'o')]
                if index == 0:
                    series.append((['G1', 'BGFA1', 'BGFA3', 'BGFA10'], '--', 's'))
                for names, style, marker in series:
                    ys = [np.nan if rows[a]['retention_of_G1_gain'] is None else rows[a]['retention_of_G1_gain'] for a in names]
                    ax.plot(xs, ys, style, marker=marker, color=color, linewidth=1.6, markersize=4)
                    for x, y, a in zip(xs, ys, names):
                        if rows[a]['false_alert_exceeds_G0_by_2pp']:
                            ax.scatter([x], [y], marker='x', s=95, linewidths=2, color='#b91c1c', zorder=5)
            ax.axhline(.5, color='#777777', linestyle=':', linewidth=1.2)
            ax.axhline(0, color='#cccccc', linewidth=.7)
            ax.set_xlabel(label)
            ax.set_xticks(xs)
            ax.grid(alpha=.18)
            if index % 2 == 0:
                ax.set_ylabel('Retention of ideal-angle timely gain')
        handles = [Line2D([0], [0], color=c, label=g) for g, c in colors.items()]
        handles += [Line2D([0], [0], color='gray', linestyle='--', marker='s', label='Background-supported FA'),
                    Line2D([0], [0], color='#b91c1c', marker='x', linestyle='', label='Actual FA > G0 + 2 pp')]
        fig.legend(handles=handles, loc='outside lower center', ncols=2, frameon=False)
        fig.suptitle('Candidate quality vs tiny near-event timely gain\n10% calibration FA budget; consumed v2 Development', fontsize=12)
        fig.savefig(output, dpi=180)
        plt.close(fig)


def render(data, json_path, plot_path=None):
    quality = data['quality_at_10pct']
    gain = '；'.join(f"{g} 理想角度增益 " + ('不可评估' if quality[g]['G1_gain_over_G0'] is None else f"{100*quality[g]['G1_gain_over_G0']:+.1f}个百分点") for g in ('HEAD', 'BODY'))
    stop = data['G4prime_decision']['abandon_this_exact_route']
    parity = data['G0_reference_parity']
    baseline = []
    for group in ('HEAD', 'BODY'):
        entries = []
        for arm in ('G0', 'G1'):
            row = quality[group]['policies'][arm]
            ti, fa = row['tiny_timely'], row['actual_false_alert']
            entries.append(f"{arm}及时{ti['numerator']}/{ti['denominator']}、假警{fa['numerator']}/{fa['denominator']}（{pct(fa['rate'])}）")
        baseline.append(group+'：'+' → '.join(entries))
    moderate = '/'.join(pct(quality[g]['policies']['COMBO_MODERATE']['retention_of_G1_gain']) for g in ('HEAD', 'BODY'))
    enabled = []
    for key, row in data['results'].items():
        if '/G4prime/' not in key or row['thresholds'][1] == 'disabled':
            continue
        group, _, budget = key.split('/')
        baseline_row = data['results'][f'{group}/G0/{budget}']['eval']
        dt = row['eval']['tiny']['timely']['rate']-baseline_row['tiny']['timely']['rate']
        df = row['eval']['all']['false_alert']['rate']-baseline_row['all']['false_alert']['rate']
        enabled.append(f'{group}@{float(budget):.0%}：及时{dt*100:+.2f}pp、假警{df*100:+.2f}pp')
    g4text = '六点均禁用，停止该具体方案' if stop else f'{6-len(enabled)}/6点禁用；启用点'+ '；'.join(enabled)+'；未触发六点全关闭停止条件'
    ideal_flags = '；'.join(f"{g} G1假警比G0高{quality[g]['policies']['G1']['false_alert_delta_over_G0']*100:.3f}pp，零剂量因此被标记，不能解释为理想候选无效"
        for g in ('HEAD', 'BODY') if quality[g]['policies']['G1']['false_alert_exceeds_G0_by_2pp'])
    generated = data.get('candidate_generation_checks', {}).get('background')
    placement_note = '' if not generated else (f"背景假候选{generated['partial_fallback']/generated['requested']:.2%}需裁切，"
        f"{1-generated['generated']/generated['requested']:.2%}无法放置；剂量为请求数。")
    lines = ['# 候选质量剂量诊断（已消费 v2 Development）', '',
        f"**描述性结果：{gain}；中等组合只保留{moderate}收益（HEAD/BODY）。** G4′ {g4text}。", '',
        '10%校准假警预算，小块近事件及时数/事件分母；假警为空查询序列数/空序列分母。'+'；'.join(baseline)+'。', '',
        '**候选质量规格（仅已测值）：**通过要求增益保留≥50%，且实际假警≤G0+2个百分点。0剂量=G1，仍须通过假警条件；G1增益≤0时保留率不可定义。', '',
        '|候选质量轴（已测剂量）|HEAD通过值|BODY通过值|',
        '|---|---|---|']
    axes = (
        ('均匀假候选/帧（0、1、3、10）', [('G1', '0'), ('FA1', '1'), ('FA3', '3'), ('FA10', '10')]),
        ('背景支持假候选/帧（0、1、3、10）', [('G1', '0'), ('BGFA1', '1'), ('BGFA3', '3'), ('BGFA10', '10')]),
        ('角度偏移区（0、0.25、0.5、1）', [('G1', '0'), ('SHIFT025', '0.25'), ('SHIFT05', '0.5'), ('SHIFT1', '1')]),
        ('膨胀区（0、0.5、1）', [('G1', '0'), ('DILATE05', '0.5'), ('DILATE1', '1')]),
        ('漏候选概率（0、20%、40%）', [('G1', '0'), ('DROP20', '20%'), ('DROP40', '40%')]),
        ('中等组合：0.5/0.5/20%/3', [('COMBO_MODERATE', '通过')]),
        ('严重组合：1/1/40%/10', [('COMBO_SEVERE', '通过')]))
    for label, arms in axes:
        cells = [label]
        for group in ('HEAD', 'BODY'):
            states = [quality[group]['policies'][arm]['meets_quality_spec'] for arm, _ in arms]
            passed = [value for (arm, value), state in zip(arms, states) if state is True]
            cells.append('、'.join(passed) if passed else '不可评估' if all(s is None for s in states) else '无')
        lines.append('|'+'|'.join(cells)+'|')
    lines += ['', '组合顺序=偏移/膨胀/漏候选/假候选。通过值可能非单调，是实测集合而非容差上界。'+ideal_flags+'。全部18策略×3预算、事件/距离分层及阈值见[完整JSON]('+json_path.as_posix()+')。', '',
        f"G0对账：{parity['all']['units']}单位相对误差最大{parity['all']['max_rel']:.2g}，eval最大{parity['eval']['max_rel']:.2g}（容差1e-5）；参考为修复输入上新生成的原版GPU2 S2，历史v2无读出文件。", '',
        '**下一步：**先检查能否产生少量干净的视觉候选，再决定RGB投入；不要把单轴通过值拼成联合规格，不插值或外推合格范围。', '',
        '**范围与局限：**使用特权信息，不是相机结果，也不是可部署方案。32个calib、63个audit单位，排除原始INCOMPLETE143；v2曾失败，修复仅作已消费Development，不能补认原正式实验。阈值只在calib选择，audit实际假警可能不同。质量规格依赖本次足迹尺寸/位置分布及逐帧独立扰动，不是通用相机要求；未测时间持续的假候选。BGFA为背景支持区，区内仍可能包含前景。'+placement_note+'G4′停止结论仅针对此方案。']
    if plot_path is not None:
        lines.insert(-2, '[质量—收益曲线PNG]('+plot_path.as_posix()+')；线段仅连接已测值，不表示插值保证。')
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis')
    parser.add_argument('output')
    args = parser.parse_args()
    source = Path(args.analysis).resolve()
    destination = Path(args.output).resolve()
    data = json.loads(source.read_text(encoding='utf-8'))
    checks = source.with_name('candidate_generation_checks.json')
    if checks.exists():
        data['candidate_generation_checks'] = json.loads(checks.read_text(encoding='utf-8'))
    plot_path = destination.with_name('candidate_quality.png')
    plot_curves(data, plot_path)
    destination.write_text(render(data, Path(source.name) if source.parent == destination.parent else source,
                                  Path(plot_path.name)), encoding='utf-8')


if __name__ == '__main__':
    main()
