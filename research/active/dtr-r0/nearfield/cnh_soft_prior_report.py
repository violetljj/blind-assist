"""Render existing soft-prior results and audit the generated score distribution."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata
from cnh_soft_prior_analysis import CONDITIONS, AUCS


def confidence_audit(root):
    samples = {s: {c: [] for c in CONDITIONS} for s in ('calib', 'audit')}
    units = {s: 0 for s in samples}
    for path in sorted((root/'scores').glob('confidence*.npz')):
        unit = int(path.stem[10:])
        split = 'calib' if unit < 128 else 'audit'
        units[split] += 1
        with np.load(path) as f:
            for condition in CONDITIONS:
                samples[split][condition].append(f[condition])
    assert units == {'calib': 32, 'audit': 63}, units
    result = {}
    for split, conditions in samples.items():
        result[split] = {}
        frames = units[split]*32*9
        for condition, parts in conditions.items():
            sample = np.concatenate(parts)
            real = sample[:, 0].astype(bool)
            nt, nf = int(real.sum()), int((~real).sum())
            aucs = {}
            for auc in AUCS:
                scores = ndtr(sample[:, 1]+real*np.sqrt(2)*ndtri(auc/100)).astype(np.float32)
                aucs[str(auc)] = (float((rankdata(scores)[real].sum()-nt*(nt+1)/2)/(nt*nf)) if nt and nf else None)
            result[split][condition] = dict(evaluated_frames=frames, true_origin_candidates=nt, false_origin_candidates=nf,
                mean_true_per_frame=nt/frames, mean_false_per_frame=nf/frames, empirical_candidate_auc=aucs,
                label='real nonBACKGROUND origin vs injected false origin, not query-positive vs negative')
    (root/'confidence_distribution.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def pct(value):
    return 'NA' if value is None else f'{100*value:.1f}%'


def count(value):
    return f"{pct(value['rate'])} ({value['numerator']}/{value['denominator']})"


def make_plot(root, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, condition, title in zip(axes, ('COMBO_MODERATE', 'COMBO_SEVERE'), ('Moderate combined errors', 'Severe combined errors')):
        for group, color in (('HEAD', '#2864b7'), ('BODY', '#d17b17')):
            rows = [results[f'{group}/G1c__{condition}__A{auc}/0.10'] for auc in AUCS]
            values = [r['retention_of_fixed_hard_G1_gain']*100 for r in rows]
            ax.plot(np.array(AUCS)/100, values, 'o-', color=color, label=group+' weighted')
            soft = results[f'{group}/G1s__{condition}/0.10']['retention_of_fixed_hard_G1_gain']*100
            ax.axhline(soft, color=color, ls=':', alpha=.7, label=group+' soft')
            for auc, value, row in zip(AUCS, values, rows):
                if row['eval_false_alert_exceeds_G0_by_2pp']:
                    ax.plot(auc/100, value, 'x', ms=12, mew=2, color='red')
        ax.axhline(50, color='#666', ls='--', lw=1)
        ax.axhline(0, color='#aaa', lw=.7)
        ax.set(title=title, xlabel='Synthetic candidate AUC', xticks=np.array(AUCS)/100)
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Retention of fixed hard-G1 gain (%)')
    axes[1].plot([], [], 'x', color='red', ms=9, label='Actual FA > G0 + 2 pp')
    axes[1].legend(fontsize=8, loc='best')
    fig.suptitle('Consumed v2 Development | 10% calibration FA budget | privileged, not RGB', fontsize=10)
    fig.tight_layout()
    fig.savefig(root/'soft_prior.png', dpi=180)
    plt.close(fig)


def report(root, tracked_report=None):
    data = json.loads((root/'analysis.json').read_text(encoding='utf-8'))
    results = data['results']
    confidence = confidence_audit(root)
    make_plot(root, results)
    requirements = {}
    requirement_lines = ['# 候选质量：仅限已测试分布与误差组合', '',
        'AUC仅指四个设定档位，未插值。G1与DROP40只有真候选，没有双类经验AUC，其分数档不是可验收的AUC要求。要求收益保留≥50%，同时单列不超过G0+2pp假警的档位；不是联合质量保证。', '',
        '|条件|HEAD达到50%的AUC档|BODY达到50%的AUC档|HEAD同时不过FA旗标|BODY同时不过FA旗标|audit假候选/帧|', '|---|---|---|---|---|---|']
    for condition in CONDITIONS:
        record = {}
        for group in ('HEAD', 'BODY'):
            passing, unflagged = [], []
            for auc in AUCS:
                row = results[f'{group}/G1c__{condition}__A{auc}/0.10']
                if row['retention_of_fixed_hard_G1_gain'] >= .5:
                    passing.append(auc/100)
                    if not row['eval_false_alert_exceeds_G0_by_2pp']:
                        unflagged.append(auc/100)
            record[group] = dict(tested_auc_retaining_half=passing, tested_auc_retaining_half_without_FA_flag=unflagged)
        record['candidate_distribution'] = confidence['audit'][condition]
        requirements[condition] = record
        lists = [record[g][k] for k in ('tested_auc_retaining_half', 'tested_auc_retaining_half_without_FA_flag') for g in ('HEAD', 'BODY')]
        texts = [', '.join(map(str, values)) or '无' for values in lists]
        if confidence['audit'][condition]['false_origin_candidates'] == 0:
            texts = ['AUC不适用；通过'+str(len(values))+'/4分数档' for values in lists]
        requirement_lines.append('|'+condition+'|'+'|'.join(texts)+f"|{confidence['audit'][condition]['mean_false_per_frame']:.3f}|")
    (root/'candidate_requirements.json').write_text(json.dumps(requirements, indent=2, allow_nan=False), encoding='utf-8')
    (root/'CANDIDATE_REQUIREMENTS.md').write_text('\n'.join(requirement_lines)+'\n', encoding='utf-8')
    appendix = ['# 全条件附表：特权软先验，非相机结果', '', '及时/报晚/没报均为小目标近事件；假警分母为组内空序列×查询盒。保留率分母固定为原硬G1收益。', '',
                '|预算|区域|策略|及时|报晚|没报|实际假警|保留率|δ / 全局阈值|假警比G0高>2pp|',
                '|---|---|---|---|---|---|---|---|---|---|']
    for key, row in results.items():
        group, policy, budget = key.split('/')
        tiny = row['eval']['tiny']
        appendix.append(f"|{budget}|{group}|{policy}|{count(tiny['timely'])}|{count(tiny['late'])}|{count(tiny['never'])}|{count(row['eval']['all']['false_alert'])}|{pct(row['retention_of_fixed_hard_G1_gain'])}|{row['delta']} / {row['threshold']}|{'是' if row['eval_false_alert_exceeds_G0_by_2pp'] else '否'}|")
    (root/'APPENDIX.md').write_text('\n'.join(appendix)+'\n', encoding='utf-8')
    selected = [('G0', '无引导'), ('G1', '原硬门控理想候选'), ('G1s__COMBO_MODERATE', '中等误差·软并集')]
    selected += [(f'G1c__COMBO_MODERATE__A{a}', f'中等误差·AUC {a/100:g}') for a in AUCS]
    selected += [('G1s__COMBO_SEVERE', '严重误差·软并集'), ('G1c__COMBO_SEVERE__A90', '严重误差·AUC .9')]
    gates = data['moderate_combo_gate_at_10pct']
    group_text = '；'.join(g+('达到' if gates[g]['passes_any_auc_at_most_090'] else '未达到')+'“AUC≤.9保留≥50%”' for g in ('HEAD', 'BODY'))
    lines = ['# 软先验与候选置信度：快速诊断（2026-09-27）', '',
        '**结论：中等组合误差下，'+group_text+'。**', '',
        '**使用了特权信息，不是相机结果，也不是可部署方案。** 同一修复v2已消费Development，32个calib选阈、63个audit评估；未重新生成传感数据，未运行视觉模型。', '',
        '|10% calib假警预算|HEAD及时（/432） / 保留率 / 实际FA|BODY及时（/404） / 保留率 / 实际FA|', '|---|---|---|']
    for policy, label in selected:
        cells = []
        for group in ('HEAD', 'BODY'):
            row = results[f'{group}/{policy}/0.10']
            tiny, fa = row['eval']['tiny']['timely'], row['eval']['all']['false_alert']
            mark = '†' if row['eval_false_alert_exceeds_G0_by_2pp'] else ''
            cells.append(f"{pct(tiny['rate'])} ({tiny['numerator']}) / {pct(row['retention_of_fixed_hard_G1_gain'])} / {count(fa)}{mark}")
        lines.append('|'+label+'|'+'|'.join(cells)+'|')
    lines += ['', '**判读与下一步。** 软并集缓解严重误差下的退步，但中等组合即使AUC .99也未保留一半收益。单独增加1个均匀假候选时，两组在AUC .9均过线；3个时需测试档AUC .95，不能推广到组合误差。建议停止本轮调参，下一步核对真实图像的小目标覆盖、错误角度候选及持续性；本线性置信度映射的失败不能推出“相机必须主检”。RGB执行与论文主线仍待决定。†表示实际FA比G0高超过2pp，按未舍入值判断。', '',
        '**已有视觉证据。** 历史COCO100文档给YOLO11n/YOLO26n为0.41/0.36误检框/图（文档重建41/100、36/100），precision .859/.872、recall .299/.294；未定位原始逐图产物。其类别/IoU口径不能验收CNH角度候选；MZ129和受控A为渲染场景提醒指标。详见[只读审计](visual_evidence_audit.md)。', '',
        '**方法与范围。** 两分支共同满足calib总FA预算；δ=0精确退回G0，仅保证calib全尺寸及时数目标不劣，不保证audit或小目标不退步。δ∈{0,.25,.5,.75,1,1.5,2,3}，全局阈值1–12步长.025及关闭；按及时数、较少假警、较小δ、较高阈值依次选定。置信度c=Φ(z)，假候选z∼N(0,1)、真候选z∼N(√2Φ⁻¹(AUC),1)，局部门槛=tg−δc；AUC不唯一决定PR曲线。候选真伪按来源而非查询标签，误差及分数逐帧独立、未模拟持续高分误报；每个条件只有一个固定随机实现。v2原失败与143排除保留；不能追认独立正式结果，v4偏差决定未改。', '',
        '[候选需求档位](CANDIDATE_REQUIREMENTS.md) · [全条件及5/20%附表](APPENDIX.md) · [分析计数](analysis.json) · [经验候选AUC及分母](confidence_distribution.json) · [曲线](soft_prior.png) · [事前选择规则](DECISIONS.md)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if tracked_report is not None:
        import re
        import os
        prefix = Path(os.path.relpath(root, tracked_report.parent)).as_posix()
        text = re.sub(r'\]\(([^):]+)\)', lambda m: ']('+prefix+'/'+m.group(1)+')', '\n'.join(lines)+'\n')
        tracked_report.write_text(text, encoding='utf-8')
    return confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--tracked-report', type=Path)
    args = parser.parse_args()
    report(args.root, args.tracked_report)
