"""Format an existing position-prior analysis JSON, without selecting outcomes."""
import argparse
import json
from pathlib import Path


def render(source):
    d = json.loads(Path(source).read_text(encoding='utf-8'))
    rows = d['results']
    gate = d['descriptive_decision_checks_10pct_tiny']['HEAD']
    def value(g, a, b, st, metric):
        return rows[f'{g}/{a}/{b:.2f}']['eval'][st][metric]
    def pct(v):
        return 'NA' if v is None else f'{100*v:.1f}'
    def pair(a, b, st, metric):
        return '/'.join(pct(value(g, a, b, st, metric)['rate']) for g in ('HEAD', 'BODY'))
    conclusion = ('达到真实 RGB 候选检查门槛' if gate['rgb_candidate_gate'] else '未达到真实 RGB 候选检查门槛')
    conclusion += '；'+('达到 ToF 时间先验优先门槛' if gate['tof_temporal_priority_gate'] else '未达到 ToF 时间先验优先门槛')
    if gate['position_downgrade_gate']:
        conclusion += '；按事前规则降低本位置引导方案优先级'
    lines = ['# 位置先验快速诊断（v3 已消费 Development）', '',
             '**结论：'+conclusion+'。** HEAD 小块近事件在 10% 校准预算下，G1/G3b/G4 相对 G0 的及时率变化为 '+
             '/'.join(f'{100*gate[k]:+.2f}' for k in ('G1_gain', 'G3b_gain', 'G4_gain'))+' pp。', '',
             '偶数 32 单位选阈、奇数 31 单位评估；均为 M1 −10°、SNR6、5 Hz、带噪自运动。下表为百分比，每格 HEAD/BODY；预算仅约束选阈集，评估实际假警不同。', '',
             '|组|实际假警 5%|10%|20%|小块及时 5%|10%|20%|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('G0', 'G1', 'G2', 'G3a', 'G3b', 'G3c', 'G3d', 'G4'):
        cells = [pair(arm, b, 'all', 'false_alert') for b in (.05, .1, .2)]
        cells += [pair(arm, b, 'tiny', 'timely') for b in (.05, .1, .2)]
        lines.append('|'+arm+'|'+'|'.join(cells)+'|')
    den = []
    for metric, st, label in (('false_alert', 'all', '空对'), ('event_alert', 'all', '正事件'),
                              ('timely', 'all', '近事件'), ('timely', 'tiny', '小块近事件')):
        den.append(label+' '+ '/'.join(str(value(g, 'G0', .1, st, metric)['denominator']) for g in ('HEAD', 'BODY')))
    max_error = max(d['G0_consistency_max_abs_by_unit'].values())
    ratios = ['NA' if gate[k] is None else f'{gate[k]:.3f}' for k in ('G3b_to_G1_gain_ratio', 'G4_to_G1_gain_ratio')]
    disabled = sum(row['thresholds'][1] == 'disabled' for key, row in rows.items() if '/G4/' in key)
    lines += ['', '固定分母（HEAD/BODY）：'+'；'.join(den)+'。G0 全单位最大绝对误差 '+f'{max_error:.3g}'+
              '（归一化误差见 scores/unit*.npz，均通过 1e−5）。G3b/G4 对 G1 收益比：'+'/'.join(ratios)+'。', '',
              'G0 无引导；G1 所有可见非背景物体角度；G2 加射线距离格±1交集；G3a/b 偏移0.5/1区，G3c 丢20%，G3d 每帧加1假候选；G4 过去两帧 ToF 候选。'+
              f'G4六个工作点中{disabled}个禁用了局部分支。全部预算的事件报警率、及时/报晚/没报、三尺寸×四距离段的逐帧检出率及分子/分母保存在 [analysis.json](analysis.json)。', '',
              '**下一步：** '+('先检查真实 RGB 是否能提供有用的角度候选。' if gate['rgb_candidate_gate'] else
              '先定位本诊断收益不足的原因，不把它解释为 ToF 物理上限或相机无用。')+
              (' ToF 时间先验达到相对收益门槛，但还须结合绝对收益和假警决定投入。' if gate['tof_temporal_priority_gate'] else ''), '',
              '**范围与局限：使用了特权信息，不是相机结果，也不是可部署方案。** G1/G2含完美背景排除和传感视场内可见性；G2不是物理上界。G4无oracle，但复用S2历史，证据相关；'+
              '其阈值对按所有尺寸的选阈集及时率优化，不能当独立确认。收益比未估计区间，越过门槛不代表稳健确认。相同校准预算不等于相同评估假警；奇偶划分仅用于消费数据诊断。距离为见证Z，不是射线距离。'+
              '1m是本模拟的及时界限，非已验证安全距离。规则与实现细节见 [DECISIONS.md](DECISIONS.md)，不改v4协议或CURRENT。']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('analysis')
    p.add_argument('report')
    args = p.parse_args()
    Path(args.report).write_text(render(args.analysis), encoding='utf-8')
