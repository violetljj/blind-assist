"""Render v4 report from completed immutable summaries; never read eval labels/models."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

METHODS = ('D_prime', 'rgb', 'tof', 'or', 'and')
NAMES = dict(D_prime='D′', rgb='RGB-only', tof='ToF-only', or_='OR', and_='AND')
NAMES['or'], NAMES['and'] = NAMES.pop('or_'), NAMES.pop('and_')
ARMS = ('native_perturbed', 'faro_rho015_ambient1', 'faro_rho060_ambient1',
        'faro_rho030_ambient3', 'faro_rho030_ambient1', 'faro_rho030_ambient10')
BANDS = ('0.3–0.8m', '0.8–1.5m', '1.5–3m')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    with Path(path).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def num(value, digits=4):
    return 'N/E' if value is None or value == '' else f'{float(value):.{digits}f}'


def pct(value):
    return 'N/E' if value is None or value == '' else f'{float(value) * 100:.3f}%'


def fraction(r, numerator='F', denominator='FREE'):
    return f"{r[numerator]}/{r[denominator]} ({pct(r.get(numerator + '_rate'))})"


def table(headers, data):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
        '| ' + ' | '.join(map(str, row)) + ' |' for row in data]


def link(root, relative, text=None):
    return f'[{text or relative}]({(root / relative).as_posix()})'


def find(data, **keys):
    matching = [r for r in data if all(str(r.get(k)) == str(v) for k, v in keys.items())]
    if len(matching) != 1:
        raise ValueError(('Non-unique report cell', keys, len(matching)))
    return matching[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--receipts', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--resource-json', required=True, type=Path)
    args = parser.parse_args(); started = time.monotonic(); root = args.root.resolve()
    summary = load(root / 'summary/summary.json'); audit = load(args.audit)
    if audit['status'] != 'PASS' or audit.get('confirmation_status') != summary['status']:
        raise ValueError('Completed independent eval audit required')
    if audit['summary_sha256'] != sha(root / 'summary/summary.json'):
        raise ValueError('Audited summary bytes changed')
    protocol = load(root / 'PLAN.json'); binding = load(root / 'protocol_commit.json')
    cal = load(root / 'cal/cal_seal.json'); oof = load(root / 'oof/oof_seal.json')
    tie_count = sum(len(load(c['all_ties_path'])['rows']) for c in cal['cells'])
    resource = load(args.resource_json)
    drift = rows(root / 'summary/drift.csv'); distributions = rows(root / 'cal/oof_vs_frozen_distribution.csv')
    manifest = load(root / 'dataset_manifest.json')
    metrics, deployed, paired = summary['pAUC'], summary['deployment'], summary['paired']
    boot = rows(root / 'summary/bootstrap.csv')
    out = [f"# 融合确认 v4：{summary['status']}", '', summary['conclusion'] + '。', '',
           '主判据只覆盖半循环 ToF 臂中带 [0,5%] FREE 区间的排序。本轮没有条件仲裁器；v3 预注册 FAIL 文本与数字保留。', '',
           '## 判据来源、冻结和指标', '',
           f"该判据是在看过 v3 事后同 FREE 分析 `7dc9bbba` 后选定的；协议 `{binding['commit'][:8]}` 在新数据构建、交叉拟合和 eval 打开前提交并推送。它是全新 visit 上的预注册确认，不是 v3 数据上的事前判据，也不改写 `b9f227e5` 的 FAIL。", '',
           'D′ 使用 v3 三个 HGB 权重的字节原件（六臂等权训练、16 个传感器特征、删除显式位置/带别输入、原最低证据门控）。RGB-only、ToF-only、OR=max、AND=min 的原始读出不变；没有冻结模型重训或概率映射。', '']
    out += table(['seed', '冻结 v3 权重 SHA256'], [[e['seed'], '`' + e['sha256'] + '`'] for e in cal['model_entries']])
    out += ['', 'W=预测 query 阳性且 POS，F=阳性且严格 sampled FREE，U=阳性且 UNKNOWN。UNKNOWN 不作为负例；分母分别为 POS、FREE、UNKNOWN。', '',
            '对全部有限分数 ties 加全拒绝点枚举，每个精确可达 F 保留最大 W；不可拆 ties、不插值。预算阶梯 R(x) 在下一个 F/FREE 点前保持当前最大 W/POS，从 F=0 开始，积分至 5%，再除以 0.05。该归一化 pAUC 的范围为 [0,1]。它不依赖部署阈值，曲线使用 eval 标签衡量排序，曲线上的切点不能作为部署选点。', '',
            '## 主指标：半循环中带', '']
    out += table(['方法', '归一化 pAUC', 'POS / FREE'], [[NAMES[m], num(r['normalized_pAUC']), f"{r['POS']} / {r['FREE']}"] for m in METHODS for r in [find(metrics, arm=ARMS[0], band=1, method=m)]])
    out += ['', '按 16 个 eval visit 成对聚类重抽样 2000 次（seed=20261011）；每次重新计算完整 ties 阶梯及面积。下表差值以百分点表示，区间为 2.5/97.5 百分位。两个区间下界都严格 >0 且 2000 次均有效才通过。', '']
    out += table(['比较', '差值 pp', '95% 区间 pp', '有效重抽样', '下界 >0'], [[f"D′−{NAMES[r['comparator']]}", num(r['delta_pAUC'] * 100, 3), f"[{num(None if r['lower'] is None else r['lower'] * 100, 3)}, {num(None if r['upper'] is None else r['upper'] * 100, 3)}]", f"{r['valid_replicates']}/2000", r['pass_lower_gt_zero']] for r in summary['primary']])
    out += ['', f"判定：**{summary['status']}**。{summary['conclusion']}。", '',
            '## 池化 cal 与部署点', '',
            '原 30 个已消费 visit 按固定种子分成 5 折，每折 24 个训练 / 6 个留出；使用与 v3 相同的 HGB 配方、三个 seed 和六臂等权，得到每个 visit 的折外分数。15 个临时模型只用于生成 cal 分数。新 4 个 cal visit 使用冻结 v3 D′。基线直接使用原分数。混合六臂的 34-visit 池按方法/带选 15 个共同切点，目标近 2%、中 5%、远 10%。最大 W、再最小 F、再最高阈值；零 POS/FREE 不可校准且全拒绝。', '',
            f'阈值、全部 {tie_count} 个候选 ties、OOF 划分/模型/cache、冻结权重、代码和新 cal cache 在唯一 OPEN 前封存；eval 始终使用 v3 冻结 D′。池化目标约束只施加于六臂混合池，单臂 cal FREE 率不必满足同一目标，FARO 基准过门会改变混合权重。', '']
    out += table(['带', '方法', '封存切点', 'cal W/POS', 'cal F/FREE'], [[BANDS[c['band']], NAMES[c['method']], c.get('threshold') if c.get('threshold_kind') != 'positive_infinity' else '+inf', fraction(r, 'W', 'POS'), fraction(r)] for c in cal['cells'] for r in [find(drift, arm='POOLED_SIX_ARMS', band=c['band'], method=c['method'], role='v4_pooled_cal')]])
    out += ['', '半循环 eval 部署点：', '']
    out += table(['带', '方法', 'W/POS', 'F/FREE', 'U/UNKNOWN', 'D′ 双向占优该基线'], [[BANDS[b], NAMES[m], fraction(r, 'W', 'POS'), fraction(r), f"{r['U']}/{r['UNKNOWN']}", '—' if m == 'D_prime' else find(paired, arm=ARMS[0], band=b, comparator=m)['deployment_bidirectional_dominance']] for b in range(3) for m in METHODS for r in [find(deployed, arm=ARMS[0], band=b, method=m)]])
    out += ['', '双向占优定义为同一 query 集上 W_D′≥W_基线且 F_D′≤F_基线，作为次指标描述，不参与主判据。', '',
            '## cal→eval 漂移与原六 visit 对照', '',
            '以下保留原 v3 cal6→eval12，并将原 v3 六 visit cal 切点作用于同一 v4 eval16，避免把数据变化与切点变化混在一起。v4 池化 cal 率是六臂混合选点池；半循环 cal 率另列。', '']
    out += table(['带', '方法', 'v4 池 cal F率', 'v4 半循环 cal F率', 'v4 eval F率', '原 v3 cal→eval F率', '原 v3 切点→同 v4 eval F率'], [[BANDS[b], NAMES[m], pct(find(drift, arm='POOLED_SIX_ARMS', band=b, method=m, role='v4_pooled_cal')['F_rate']), pct(find(drift, arm=ARMS[0], band=b, method=m, role='v4_pooled_cal')['F_rate']), pct(find(drift, arm=ARMS[0], band=b, method=m, role='v4_eval16')['F_rate']), pct(find(drift, arm=ARMS[0], band=b, method=m, role='v3_cal6')['F_rate']) + '→' + pct(find(drift, arm=ARMS[0], band=b, method=m, role='v3_eval12')['F_rate']), pct(find(drift, arm=ARMS[0], band=b, method=m, role='v3_cuts_on_v4_eval16')['F_rate'])] for b in range(3) for m in METHODS])
    out += ['', '同一 v4 半循环 eval 的切点对照（原 v3 六 visit cal → v4 池化 cal）：', '']
    out += table(['带', '方法', '原切点 W/POS', '池化切点 W/POS', '原切点 F/FREE', '池化切点 F/FREE'], [[BANDS[b], NAMES[m], fraction(old, 'W', 'POS'), fraction(new, 'W', 'POS'), fraction(old), fraction(new)] for b in range(3) for m in METHODS for old in [find(drift, arm=ARMS[0], band=b, method=m, role='v3_cuts_on_v4_eval16')] for new in [find(drift, arm=ARMS[0], band=b, method=m, role='v4_eval16')]])
    out += ['', f"所有六臂、旧30 OOF / 新4冻结 / 池cal / eval / 原v3对照的 W/F 分母见 {link(root, 'summary/drift.csv')}；逐 visit 见 {link(root, 'summary/drift_visit.csv')}。池化校准只能改变工作点估计，不能保证新 visit 上的 FREE 率；观察到的变化不归因为单一物理机制。", '',
            '## OOF 与冻结 D′ 的分布差异', '',
            '下面在同一旧 30-visit 行上对比 OOF 与冻结 v3 分数。冻结对比为训练内分数，OOF 为折外分数，且临时模型使用 24 visit 而冻结模型使用 30 visit；该差异是阈值迁移的局限，不是 eval 性能差。禁用分数 −inf 不进入有限分数分位数。', '']
    drows = []
    for arm in ('POOLED_SIX_ARMS', ARMS[0]):
        for b in range(3):
            for label in ('ALL', '1', '0', '-1'):
                a = find(distributions, arm=arm, band=b, label=label, score_source='OOF')
                z = find(distributions, arm=arm, band=b, label=label, score_source='FROZEN_IN_SAMPLE')
                q = lambda r: '/'.join(num(r[k]) for k in ('q05', 'q50', 'q95'))
                drows.append([arm, BANDS[b], dict(ALL='全部', **{'1':'POS', '0':'FREE', '-1':'UNKNOWN'})[label], f"{a['finite_rows']}/{a['rows']}", q(a), q(z), num(a['paired_OOF_minus_frozen_mean'])])
    out += table(['臂', '带', '标签', '有限/总行', 'OOF p05/p50/p95', '冻结训练内 p05/p50/p95', '成对 OOF−冻结均值'], drows)
    out += ['', f"全部臂/标签的精确 p00/p05/p25/p50/p75/p95/p100、禁用数见 {link(root, 'cal/oof_vs_frozen_distribution.csv')}；新4冻结 cal 分布见 {link(root, 'cal/new4_frozen_distribution.csv')}。不对 OOF 与冻结分数进行概率重映射。", '',
            '## 近/远带及六个 ToF 臂：描述结果', '',
            '除半循环中带外均不进入判据。FARO 基准仅包括固定 K0 联合过门帧；缺测几何保持 UNKNOWN，不由 native 填充。ambient×10 与反射/环境退化臂沿用固定生成配方。', '']
    out += table(['臂', '带', 'POS/FREE', 'D′ pAUC', 'RGB pAUC', 'OR pAUC', 'D′−RGB 95% pp', 'D′−OR 95% pp'], [[arm, BANDS[b], f"{d['POS']}/{d['FREE']}", num(d['normalized_pAUC']), num(find(metrics, arm=arm, band=b, method='rgb')['normalized_pAUC']), num(find(metrics, arm=arm, band=b, method='or')['normalized_pAUC']), *[f"[{num(None if r['lower']=='' else float(r['lower'])*100, 3)}, {num(None if r['upper']=='' else float(r['upper'])*100, 3)}] ({r['valid_replicates']}/2000)" for m in ('rgb', 'or') for r in [find(boot, arm=arm, band=b, comparator=m)]]] for arm in ARMS for b in range(3) for d in [find(metrics, arm=arm, band=b, method='D_prime')]])
    out += ['', '全部方法部署点（W/F/U 与各分母）：', '']
    out += table(['臂', '带', '方法', 'W/POS', 'F/FREE', 'U/UNKNOWN'], [[r['arm'], BANDS[r['band']], NAMES[r['method']], fraction(r, 'W', 'POS'), fraction(r), f"{r['U']}/{r['UNKNOWN']}"] for r in deployed])
    out += ['', f"全部方法 pAUC：{link(root, 'summary/pauc.csv')}；曲线：{link(root, 'summary/curves.csv')}；bootstrap：{link(root, 'summary/bootstrap.csv')}；2000 次成对重抽样缓存：{link(root, 'summary/bootstrap_replicates.npz')}；逐 visit：{link(root, 'summary/visit_pauc.csv')}；部署配对救/损和 FREE 增/减：{link(root, 'summary/deployment_paired.csv')}。", '',
            f"![半循环、FARO过门、ambient×10低FREE阶梯曲线]({(root / 'summary/low_free_curves.png').as_posix()})", '',
            '## 新 visit、核验、资源及证据边界', '',
            f"新 cohort {len(manifest['entries'])} 个 visit，每个 32 帧；官方顺序前4 cal、后16 eval，两个独立16帧5Hz窗口和 v3 相同 reference 门槛。历史消费 visit 与失败候选整 visit 排除，不调窗口迁就模型。接受/跳过证据：{link(root, 'candidate_gate_records.json')}；完整 manifest：{link(root, 'dataset_manifest.json')}。", '',
            f"独立核验 **{audit['status']}**；核验细项和实际数量保存在 {link(root, str(args.audit.resolve().relative_to(root)), '独立 eval 审计')}。审计独立重算 ties 阶梯、pAUC、2000 visit bootstrap、部署点、全部漂移和 OOF/封存身份；报告只读已审计汇总，不再次读取 eval reference 或重新加载模型。", '',
            f"资源：{resource.get('description', json.dumps(resource, ensure_ascii=False))}。预算 CPU≤2400 command-wall s、GPU≤600s、下载≤4,000,000,000B；统计与报告预算包含于总账。", '',
            '该证据是公开真实 RGB＋半合成 ToF 的 Development 新 visit 确认。半循环 ToF 不提供独立几何真值；FARO 是有限下载前缀和固定过门子集。reference 门槛富集、名义共位/45°、静态表面与轨迹误差保留。16 个 visit 的区间不保证其他场景或域迁移；OOF→冻结分数分布差会影响部署点。本轮不访问保护480/test，不证明实机、整身体清空或导航安全，不升级 App/M3/5格读出。', '',
            f"复现脚本：`sync_fusion_confirm_v4_models.py`（crossfit→cal→唯一eval→cache summarize）；报告脚本 `{Path(__file__).name}`；协议、封存、表格与 hash 收据见 `{args.receipts.name}`。", '']
    args.report.write_text('\n'.join(out), encoding='utf8')
    files = [root / x for x in ('PLAN.json', 'protocol_commit.json', 'dataset_manifest.json',
        'candidate_gate_records.json', 'oof/oof_seal.json', 'cal/cal_seal.json',
        'eval/eval_open.json', 'eval/cache_seal.json', 'summary/summary.json')]
    files += [Path(c['all_ties_path']) for c in cal['cells']]
    files += list((root / 'summary').glob('*.csv')) + [root / 'summary/low_free_curves.png',
        root / 'summary/bootstrap_replicates.npz', root / 'cal/oof_vs_frozen_distribution.csv',
        root / 'cal/new4_frozen_distribution.csv', args.audit, args.resource_json, args.report, Path(__file__)]
    receipt = dict(task='SYNC_FUSION_CONFIRM_V4_DEV_20261011', status=summary['status'],
        conclusion=summary['conclusion'], primary=summary['primary'], protocol_commit=binding,
        frozen_v3_model_entries=cal['model_entries'], cal_cells=cal['cells'],
        oof_folds=oof['folds'], model_cal_eval_distinction='OOF15calibrators only; byte-frozen v3 eval; no remapping',
        artifact_root=str(root), resources=resource, independent_audit=audit,
        files=[dict(path=str(f.resolve()), bytes=f.stat().st_size, sha256=sha(f)) for f in files],
        report_CPU_command_wall_s=time.monotonic()-started, report_GPU_s=0,
        report_eval_reference_reads=0, report_model_reads=0, protected_480_test_reads=0)
    args.receipts.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(dict(report=str(args.report), receipt=str(args.receipts),
                         status=summary['status'], CPU_command_wall_s=time.monotonic()-started), ensure_ascii=False))


if __name__ == '__main__':
    main()
