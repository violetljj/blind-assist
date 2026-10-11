"""Publish sealed no-bin0 DR results; this module never trains or selects a cut."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-no-bin0-dr-retrain-dev-20261011'
REPORT = Path(__file__).with_name('CNH_NO_BIN0_DR_RETRAIN_DEV_20261011.md')
ARMS = ('original_old5', 'old_default_S', 'no_bin0_M3', 'no_bin0_old5', 'no_bin0_S', 'no_bin01_M3', 'no_bin01_old5', 'no_bin01_S')
LABELS = dict(original_old5='旧原5格', old_default_S='旧默认S', no_bin0_M3='主臂M3', no_bin0_old5='主臂原5格', no_bin0_S='主臂S', no_bin01_M3='敏感性M3', no_bin01_old5='敏感性原5格', no_bin01_S='敏感性S')
CONDITIONS = ('1', '2', '3', '4', '4b', '5', '6', '7', '8', '9')
SCENES = dict(zip(CONDITIONS, ('阳台UNKNOWN', '近墙', '远墙', '空通道边界', '暗空通道边界', '近杆含筒', '远杆含筒', '外侧杆', '头横杆', '空通道边界')))


def read(out, name):
    return json.loads((out/name).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(path, label=None):
    path = Path(path)
    return f'[{label or path.name}]({path.as_posix()})'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |', *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def index_real(receipt):
    return {(r['arm'], r['path'], str(r['condition']), r['height'], r['model']): r for r in receipt['tables']}


def cell(row):
    n = row['ready_frames']
    return f"{row['strong']}/{n} ({100*row['strong_rate']:.1f}%)；{row['light']}/{n} ({100*row['light_rate']:.1f}%)"


def summarize_real(index, arm='main', path='v2'):
    empty = [index[(arm, path, 'BIN0_BACKGROUND_ONLY', height, model)] for height in ('HEAD', 'BODY') for model in ('M3', 'old5', 'S')]
    bin0_pass = all(r['strong'] == 0 and r['light'] == 0 for r in empty)
    checks = {}
    for condition in ('2', '3', '5', '6', '8'):
        heights = ('HEAD',) if condition == '8' else ('HEAD', 'BODY')
        rows = [index[(arm, path, condition, height, 'S')] for height in heights]
        checks[condition] = any(r['strong']+r['light'] > 0 for r in rows)
    return dict(bin0_pass=bin0_pass, reminder_present=checks, walls_rods_pass=all(checks.values()))


def figures(out, stats, real):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    arms = ('original_old5', 'old_default_S', 'no_bin0_S', 'no_bin01_S')
    labels = ('Original old5', 'Old default S', 'Drop bin0 + DR S', 'Drop bins0/1 + DR S')
    x = np.arange(len(arms))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
    for i, height in enumerate(('HEAD', 'BODY')):
        vals = [100*stats[a][height]['timely']/stats[a][height]['denominator'] for a in arms]
        axes[0].bar(x + (i-.5)*.35, vals, width=.35, label=height)
        for pos, val, arm in zip(x+(i-.5)*.35, vals, arms):
            axes[0].text(pos, val+1, str(stats[arm][height]['timely']), ha='center', fontsize=8)
    axes[0].set(ylabel='Timely contact events (%)', ylim=(0, 108), title='Same fresh nominal hold')
    axes[0].legend(fontsize=8)
    for panel, key, title, ylabel in ((1, 'far_clear', 'Far + clear clips with any notice', 'Notified clips (%)'), (2, 'weighted_cost', 'Joint notification cost (v2)', 'Weighted cost units')):
        vals = [100*stats[a][key]['rate'] if key == 'far_clear' else stats[a][key] for a in arms]
        axes[panel].bar(x, vals, color=('#777777', '#bb6666', '#4488bb', '#77aa77'))
        axes[panel].set(title=title, ylabel=ylabel)
        for pos, val in zip(x, vals): axes[panel].text(pos, val, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=22, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.savefig(out/'simulation_comparison.png', dpi=170)
    plt.close(fig)
    index = index_real(real)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout='constrained')
    for pi, path in enumerate(('v2', 'bias_only')):
        for hi, height in enumerate(('HEAD', 'BODY')):
            ax = axes[hi, pi]
            rows = [[index[(arm, path, c, height, 'S')] for arm in ('old', 'main', 'drop01')] for c in CONDITIONS]
            values = np.array([[100*r['strong_rate'] for r in row] for row in rows])
            im = ax.imshow(values, vmin=0, vmax=100, cmap='YlOrRd', aspect='auto')
            for ci, row in enumerate(rows):
                for ai, r in enumerate(row):
                    ax.text(ai, ci, f"{100*r['strong_rate']:.0f} / {100*r['light_rate']:.0f}", ha='center', va='center', fontsize=8, color='white' if r['strong_rate'] > .65 else 'black')
            ax.set(title=f'{height}: {path}', yticks=np.arange(len(CONDITIONS)), yticklabels=CONDITIONS, xticks=np.arange(3), xticklabels=('Old S', 'Drop bin0 S', 'Drop bins0/1 S'))
            ax.tick_params(axis='x', labelsize=8)
    fig.colorbar(im, ax=axes, label='Strong alarm query-frame rate (%)', shrink=.7)
    fig.suptitle('Consumed real pilot: labels = strong % / exclusive light %; clear 4/4b/9 are boundary scenes', fontsize=10)
    fig.savefig(out/'real_alarm_comparison.png', dpi=170)
    plt.close(fig)


def run(out=OUT, report=REPORT):
    began = time.monotonic()
    seal = read(out, 'sealed_candidates.json')
    stats = read(out, 'simulation_evaluation.json')
    real = read(out, 'real_receipt.json')
    manifests = {a: read(out, a+'_real_manifest.json') for a in ('main', 'drop01')}
    for arm in manifests:
        assert sha(out/(arm+'_real_manifest.json')) == seal['real_manifest_hashes'][arm]
    assert real['model_seal_sha256'] == sha(out/'sealed_candidates.json')
    audits = {name: read(out, name) for name in ('independent_audit_real.json', 'independent_forward_audit.json')}
    assert all(row['status'] == 'PASS' for row in audits.values())
    index = index_real(real)
    interpretations = {path: summarize_real(index, path=path) for path in ('v2', 'bias_only')}
    accept = stats['acceptance']
    cal_valid = seal['cuts']['no_bin0']['s_calibration']['status'] == 'SEALED'
    passed = cal_valid and accept['timely_pass'] and accept['far_clear_pass'] and interpretations['v2']['bin0_pass'] and interpretations['v2']['walls_rods_pass']
    verdict = '建议新骨干+S替换为默认候选（待独立实机确认）' if passed else '本次不建议替换默认候选，保留旧S候选'
    failures = []
    if not cal_valid: failures.append('主S强档已超出池化校准预算，无有效S工作点')
    if not accept['timely_pass']: failures.append('模拟及时净增保留不足80%')
    if not accept['far_clear_pass']: failures.append('far+clear有通知clip增幅超过1pp')
    if not interpretations['v2']['bin0_pass']: failures.append('bin0背景空输入仍有提醒')
    if not interpretations['v2']['walls_rods_pass']: failures.append('v2路径有墙/杆条件完全失去提醒')
    figs = figures(out, stats, real)
    rows = []
    for arm in ARMS:
        s = stats[arm]
        rows.append([LABELS[arm], *[f"{s[h]['timely']}/{s[h]['denominator']}；+{s[h]['rescue']}/−{s[h]['loss']}；净{s[h]['net']:+d}" for h in ('HEAD','BODY')], f"{s['far_clear']['notified_clips']}/{s['far_clear']['denominator']} ({100*s['far_clear']['rate']:.3f}%)", f"{s['weighted_cost']:.2f}"])
    text = [f'# 去bin0与仪器域随机化骨干重训：开发评价\n\n2026-10-11；EXPLORE。\n\n**{verdict}。**']
    if failures: text.append('失败项：'+'；'.join(failures)+'。')
    text.append(f"共同参照为同一新hold上的旧原5格：旧默认S及时净增{accept['old_gain_vs_common_old5']:+d}，主臂S净增{accept['new_gain_vs_common_old5']:+d}；预定80%收益保留门槛{'通过' if accept['timely_pass'] else '未通过'}。主臂相对旧S的far+clear有通知clip率变化{accept['far_clear_increment_pp']:+.3f}pp，≤+1pp门槛{'通过' if accept['far_clear_pass'] else '未通过'}。这些是预定点估计工程门槛，描述区间不构成统计非劣性确认。")
    text.append(f"设计由已消费宿舍pilot诊断驱动；真实十段1507帧、满8帧历史1362帧只作开发观察，不能把本次新模拟留出解释成独立实机确认。协议在训练/新hold前提交：73a28096→167b3c69；{link(Path(__file__).with_name('CNH_NO_BIN0_DR_RETRAIN_PLAN_20261011.md'),'PLAN')}。")
    text.append('## 输入、仪器随机化与训练\n\n全部64 zone去bin0，保留16-bin张量形状并将该bin置零。mask在CVR、ordinary、LOCAL、S当前峰特征之前统一应用；敏感性臂另去任务范围内bin1。其余训练合同继承。每个训练物理场景×噪声副本独立抽取仪器参数，历史内固定、epoch复用，不按真实回放重采样或选点。')
    drrows = [('有效增益','U[0.42,0.73]'),('Gaussian σ','U[70,115] mm'),('正向拖尾比例/尺度','U[0,5%] / 固定0.50 m'),('range零偏','U[−100,+100] mm'),('noise a','U[0,0.25357091365]'),('noise b','U[0.08671592722,0.09326653791]'),('noise c','U[0.58070959930,0.68960300094]'),('ambient','名义4 × U[1,2.6]')]
    text.append(table(('参数','预定范围'),drrows))
    text.append('噪声代理为Var=a·positive_signal+b·ambient+c；墙联合校准R²=0.182496，暗档单独约0.0047。系数来自decoded单位，将其桥接到模拟器能量单位是本次未经独立确认的假设。先重渲染未卷积表面能量、128原始bin脉冲积分后聚合，避免重复展宽。真实pilot不进入训练、随机化范围或切点选择；墙2背景不是完整空间真值，不插入训练。')
    text.append('训练前公式核对发现：忽略表面与bias时，该噪声代理经过原√(16·ambient+bias)归一化，背景标准差约0.09–0.13，而原名义模拟约1。墙面有效gain提供条件均值对应，但未识别电子噪声单位；本轮不扩大已写定的系数范围。若模拟性能下降，不能把联合设计的损失全部归因于去bin0；噪声口径或名义噪声覆盖对照应另立协议，不用本轮hold调参。')
    recipe=[]
    for key in ('main','drop01'):
        m=manifests[key]
        recipe.append([key, len(m['m3_paths']), len({p['sha256'] for p in m['ordinary_paths']}),len(m['s_paths']),','.join(map(str,m['drop_bins']))])
    text.append(table(('臂','M3独立权重数','ordinary独立权重数','S seed数（每seed HEAD/BODY）','去除bin'),recipe))
    text.append('主臂M3原20epoch、47,424训练行，ordinary原32epoch/7488step、59,904行；含现有窄通道与侧物族。M3 AdamW lr0.002/wd0.0001、cosine20；ordinary lr0.001/wd0.0001、masked weighted BCE原调度。S三seed使用task-cost原47维实现、HGB100iter/lr0.1/7leaves/depth3/minleaf50/l2=1，无earlystop。敏感性M3/ordinary各一模型，复制接口槽不增加独立seed；其结果不作等规模集成优劣判断。')
    if (out/'profile.json').exists():
        p=read(out,'profile.json')
        text.append(f"长训练前profile：主M3预计{p.get('estimate_main_m3_train_seconds',0):.1f}s、主ordinary{p.get('estimate_main_ordinary_train_seconds',0):.1f}s、双臂投影{p.get('estimate_two_arm_projection_seconds',0):.1f}s；共享渲染和特征，避免按seed重复生成。实际时长以最终receipt计费为准。")
    feature_receipts=[]
    for arm in ('no_bin0','no_bin01'):
        path=out/'features'/arm/'receipt.json'
        if path.exists():
            r=read(path.parent,path.name)
            feature_receipts.append([arm,r.get('status','COMPLETE'),r.get('ordinary_rows','—'),r.get('m3_rows','—'),str(r.get('paired_projection_max_abs','—')),str(r.get('resumed_ordinary',False))])
    if feature_receipts:
        text.append(table(('特征臂','状态','ordinary行','M3行','双臂投影vs独立投影最大绝对差','恢复既有ordinary缓存'),feature_receipts))
    text.append('过程失败与恢复保留：首次profile在PLAN.json缺少字段时中止（3.22s），首次冷渲染耗时不宜外推，另存原receipt并改用热态profile。双臂特征阶段完成ordinary缓存后，因paired_errors初始化遗漏发生UnboundLocalError（20.165s）；仅修复初始化，显式resume对两臂缓存首尾各4行逐值验证，将未完成M3缓存按时间戳保留，再续作投影。失败耗时计入预算，训练配方、数据、随机化范围及判读门槛未因此改变。')
    text.append('## 模拟留出与成本\n\n切点仅模拟cal。新M3/raised按whole-score ties匹配原clear成本，解析LOCAL保留原cut；S池化六组已消费负例，contact不选点，选择相对旧原5格far+clear有通知clip增量≤3pp的最低tau。全部权重/切点hash封存后才生成新hold，旧新候选用同一名义光子观测配对。')
    source=read(out,'fresh_hold_source_receipt.json')
    text.append(f"新hold {source['physical_scenes']}物理scene×K{source['replicas']}，四新背景family；HEAD/BODY接触event各{source['contact_events_per_height'][0]}/{source['contact_events_per_height'][1]}。对{source['inventory_file_count']}份允许库存metadata、{source['inventory_keys']}物理key预检，物理key、背景几何、family、目标尺寸和rho交集均为0。未读取保护480/test。")
    text.append(table(('候选','HEAD及时；救/损/净（vs旧原5格）','BODY及时；救/损/净','far+clear有通知clip','v2加权成本'),rows))
    text.append(f"![同一新模拟留出对照]({(out/'simulation_comparison.png').as_posix()})")
    paired=stats['new_vs_old_paired']
    text.append('直接对照旧默认S：'+ '；'.join(f"{h}救{paired[h]['rescue']}/损{paired[h]['loss']}，净{paired[h]['net']:+d}/{paired[h]['denominator']}" for h in ('HEAD','BODY'))+'。及时为f3–13，f14–15为晚；gap1逐query重放、同帧联合最高档。near-pass≤10cm轻档成本0.25/强档1，far/clear任何通知1；不是slot计数替代clip率。')
    boot=stats.get('new_vs_old_clustered_bootstrap',{})
    if 'intervals' in boot:
        rr=[]
        for kind,interval in boot['intervals'].items():
            rr.append([kind,*[f"[{interval[q]['lower']:+.3f},{interval[q]['upper']:+.3f}]" for q in ('HEAD_BODY_net_pp','far_clear_increment_pp')]])
        text.append(table(('配对95%描述区间','主S−旧S及时差（pp）','far+clear率差（pp）'),rr))
        text.append('K副本保留在物理scene内；仅四背景family，family区间有限，不作确认性推断。')
    group=[]
    for arm in ('old_default_S','no_bin0_S','no_bin01_S'):
        for name in ('dark_thin','sign_edge'):
            v=stats[arm]['groups'][name]
            group.append([LABELS[arm],name,*[f"{r['timely']}/{r['denominator']}（旧原5格{r['baseline']}）" for r in v]])
    text.append(table(('候选','子组','HEAD及时','BODY及时'),group))
    text.append('子组只描述，不据此选点；完整模拟表：'+link(out/'simulation_comparison.csv')+'、'+link(out/'simulation_groups.csv')+'。')
    text.append('## 真实已消费pilot开发回放\n\n两条路径都保持v2 denominator：(1)v2原numerator；(2)只减墙2拟合bin0 bias，不减场景表面。适配后统一mask，名义−10°、恒等历史输运，满8历史为分母。每个表格cell为“强计数/帧数（%）；纯轻计数/帧数（%）”；轻不包括强。')
    text.append(f"![已消费真实pilot两输入路径S报警]({(out/'real_alarm_comparison.png').as_posix()})")
    for path in ('v2','bias_only'):
        for model in ('M3','old5','S'):
            rr=[]
            for c in CONDITIONS:
                rr.append([c+' '+SCENES[c],*[cell(index[(a,path,c,h,model)]) for a in ('old','main','drop01') for h in ('HEAD','BODY')]])
            text.append(f'### {path} / {model}\n\n'+table(('条件','旧HEAD','旧BODY','主HEAD','主BODY','敏HEAD','敏BODY'),rr))
    text.append('4/4b/9的尽头约在3m query窗边缘，单列为边界场景；不是干净clear，不能据此报告实机误报率。7外侧杆按预期不应强；1阳台UNKNOWN。2/3墙、5/6含筒杆应提醒、8主要看HEAD；不额外引入最低报警率门槛。推荐条件中的“仍有提醒”只要求每段至少一个满窗响应（8取HEAD），属于响应存在性检查；若比例很低，不能据此声称目标提醒稳定保留。')
    text.append('## bin0背景空输入与机制核验\n\n墙2拟合背景、无表面、无噪声，只是合成空输入，不能代替真实clear录制。')
    rr=[]
    for path in ('v2','bias_only'):
        for model in ('M3','old5','S'):
            rr.append([path,model,*[cell(index[(a,path,'BIN0_BACKGROUND_ONLY',h,model)]) for a in ('old','main','drop01') for h in ('HEAD','BODY')]])
    text.append(table(('路径','模型','旧HEAD','旧BODY','主HEAD','主BODY','敏HEAD','敏BODY'),rr))
    text.append('所有新score和grade对被删除bin的任意大幅污染逐值一致（主臂bin0、敏感性bin0/1），覆盖M3、LOCAL、ordinary、S每seed与均值。这里检验信息确实删除，不证明真实全域迁移。')
    text.append('## 判读、核验与下一步\n\n'+verdict+'。'+('失败项：'+'；'.join(failures)+'。' if failures else '主模拟门槛、v2路径bin0空输入无提醒、墙/杆仍有提醒均满足预定工程条件。'))
    text.append('分别看v2和bias_only，bin0删除可以验证已知失配来源；空通道行为仍受3m窗边缘幕布和剩余bin分布影响。敏感性去bin1损失任务范围内信息，且backbone seed数不同，结果只描述。建议后续先独立录制尽头>4.5m楼道clear及墙/细杆，再决定默认落地；保留主臂参数范围，若剩余失配仍突出，再按bin/zone定位空间背景、固件处理、相关噪声与多径，而不通过pilot调cut。')
    text.append('独立核验从封存权重/切点、池化whole-tie选点、成本账、模拟逐event与真实逐query计数复算，并以fresh CUDA FP32窗重放两路径及空输入。'+ '；'.join(f"{name}: {r['status']}" for name,r in audits.items())+'。'+link(out/'independent_audit_real.json')+'、'+link(out/'independent_forward_audit.json')+'。')
    if (out/'budget_summary.json').exists():
        budget=read(out,'budget_summary.json')
        text.append(f"最终预算：GPU command-wall {budget['gpu_command_wall_seconds']:.3f}/5400 s；CPU科学计算/评价command-wall {budget['cpu_command_wall_seconds']:.3f}/1800 s。报告绘图另有轻量receipt，由主任务汇总。")
    else:
        gpu=sum(read(out/'gpu_receipts',p.name)['seconds'] for p in (out/'gpu_receipts').glob('*.json'))
        cpu=sum(read(out/'cpu_receipts',p.name)['seconds'] for p in (out/'cpu_receipts').glob('*.json'))
        text.append(f"阶段receipt累计GPU {gpu:.3f}/5400 s、CPU {cpu:.3f}/1800 s；审计/报告等最终整合计费以主任务预算summary为准。")
    text.append('本次不接入App、不读取保护480/test。模拟新hold为Development；宿舍pilot已消费且设计由其驱动，不能称为独立实机性能、安全效果或正式非劣性证明。旧冻结候选及失败结果保留。')
    text.append('## 权重、切点与复现\n\n'+link(out/'sealed_candidates.json')+'；'+link(out/'sealed_calibration.json')+'；'+link(out/'domain_randomization_draws.json')+'；'+link(out/'real_alarm_table.csv')+'。')
    hashes=[]
    for key,m in manifests.items():
        hashes.append([key,'manifest',link(out/(key+'_real_manifest.json')),seal['real_manifest_hashes'][key]])
        entries=m['m3_paths']+m['ordinary_paths']+[p for pair in m['s_paths'] for p in pair]
        seen=set()
        for e in entries:
            if e['path'] in seen:continue
            seen.add(e['path']);hashes.append([key,'权重',link(ROOT/e['path']),e['sha256']])
    text.append(table(('臂','类型','ignored artifact路径','SHA256'),hashes))
    text.append('主切点：'+ json.dumps(manifests['main']['cuts'],ensure_ascii=False)+'。敏感性切点见其manifest；没有按真实结果调整。\n\n复现生成器：'+link(Path(__file__))+'；已封存完成后运行：`python research/active/dtr-r0/nearfield/cnh_no_bin0_dr_report_20261011.py`。')
    report.write_text('\n\n'.join(text)+'\n',encoding='utf8')
    receipt=dict(status='COMPLETE',seconds=time.monotonic()-began,verdict=verdict,recommend=bool(passed),failures=failures,real_gate_interpretations=interpretations,report_sha256=sha(report),figures={name:sha(out/name) for name in ('simulation_comparison.png','real_alarm_comparison.png')},source_sha256=sha(Path(__file__)))
    (out/'report_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--report',type=Path,default=REPORT);a=p.parse_args();run(a.out,a.report)
