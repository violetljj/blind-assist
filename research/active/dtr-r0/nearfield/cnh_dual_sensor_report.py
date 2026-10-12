"""Generate the dual-sensor alarm report and one standalone scientific plot."""
import argparse
from pathlib import Path

import cnh_dual_sensor_evaluate as E
import cnh_unknown_target_reference_evaluate as U

NAMES = {'single':'single', 'dual_frozen':'dual OR 冻结阈值',
         'dual_matched':'dual OR 匹配误停', 'alternating_frozen':'dual 交替2.5Hz'}


def metric(value):
    rate = value['rate']
    return f"{value['stops']}/{value['n']}" + (f"（{rate*100:.1f}%）" if rate is not None else '（无分母）')


def delta(value):
    rate = value['delta_rate']
    if rate is None:
        return '无分母'
    ci = value['ci95']
    text = f"{rate*100:+.1f}pp，净{value['delta_stops']:+d}/{value['n']}"
    if ci[0] is not None:
        text += f"，场景95%区间[{ci[0]*100:+.1f},{ci[1]*100:+.1f}]pp"
    return text


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells = result['sequence']['evaluation']
    colors = {'single':'#4f5b66','dual_frozen':'#e58b2b','dual_matched':'#168a74','alternating_frozen':'#8b5cb6'}
    fig, axes = plt.subplots(1, 2, figsize=(10,4.4), sharey=True)
    for ax, group in zip(axes, ('FOV_OUT','FOV_IN')):
        for arm in E.ARMS:
            clear = cells['all'][arm]['joint_clear']
            timely = cells[group][arm]['branches']['shallow']['timely']
            if clear['rate'] is None or timely['rate'] is None:
                continue
            ci = timely['ci95']
            err = [[(timely['rate']-ci[0])*100],[(ci[1]-timely['rate'])*100]] if ci[0] is not None else None
            x, y = clear['rate']*100, timely['rate']*100
            ax.errorbar(x,y,yerr=err,fmt='o',color=colors[arm],capsize=4,label=arm)
            ax.annotate(f"{timely['stops']}/{timely['n']}",(x,y),xytext=(5,5),textcoords='offset points',fontsize=8)
        ax.set_title(group+' shallow timely')
        ax.set_xlabel('Joint clear first-stop rate (%)\n(all evaluation physical episodes)')
        ax.grid(alpha=.2)
        ax.set_ylim(-5,105)
    axes[0].set_ylabel('Timely rate (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,fontsize=8,bbox_to_anchor=(.5,.01))
    fig.suptitle('Dual +/-15 deg frozen M3: calibration-disjoint evaluation\nReused synthetic Development; vertical bars: scene-bootstrap 95% CI',fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.9))
    folder = Path(out)/'figures'
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/'timely_vs_joint_clear.png'
    fig.savefig(path,dpi=180)
    fig.savefig(folder/'timely_vs_joint_clear.svg')
    plt.close(fig)
    return path


def report(out=E.OUT, document=None):
    out = Path(out)
    result = U.read(out/'result.json')
    picture = plot(result,out)
    document = Path(document) if document else out/'REPORT.md'
    decision = result['decision']
    lines = [f"# 双 ±15° 传感器报警级验证（2026-10-05）",'',
        f"**判读：{decision['branch']}。** 冻结M3，48个既有pilot2合成场景、K4、7个位移、13个决策帧；按整场景分层冻结24校准/24评价。结果只适用于已消费模拟Development。",'',
        '主判读使用评价集dual匹配误停臂；单传感器阈值0.8557642486787612，dual只在校准集标定一条全局阈值。',
        f"dual匹配阈值：{result['thresholds']['dual_matched']['threshold']:.15g}（>=）；校准clear首停{result['thresholds']['dual_matched']['actual_stops']}/{result['thresholds']['dual_matched']['n']}，single预算{result['thresholds']['dual_matched']['allowed_stops']}，因完整tie组未用预算{result['thresholds']['dual_matched']['unattained_stops']}次。",'',
        '## 评价集主表','', '| 臂 | FOV_OUT浅及时 | FOV_IN浅及时 | 总体深及时 | 联合clear首停 |', '|---|---:|---:|---:|---:|']
    cells = result['sequence']['evaluation']
    for arm in E.ARMS:
        a = cells['FOV_OUT'][arm]['branches']['shallow']['timely']
        b = cells['FOV_IN'][arm]['branches']['shallow']['timely']
        d = cells['all'][arm]['branches']['deep']['timely']
        c = cells['all'][arm]['joint_clear']
        lines.append(f"| {NAMES[arm]} | {metric(a)} | {metric(b)} | {metric(d)} | {metric(c)} |")
    lines += ['',f"评价集FOV_OUT {cells['FOV_OUT']['single']['scenes']}个独立场景，浅分母{cells['FOV_OUT']['single']['branches']['shallow']['timely']['n']}；FOV_IN {cells['FOV_IN']['single']['scenes']}个独立场景，浅分母{cells['FOV_IN']['single']['branches']['shallow']['timely']['n']}。K4重复不扩大独立场景数。",'',
        f"匹配dual对single：FOV_OUT {delta(cells['FOV_OUT']['dual_matched']['branches']['shallow']['paired_minus_single'])}；FOV_IN {delta(cells['FOV_IN']['dual_matched']['branches']['shallow']['paired_minus_single'])}。",
        f"评价联合clear首停净变化{decision['clear_delta_stops']:+d}/{decision['clear_n']}；冻结容差single+1次，{'通过' if decision['clear_cost_pass'] else '未通过'}。",'',
        '支持：OUT浅及时≥+30pp、IN≥−3pp且评价clear≤single+1；不支持：OUT<+10pp或IN<−3pp；其他MIXED。交替采样单独报告，不进入主判读。', '',
        f"![及时率与联合clear首停]({picture.as_posix()})",'',
        '联合clear：外侧15/20cm位移，HEAD和BODY两个公开查询全13帧均实际clear，任一查询报警计一次物理episode首停。每episode曝光代理为13×0.2秒；两查询不重复计时。','',
        '## 全批、校准与评价分层','', '| 批 | 分层 | 臂 | 场景 | 浅及时 | 深及时 | 联合clear首停 |', '|---|---|---|---:|---:|---:|---:|']
    for split in ('all','calibration','evaluation'):
        for group in ('all','FOV_OUT','FOV_IN','mode0','mode1','mode2','HEAD','BODY'):
            for arm in E.ARMS:
                cell = result['sequence'][split][group][arm]
                lines.append(f"| {split} | {group} | {NAMES[arm]} | {cell['scenes']} | {metric(cell['branches']['shallow']['timely'])} | {metric(cell['branches']['deep']['timely'])} | {metric(cell['joint_clear'])} |")
    lines += ['', '## 交替采样与首报来源','',
        '交替臂每0.2秒tick只有一路新观测：L在绝对偶数帧、R在奇数帧。推理按最近1.4秒真实采样观测构造历史；平滑只纳入最近5次新logit，另一路分数保持至多0.4秒。起始未采样支路无分数，不能触发报警。','',
        '| 评价分层 | 交替对single浅及时差 | 全速来源L/R/tie | 交替来源L/R/tie |','|---|---:|---:|---:|']
    for group in ('all','FOV_OUT','FOV_IN'):
        alt = cells[group]['alternating_frozen']['branches']['shallow']
        full = cells[group]['dual_frozen']['branches']['shallow']['first_report_source']
        source = alt['first_report_source']
        lines.append(f"| {group} | {delta(alt['paired_minus_single'])} | {full['L']}/{full['R']}/{full['tie']} | {source['L']}/{source['R']}/{source['tie']} |")
    lines += ['', '来源按目标高度查询首报时两支路较大平滑分数归属；tie为严格等分。它描述首次报警来源，不等于目标物体归因。','', '## Splay几何预扫','']
    splay_path = out/'splay/table.md'
    if splay_path.exists():
        lines += [splay_path.read_text(encoding='utf8').strip(), '',
            '报警主臂固定±15°；几何筛选通过的更大splay仍须另做报警与误停检验，不能从射线可达直接推断报警收益。','']
    else:
        lines += ['几何预扫收据由主任务补入；本报警主臂固定±15°。','']
    secondary_path = out/'secondary22p5/result.json'
    if secondary_path.exists():
        secondary = U.read(secondary_path)
        sc = secondary['sequence']['evaluation']
        lines += ['## 几何筛选后的±22.5°次臂','',
            '±22.5°由报警指标读取前的几何扫描选定；补充PLAN写入时主结果已产生，冻结时点如实记录。完整48场景、相同光子seed与共同位姿误差、原截止时间均继承主臂。次臂未新标定阈值，不参与主判读。','',
            '| 次臂工作点 | FOV_OUT浅及时 | FOV_IN浅及时 | 总体深及时 | 联合clear首停 |',
            '|---|---:|---:|---:|---:|']
        for arm, label in [('secondary_frozen','±22.5°原冻结阈值'),
                           ('secondary_transferred_main_threshold','±22.5°移植主臂阈值'),
                           ('secondary_alternating_frozen','±22.5°交替2.5Hz')]:
            lines.append(f"| {label} | {metric(sc['FOV_OUT'][arm]['branches']['shallow']['timely'])} | {metric(sc['FOV_IN'][arm]['branches']['shallow']['timely'])} | {metric(sc['all'][arm]['branches']['deep']['timely'])} | {metric(sc['all'][arm]['joint_clear'])} |")
        lines += ['',
            '更大的splay提高几何缺口消除，但没有提高本批浅及时率；原冻结阈值下clear更少，是检测—成本权衡，不能简单宣称角度更大更好或更差。移植阈值只作诊断，不能称次臂已完成等误停校准。保留±15°为已通过主规则的模拟参照。','',
            '次臂冻结PLAN继承的母extrinsics文字含±15选择器；实际±22.5外参由splay_deg、selector_mapping及保存矩阵明确，工程与独立检查验证。','']
    lines += ['## 计算、验证与边界','',
        f"CPU评价耗时{result['evaluation_s']:.2f}秒；2000次整场景配对bootstrap，在各批内重采样，阈值固定。渲染和推理墙钟见任务生命周期收据。",'',
        '原单传感器两公开查询分数和已封存M3结果逐值核对；FOV分类固定于原单传感器。保留所有48场景和K4重复，无结果驱动删样。校准/评价单位见result.json和PLAN.json。', '',
        '工程核对：原single重新推理raw logit最大绝对误差0；CPU/GPU期望值最大差2.84e−14。两支路共享原头部位姿估计误差并应用各自固定外参；安装无横向平移，俯仰−10°。', '',
        '输入边界：公开查询继承原名义头部/查询框；估计误差只作用于历史相对重投影。这不能证明完整实机头姿误差鲁棒性。', '',
        '该批仅mode0/1；mode2分母0，不能解释为阴性结果。K4是噪声/位姿重复而非独立场景。', '',
        '双路在每路原平滑后取max OR；误停代价按实际计数保留，匹配阈值只在校准集选定，不能由评价及时率重新选择。2.5Hz是压力测试，尚未测双传感器实际吞吐。', '',
        '结论限于当前模拟、冻结M3和公开查询语义。未证明真实串扰、SNR、功耗、佩戴反应、报警安全性或自然走路误停负担；近低位覆盖范围也不构成盲杖补齐漏检的实测证据。', '',
        ]
    natural_path = out/'natural-precheck/admission.json'
    if natural_path.exists():
        lines += ['自然检查跑前固定96000/96001/96002的config0，三条共49,152个float16 z1逐位复现，差0。原signed/raw CNH未保留，无原始逐值比较参考，故raw gate为NOT_EVALUABLE、自然整批NOT_RUN。z1成功保留为模型输入复现证据，不替代raw准入。','']
    completion_path = out/'completion.json'
    if completion_path.exists():
        completion = U.read(completion_path)
        lines += [f"本轮PLAN冻结至科学交付检查完成墙钟{completion['wall_minutes_from_freeze']:.2f}分钟（两小时上限）；主渲染/推理{completion['main_render_s']:.2f}/{completion['main_inference_s']:.2f}秒，次臂渲染/推理{completion['secondary_render_s']:.2f}/{completion['secondary_inference_s']:.2f}秒。运行与源码快照保留于本轮目录，任务GPU进程已结束。",'']
    lines += ['复现入口：`cnh_dual_sensor_alarm.py freeze / engineering / render / infer`，随后 `cnh_dual_sensor_evaluate.py`、`cnh_dual_sensor_check.py`、`cnh_dual_sensor_report.py`。本轮freeze已消费，新的科学队列需新授权预算，不覆盖现有PLAN/result。','']
    document.parent.mkdir(parents=True,exist_ok=True)
    document.write_text('\n'.join(lines),encoding='utf8')
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=E.OUT)
    parser.add_argument('--document',type=Path)
    args = parser.parse_args()
    print(report(args.out,args.document))
