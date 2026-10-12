"""Render one scientific curve and a consolidated A/B/C simulation report."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

import cnh_dual_sensor_envelope_cpu as C

ROOT=C.ROOT
OUT=C.OUT
REPORT=ROOT/'research/active/dtr-r0/nearfield/CNH_DUAL_SENSOR_ENVELOPE_20261005.md'
LABELS=dict(single='single',dual15='dual ±15° / 匹配阈值',dual22p5_frozen='dual ±22.5° / 原阈值',dual22p5_transferred='dual ±22.5° / 移植阈值')


def frac(value):
    return f"{value['stops']}/{value['n']}（{100*value['rate']:.1f}%）" if value.get('rate') is not None else f"{value['stops']}/{value['n']}"


def figure(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=dict(single='#6b7280',dual15='#2563eb',dual22p5_frozen='#d97706',dual22p5_transferred='#059669')
    names=dict(single='Single / frozen',dual15='Dual +/-15 / matched',dual22p5_frozen='Dual +/-22.5 / frozen',dual22p5_transferred='Dual +/-22.5 / transferred')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True,layout='constrained')
    for ax,side,title in zip(axes,('opposite','same'),('Opposite side (11 scenes, n=88)','Same side (13 scenes, n=104)')):
        for arm in names:
            xs,ys,lo,hi=[],[],[],[]
            for psi in result['psi_deg']:
                cell=result['sequence'][str(psi)][side].get(arm)
                if cell is None:continue
                met=cell['branches']['shallow']['timely']
                xs.append(psi);ys.append(100*met['rate']);lo.append(100*met['ci95'][0]);hi.append(100*met['ci95'][1])
            ax.plot(xs,ys,'o-',label=names[arm],color=colors[arm],linewidth=2,markersize=5)
            if arm=='dual15':ax.fill_between(xs,lo,hi,color=colors[arm],alpha=.12,linewidth=0)
        ax.set(title=title,xlabel='Constant physical head yaw ψ (degrees)',ylim=(-3,103),xticks=result['psi_deg'])
        ax.grid(axis='y',alpha=.2);ax.axvline(15,color='#d1d5db',linestyle=':',linewidth=1)
    axes[0].set_ylabel('Shallow timely alarm (%)')
    axes[1].legend(loc='lower left',fontsize=8)
    fig.suptitle('Frozen M3 alarm envelope: dual sensor splay versus head yaw',fontsize=13,fontweight='bold')
    path=out/'figures/timely_vs_psi.png';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Completed curve immutable')
    fig.savefig(path,dpi=190);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)
    return path


def make_figure(out=OUT):
    out=Path(out)
    receipt_path=out/'figure_receipt.json'
    if receipt_path.exists():raise FileExistsError('Figure receipt already sealed')
    angular=out/'angular/result.json'
    plot=figure(C.read(angular),out)
    C.write_new(receipt_path,dict(status='COMPLETE',angular_result_sha256=C.sha(angular),
        png=str(plot),png_sha256=C.sha(plot),pdf=str(plot.with_suffix('.pdf')),
        pdf_sha256=C.sha(plot.with_suffix('.pdf')),source_sha256=C.sha(__file__)))
    print('FIGURE',plot)
    return plot


def build(out=OUT):
    out=Path(out);plan=C.read(out/'PLAN.json');plan_sha=C.sha(out/'PLAN.json')
    a=C.read(out/'angular/result.json');b=C.read(out/'natural/result.json');c=C.read(out/'alternating/result.json')
    assert all(v['status']=='COMPLETE' for v in (a,b,c))
    assert a['provenance']['plan_sha256']==c['provenance']['parent_plan_sha256']==b['provenance']['parent_plan_sha256']==plan_sha
    if REPORT.exists():raise FileExistsError('Consolidated report immutable; parent owns any subsequent edit')
    figure_receipt=out/'figure_receipt.json'
    if figure_receipt.exists():
        sealed=C.read(figure_receipt)
        assert sealed['status']=='COMPLETE' and sealed['angular_result_sha256']==C.sha(out/'angular/result.json')
        plot=Path(sealed['png'])
        assert C.sha(plot)==sealed['png_sha256'] and C.sha(Path(sealed['pdf']))==sealed['pdf_sha256']
    else:
        plot=make_figure(out)
    out20=a['sequence']['20']['opposite']['dual15']['branches']['shallow']['timely']
    out25=a['sequence']['25']['opposite']['dual15']['branches']['shallow']['timely']
    out30=a['sequence']['30']['opposite']['dual15']['branches']['shallow']['timely']
    cn=c['sequence']['evaluation']['all']['alternating_matched']
    nc=b['cells']['single']['all']['clear'];nd=b['cells']['dual15']['all']['clear']
    batch_contract=out/'natural/ENGINEERING_BATCH_CONTRACT.json'
    batch_text=('自然批新增batch执行合同已保存于`natural/ENGINEERING_BATCH_CONTRACT.json`，与模型/渲染器源哈希冻结记录一起保留。'
                if batch_contract.exists() else '自然批batch工程合同尚未写入；现有模型/渲染器源哈希冻结记录在`natural/PLAN.json`。')
    worker_path=out/'worker/terminal.json'
    worker_text=''
    if worker_path.exists():
        worker=C.read(worker_path)
        assert worker['status']=='NOT_USED_PROVISION_ABORTED' and worker['returned_units']==[], 'Report worker contribution from actual evidence before finalization'
        worker_text='曾尝试副机部署，按部署时限中止，没有启动Python/科学任务、没有返回科学unit；全部科学GPU结果仍来自本机RTX5060。副机尝试收据见`worker/terminal.json`。'
    elapsed=(time.time()-plan['started_unix'])/60
    lines=['# 双传感器偏头容限与模拟杂物成本补完（2026-10-05）','',
        f"**±15°的反侧浅及时在ψ=20/25/30°分别为{frac(out20)}、{frac(out25)}、{frac(out30)}。** 自然96000模拟clear首停single/dual匹配移植阈值为{nc['stops']}/{nc['n']}与{nd['stops']}/{nd['n']}，{nc['stops_per_proxy_minute']:.3f}→{nd['stops_per_proxy_minute']:.3f}次/查询曝光代理分钟。交替2.5Hz重新匹配校准误停后评价浅及时{cn['branches']['shallow']['timely']['stops']}/192，FOV_IN仍低于single5.56pp。本轮全部为已消费合成Development的EXPLORE诊断。",'',
        '原`DUAL_ALARM_SUPPORTED_SIM`判读保留在其冻结条件内；原mode0头偏+15°与−15°左光轴正好相消，FOV_OUT的48次左路首报包含47次及时，只支持该对齐构造的报警恢复。本轮曲线提供额外偏头容限信息。mode1物理头扫视±20°，双路外侧光轴到±35°；92/96→96/96是原评价集匹配阈值结果，原阈值95/96。','',
        '## A：恒定偏头容限','',
        '24个mode0场景，原反侧FOV_OUT为11场景、浅88/深44；同侧FOV_IN为13场景、浅104/深52。保留原ψ=15°侧别，不随新角度重分类。K4、7位移、13帧；模型、阈值、因果平滑、−10°俯仰冻结。每场景全部位移和重复共同进入2000次配对场景bootstrap。','',
        '![偏头容限曲线](../../../../artifacts.local/work/cnh-dual-sensor-envelope-20261005/figures/timely_vs_psi.png)','',
        '图中蓝色带为±15°浅及时的场景bootstrap95%区间。±22.5°使用原单阈值与±15°匹配阈值的移植口径，**尚未做自身等误停标定**。','',
        '| ψ | 臂 | 反侧浅及时 | 反侧深及时 | 反侧联合clear | 同侧浅及时 | 同侧深及时 | 同侧联合clear | 全体联合clear |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for psi in a['psi_deg']:
        for arm in LABELS:
            cells=a['sequence'][str(psi)]
            if arm not in cells['all']:
                lines.append(f'| {psi}° | {LABELS[arm]} | NOT_RUN_BUDGET | — | — | — | — | — | — |');continue
            opposite,same,allcell=[cells[g][arm] for g in ('opposite','same','all')]
            values=[]
            for cell in (opposite,same):
                values.extend([frac(cell['branches']['shallow']['timely']),frac(cell['branches']['deep']['timely']),frac(cell['joint_clear'])])
            values.append(frac(allcell['joint_clear']))
            lines.append(f"| {psi}° | {LABELS[arm]} | "+' | '.join(values)+' |')
    lines.extend(['','联合clear仍是外侧15/20cm、双公开查询在全部13帧均clear的物理episode；任一查询报警计一次，分母192（反侧88、同侧104），曝光代理8.32分钟。没有把两个查询重复计为两个物理事件。','',
        '### 跑前几何预期与报警对照','',
        '原提议距离为`0.29 / tan(22.5° − |ψ − splay|)`，以1.3m比较。对反侧x<0，只有ψ≥splay时它对应朝外侧的水平边界；ψ<splay时绝对值写法是保守界，并不能据此断言中心缺口。该式忽略目标宽度、纵向边界、8×8离散射线和M3历史；它是曝光提示，无法保证报警及时。1.3m是此前回正剂量参照，本轮及时截止为0.9m；此前更远距离见过目标的历史证据仍可能在更大偏头角及时首报。',
        f"相对ψ20°，±15°反侧浅及时在ψ25°变化{100*(out25['rate']-out20['rate']):+.1f}pp，在ψ30°变化{100*(out30['rate']-out20['rate']):+.1f}pp。因此应按实测曲线选择可讨论范围，不能直接把几何1.3m界当成报警容限。",'',
        '| ψ | splay | 原绝对值界/m | 1.3m内 | 反侧浅及时（冻结口径） |','|---:|---:|---:|---|---:|'])
    for pred in a['geometry_predictions']:
        psi,splay=pred['psi_deg'],pred['splay_deg'];arm={0:'single',15:'dual15',22.5:'dual22p5_transferred'}[splay]
        cell=a['sequence'][str(psi)]['opposite'].get(arm)
        bound='无正角裕量' if pred['proposed_abs_bound_m'] is None else f"{pred['proposed_abs_bound_m']:.3f}"
        lines.append(f"| {psi}° | ±{splay:g}° | {bound} | {'是' if pred['within1p3m'] else '否'} | {frac(cell['branches']['shallow']['timely']) if cell else 'NOT_RUN_BUDGET'} |")
    diagnostic_path=out/'angular/geometry_diagnostic.json'
    if diagnostic_path.exists():
        diagnostic=C.read(diagnostic_path)
        lines.extend(['','结果封存后只读取实际目标射线支持，未加渲染或判据：','',
            '| ψ / 左光轴 | 反侧浅分支末次可见距离，中位[范围]/m | 及时首报距离，中位[范围]/m | 首报在末次可见时或之前 / 之后 |',
            '|---|---:|---:|---:|'])
        for psi,groups in diagnostic['diagnostics'].items():
            d=groups['opposite'];v=d['last_visible_front_distance'];r=d['fused_timely_first_report_distance']
            lines.append(f"| {psi}° / {d['absolute_left_axis_deg']}° | {v['median_m']:.2f} [{v['min_m']:.2f},{v['max_m']:.2f}] | {r['median_m']:.2f} [{r['min_m']:.2f},{r['max_m']:.2f}] | {d['fused_timely_reports_before_or_at_last_visible_distance']} / {d['fused_timely_reports_after_last_visible_distance']} |")
        lines.extend(['','可见距离的分母为11场景×2浅分支=22，不复制K；首报距离含K4重复。末次可见只在原13决策帧中计算；ψ30直接读取旧目标object_id=0。ψ25仍及时与实际目标宽度及更早首报相容，当前关联没有分离单一成因；数据与来源哈希见`angular/geometry_diagnostic.json`。'])
    lines.extend(['','ψ=15°直接复用原single/±15°/±22.5°载荷并逐值核对；新网格中相同绝对光轴共享观测与分数。0/15/30°光轴复用原左/中心/右流，−7.5/37.5°复用原±22.5°流；其他绝对角使用新光子前缀2026100520。共享同一条旧位姿误差再按固定外参变换，避免独立位姿估计带来的额外收益。复用合同见`ENGINEERING_INPUT_CONTRACT.json`。本轮ψ0 single为86/88，与旧全程正视回正参照的88/88分开：本轮使用旧dual左流的光子实现和共享误差外参，未宣称复跑旧回正输入逐位相同。','',
        '## B：自然96000合成批的clear代价','',
        f"本轮{len(b['units'])}个unit × {len(b['configs'])}个均匀config，共{b['scenes']}场景、{b['query_episodes']}查询episode。实际config IDs：`{b['configs']}`。原始single分数复用，左右新渲染和冻结M3推理；所有阈值冻结，移植阈值未在自然批重新标定。",'',
        f"自然模拟查询clear首停净增{nd['stops']-nc['stops']}次，成本增加{100*(nd['stops_per_proxy_minute']/nc['stops_per_proxy_minute']-1):.1f}%。因此pilot2中的clear下降不能外推为自然杂物场景无额外误停；本轮新增成本与方向、阈值和观测改变同时存在，未分离因果来源。",'',
        '准入按用户本轮明确改为z1模型输入逐位复现：原3条序列49,152/49,152值bitwise相等；这是模型输入管线准入，未声称raw CNH逐位相同。该批包含边界目标、较亮混合表面和侧墙。clear按所有物体的固定查询标签判断，新增误停不能仅凭首报来自侧路就归因于路边杂物。',batch_text,'',
        '| mode | 臂 | clear首停 | 查询曝光代理/min | 次/代理min | 浅0–2cm及时 | 2–5cm及时 | 深>5cm及时 | 首报来源 |','|---|---|---:|---:|---:|---:|---:|---:|---|'])
    for mode in ('all','mode0','mode1','mode2'):
        for arm in ('single','dual15','dual15_original_threshold'):
            cell=b['cells'][arm][mode];clear=cell['clear']
            labels={'single':'single','dual15':'dual15 / 移植匹配阈值','dual15_original_threshold':'dual15 / 原阈值'}
            lines.append(f"| {mode} | {labels[arm]} | {clear['stops']}/{clear['n']} | {clear['proxy_minutes']:.3f} | {clear['stops_per_proxy_minute']:.3f} | {frac(cell['contact0-2cm'])} | {frac(cell['contact2-5cm'])} | {frac(cell['contact>5cm'])} | {cell['first_report_source']} |")
    lines.extend(['','上表成本分母是clear查询曝光，与A的双查询联合clear物理episode成本分别报告，不能直接拼合。自然批浅、深分母按实际几何类别与可插值0.9m截止点，截尾单列。31条浅事件属于可评价几何子集，不是全部7680查询episode。clear成本使用全部固定clear查询曝光，不要求目标轨迹跨过0.9m。','',
        '| mode | 全查询episode | covered | single截尾n / 已报警 | dual15移植阈值截尾n / 已报警 | dual15原阈值截尾n / 已报警 |',
        '|---|---:|---:|---:|---:|---:|'])
    for mode in ('all','mode0','mode1','mode2'):
        total=b['query_episodes'] if mode=='all' else sum(u%3==int(mode[-1]) for u in b['units'])*len(b['configs'])*2
        censored=[b['cells'][arm][mode]['censored'] for arm in ('single','dual15','dual15_original_threshold')]
        assert len(set(cell['n'] for cell in censored))==1
        covered=total-censored[0]['n']
        values=[f"{cell['n']} / {cell['already_alarm']}" for cell in censored]
        lines.append(f'| {mode} | {total} | {covered} | '+' | '.join(values)+' |')
    lines.extend(['','截尾表示当前目标几何轨迹无法在既定观察区间内插值评价0.9m截止点，不是已知漏报；已报警仅描述事件，不能把截尾全部视作negative。','',
        '| 臂 | 双查询联合clear物理首停 | 代理分钟 | 次/代理分钟 |','|---|---:|---:|---:|'])
    for arm,met in b['joint_clear'].items():
        lines.append(f"| {arm} | {met['stops']}/{met['n']} | {met['proxy_minutes']:.3f} | {met['stops_per_proxy_minute']:.3f} |")
    lines.extend(['','自然批clear新增/移除与配对区间：'])
    for arm,comparisons in b['comparisons'].items():
        met=comparisons['clear'];lo,hi=met['paired_unit_delta_ci95']
        lines.append(f"- {arm}：新增{met['added']}，移除{met['removed']}；每查询曝光代理分钟差的配对unit95%区间[{lo:.3f}, {hi:.3f}]。")
    lines.extend(['','## C：2.5Hz交替重新匹配校准误停','',
        f"单一新阈值{c['matching']['threshold']:.17g}（logit，>=），原24校准场景的clear17/192与single17一致；在原24评价场景冻结使用。该臂已经用采样后的观测重新构建输入并推理，本轮只用已有raw logits重建平滑和标定阈值。",'',
        '| 评价指标 | single | 交替原阈值 | 交替匹配阈值 |','|---|---:|---:|---:|'])
    for group in ('all','FOV_OUT','FOV_IN'):
        for branch in ('shallow','deep'):
            vals=[c['sequence']['evaluation'][group][arm]['branches'][branch]['timely'] for arm in ('single','alternating_frozen','alternating_matched')]
            lines.append('| '+group+'/'+branch+' | '+' | '.join(frac(v) for v in vals)+' |')
    vals=[c['sequence']['evaluation']['all'][arm]['joint_clear'] for arm in ('single','alternating_frozen','alternating_matched')]
    lines.append('| 联合clear首停 | '+' | '.join(frac(v) for v in vals)+' |')
    lines.extend(['','阈值重评恢复部分及时性，但评价FOV_IN浅及时132/144仍低于single140/144，损失8条、5.56pp；配对95%区间[−12.5,0]pp。总体浅净+18/192，配对区间约[−2.1,+21.4]pp。不能因原阈值成本为0就判降频整体无效，也不能因重新标定后有改善就推导硬件可用。','',
        '这是5Hz训练的冻结M3输入2.5Hz观测的压力测试：每路历史观测更稀疏，5个新logit平滑的端点跨度从0.8s变成1.6s（按五采样窗口习惯可称1s→2s）；旧分数最多保留0.4s。它没有测硬件吞吐，也未对2.5Hz专门训练或调整平滑。','',
        '## 验证、预算和范围','',
        f"A：ψ15四臂旧结果一致、复用光轴逐值atol1e−12核对、{a['checks']['independent_integer_comparisons']}项计数独立循环核对；C：raw acquired-logit重建与旧ledger核对、{c['checks']['integer_comparisons']}项计数核对。自然批原single保留分母、所有传感器同场景配对，具体准入和运行收据在本任务树中。报告生成时累计约{elapsed:.1f}分钟；最终交付耗时以completion.json为准，冻结90分钟预算未扩展。",'',
        worker_text,'',
        '证据范围统一限于当前模拟：新曲线覆盖恒定正向头偏的24个mode0场景；扫描头动已有原mode1诊断。背景、光子、共享位姿误差、零横向基线和冻结查询定义均为条件。真实头动分布、人体提示执行、SNR、串扰、功耗和双路吞吐尚无本轮证据。±22.5°的两阈值口径没有自身等误停标定，不能作为无成本的硬件夹角优势。','',
        '硬件恢复后的待办：第二个L8CH；双路I2C调度/双总线或SPI的实测吞吐与延迟；±15°两发射视场15°重叠时的串扰与信号质量；真实负例和误停体验。本轮保持硬件暂停。','',
        f"PLAN SHA256：`{plan_sha}`。完整分层、首报来源、配对区间、载荷哈希在`artifacts.local/work/cnh-dual-sensor-envelope-20261005/`的angular/natural/alternating结果中。"])
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf8')
    C.write_new(out/'report_receipt.json',dict(status='COMPLETE',report=str(REPORT),report_sha256=C.sha(REPORT),figure=str(plot),figure_sha256=C.sha(plot),
        plan_sha256=plan_sha,result_sha256={name:C.sha(out/name/'result.json') for name in ('angular','natural','alternating')},source_sha256=C.sha(__file__),elapsed_minutes=elapsed))
    print('REPORT',REPORT)
    return REPORT


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--figure-only',action='store_true')
    args=parser.parse_args()
    make_figure(args.out) if args.figure_only else build(args.out)
