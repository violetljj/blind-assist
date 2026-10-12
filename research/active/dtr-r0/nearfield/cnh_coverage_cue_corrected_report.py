"""Describe frozen corrected results without altering any experiment input."""
from collections import Counter
from pathlib import Path
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cnh_coverage_cue_corrected as R

OUT=R.OUT
COLORS={'E_orig':'#b98332','E_oracle':'#316fb4','E_grav':'#258267'}


def interval_stats(streams,model,name):
    rows=[next(r for r in s['rows'] if r['name']==name) for s in streams if s['model']==model]
    intervals=np.array([v for r in rows for v in r['cue_intervals_frames']])
    counts=Counter(intervals.tolist());n=len(intervals);events=sum(r['auditory_events'] for r in rows)
    return dict(model=model,name=name,streams=len(rows),events=events,intervals=n,
                both_side_fraction=sum(r['dual_tag_events'] for r in rows)/events if events else None,
                quantiles_s={str(q):float(np.quantile(intervals*.2,q)) for q in (.1,.5,.9)} if n else {},
                distribution=[dict(frames=int(k),seconds=k*.2,n=int(v),fraction=v/n) for k,v in sorted(counts.items())],
                cooldown_fraction=counts[9 if '_p0_' in name else 12]/n if n else None)


def plot(cells):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,4,figsize=(15,10),sharex=True,sharey=True)
    for si,sigma in enumerate((5,10,15)):
        for ti,tau in enumerate((.5,1.,2.,4.)):
            ax=axes[si,ti];ax.grid(alpha=.15)
            for model,color in COLORS.items():
                cc=[c for c in cells if c['model']==model and c['sigma']==sigma and c['tau']==tau and c['method'] in ('A','B','v1') and not c['diagnostic']]
                ax.scatter([c['auditory_cues_per_min'] for c in cc],[100*c['elimination'] for c in cc],color=color,alpha=.15,s=9)
                ordered=sorted(cc,key=lambda c:(c['auditory_cues_per_min'],-c['elimination']))
                efficient=[];best=-float('inf')
                for c in ordered:
                    if c['elimination']>best:
                        efficient.append(c);best=c['elimination']
                ax.plot([c['auditory_cues_per_min'] for c in efficient],[100*c['elimination'] for c in efficient],
                        color=color,ls='--' if model=='E_orig' else '-',marker='o',ms=3,lw=1.5)
            dual=next(c for c in cells if c['model']=='E_oracle' and c['sigma']==sigma and c['tau']==tau and c['method']=='dual')
            ax.axhline(100*dual['elimination'],color='#303b47',ls='-.',lw=1.3)
            ax.axhline(100,color='#78828a',ls=':',lw=1)
            ax.axvline(10,color='#8c9299',ls=':',lw=.8)
            ax.scatter([0],[0],color='#303b47',s=15)
            ax.set_title(f'sigma={sigma} deg, tau={tau:g} s',loc='left',fontsize=11)
            ax.set_xlim(-1,40);ax.set_ylim(-15,105)
            if ti==0:ax.set_ylabel('Excess yaw-zero deficit eliminated (%)')
            if si==2:ax.set_xlabel('Actual auditory events / min')
    handles=[Line2D([0],[0],color=col,ls='--' if m=='E_orig' else '-',marker='o',ms=4,label=m+(' (legacy v1)' if m=='E_orig' else '')) for m,col in COLORS.items()]
    handles += [Line2D([0],[0],color='#303b47',ls='-.',label='Dual +/-15 deg, no cue'),Line2D([0],[0],color='#78828a',ls=':',label='Permanent yaw-zero geometry (100%)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.937),ncol=3,frameon=False)
    fig.suptitle('Corrected coverage geometry: benefit versus auditory cue burden',x=.065,ha='left',fontsize=20,y=.985)
    fig.text(.065,.952,'12 yaw-restorable cells; K4 per cell. Pale dots: all 66 delayed policies. Curves: observed efficient configurations.',fontsize=10)
    fig.text(.065,.025,'Synthetic motion/control only. Lines connect observed configurations; budget selection never interpolates.\n'
             'E_orig reuses historical closed loops, including the 16-cell v1 defect. Permanent yaw-zero is a geometry reference, not a zero-cost software action.',fontsize=9)
    fig.subplots_adjust(left=.065,right=.985,top=.845,bottom=.11,hspace=.31,wspace=.16)
    fig.savefig(OUT/'coverage_cue_corrected_tradeoff.png',dpi=160)
    fig.savefig(OUT/'coverage_cue_corrected_tradeoff.svg')
    plt.close(fig)


def main():
    start=time.monotonic();result=R.S.read(OUT/'result.json');assert result['status']=='COMPLETE'
    cells=result['cells'];front=result['frontier'];streams=result['per_stream']
    # Exact policy pairing is an additional descriptor, not frontier selection.
    matched=[]
    lookup={(c['model'],c['sigma'],c['tau'],c['name']):c for c in cells}
    for c in cells:
        if c['model']!='E_grav' or c['method'] not in ('A','B','v1') or c['diagnostic']:continue
        oracle=lookup['E_oracle',c['sigma'],c['tau'],c['name']]
        matched.append(dict(sigma=c['sigma'],tau=c['tau'],name=c['name'],elimination_delta_pp=100*(c['elimination']-oracle['elimination']),
                            cue_rate_delta_per_min=c['auditory_cues_per_min']-oracle['auditory_cues_per_min']))
    stats=[interval_stats(streams,m,n) for m in R.MODELS for n in ('v1_p0_L0.6_T0.2','v1_p3_L0.6_T0.2')]
    R.S.save(OUT/'supplement.json',dict(interval_stats=stats,matched_grav_minus_oracle=matched))
    plot(cells)
    with (OUT/'frontiers.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['model','sigma_deg','tau_s','budget_per_min','status','selected','elimination_pct','actual_cues_per_min','dual_elimination_pct'])
        for row in front:
            b=row['best'];w.writerow([row['model'],row['sigma'],row['tau'],row['budget'],row['status'],b['name'] if b else '',100*b['elimination'] if b else '',b['auditory_cues_per_min'] if b else '',100*row['dual_elimination']])
    branches=result['branches'];verdict=branches['E_oracle']['verdict']
    gaps=[g['delta_pp'] for g in result['estimator_gap'] if g['budget']==10 and g['delta_pp'] is not None]
    dual=[100*c['elimination'] for c in cells if c['model']=='E_oracle' and c['method']=='dual']
    recommendation={'GEOMETRY_DOMINATED_SYNTH':'按冻结规则，覆盖主手段建议转向传感器配置验证，软件提示保留为UNKNOWN/兜底；本轮不启动硬件。',
                    'SOFTWARE_COMPETITIVE_SYNTH':'软件在本合成预算下有竞争力，下一步优先验证可部署估计与实际提示遵从率；本轮不直接决定产品工作点。',
                    'MIXED':'未达到任一8/12分支，不能把该结果写成软件已竞争或硬件已胜出；先区分预算不可行格与可比格，再由用户决定是否取得小量真实头姿。'}[verdict]
    lines=['# 覆盖提示修正重跑（2026-10-05）','',
           f'**修正后 E_oracle 的 ≤10 次/min 判读为 `{verdict}`。** 三档均完成48流，保留原48流台账并另跑96条更正闭环；每流80配置，共2,880个格×配置聚合行。双±15°同口径消除比例为{min(dual):.1f}%–{max(dual):.1f}%（12格均值{np.mean(dual):.1f}%）。结论仅适用于同批合成头动的有限几何控制诊断。','',
           recommendation+' 本轮按用户要求先修再比较，先不下载头姿数据。软件结果不直接等于人体可执行收益，双传感器仍是联合射线几何，未模拟串扰、SNR、功耗或融合报警。M3、阈值、安装保持原样。','',
           '## 修正与冻结口径','',
           '[跑前协议](CNH_COVERAGE_CUE_CORRECTED_PLAN_20261005.md)；机器PLAN SHA256：`'+result['plan_sha256']+'`。旧601358bd载荷及曝光门保留，原A/B/v1成本、残余、旧frontier与SIMPLE_RULE_RETAINED不再作为路线判断依据。','',
           '主指标仅含12个可回正球（半径2.5cm）：每侧两个x，HEAD y=−.045/.265m、BODY y=.54m。BODY .78m单列纵向边界。每侧每精确高度600到达/min，HEAD合计每侧1200、BODY600，全组3600/min；每流10–118s计6480单元到达。zero主k3缺口在所有流严格为0。k3不足是[.9,1.4]m内未得3帧可见的有限单元到达数，不是漏报警或真实危险数。','',
           'A/B公式不变，v1更正为只跟踪12单元。E_orig仅重算旧台账的新主指标，**其旧v1仍包含16单元缺陷**，是污染历史参照而非修好控制器后的E_orig复跑。E_oracle给控制器当前真姿态和真位置（软件逻辑参照、不可部署）；E_grav去掉X/Z常偏置，仅把原Y符号的±1°/s偏置绕世界重力Y轴左乘，原局部三轴0.1°逐步噪声、平移尺度和抽样不变。E_grav没有额外滤波锁定逐步随机游走，属于重力参照AHRS合成近似，不是实机模型。未来运动仅进入评价fork。','',
           '同seed、σ/τ/K4、holds/scans、p/L/T网格，无缩量。66个有延迟软件配置进入前沿；11个L0/T0诊断不参加，3参照另列。逐格预算施加在四流平均听觉事件/min上，双侧同时提示计一次事件；不插值，无可行配置标NOT_COMPARABLE。多余侧提示以实际提示时无后续干预的真未来fork、仅可回正单元评价，保留左右标签。全部精确高度、配置和逐流数据在JSON。','',
           '## 三档判读与≤10/min前沿','',
           '|估计档|分支|≥dual−15pp格|≤dual−30pp格|可比格|','|---|---|---:|---:|---:|']
    for m,b in branches.items():lines.append(f'|{m}{"（历史）" if m=="E_orig" else ""}|{b["verdict"]}|{b["competitive_cells"]}/12|{b["dominated_cells"]}/12|{b["comparable_cells"]}/12|')
    lines += ['', '表中软件单元为“消除比例%；实际听觉事件/min；所选配置”。—表示该预算无可行延迟软件配置，不能当成0%收益。', '',
              '|σ / τ|dual消除比例|E_orig ≤10/min（历史）|E_oracle ≤10/min|E_grav ≤10/min|','|---|---:|---|---|---|']
    for sigma in (5,10,15):
        for tau in (.5,1.,2.,4.):
            ff=[next(f for f in front if f['model']==m and f['sigma']==sigma and f['tau']==tau and f['budget']==10) for m in R.MODELS]
            values=[f'{100*f["best"]["elimination"]:.1f}%；{f["best"]["auditory_cues_per_min"]:.2f}；`{f["best"]["name"]}`' if f['best'] else '— NOT_COMPARABLE' for f in ff]
            lines.append(f'|{sigma}° / {tau:g}s|{100*ff[0]["dual_elimination"]:.1f}%|'+'|'.join(values)+'|')
    lines += ['', '![消除比例与实际听觉事件，三档估计和dual/zero参照](../../../../artifacts.local/work/cnh-coverage-cue-corrected-20261005/coverage_cue_corrected_tradeoff.png)', '',
              '图线连接观测到的有效配置，不代表中间预算可插值实现。zero=100%是恒定正视几何参照，不能解释为无成本软件工作点。完整5/10/20预算逐格选择见[CSV](../../../../artifacts.local/work/cnh-coverage-cue-corrected-20261005/frontiers.csv)，所有配置见[result.json](../../../../artifacts.local/work/cnh-coverage-cue-corrected-20261005/result.json)。','',
              '## 估计误差与修正v1间隔','',
              f'≤10/min E_grav−E_oracle前沿差的可比格数为{len(gaps)}/12，绝对差≤5pp为{result["estimator_close_cells_at10"]}/12，判为`{result["estimator_branch_at10"]}`。'+(f'差范围[{min(gaps):+.2f},{max(gaps):+.2f}]pp，中位数{np.median(gaps):+.2f}pp。' if gaps else '该预算无成对可行点，不能判断估计是否瓶颈。'), '',
              '逐相同配置E_grav−E_oracle差、全部5/10/20前沿差和间隔分布分别在supplement.json/result.json。前沿差允许两档选择不同配置，不能解释成单一配置的因果误差。', '',
              '|估计档 / v1配置|事件数|同流相邻间隔数|双侧事件|恰冷却周期占比|间隔P10 / 中位 / P90 (s)|','|---|---:|---:|---:|---:|---|']
    for st in stats:
        q=st['quantiles_s'];quant=' / '.join(f'{q[str(x)]:.1f}' for x in (.1,.5,.9)) if q else 'NOT_EVALUABLE'
        lines.append(f'|{st["model"]} / {st["name"]}|{st["events"]}|{st["intervals"]}|{100*st["both_side_fraction"]:.1f}%|{100*st["cooldown_fraction"]:.1f}%|{quant}|')
    lines += ['', '相邻间隔仅在同一流10–118s统计窗内计算，不跨流连接。p0/L.6/T.2冷却最短9帧=1.8s，p3对应12帧=2.4s；完整频数分布保留，不仅报告最短间隔。','',
              '## 纵向视场边界','']
    vertical=(OUT/'vertical_fov.md').read_text(encoding='utf8')
    lines.append(vertical)
    lines += ['', '## 验证、计时与使用边界','',
              f'机器摘要状态{result["status"]}；冻结至科学摘要{result["elapsed_s"]/60:.2f}分钟，96条新流全部完成、无删T/p/L；旧48条台账逐hash验证未改。原几何引擎/曝光门复用，zero在144流严格断言为0；独立engineering核对exogenous/noise抽样与纯重力漂移、精确高度指标和听觉事件口径。控制器只新增修正所需输入，不推理或训练。CPU纯NumPy，使用既有运行时，不导入Torch/CUDA。','',
              '四条合成重复不是人群样本；采样、站间距、球形离散单元、无遮拦/无光子模型及假设回正遵从率限制外推。曝光门M3及时438/440仍按原目标条件保留，不是本轮新控制策略的报警收益。双传感器为几何参照；E_oracle是不可部署软件参照；E_grav误差未实机标定。原0.6s时效报告保持原口径，提示段数受生成器结构限制。','',
              '源代码：[闭环](cnh_coverage_cue_corrected_core.py) · [冻结与汇总](cnh_coverage_cue_corrected.py) · [报告/图](cnh_coverage_cue_corrected_report.py)。完整载荷：[输出目录](../../../../artifacts.local/work/cnh-coverage-cue-corrected-20261005/)。进程由主runner的ProcessPoolExecutor管理，完成后释放；台账、PLAN和证据文件保留供复核。']
    path=Path(__file__).parent/'CNH_COVERAGE_CUE_CORRECTED_20261005.md'
    path.write_text('\n'.join(lines)+'\n',encoding='utf8')
    R.S.save(OUT/'report_receipt.json',dict(report=str(path),report_sha256=R.S.sha(path),figure_sha256=R.S.sha(OUT/'coverage_cue_corrected_tradeoff.png'),seconds=time.monotonic()-start))
    print(path,flush=True)


if __name__=='__main__':main()
