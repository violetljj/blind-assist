"""Render v4 signed-bias contract receipts; no fitting, scoring or cut selection."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/local-signed-bias-contract-dev-20261011'
PRIOR=ROOT/'artifacts.local/work/cnh-no-bin0-dr-retrain-dev-20261011'
REPORT=Path(__file__).resolve().parent.parent/'LOCAL_SIGNED_BIAS_CONTRACT_DEV_20261011.md'
CONDITIONS=('1','2','3','4','4b','5','6','7','8','9')
SCENES=dict(zip(CONDITIONS,('阳台UNKNOWN','近墙','远墙','空通道边界','暗空通道边界','近杆含筒','远杆含筒','外侧杆','头横杆','空通道边界')))
CONTROLS=('ZERO_RAW','PHYSICAL_BACKGROUND','SIGNED_BACKGROUND','PURE_AMBIENT','PURE_CROSSTALK','NEGATIVE_ONLY','NEGATIVE_BIAS_ZERO','BIN0_BACKGROUND_ONLY')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def link(path,label=None):return f'[{label or Path(path).name}]({Path(path).as_posix()})'
def table(head,rows):return '\n'.join(['| '+' | '.join(map(str,head))+' |','| '+' | '.join(['---']*len(head))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
def cell(r):return f"{r['strong']}/{r['ready_frames']} ({100*r['strong_rate']:.1f}%)；{r['light']}/{r['ready_frames']} ({100*r['light_rate']:.1f}%)"
def control_cell(r):return f"{r['strong']}/{r['ready_frames']}；{r['light']}/{r['ready_frames']}"
def index(rows):return {(r['arm'],str(r['condition']),r['height'],r['model']):r for r in rows}


def render_figures(out,old,new,simulation,prior_sim):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8.5),layout='constrained')
    for hi,height in enumerate(('HEAD','BODY')):
        for ai,arm in enumerate(('old','main')):
            ax=axes[hi,ai]
            rows=[[old[(arm,c,height,'S')],new[(arm,c,height,'S')]] for c in CONDITIONS]
            values=np.array([[100*r['strong_rate'] for r in row] for row in rows])
            im=ax.imshow(values,cmap='YlOrRd',vmin=0,vmax=100,aspect='auto')
            for ci,row in enumerate(rows):
                for pi,r in enumerate(row):ax.text(pi,ci,f"{100*r['strong_rate']:.0f} / {100*r['light_rate']:.0f}",ha='center',va='center',fontsize=8,color='white' if r['strong_rate']>.65 else 'black')
            ax.set(title=f'{arm}: {height}',xticks=(0,1),xticklabels=('Previous v2','Typed v4'),yticks=np.arange(10),yticklabels=CONDITIONS)
    fig.colorbar(im,ax=axes,label='Strong query-frame rate (%)',shrink=.7)
    fig.suptitle('Consumed dorm pilot S: strong % / exclusive light %; 3 m curtain boundary',fontsize=11)
    fig.savefig(out/'real_v2_v4_comparison.png',dpi=170);plt.close(fig)
    names=('Old default S','Drop bin0 DR + S (v2)','Same candidate typed v4')
    stats=(prior_sim['old_default_S'],prior_sim['no_bin0_S'],simulation['statistics'])
    x=np.arange(3)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for qi,height in enumerate(('HEAD','BODY')):
        y=[100*r[height]['timely']/r[height]['denominator'] for r in stats]
        axes[0].bar(x+(qi-.5)*.35,y,width=.35,label=height)
        for pos,val,r in zip(x+(qi-.5)*.35,y,stats):axes[0].text(pos,val+1,str(r[height]['timely']),ha='center',fontsize=8)
    axes[0].set(title='Timely contact events',ylabel='Timely events (%)',ylim=(0,108));axes[0].legend(fontsize=8,loc='lower left')
    for i,key,title,ylabel in ((1,'far_clear','Far + clear clips with notice','Notified clips (%)'),(2,'weighted_cost','Gap1 joint weighted cost','Cost units')):
        y=[100*r[key]['rate'] if key=='far_clear' else r[key] for r in stats]
        axes[i].bar(x,y,color=('#aa6666','#5599bb','#559977'))
        axes[i].set(title=title,ylabel=ylabel)
        for pos,val in zip(x,y):axes[i].text(pos,val,f'{val:.2f}',ha='center',va='bottom',fontsize=8)
    for ax in axes:ax.set_xticks(x,names,rotation=15,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same consumed simulated hold; v4 input and output closure, no reselection',fontsize=10)
    fig.savefig(out/'sim_v4_preservation.png',dpi=170);plt.close(fig)


def run(out=OUT,report=REPORT):
    began=time.monotonic()
    real=read(out/'real_receipt.json');prior=read(PRIOR/'real_receipt.json')
    sim=read(out/'sim_evaluation.json');prior_sim=read(PRIOR/'simulation_evaluation.json')
    closure=read(out/'sim_input_closure.json');seal=read(out/'v4_input_seal.json')
    assert real['status']=='COMPLETE' and sim['status']=='COMPLETE'
    assert real['input_seal_sha256']==sha(out/'v4_input_seal.json')
    audits={name:read(out/name) for name in ('independent_units.json','independent_audit.json','independent_forward.json','sim_forward_receipt.json','sim_controls_receipt.json')}
    assert all(v['status']=='PASS' for v in audits.values())
    new=index(real['tables']);old=index([r for r in prior['tables'] if r['path']=='v2' and r['arm'] in ('old','main')])
    control_pass=all(new[('main',c,h,m)]['strong']==new[('main',c,h,m)]['light']==0 for c in CONTROLS for h in ('HEAD','BODY') for m in ('M3','old5','S'))
    reminder={c:any(new[('main',c,h,'S')]['strong']+new[('main',c,h,'S')]['light']>0 for h in (('HEAD',) if c=='8' else ('HEAD','BODY'))) for c in ('2','3','5','6','8')}
    accepted=sim['acceptance']
    recommend=bool(control_pass and all(reminder.values()) and all(accepted[k] for k in ('gain_pass','original_default_plus_1pp_pass','timely_small_change_pass','far_clear_small_change_pass')))
    assert audits['independent_audit.json']['independent_recommendation']==recommend
    verdict='去bin0+仪器域随机化骨干+新S满足模拟Development默认候选的替换条件' if recommend else '本轮修复未满足默认替换条件，保留旧候选'
    merged=[]
    for path,source in (('v2_historical',old),('v4',new)):
        for arm in ('old','main'):
            for c in CONDITIONS:
                for h in ('HEAD','BODY'):
                    for m in ('M3','old5','S'):merged.append(dict(source[(arm,c,h,m)],path=path))
    columns=('arm','path','condition','height','model','ready_frames','strong','light','strong_rate','light_rate')
    with (out/'real_v2_v4_full_comparison.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(merged)
    render_figures(out,old,new,sim,prior_sim)
    text=[f'# signed bias / LOCAL输入合同修复\n\n2026-10-11；EXPLORE。\n\n**{verdict}。**']
    text.append('主候选八类确定性无表面构造，M3/原5格/S的HEAD/BODY各9满窗均0强/0轻；因果历史和零证据检查通过。相同已消费模拟hold仍及时净增185/1024，far+clear有通知152/1536，加权成本291.75；输入FP16逐值相同、grade与成本逐值相同，未改任何权重、LOCAL或模型切点。')
    text.append('真实宿舍空通道没有由此安静下来：主v4的4/4b/9 HEAD381/381、BODY373/381强，外侧杆7 HEAD/BODY各146/146强。合同零输入修复与真实sim→real闭合是两个证据层；本次只提升模拟Development候选，真实效果仍待独立录制，不接入App。')
    text.append('继承'+link(ROOT/'research/active/dtr-r0/nearfield/CNH_NO_BIN0_DR_RETRAIN_DEV_20261011.md','193e4a21去bin0重训报告')+'及其冻结权重/切点。该历史报告的v2全链空输入FAIL保持原样；本任务是后续显式typed输入与无证据合同修复，没有重训或据宿舍回放选点。'+link(REPORT.with_name('LOCAL_SIGNED_BIAS_CONTRACT_PLAN_20261011.md'),'本轮PLAN')+'。')
    text.append('## 两种typed入口与因果历史\n\n实机decoded_raw入口将旧signed模板B拆成Bphys=max(B,0)、Bstat=min(B,0)，扣除量min(Bphys,max(H,0))不超过已有非负decoded计数。H=0或H=Bphys得到零；负decoded残差保留，Bstat不再加回分子制造正回波。沿用v2拟合方差和floor作统计分母，不减实际场景，不重新拟合模板。')
    text.append(f"模板仅旧覆盖{seal['template_covered']}/{seal['template_total']} zone×bin；其中非负物理代理正单元{seal['physical_positive_cells']}、signed统计负单元{seal['statistical_negative_cells']}。Bphys是既有录制的非负背景代理，不是已识别的纯串扰；未覆盖保持0/NOT_EVALUABLE，不插补。")
    text.append('模拟training_signed_z入口沿用原(H−Bsim)/sqrt(16A+max(Bsim,0))与FP32→FP16运算。Bsim非负，H为两次Poisson差分，ambient处在方差而非均值；正常逐cell signed负残差保留，不套实机截断，也不做全局ReLU。确切零raw或等于背景的整帧显式编码为零。')
    text.append('mask只继承主候选全部64 zone的bin0删除，旧候选不新增mask。某帧在既有mask之后没有任何>0证据，才编码零native张量；不设幅度、比例、zone数或模型score门槛。在同一sequence/time-reset内，实际最多8帧历史全部无正证据时，最终M3/原5格/S静默，保存未门控分数。历史内任一帧有正证据就运行原链；当前零帧不覆盖先前仍有效的正帧，8帧过期或reset后才清空。几何/coverage/缺失标记仅表示可用性，不算回波；非有限输入拒绝标UNKNOWN，不当clear。')
    text.append('本轮没有完成真实零均值噪声或光子单位识别。实机饱和扣除避免减负bias加正，但在0<H<Bphys时裁掉负残差，可能抬高LOCAL均值；这与保留完整signed随机噪声不等价。LOCAL继续原几何投影patch范数和cut，signal-dependent/相关噪声不在该几何范数中，不能称真实LOCAL已单位方差校准。墙联合拟合噪声R²=0.182496、decoded→模拟单位桥接仍弱；本轮不扩随机化范围、无新噪声拟合。')
    text.append('## 确定性全链控制\n\n每个控制16输入帧、9满8历史；表格cell为“强计数/9；纯轻计数/9”。主八控制×三模型×两高度共432零判定，属重复工程检查，不是432个独立实机样本。')
    rows=[]
    for c in CONTROLS:
        for m in ('M3','old5','S'):rows.append([c,m,*[control_cell(new[(a,c,h,m)]) for a in ('old','main') for h in ('HEAD','BODY')]])
    text.append(table(('控制','模型','旧v4 HEAD','旧v4 BODY','主v4 HEAD','主v4 BODY'),rows))
    text.append('墙2bin0背景与旧v2模板不是同一模板；旧候选保留bin0，残留正信号可能触发，须按上述分支计数描述。主候选bin0本已设计删除，八控制全链静默；不回溯改旧模型或193e4a21的失败。纯ambient为无噪声差分，不是任意随机ambient都严格静默；随机signed噪声可有正证据，不受零证据保证。')
    text.append('历史构造POSITIVE_THEN_ZERO首帧正回波、随后15零帧，evidence_present应为前8帧True、后8帧False；正回波仍在历史时保留原链，不能以当前帧为零压掉已有有效历史。主/旧S在满窗分母中各HEAD/BODY1/9强、0/9轻；该一次响应来自未过期历史，不是控制失败。单位/独立审计逐值核对历史长度、过期及sequence/time reset。')
    text.append('## 同一模拟hold保性能\n\n复用193e4a21的1536物理scene×K2已消费hold，HEAD/BODY接触event各512。新合同不重新选点；先证明全输入/分支与因果门控相同，再复用封存score，另固定首8scene×K2 fresh CUDA前向复核。不能称为新的独立确认。')
    rows=[]
    for name,s in (('旧默认S',prior_sim['old_default_S']),('193e4a21主S/v2',prior_sim['no_bin0_S']),('同主S/typed v4',sim['statistics'])):
        rows.append([name,*[f"{s[h]['timely']}/{s[h]['denominator']}；救{s[h]['rescue']}/损{s[h]['loss']}；净{s[h]['net']:+d}" for h in ('HEAD','BODY')],f"{s['far_clear']['notified_clips']}/{s['far_clear']['denominator']} ({100*s['far_clear']['rate']:.3f}%)",f"{s['weighted_cost']:.2f}"])
    text.append(table(('候选','HEAD及时（vs共同旧原5格）','BODY及时','far+clear有通知clip','加权成本'),rows))
    text.append(f"![相同已消费模拟hold]({(out/'sim_v4_preservation.png').as_posix()})")
    text.append(f"全量表示核对{closure['physical_scenes']} scene×K{closure['replicas']}、{closure['native_frames']} native帧：sim_adapter改变cell {closure['sim_adapter_changed_cells']}，mask后改变cell {closure['masked_input_changed_cells']}，最大绝对差{closure['max_abs_input_difference']:.1f}；{closure['history_slots']}因果历史slot无正证据slot {closure['history_no_positive_slots']}，所以该hold无额外门控。再生grade/通知/加权成本与旧归档一致，cut改变0。{link(out/'sim_input_closure.json')}、{link(out/'sim_forward_receipt.json')}。")
    text.append('预定工程门槛全部通过：共同旧原5格参照下旧S净增170/1024，主S185≥136（保留80%）；far+clear152/1536≤旧168/1536+1pp（最多183）。相对193e4a21及时损失0≤10/1024、far+clear增量0≤15/1536。gap1逐query通知、同帧最高档联合，near-pass≤10cm轻0.25/强1，far/clear任意通知1；不以slot次数代替clip率，不称统计非劣性。')
    text.append('193e4a21主强档HEAD345/512、BODY272/512低于旧原5格389/512、312/512，主S轻档将及时总数补至476/512、410/512。v4输入相同，未恢复这项强档损失；模拟默认候选指完整分级S链，不代表骨干/强档所有指标均提升。暗薄主S60/96、36/96，sign-edge128/128、105/128，均沿用旧主S结果。')
    text.append('## 十段真实已消费pilot：历史v2与typed v4\n\n原1507帧、主分母1362满8历史帧，HEAD/BODY各1362 query-frames。各cell为“强计数/帧（%）；纯轻计数/帧（%）”。旧/主均使用各自冻结权重和切点；v2列取193e4a21历史归档，v4为本任务新回放，历史不改。')
    text.append(f"![已消费真实pilot合同前后S]({(out/'real_v2_v4_comparison.png').as_posix()})")
    for m in ('S','M3','old5'):
        for h in ('HEAD','BODY'):
            rows=[[c+' '+SCENES[c],cell(old[('old',c,h,m)]),cell(old[('main',c,h,m)]),cell(new[('old',c,h,m)]),cell(new[('main',c,h,m)])] for c in CONDITIONS]
            text.append(f'### {m} / {h}\n\n'+table(('条件','历史旧v2','历史主v2','旧typed v4','主typed v4'),rows))
    text.append('全部旧新M3/原5格/S强轻表见'+link(out/'real_v2_v4_full_comparison.csv')+'；本轮完整控制/真实表见'+link(out/'real_alarm_table.csv')+'。4/4b/9及4–9的尽头幕布约在3m窗边缘，是边界场景，不能把这组强报警率当实机误报率。1阳台UNKNOWN；7外侧杆只描述。墙2/3、杆5/6、8HEAD保留至少一个满窗提醒只是响应存在性，不代表稳定实机检出能力；低BODY响应须看完整分母。')
    text.append('主v4墙2/3 HEAD/BODY各146/146强；近/远杆5/6 HEAD仍146/146强、BODY无强，仅各1/146轻；8 HEAD146/146强。主S阳台HEAD/BODY0/105。真实空通道与外侧7仍近全强，不宣称sim→real闭合；下一独立clear录制应选尽头>4.5m楼道，同时补墙/细杆，保留旧新链并行对照。')
    text.append('## 核验、默认候选与证据界限\n\n'+verdict+'。'+ '；'.join(f'{name} {v["status"]}' for name,v in audits.items())+'。')
    text.append('独立单位核验、全量模拟FP16/grade/成本复算、真实逐query计数/模板/因果历史、固定fresh CUDA窗口全部完成。独立CPU审计覆盖50,331,648模拟cell/49,152帧、79,872通知slot、228真实表行/38输出及3,381,248真实重建cell；审计单独得出recommend=True。没有保护480/test访问、训练或切点改变。审计材料：'+'、'.join(link(out/name) for name in audits)+'。')
    promotion=out/'promotion_receipt.json'
    if promotion.exists():
        promoted=read(promotion)
        assert promoted['status']=='PROMOTED_SIM_DEVELOPMENT_DEFAULT' and promoted['all_gates_pass']==recommend
        text.append('已按预定条件替换模拟Development默认引用：'+link(ROOT/'research/active/dtr-r0/nearfield/CNH_SIM_DEFAULT_CANDIDATE.json','tracked默认指针')+'→'+link(out/'candidate_manifest.json','v4候选manifest')+'。'+link(promotion,'promotion receipt')+'状态PROMOTED_SIM_DEVELOPMENT_DEFAULT；权重和切点未改变，只绑定已封存去bin0+DR+新S与typed v4输入合同。App与正式实机结论不随之升级。')
    else:text.append('默认候选引用与manifest更新由主任务按上述工程条件完成；模拟默认引用不等于App部署，也不是独立实机性能或安全确认。')
    budget=out/'budget_summary.json'
    if budget.exists():
        b=read(budget);text.append(f"预算保守上界：GPU command-wall {b['gpu_command_wall_seconds']:.3f}/300 s，CPU科学计算/评价command-wall {b['cpu_command_wall_seconds']:.3f}/1200 s。失败/启动计费、内部receipt与wrapper外部wall不重复加；轻量报告绘图按主任务reserve/汇总计入。{link(budget)}。")
    text.append('保留开发失败及机械恢复receipt：首次fresh模拟前向因CPU输出仍绑定旧adapter hash被拒绝（5.866s），旧四份CPU输出在同OUT保留_pre_final_adapter后缀，再在最终封存adapter上全量CPU核对（12.666s）和fresh GPU（9.336s）。独立CPU审计自身曾误用strict-pilot路径及CSV编码名utf8-sig，分别更正为relaxed-pilot与utf-8-sig后重算通过，两次失败receipt保留。最终输入差异0、全分支差异0，不以旧hash输出冒充最终复核；失败计费，科学结果、模型与cut不变。实际随机噪声、多径、未覆盖空间背景、decoded噪声单位与LOCAL真实均值仍待独立识别；本次保证的是无正证据历史静默及既定模拟观测上的行为保留。未来若需改变LOCAL尺度或cut，应另用模拟cal whole-tie成本协议，不按宿舍输出选点。')
    text.append('## 复现与绑定\n\n'+link(out/'v4_input_seal.json')+'、'+link(out/'v4_template.npz')+'、'+link(PRIOR/'main_real_manifest.json')+'。')
    text.append(table(('绑定','SHA256'),[(k,v) for k,v in seal.items() if k.endswith('sha256')]))
    text.append('adapter/replay封存代码保持不变：'+link(Path(__file__).with_name('local_signed_bias_adapter.py'))+'、'+link(Path(__file__).with_name('local_signed_bias_replay.py'))+'。报告生成器：'+link(Path(__file__))+'；运行只读取已完成receipt并绘图，不训练、拟合或评分。')
    report.write_text('\n\n'.join(text)+'\n',encoding='utf8')
    receipt=dict(status='COMPLETE',seconds=time.monotonic()-began,recommend=recommend,verdict=verdict,
        deterministic_controls_pass=control_pass,reminder_present=reminder,report_sha256=sha(report),
        figures_sha256={n:sha(out/n) for n in ('real_v2_v4_comparison.png','sim_v4_preservation.png')},
        complete_comparison_sha256=sha(out/'real_v2_v4_full_comparison.csv'),source_sha256=sha(__file__),training=0,cut_changes=0)
    (out/'report_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--report',type=Path,default=REPORT);a=p.parse_args();run(a.out,a.report)
