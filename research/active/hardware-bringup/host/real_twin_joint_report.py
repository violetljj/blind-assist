"""Figures and descriptive joint calibration readout, no selection/refit."""
import csv,json
from pathlib import Path
import numpy as np
import real_twin_joint_calibration as J
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def pair(r,arm='old5',field='strong_rate'):return '/'.join(f'{100*q[field]:.1f}' for q in r['arms'][arm])
def interval(x):return '['+', '.join(f'{v:.4f}' for v in x)+']' if x is not None else 'NOT_EVALUABLE'
def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    instrument=np.load(J.OUT/'instrument_fit_arrays.npz');params,bg=J.instrument();bg=bg.reshape(64,16)
    for sid in ('2','3','4','4b','9','5','6','7','8'):
        if sid in ('2','3'):
            obs=instrument[f'wall{sid}_mean'];pred=instrument[f'wall{sid}_prediction']
        else:
            with np.load(J.OUT/(f'c{sid}_centre_input.npz')) as d:obs=d['real_mean'];pred=d['expected']
        signal=obs-bg;synthetic=pred-bg;res=pred-obs;lim=max(np.percentile(abs(signal),99),np.percentile(abs(synthetic),99),1)
        fig,axs=plt.subplots(2,3,figsize=(13,7))
        for ax,x,name in zip(axs[0],[signal,synthetic,res],['Real - shared bin0 background','Twin - shared bin0 background','Twin - real']):
            im=ax.imshow(x,aspect='auto',cmap='coolwarm',vmin=-lim,vmax=lim);ax.set(title=name,xlabel='Aggregated bin',ylabel='Raw zone');fig.colorbar(im,ax=ax)
        axs[1,0].plot(obs.mean(0),label='real raw');axs[1,0].plot(pred.mean(0),label='twin raw');axs[1,0].legend();axs[1,0].set_title('Raw mean: bin0 can dominate L2')
        axs[1,1].plot(signal.mean(0),label='real signal');axs[1,1].plot(synthetic.mean(0),label='twin signal');axs[1,1].legend();axs[1,1].set_title('Signal mean')
        axs[1,2].bar(range(16),abs(res).sum(0));axs[1,2].set_title('Absolute residual sum / bin')
        fig.suptitle(f'Condition {sid}: fixed raw 180-degree contract, weak geometry');fig.tight_layout();fig.savefig(J.OUT/(sid+'_residual.png'),dpi=140);plt.close(fig)
    f=J.read('geometry_fit.json');fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,name in zip(axs,['pitch_deg','curtain_distance_m']):
        rows=sorted(f['profiles'][name]['grid'],key=lambda q:q['value']);ax.plot([q['value'] for q in rows],[q['objective'] for q in rows],'o-');ax.axhline(1.05*f['objective'],color='red',linestyle='--');ax.set(xlabel=name,ylabel='Conditional weighted objective',title='Descriptive profile, not confidence interval')
    fig.tight_layout();fig.savefig(J.OUT/'geometry_profiles.png',dpi=140);plt.close(fig)

def main():
    plot();i=J.read('instrument_fit.json');refined=J.read('instrument_profile_refined.json');n=J.read('noise_fit.json');g=J.read('geometry_fit.json');inputs=J.read('closure_inputs.json');r=J.read('closure_replay.json');comp=J.read('component_report.json')
    # Store complete alarm tables including all position scan cells and paths.
    rows=[]
    for domain,datasets in [('real',r['real']),('twin',r['twin'])]:
        for key,paths in datasets.items():
            for path,result in paths.items():
                for arm,queries in result['arms'].items():
                    for q in queries:rows.append(dict(domain=domain,cell=key,path=path,arm=arm,ready=result['ready'],**q))
    with (J.OUT/'alarm_table.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    centre=[c for c in inputs['cells'] if c['position']=='centre'];wall=np.load(J.OUT/'instrument_fit_arrays.npz');bg=wall['bias']
    wall3_signal=float(np.linalg.norm(wall['wall3_prediction']-wall['wall3_mean'])/np.linalg.norm(wall['wall3_mean']-bg))
    decisions=[]
    for c in centre:
        sid=c['condition'];real=r['real'][sid]['v2'];twin=r['twin'][c['key']]['v2'];differences={}
        for arm in ('m3','old5','Sensemble'):
            differences[arm]=dict(strong_pp=[100*(b['strong_rate']-a['strong_rate']) for a,b in zip(real['arms'][arm],twin['arms'][arm])],light_pp=[100*(b['light_rate']-a['light_rate']) for a,b in zip(real['arms'][arm],twin['arms'][arm])])
        passes=dict(histogram=c['metrics']['signal_relative_l2']<=.35 and c['metrics']['weighted_rmse']<=2,
                    alarms=all(abs(x)<=15 for a in differences.values() for x in a['strong_pp']) and all(abs(x)<=10 for x in differences['Sensemble']['light_pp']))
        decisions.append(dict(condition=sid,metrics=c['metrics'],alarm_differences=differences,passes=passes))
    closed=wall3_signal<=.35 and all(all(c['passes'].values()) for c in decisions)
    J.save('closure_decision.json',dict(closed=closed,wall3_signal_relative_l2=wall3_signal,conditions=decisions,rule='PLAN: histogram<=.35 and WRMSE<=2, strong discrepancy<=15pp each arm/query, S light<=10pp'))
    root='../../../artifacts.local/work/real-twin-joint-calibration-dev-20261011/'
    lines=['# 实机–孪生联合仪器/几何校准', '', '2026-10-11 · EXPLORE · 现有宿舍pilot复用 · 不重训', '',
    '**尚未闭合；不能写“主要差距可由零偏/脉冲/增益/噪声解释”。** 墙2拟合与墙3外推的峰形/幅值较好，但空通道和杆段的背景扣除后直方图仍严重不一致，M3 BODY也有较大差异。原5格/S的强档全部100%看似一致，是饱和读出，并非物理闭合证明。', '',
    f'墙3拟合外仪器检验signal L2={wall3_signal:.4f}（全raw L2={i["wall3_check"]["relative_l2"]:.4f}）通过预写.35描述门槛；7个空/杆中心的signal L2均>.35。4/4b/9原raw L2仅约4–9%，因bin0背景占主能量，扣除共同背景后残差约100%。最终主几何点pitch={g["parameters"]["pitch_deg"]:.3f}°、幕距={g["parameters"]["curtain_distance_m"]:.3f}m在下界，幕距剖面容许整个2.8–4.5m，**真实尽头位置与零偏仍不可独立确定**。', '',
    '## 数据分工、协议与读出', '',
    '[PLAN](REAL_TWIN_JOINT_CALIBRATION_PLAN_20261011.md)先写；墙2稳定152帧fit、墙3稳定152帧仅几何check，仪器在开墙3前已保存；4/4b noise，4/4b/9几何；5/6/7/8仅evaluation。全64zone×16bin保留，无模型报警拟合，无阈值/权重改动，不读保护480/test。所有已有帧已消费，本次是描述性复用；空段是拟合内，杆段仅本轮拟合外，不称独立实机性能确认。', '',
    '物理forward采用冻结tier6能量606.1126、ρ cos/r²、4邻leak.02；offset移动radial，Gaussian＋单向exp混合核连续聚合积分，不折回窗外能量。墙ρ=.5锚定gain；背景仅墙2profile出的64个bin0非负值，其他bins为0，不借v3，不减整场景均值。拟合4×4ray、最终16×16，墙积分差0.0131%、主几何积分差'+f'{g["quadrature_4_to_16_relative_l2"]*100:.3f}%'+ '。墙权重含经验均值方差与5%response discrepancy floor，先验单独附加。脉冲尾是弱proxy，非真实完整脉冲响应测量。', '',
    '主报警：真实和合成decoded CNH共用继承v2 adapter、raw顺序/旧−10°/恒等历史，合成ray物理姿态为拟合pitch。固定180°粗映射只用于生成/对比raw，非metric外参；不会按结果挑重排。辅助joint路径两端共用新background/新噪声底，不反卷积，不选好看路径替代主结果。每合成cell三seed×16帧/full8分母27，真实沿用实际seq/time重置。Gaussian有效噪声是decoded proxy，不伪称Poisson光子模型已校准。', '',
    '首轮误把上轮工程扫描55cm作为现场横向先验，拟合g约55cm；原完整输出、23cell回放、18组件、审计与PLAN快照保留于broad-gap-sensitivity。来源复核后，最终主先验按记录改为20–30cm，仅用同直方图目标重做geometry/closure；墙仪器/噪声不改，杆结果不回流。宽范围分支的幕距3.078m/pitch−.815°与最终主不相同，说明几何不稳定；不把前分支当真实测量或删除失败。', '',
    '## 仪器参数与不确定性', '',
    table(['参数','墙2有效拟合','5%剖面容许范围/局限'],[[k,f'{v:.6f}',interval(refined['profiles'][k]['descriptive_5percent_set'])] for k,v in i['instrument_parameters'].items()]), '',
    '零偏−4.800mm，Gaussian σ91.465mm（FWHM约215mm）、尾比例2.52%、尾尺度.50m上界、有效gain.532312。零偏粗剖面−.10…+.10m；σ局部范围.087965–.094965m。tail scale局部.4765–.50不能当完整区间，因为尾弱/上界；gain粗+局部容许.4167–.7333。ρ不可独立辨识，不能直接对比旧ρ=.3条件下gain=.749185而说硬件增益下降。', '',
    f'墙2d={i["wall2_geometry"]["wall_distance_m"]:.6f}m、normal pitch/yaw={i["wall2_geometry"]["normal_pitch_deg"]:.3f}/{i["wall2_geometry"]["normal_yaw_deg"]:.3f}°；d+offset=.667381m。墙距与零偏Jacobian相关−.999341，与gain也高度相关，先验改变会平移解，不能声称δ≈0已测准。3独立几何d={i["wall3_check_geometry"]["wall_distance_m"]:.6f}m、法向={i["wall3_check_geometry"]["normal_pitch_deg"]:.3f}/{i["wall3_check_geometry"]["normal_yaw_deg"]:.3f}°，没有重拟合仪器。法向包括墙姿态与佩戴方向，不能代替空段sensor俯角。', '',
    table(['条件','全raw L2','signal L2（扣同背景）','weighted RMSE'],[['2',f'{i["wall2"]["relative_l2"]:.5f}',f'{np.linalg.norm(wall["wall2_prediction"]-wall["wall2_mean"])/np.linalg.norm(wall["wall2_mean"]-bg):.5f}',f'{i["wall2"]["weighted_rmse"]:.3f}'],['3 check',f'{i["wall3_check"]["relative_l2"]:.5f}',f'{wall3_signal:.5f}',f'{i["wall3_check"]["weighted_rmse"]:.3f}']]), '',
    '墙2bin0背景min/median/max约17.614/58.400/121.094 decoded；只来自近墙，可能有early pulse/多径/固件处理，不等于纯串扰。profiler容许目标≤1.05×最小值，粗/局部conditional grid及相关矩阵/SVD全部保留；区间不是95%后验/置信区间。', '',
    '## 噪声与ambient', '',
    f'38912稳定窗-cell，Var={n["coefficients_a_b_c"][0]:.6f}·positive_signal+{n["coefficients_a_b_c"][1]:.6f}·ambient+{n["coefficients_a_b_c"][2]:.6f}；R²={n["fit"]["r_squared"]:.6f}、relative RMSE={n["fit"]["relative_rmse"]:.4f}。4单段R²={n["segment"]["4"]["r_squared"]:.4f}，4b={n["segment"]["4b"]["r_squared"]:.4f}；奇偶窗外推R²分别'+ '/'.join(f'{q["test"]["r_squared"]:.4f}' for q in n['odd_even_window_check'])+'。numerical floor仅防数值除零。', '',
    '重拟合包含所有稳定cell，与旧排除固件峰附近/ambient不稳cell的31934样本不同；旧R²=.5337不能作为同样本优劣比较。新signal系数奇偶训练0/.25357漂移，暗段方差几乎未解释，表明简单a/b/c未闭合。模拟器16×ambient+bias是光子单位，当前b≈.08988为decoded单位，不能直接数字替换；还缺单位映射、空间/跨bin相关噪声与固件处理。', '',
    '## 空通道几何与剖面', '',
    table(['参数','最终主中心','5%剖面容许','边界纳入'],[[k,f'{v:.6f}',interval(g['profiles'][k]['descriptive_set']),str(g['profiles'][k]['bound_included'])] for k,v in g['parameters'].items()]), '',
    'roll固定0、高1.55、ceil2.8；主要物体形状来自记录约束AABB假设。最终主幕距到下界、effective幕ρ≈.022很低且下界也容许；左右间隙整个记录范围都容许。pitch条件剖面较窄，但同ρ/背景/零偏/几何形状纠缠，不能称真实佩戴俯角测准；没有把仪器全profile传播成完整几何后验。仍需未知多径、side geometry、raw外参等解释。', '',
    f'![几何条件剖面]({root}geometry_profiles.png)', '',
    '## 拟合外杆段直方图与报警闭合', '',
    table(['条件','raw L2','signal L2','weighted RMSE','最大cell zone/bin：decoded差','直方图门槛'],[[c['condition'],f'{c["metrics"]["raw_relative_l2"]:.3f}',f'{c["metrics"]["signal_relative_l2"]:.3f}',f'{c["metrics"]["weighted_rmse"]:.3f}',f'{c["metrics"]["top_cells"][0]["zone"]}/{c["metrics"]["top_cells"][0]["bin"]}: {c["metrics"]["top_cells"][0]["difference"]:.2f}','PASS' if c['passes']['histogram'] else '未闭合'] for c in decisions]), '',
    '主v2路径，强/轻单位%且H/B分开：', '',
    table(['条件','real ready','twin ready','M3 real强','M3 twin强','原5格/S real强','原5格/S twin强','S real轻','S twin轻'],[[sid,r['real'][sid]['v2']['ready'],r['twin'][f'c{sid}_centre']['v2']['ready'],pair(r['real'][sid]['v2'],'m3'),pair(r['twin'][f'c{sid}_centre']['v2'],'m3'),pair(r['real'][sid]['v2']),pair(r['twin'][f'c{sid}_centre']['v2']),pair(r['real'][sid]['v2'],'Sensemble','light_rate'),pair(r['twin'][f'c{sid}_centre']['v2'],'Sensemble','light_rate')] for sid in r['real']]), '',
    'M3/原5格轻档均0。主旧5格/S强的全100%一致并没有消除直方图或分支差异；尤其M3 BODY在5/6/9差异显著，超过15pp。本轮不挑位置或输入路径救主结果，0报警取消不等于物理解释成立。', '',
    '每杆中心＋4个预定位置扫描，signal残差和M3强率包络：', '',
    table(['条件','signal L2中心 / 全5位置min–max','M3 twin强HEAD min–max %','BODY min–max %'],[[sid,f'{next(c["metrics"]["signal_relative_l2"] for c in centre if c["condition"]==sid):.3f} / '+f'{min(c["metrics"]["signal_relative_l2"] for c in inputs["cells"] if c["condition"]==sid):.3f}–{max(c["metrics"]["signal_relative_l2"] for c in inputs["cells"] if c["condition"]==sid):.3f}']+[f'{min(r["twin"][c["key"]]["v2"]["arms"]["m3"][q]["strong_rate"] for c in inputs["cells"] if c["condition"]==sid)*100:.1f}–{max(r["twin"][c["key"]]["v2"]["arms"]["m3"][q]["strong_rate"] for c in inputs["cells"] if c["condition"]==sid)*100:.1f}' for q in (0,1)] for sid in ('5','6','7','8')]), '',
    '辅助joint-noise路径只作读出敏感性：', '',
    table(['条件','M3 real/twin HEAD/BODY强%','原5格/S real强%','原5格/S twin强%','S real轻%','S twin轻%'],[[sid,pair(r['real'][sid]['joint'],'m3')+' / '+pair(r['twin'][f'c{sid}_centre']['joint'],'m3'),pair(r['real'][sid]['joint']),pair(r['twin'][f'c{sid}_centre']['joint']),pair(r['real'][sid]['joint'],'Sensemble','light_rate'),pair(r['twin'][f'c{sid}_centre']['joint'],'Sensemble','light_rate')] for sid in r['real']]), '',
    '减除完整墙2bin0背景并换新noise floor后，4真实原5格仍100/100%，孪生仅14.8/11.1%；7真实100/100%，孪生22.2/11.1%。因此继承v2的背景支持是一个强报警来源，但补这一项后仍有明显真实/合成分歧，不能把“背景only在旧合同下全报”写成所有真实强报警的单一物理原因。', '',
    '## 窗边缘与组件充分性（固定拟合参数后追加）', '',
    table(['条件','组件','M3强H/B %','原5格=S强H/B %','S轻H/B %'],[[c['condition'],c['component'],pair(c['counts'],'m3'),pair(c['counts']),pair(c['counts'],'Sensemble','light_rate')] for c in comp['rows']]), '',
    '组件不是新fit：幕4.8、offset0、nominal核proxy、mean-only、幕移出＋mean-only、背景only及已容许幕距剖面端点，全部保留历史/zone/阈值。若幕移出后仍强，则“幕在边缘＋脉冲”不是本合同中全报警的必要解释；若背景only已强，则说明墙2模板经继承适配器/冻结表示具有充分触发支持，仍不能把近墙背景认定为真实纯仪器项。', '',
    '## 残差图与剩余物理项', '',
    '\n\n'.join(f'![条件{sid}逐zone/bin残差]({root}{sid}_residual.png)' for sid in ('2','3','4','4b','9','5','6','7','8')), '',
    '残差最大的bin/zone详见closure_inputs/decision；应优先看bin0空间变化、4/4b bin7/10/11弱次返回、5杆bin3支持、6的远bin、8的近bin2大回波。单平面核的良好墙外推不覆盖宿舍多径：墙/地面/侧物二次路径、空间非均匀串扰随ambient/启动变化、固件去背景/峰选/累加与量化都可能缺失。8记录位置与实际近峰也未独立确认；不据此擅自移动横杆到实测峰来提高相似度。', '',
    '宿舍空通道有远bin10/11（约3.0–3.6m）回波，尽头在3m窗边缘是合理候选解释，并非干净clear。但幕距剖面平坦/边界，不能把本次点估计2.8或前分支3.08当测量真值。未来clear应选尽头>4.5m楼道并记录姿态；即使如此仍需独立模板/噪声合同，不能承诺换场景就无报警。', '',
    '## 模拟器更新与重训草案（未执行）', '',
    '优先顺序：①空间/ambient/启动相关的signed背景及固件输出链、raw外参合同；②decoded噪声单位、跨zone/bin相关及head-motion有效方差；③物理多径与side/幕/杆支持；④已拟合range/脉冲/gain作为有先验的随机参数。这里仍未闭合，不能用当前高幅bin0模板原样写入模拟器并声称已识别串扰。', '',
    '域随机化候选：offset先保留−.10…+.10m先验约束范围，联合使用剖面条件解（同时变gain/墙距相关项，不能仅独立δ±.1）；σ≈.088–.095m，tail fraction约.025–.030，tail scale至少保留.03–.50全弱先验以反映不可辨识，gain≈.417–.733（ρ=.5锚定）。噪声a使用奇偶0… .254范围，b约.087… .094，c约.58… .69作有效单位开发压力候选，必须先检查生成方差/相关性，不作为已校准物理范围。幕2.8–4.5、g20–30cm、pitch全原−25…+5含对照及局部0–2.1不确定集；不要用窄profile假装覆盖真实姿态。', '',
    '具体路线：先完善仪器/固件/多径生成器并复核新合成对真实墙与新clear的分布；固定新train/cal/hold场景和上述随机化绑定，再重训骨干及强档读出，用cal封存阈值后一次hold比较空通道/窗内墙杆/横杆及姿态成本。当前M3/原5格/S保留冻结对照。新骨干强档稳定后再重拟合S轻档，报告救/损与通知成本；单独微调S不能撤继承强档。本轮不执行训练或切点选择，原十段不得重新命名为新确认。', '',
    '## 独立核验、成本与证据', '',
    '独立实现erf聚合核/邻格leak、背景/墙3signal残差、scipy NNLS噪声/奇偶窗、geometry/生成均值方差/固定seed、强轻计数与CUDA fresh短窗；最终主与broad分支审计分开。详见INDEPENDENT_AUDIT.md。工程失败：generate数组truth判据错误、随后提前启动replay缺输入，均保留failed receipt，修复后依赖顺序运行；独立噪声计数变量覆写的审计失败亦留记录，不改变科学参数。', '',
    '预算含两几何分支、墙profile、所有GPU/独立核验/失败/绘图读写，保守CPU≤400/1800、GPU≤100/600累计command-wall s；GPU command整个wall亦计CPU。模型finally释放，无新硬件进程或网络录制。', '',
    '\n'.join(f'- [{name}]({root}{name})' for name in ['instrument_fit.json','instrument_profile_refined.json','instrument_parameters_frozen.json','instrument_fit_arrays.npz','noise_fit.json','geometry_fit.json','closure_inputs.json','closure_replay.json','closure_decision.json','alarm_table.csv','component_report.json','INDEPENDENT_AUDIT.md','independent_joint_cpu_audit.json','independent_joint_cuda_audit.json','closeout_receipt.json','broad-gap-sensitivity/PLAN_snapshot.md']), '',
    '代码：[墙/通用forward](host/real_twin_joint_instrument.py)、[噪声/几何/闭合](host/real_twin_joint_calibration.py)、[组件](host/real_twin_joint_components.py)、[表图/报告](host/real_twin_joint_report.py)。继承[宿舍孪生判别](DORM_TWIN_DISCRIMINATION_DEV_20261011.md)。', '',
    '局限：完整照片、独立距离/姿态/外参缺失；有效参数与ρ/几何共线；profile不是CI，部分平坦/边界；背景源于近墙可能有物理返回；空段拟合内且帧/历史相关；Gaussian和AABB不是完整固件光学；杆位置只记录估计；报警饱和能掩盖输入差异。全部结论为pilot机制诊断，非实机检出率/误报率、安全或迁移性能结论。', '']
    path=Path(__file__).resolve().parents[1]/'REAL_TWIN_JOINT_CALIBRATION_DEV_20261011.md';path.write_text('\n'.join(lines),encoding='utf8');print(path)
if __name__=='__main__':main()
