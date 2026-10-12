"""Readable report from fixed OOF results; no fitting or threshold selection."""
import json
from pathlib import Path
import numpy as np
from sync_fusion_robust_dev import ROOT, PLAN, ARMS, MODELS, FEATURES, BANDS, sha

REPORT=Path(__file__).with_name('SYNC_FUSION_ROBUST_DEV_20261011.md')
LABELS={ARMS[0]:'半循环',ARMS[1]:'rho .15',ARMS[2]:'rho .6',ARMS[3]:'ambient ×3',ARMS[4]:'过门 FARO',ARMS[5]:'ambient ×10'}
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def interval(c):return 'NA' if c['lo'] is None else f"[{c['lo']:.2f}, {c['hi']:.2f}]"
def ratio(v):return 'N/E' if v is None else f'{100*v:.1f}%'
def ratio_interval(c):return 'NA' if c['lo'] is None else f"[{100*c['lo']:.1f}%, {100*c['hi']:.1f}%]"
def main():
    s=load(ROOT/'summary.json');p=load(PLAN);audit=load(ROOT/'independent_audit.json');resources=load(ROOT/'resource_receipt.json')
    assert audit['status']=='PASS';M={(r['fold'],r['arm'],r['band'],r['model']):r for r in s['metrics']};P={(r['fold'],r['arm'],r['band'],r['model'],r['baseline']):r for r in s['paired']}
    lines=['# 融合器稳健性：混合 ToF 质量训练（EXPLORE）','',
      '执行日期：2026-10-11；输入基线 `1d16bde2`。全部30个 visit 已消费 Development，本报告不改变融合确认 v2 的结论。', '',
      '## 结果与下一轮选择','',
      '| 融合器 | FARO近带净增 vs RGB | 压力近带净增 vs RGB | 半循环中带净增 vs 最佳单源 | 相对A保留率 | 原计数稳健候选 |',
      '| --- | ---: | ---: | ---: | ---: | --- |']
    original_candidates=[]
    for r in s['candidates']:
        passed=r['model']!='A' and r['FARO_near_net']>=0 and r['stress_near_net']>=0 and r['native_middle_net']>=.8*r['A_middle_net']
        original_candidates.append(dict(model=r['model'],original_count_candidate=passed,retention_status='NOT_EVALUABLE' if r['retention'] is None else 'EVALUABLE'))
        lines.append(f"| {r['model']} | {r['FARO_near_net']:+d} | {r['stress_near_net']:+d} | {r['native_middle_net']:+d} | {ratio(r['retention'])} | {'是（比例N/E）' if passed and r['retention'] is None else '是' if passed else '否' if r['model']!='A' else 'A对照'} |")
    lines+=['','净增为POS命中数（救−损），不是相对全部query的正确率。保留率使用本CV的A与同一个固定最佳单源，分母不是v2的+385。按用户原始计数描述 Δ模型≥0.8×ΔA，B/C/D同时满足两条；但A合计净增−122，80%计数下界为−97.6，成为空约束，**不能据此声称保住80%的正收益**。保留率N/E。',
      '事先PLAN额外规定A≤0时比例N/E且不登记有效保留型候选；原始summary仍保留False。这里在结果后明确报告用户原始计数规则和PLAN比例保护的差别，是判读澄清，没有改训练/阈值/数据。两种记录均不意味着统计非劣或FREE已达标。','',
      '| 臂/带 | 融合器 | W/POS | F/FREE | RGB救/损 | 净增95%区间(pp) |',
      '| --- | --- | --- | --- | --- | --- |']
    for a,b in ((ARMS[4],0),(ARMS[5],0),(ARMS[0],1)):
        for model in 'ABCD':
            r=M[-1,a,b,model];x=P[-1,a,b,model,'rgb'];lines.append(f"| {LABELS[a]} / {BANDS[b]} | {model} | {r['W']}/{r['POS']} | {r['F']}/{r['FREE']} | {x['rescue']}/{x['loss']} | {interval(x['net_pp_ci'])} |")
    recommendation=load(ROOT/'recommendation.json')
    lines+=['',recommendation['text'],'','## 固定 PLAN、数据与公平比较','',
      f"[PLAN](SYNC_FUSION_ROBUST_PLAN_DEV_20261011.json) SHA `{sha(PLAN)}` 在构建和拟合前写定。12旧visit（train6/cal2/eval4）与18个v2 visit（cal6/eval12）合并；960真实RGB帧、每帧27query、每query K0/K1聚为一行。外层5折各6visit，内层cal5/train19，5/24为1/5的整数近似。随机种子20261011，内cal seed2026101100+fold；不按标签/结果分层。",'',
      '| 折 | 外层评价 visit | 内层 cal visit | 训练visit数 |','| --- | --- | --- | ---: |']
    for f in p['data']['folds']:lines.append(f"| {f['fold']} | {', '.join(f['eval'])} | {', '.join(f['cal'])} | 19 |")
    lines+=['','完整训练名单也在PLAN；payload `folds.csv`列出每折所有角色。所有评价query只出现于一个外层fold。', '',
      '半循环保持冻结native扰动和残差池。rho .15/.6、ambient×3沿用v1.1的FARO几何质量臂：覆盖不足及缺pose帧仍保留为UNKNOWN输入；只有FARO主臂按既有K0 joint_pass_zones≥52筛帧。新压力臂rho .3/ambient×10从同一FARO期望生成，不native填洞或人为dropout。已有臂逐字节复用；v2缺失的rho/ambient臂按缓存FARO rho .3期望乘0.5/2、背景乘1/3/10，用相同帧/K的seed和冻结Poisson差采样器补齐。45°FOV、8×8×16、bin约0.3002784m与16×16子射线、SNR3/coverage.75/16th余量读出不变。','',
      'A使用半循环POS/FREE训练；B/C/D混合所有可用臂，UNKNOWN不拟为负例。每臂已知行总权重=Nknown/6，行权重=Nknown/(6×本臂known数)，均值1；FARO较少但不欠权。Logistic保持C1/lbfgs/max1000、无class_weight，StandardScaler仅fit本折train known，B/C同权重标准化。D保持v1 HGB100轮/lr.1/7叶/深3/minleaf50/L2=1/不早停。各955/956/957概率平均；确定性种子可相同，不是独立证据。混合行数与权重和较A改变正则化相对数据规模，B与A也同时改变来源构成，不能当纯物理噪声因果隔离。','',
      '基础20维与v2一致。C仅额外加入ToF模型余量×UNKNOWN比例、×支持像素射线数、×峰SNR均值三个交互；峰SNR由当前hist/background同冻结公式计算，先64zone均值再K均值，未把质量单独加入基础特征。模型缺测余量−10带标志、米量clip[-10,10]；ToF-only保留原始−∞缺测。支持数为投影像素射线数。无标签/参考深度/几何真值/期望值/visit身份特征。','',
      '所有六臂cal行按原始query数混合，逐折/方法/带只一个阈值；近2%/中5%/远10% FREE预算，枚举所有有限ties及+∞，依次最大W、最小F、最高阈值，FREE或POS零分母则拒绝全部。RGB-only（近Uni/中远DAV）和ToF-only也用相同混合cal与跨臂单阈值。冻结的是传感器读出；本任务的融合及单源比较阈值重新CV校准。因此既有v2独立分源切点的数字不可直接纵比。', '',
      '## 压力强度与覆盖','',
      '| 臂 | 帧 | query行 | ToF余量缺测比例 | 平均过门zone比例 |',
      '| --- | ---: | ---: | ---: | ---: |']
    with np.load(ROOT/'features.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    for a in ARMS:
        mask=d['arm']==a;lines.append(f"| {LABELS[a]} | {len(set(d['frame_id'][mask]))} | {int(mask.sum())} | {100*d['X'][mask,1].mean():.2f}% | {100*d['accepted_zone_fraction'][mask].mean():.2f}% |")
    # Compare stress severity on identical frames, never assuming complete sensor failure.
    lines+=['','压力臂仅模拟10倍背景光；上表实测UNKNOWN/过门率决定退化程度，不能把名称当作基本失效证据。FARO主臂分母受固定门槛筛选，不能与其他全帧臂直接横向比较来源优劣。', '',
      '## 质量特征是否降低 ToF 权重','',
      'Logistic B没有交互，UNKNOWN等质量项能下调总log-odds，但ToF分数偏导仍为固定系数，不能称自适应降权。C可计算原尺度有效偏导 βToF+β交互U×U+β交互支持×支持+β交互质量×质量；其方向见下表。D的特征重要性仅说明使用了质量，局部响应只是观测关联，均不单独证明真实传感器退回RGB。','',
      '| 模型 | 项 | 标准化系数折均值 | 原尺度系数折均值 |','| --- | --- | ---: | ---: |']
    co=load(ROOT/'coefficients.json')
    for model in 'ABC':
        names=['tof_margin16_m','tof_unknown_fraction','tof_support_pixels']+(list(p['features']['C_extra']) if model=='C' else [])
        for name in names:
            rows=[r for r in co if r['model']==model and r['feature']==name];lines.append(f"| {model} | {name} | {np.mean([r['standardized_coefficient'] for r in rows]):.5g} | {np.mean([r['raw_coefficient'] for r in rows]):.5g} |")
    lines+=['','| 折 | C原尺度ToF有效偏导中位数 | UNKNOWN低→高（其他质量固定中位） | 支持低→高 | 峰SNR低→高 |','| --- | ---: | --- | --- | --- |']
    with np.load(ROOT/'oof.npz',allow_pickle=False) as z:foldids=z['fold']
    for f in range(5):
        beta={n:np.mean([r['raw_coefficient'] for r in co if r['model']=='C' and r['fold']==f and r['feature']==n]) for n in ['tof_margin16_m',*p['features']['C_extra']]}
        mask=foldids==f;uu=d['X'][mask,3];ss=d['X'][mask,2];qq=d['peak_quality'][mask];med=[np.median(v) for v in (uu,ss,qq)]
        b0=beta['tof_margin16_m'];bb=np.array([beta[n] for n in p['features']['C_extra']]);eff=b0+bb[0]*uu+bb[1]*ss+bb[2]*qq;pairs=[]
        for j,v in enumerate((uu,ss,qq)):
            low,high=np.quantile(v,[.25,.75]);m=np.asarray(med);base=b0+bb@m;pairs.append(f"{base+bb[j]*(low-m[j]):.3g}→{base+bb[j]*(high-m[j]):.3g}")
        lines.append(f"| {f} | {np.median(eff):.3g} | {' | '.join(pairs)} |")
    importance={name:np.mean([r['cal_permutation_logloss_increase'] for r in co if r['model']=='D' and r['feature']==name]) for name in FEATURES}
    lines+=['','D固定单次排列重要性：内cal已知行**未加权**平均logloss差，seed20261011+fold，无评价选特征。前8项为 '+', '.join(f'`{n}` {v:.5g}' for n,v in sorted(importance.items(),key=lambda x:-x[1])[:8])+'。',
      '这次B的原尺度ToF余量系数高于A；C的余量×UNKNOWN交互折均值为正，余量×支持数为负，余量×峰SNR为小正值，不符合简单“UNKNOWN越多就单调降低ToF分数权重”的预期。混合训练的效果不能解释为已经证实这一降权机制。D确实使用质量/支持特征，但其作用须与实际paired收益和FREE成本一起判断。',
      '四模型按UNKNOWN/支持/峰SNR低高四分位的ToF余量+0.01m概率响应均值见payload `influence.csv`；C同步重算交互，其他特征固定。这里含−10缺测placeholder局部变化，不是可部署硬件扰动，不能以此单独声称因果退回RGB。', '',
      '## 各折中带收益保留','',
      '| 折 | 模型 | 最佳单源 | A净增 | 模型净增 | 保留率 | 保留率95%区间 | 有效bootstrap |','| --- | --- | --- | ---: | ---: | ---: | --- | ---: |']
    for r in s['retention']:lines.append(f"| {'合计' if r['fold']==-1 else r['fold']} | {r['model']} | {r['best_single']} | {r['A_net']} | {r['model_net']} | {ratio(r['retention'])} | {ratio_interval(r['retention_ci'])} | {r['retention_ci']['valid']} |")
    lines+=['','最佳单源在对应fold/合计半循环中带按W高、F低、同分RGB确定，再固定比较器进行bootstrap。合计保留率是合计净增之比，不平均各折比例。A净增≤0时N/E；bootstrap A净增≤0的抽样不纳入比例区间。','',
      '## 逐折与合计完整表','',
      'W=预测query为正且参考POS，F=严格sampled FREE上预测正；UNKNOWN不作负例，U计入载荷。救/损基于相同POS query，FREE增/减基于相同FREE query。括号区间为visit聚类2000次的95%区间(pp)：合计抽30visit，每折抽其6visit，包括零行FARO visit；分母为0的抽样略过，有效次数保存在summary。这里只重采样固定OOF预测，未重训CV。']
    for fold in (-1,0,1,2,3,4):
        lines+=['',f"### {'合计' if fold==-1 else '折 '+str(fold)}",'',
          '| 臂 | 带 | 融合器 | W/POS (W率CI) | F/FREE (F率CI) | vsRGB 救/损=净 (CIpp) | vsToF 救/损=净 (CIpp) | FREE增/减 vsRGB;ToF |',
          '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for a in ARMS:
            for b in range(3):
                for model in 'ABCD':
                    r=M[fold,a,b,model];ps=[P[fold,a,b,model,base] for base in ('rgb','tof')]
                    pp=[f"{x['rescue']}/{x['loss']}={x['net']:+d} {interval(x['net_pp_ci'])}" for x in ps];ff='; '.join(f"{x['FREE_added']}/{x['FREE_removed']}" for x in ps)
                    lines.append(f"| {LABELS[a]} | {BANDS[b]} | {model} | {r['W']}/{r['POS']} {interval(r['W_rate_ci'])} | {r['F']}/{r['FREE']} {interval(r['F_rate_ci'])} | {pp[0]} | {pp[1]} | {ff} |")
        lines+=['','单源工作点（同折同带跨臂阈值）：','', '| 臂 | 带 | RGB W/POS,F/FREE | ToF W/POS,F/FREE |','| --- | --- | --- | --- |']
        for a in ARMS:
            for b in range(3):
                rs=[M[fold,a,b,m] for m in ('rgb','tof')];texts=[f"{r['W']}/{r['POS']}, {r['F']}/{r['FREE']}" for r in rs];lines.append(f"| {LABELS[a]} | {BANDS[b]} | {' | '.join(texts)} |")
    addon=load(ROOT/'evidence_summary.json');addon_audit=load(ROOT/'independent_addon_audit.json');assert addon_audit['status']=='PASS'
    lines+=['','## 追加：先验基线与最低局部证据（事后）','',
      '用户在另一聊天明确要求未完成的本run补先验/证据审计，已核对原始人类请求。[追加PLAN](SYNC_FUSION_ROBUST_EVIDENCE_ADDENDUM_PLAN_DEV_20261011.json)在补充拟合前固定；此时A–D结果已知，追加属于事后开发分析，不冒充预注册。只用原列16–19（距离带和query x/y/z）的逻辑回归，复用B各折train-known等权、加权StandardScaler、C1/L2/lbfgs1000/三seed概率均值与混合cal的全部ties/目标；不改原模型、切点、原表或预算。','',
      '统一最低证据合同为**封存unclipped raw余量finite且≥0**：RGB近Uni/中远DAV，单K对应≥16条区间内像素射线；ToF沿用K0/K1余量均值并传播missing，因此K2均值≥0并不要求每K都单独≥16。追加PLAN最初写的双K各≥16另保留为更严格敏感性，全部逐K支持数已复算；跨任务合同澄清后同时展示两者，未改变预测。另列原cal单源二值分类，负cut会支持无区间内侵入，不能与最低阳性证据混同。',
      '分类顺序为①双方都有、②仅ToF、③仅RGB、④双方均无最低局部证据。④不能直接称“纯先验”：融合器还看到其他RGB模型、区间前后深度/比例等信息。简先验logit的结果也不能排除HGB中更复杂的场景先验。', '',
      '| 折 | 臂 | 带 | 先验W/POS | 先验F/FREE |','| --- | --- | --- | --- | --- |']
    for r in addon['prior_metrics']:lines.append(f"| {'合计' if r['fold']==-1 else r['fold']} | {LABELS[r['arm']]} | {BANDS[r['band']]} | {r['W']}/{r['POS']} | {r['F']}/{r['FREE']} |")
    E={(r['fold'],r['arm'],r['band'],r['model'],r['variant'],r['category']):r for r in addon['evidence_split']}
    cats=('both','tof_only','rgb_only','neither')
    lines+=['','原K2余量合同的合计见证与误支持拆分，四个数分别为①/②/③/④；每行相加严格等于原W或F。','',
      '| 臂 | 带 | 模型 | W构成①/②/③/④ | F构成①/②/③/④ |','| --- | --- | --- | --- | --- |']
    for a in ARMS:
        for b in range(3):
            for model in 'ABCD':
                rr=[E[-1,a,b,model,'raw_mean_margin',c] for c in cats]
                lines.append(f"| {LABELS[a]} | {BANDS[b]} | {model} | {'/'.join(str(r['W']) for r in rr)} | {'/'.join(str(r['F']) for r in rr)} |")
    lines+=['','D重点端点的净增拆分（同最低证据合同）：','',
      '| 臂/带 | 证据 | W | F | vsRGB救/损=净 | vsToF救/损=净 |','| --- | --- | ---: | ---: | --- | --- |']
    for a,b in ((ARMS[4],0),(ARMS[5],0),(ARMS[0],1)):
        for c in cats:
            r=E[-1,a,b,'D','raw_mean_margin',c];lines.append(f"| {LABELS[a]} / {BANDS[b]} | {c} | {r['W']} | {r['F']} | {r['rescue_rgb']}/{r['loss_rgb']}={r['rescue_rgb']-r['loss_rgb']:+d} | {r['rescue_tof']}/{r['loss_tof']}={r['rescue_tof']-r['loss_tof']:+d} |")
    lines+=['',f"共有{addon['bothK_missing_but_mean_supported']}条query达到K2平均余量≥0但未同时满足双K各≥16；三种合同、逐折完整拆分及救/损见 `evidence_split.csv`，逐query两K支持数/类别/先验预测见 `per_query_evidence.csv`。独立补充核验PASS，精确数量见[收据](SYNC_FUSION_ROBUST_EVIDENCE_INDEPENDENT_AUDIT_DEV_20261011.json)。原features/oof/summary哈希未变。", '',
      '## 独立核验、资源与交付','',
      f"独立核验 **PASS**，见[审计收据](SYNC_FUSION_ROBUST_INDEPENDENT_AUDIT_DEV_20261011.json)。审计不导入本任务训练/选点/汇总函数，核对折隔离、训练等权与标准化、模型配方及独立预测、所有ties/阈值、逐query缓存、逐折/合计指标与paired bootstrap、保留率和候选；新增采样尺度及固定跨度代表帧逐数组重放。旧半循环与FARO20维特征逐query复核，冻结依赖hash留在payload。精确审计覆盖量以收据为准。",'',
      f"CPU command-wall保守记账 {resources['CPU_command_wall_conservative_s']:.3f}/1800s；GPU {resources['GPU_s']}/300s，无新RGB推理、训练仅CPU，无下载。分阶段实测/失败及预算估计见[交付收据](SYNC_FUSION_ROBUST_RECEIPTS_DEV_20261011.json)。初次PLAN创建遇pycache预建目录，验证同字节PLAN后机械恢复；第一次特征前缀因ambient参数映射代码缺陷停止并保留，修正1/3/10映射后原PLAN重建。第二次因逐行重新解压NPZ并保留整表引用而MemoryError，改为每数组只读一次，原源/失败保留，已有CNH逐数组一致后复用。这些修复均在训练或选点前；未按OOF结果调整配方。折2/4 B/C标准化恒定dav/uni_finite_fraction的variance舍入至约−4e−37/−3e−37，scale均1，全部train/cal/eval变换有限；独立核验确认无NaN。首次审计把sklearn1.9默认penalty='deprecated'误判为非L2，核对安装版fit映射l1_ratio0为L2后只修审计兼容，失败也计费保留。",'',
      '逻辑payload：`artifacts.local/work/sync-fusion-robust-dev-20261011/`，物理落盘 `F:/ba-data/blindassist-artifacts-20260805/work/sync-fusion-robust-dev-20261011/`。保留 `features.npz`、`per_query.csv`、`oof.npz`、`folds.csv`、60模型、每折训练weights/cal_scores/full ties、summary/metrics/paired、coefficients/influence、新增CNH与输入hash、独立audit和失败前缀。Git提交脚本、PLAN、报告、精简收据；不上传原数据/模型/逐query大载荷。没有任务常驻进程、GPU分配或会话需要保留；这些磁盘载荷由本任务留作复算证据。','',
      '[逐query预测表](../../../../artifacts.local/work/sync-fusion-robust-dev-20261011/per_query.csv) · [逐query证据/先验表](../../../../artifacts.local/work/sync-fusion-robust-dev-20261011/per_query_evidence.csv) · [折划分](../../../../artifacts.local/work/sync-fusion-robust-dev-20261011/folds.csv) · [完整指标与bootstrap](../../../../artifacts.local/work/sync-fusion-robust-dev-20261011/summary.json)','',
      '## 结论边界','',
      '全部30visit已消费、参考门槛富集、既有残差池早于本CV。半循环仍与参考native共源；FARO静态几何、理想rho、缺覆盖/pose和官方筛选不能替代真实ToF。峰质量是全局zone统计，不能自动识别所有近带伪影；ambient×10模拟强度只覆盖该光子模型。各折阈值在合并cal满足目标不保证每评价臂FREE达标；净增≥0不是统计非劣。报告只选下一轮独立确认的方法，不升级App、冻结ToF/RGB默认或实机/接触/安全结论，保护480/test未访问。v2确认中带通过、远FREE失败及FARO近带负增益均原样保留。','']
    REPORT.write_text('\n'.join(lines),encoding='utf8')
    receipt=dict(task=p['task'],PLAN_sha256=sha(PLAN),report_sha256=sha(REPORT),resource=resources,
      candidates=s['candidates'],original_count_descriptor_candidates=original_candidates,retention=s['retention'],pooled_metrics=[r for r in s['metrics'] if r['fold']==-1],
      pooled_paired=[r for r in s['paired'] if r['fold']==-1],audit=audit,
      addon=dict(audit=addon_audit,PLAN_sha256=sha(Path(__file__).with_name('SYNC_FUSION_ROBUST_EVIDENCE_ADDENDUM_PLAN_DEV_20261011.json')),prior_metrics=addon['prior_metrics']),
      outputs={n:sha(ROOT/n) for n in ('features.npz','per_query.csv','oof.npz','summary.json','cuts.json','models_manifest.json','feature_manifest.json','coefficients.json','influence.csv','independent_audit.json','executed_source.py','dependency_seal.json','evidence.npz','evidence_summary.json','evidence_split.csv','per_query_evidence.csv','independent_addon_audit.json','prior/oof.npz','prior/cuts.json','prior/models.json')},
      failures=load(ROOT/'engineering_failures.json'),payload=str(ROOT.resolve()))
    Path(__file__).with_name('SYNC_FUSION_ROBUST_RECEIPTS_DEV_20261011.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf8')
    Path(__file__).with_name('SYNC_FUSION_ROBUST_INDEPENDENT_AUDIT_DEV_20261011.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    Path(__file__).with_name('SYNC_FUSION_ROBUST_EVIDENCE_INDEPENDENT_AUDIT_DEV_20261011.json').write_text(json.dumps(addon_audit,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(REPORT),rows=len(s['metrics']),candidates=s['candidates'])))
if __name__=='__main__':main()
