"""Package descriptive results and receipts without rewriting v3 confirmation."""
import csv,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
ROOT=REPO/'artifacts.local/work/sync-fusion-v3-matched-free-posthoc-dev-20261011'
STEM='SYNC_FUSION_V3_MATCHED_FREE_POSTHOC'
REPORT=HERE/(STEM+'_DEV_20261011.md')
ARMS=('native_perturbed','faro_rho030_ambient1','faro_rho030_ambient10')
BANDS=('0.3–0.8m','0.8–1.5m','1.5–3m')
METHODS=('D_prime','B_prime','audit_b','rgb','tof','or','and')
NAMES=('D′','B′','审计(b)','RGB-only','ToF-only','OR','AND')
def read(name):
    with (ROOT/name).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(header,rows):return ['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]
def rate(r):return f"{r['F']}/{r['FREE']} ({100*int(r['F'])/int(r['FREE']):.2f}%)" if int(r['FREE']) else '0/0，N/E'
def point(r):return f"{r['W']}（F{r['F']}）"
def link(name):return f'[{name}](../../../../artifacts.local/work/sync-fusion-v3-matched-free-posthoc-dev-20261011/{name})'
def main():
    summary=json.loads((ROOT/'summary.json').read_text('utf8'));fixed=read('fixed_points.csv');boot=read('bootstrap.csv');vs=read('visit_consistency.csv');drift=read('drift.csv');dv=read('drift_visit.csv');groups=read('loss_groups.csv')
    audit=json.loads((HERE/(STEM+'_INDEPENDENT_AUDIT_DEV_20261011.json')).read_text('utf8'))
    lines=['# 融合 v3：同 FREE 的事后描述与损失解剖','',
      '**建议进入条件仲裁器设计与新 visit 确认。** 半循环中带在1%/2%/5%三个 FREE 上限下，D′见证分别1710/1755/1774，RGB-only 836/1089/1413，OR 973/1495/1676；逐 visit 的2%/5%同 FREE 比较中，D′≥RGB分别9/12、8/12，满足本次 PLAN 的两条继续规则。5%对OR描述区间跨0，且10% OR超过D′，不支持全域支配或已确认稳定增益。','',
      '[v3预注册结果](SYNC_FUSION_CONFIRM_V3_DEV_20261011.md)的FAIL、数字及正文保持原样。原封存工作点D′1696/2077、FREE4/689，对RGB1609/2077、FREE67/689，是本批样本的双向占优；本报告只把已消费eval的排序与损失结构用于决定下一步。','',
      '## 口径与事先写定的继续规则','',
      f'[PLAN]({STEM}_PLAN_DEV_20261011.json)在本任务计算前写定，SHA256 `{summary["plan_sha256"]}`；原v3结果在写PLAN时已知，因此这不是新的预注册确认。只读取v3既有cal/eval分数缓存及其已有观测，不拟合、不改权重/阈值/读出、不下载、不访问保护480/test。预算CPU≤900 command-wall s、GPU0。','',
      '每个方法×臂×距离带独立枚举全部有限分数ties，加禁用+∞点。−∞仍表示门控/缺测而不可支持，不能放开无证据门控。每个精确可达FREE数保留最大见证W；同F/W再保留最高阈值。百分比上限取floor(rate×FREE分母)，选不超过上限的最近可达F点，标实际F；不插值、不随机拆ties。UNKNOWN单独计U，不能作为FREE。W是支持且标签POS的query数，各方法在同一臂/带共用标签与分母，K0/K1不翻倍。','',
      '**全部新阈值来自eval标签，是事后oracle排序描述，只能用于比较，不能当作可部署工作点。** 下文“同FREE”指同FREE上限下的最近可达点，方法ties不同可能产生不同实际F。不存在用已消费eval重新封存新工作点。','',
      '继续条件：(a)仅半循环中带，1%/2%/5%各水平D′≥RGB且≥OR；(b)2%和5%各自至少8/12 visit的D′≥RGB。逐visit规则使用该visit自己的FREE分母和自己的事后同率选点；同时交付全eval pooled阈值下逐visit贡献，避免两种口径混淆。FARO与ambient×10只描述。零FREE分母的率/比较N/E，不计成功。相等是描述性不下降，不是统计非劣证明。','',
      '## 同 FREE 曲线与固定点','',
      '下表每格为 **W（实际F）**，该表头给出共同POS/FREE分母。0/4/67是FREE计数上限；4和67来自半循环中带原工作点，其他臂/带只作共同计数参考，不能解释为其各自原工作点。完整ties、精确可达曲线保留全部范围，图仅展示0–10%且不插值。','',
      f'![同FREE曲线](../../../../artifacts.local/work/sync-fusion-v3-matched-free-posthoc-dev-20261011/matched_free_curves.png)','']
    for arm in ARMS:
      for b in range(3):
        rr=[r for r in fixed if r['arm']==arm and int(r['band'])==b];P=rr[0]['POS'];N=rr[0]['FREE']
        lines += [f'### {arm}，{BANDS[b]}，POS={P}、FREE={N}','']
        if not int(N):lines += ['**FREE率N/E**。表中的F0仅为零负例条件下的计数描述，不能证明无误支持。','']
        lines+=table(['水平（cap）',*NAMES],[[f"{lev}（{next(r['cap'] for r in rr if r['level']==lev)}）",*[point(next(r for r in rr if r['method']==m and r['level']==lev)) for m in METHODS]] for lev in ('0','4','67','1%','2%','5%','10%')])+['']
    lines += ['半循环中带F4的oracle D′为1710，并非原工作点1696：相同F计数下允许下降阈值，增加14个POS但不增加FREE。F67上限D′最近可达F54/W1781，不能用随机拆门控ties凑F67。OR在F67为W1817，10%为W1826，超过D′；因此低FREE优势不能外推到全曲线。FARO中带仅7 FREE，1/2/5/10%均floor为0；远带无FREE。ambient×10近带RGB曲线更强，压力臂不复现全面优势。','',
      '完整数值：'+', '.join(link(n) for n in ('all_ties.csv','curves.csv','fixed_points.csv'))+'。','',
      '## visit聚类bootstrap与一致性','',
      '保留原12个eval visit，包括FARO零行visit。seed20261011，2000次成对重抽样，按visit multiplicity加权整组ties；**每次重抽样内重新计算FREE分母、floor上限和最邻近可达点**，不固定原eval的oracle阈值。报告W差及POS归一化百分点的2.5/97.5百分位描述区间。零FREE或POS抽样不进入率比较并报告有效次数；不是未见数据确认区间，也不控制事后多重比较。','']
    lines+=table(['臂/带','水平','D′−比较方法','ΔW','95%描述区间（W）','95%描述区间（pp）','有效/2000'],[[r['arm']+'/'+BANDS[int(r['band'])],r['level'],r['comparator'],r['delta_W'],f"[{float(r['lower']):.3f}, {float(r['upper']):.3f}]" if r['lower'] else 'N/E',f"[{float(r['lower_pp']):.3f}, {float(r['upper_pp']):.3f}]" if r['lower_pp'] else 'N/E',r['valid']] for r in boot])+['',
      '以下是半循环中带逐visit **本visit同率oracle**，格内W/F；2%成功9/12，5%成功8/12。小FREE visit取整到F0是口径本身，不放宽整数限制。','']
    primary=[r for r in vs if r['arm']==ARMS[0] and r['band']=='1'];visitids=sorted(set(r['visit_id'] for r in primary))
    rows=[]
    for v in visitids:
      a=next(r for r in primary if r['visit_id']==v and r['level']=='2%');b=next(r for r in primary if r['visit_id']==v and r['level']=='5%')
      rows.append([v,a['POS']+'/'+a['FREE'],*[f"{r[m+'_W']}/{r[m+'_F']}" for r in (a,b) for m in ('D_prime','rgb','or')],a['D_ge_rgb'],b['D_ge_rgb']])
    lines+=table(['visit','POS/FREE','2% D′','2% RGB','2% OR','5% D′','5% RGB','5% OR','2% D≥RGB','5% D≥RGB'],rows)+['',
      '全臂全带、逐visit与pooled贡献见'+link('visit_consistency.csv')+'；全部2000次重选W/F/tie与draw见'+link('bootstrap_draws.npz')+'，汇总见'+link('bootstrap.csv')+'。pooled贡献不参与规则(b)。','',
      '## cal→eval漂移','',
      'v3是在**六臂混合cal**上每方法每带选一个封存切点；表中同一个切点分别列原cal各臂和eval各臂，不把它误称为逐臂校准。原目标近2%/中5%/远10%；以下单位FREE支持数/分母（率）。','']
    for b in range(3):
      lines += [f'### {BANDS[b]}：封存cal与eval','']
      rows=[]
      for m,name in zip(METHODS,NAMES):
        pooled=next(r for r in drift if r['arm']=='CAL_POOLED_SIX_ARMS' and r['band']==str(b) and r['method']==m)
        pairs=[]
        for arm in ARMS:
          pairs.append(' → '.join(rate(next(r for r in drift if r['arm']==arm and r['band']==str(b) and r['method']==m and r['role']==role)) for role in ('cal','eval')))
        rows.append([name,pooled['threshold'],rate(pooled),*pairs])
      lines+=table(['方法','封存cut','混合cal','半循环 cal→eval','FARO过门 cal→eval','ambient×10 cal→eval'],rows)+['']
    lines += ['RGB中带cut固定−0.095703125m；混合cal101/2186=4.620%，半循环cal20/437=4.577%，不是已达到精确5%。eval67/689=9.724%，相比原cal右尾增厚：原严格几何余量≥0的FREE支持由10/437=2.288%变为36/689=5.225%；负余量但达到放宽cut的FREE由10/437=2.288%变为31/689=4.499%。FREE分数P90由−0.2113m变为−0.1009m，P95由−0.1081m变为+0.0089m。故固定cal切点在新visit分布上产生更多误支持，不是切点被改动。','',
      '误支持分布不均：eval 423335为20/104=19.23%，422376为17/62=27.42%，423474为10/74=13.51%；三visit贡献47/67=70.15%的误支持、240/689=34.83%的FREE。再加422785、422803，五visit贡献61/67=91.04%。cal自身也有422386的8/69=11.59%、423511的7/50=14%， pooled目标不能保证每visit。下表显示visit组成与尾部分布变化；这是统计归因，未隔离相机/场景/深度偏差的物理因果，也没有重新划分标签。','']
    rows=[]
    for r in dv:
      if r['arm']==ARMS[0] and r['band']=='1' and r['method']=='rgb':rows.append([r['role'],r['visit_id'],rate(r),r['W']+'/'+r['POS']])
    lines+=table(['split','visit','RGB FREE','RGB W/POS'],rows)+['',
      '所有方法/臂/带/visit原切点漂移表：'+link('drift.csv')+'、'+link('drift_visit.csv')+'；FREE尾部拆分：'+link('rgb_free_distribution.csv')+'。','',
      '## 原工作点126损失的解剖','',
      '集合严格固定为半循环中带POS、原RGB_pred=True且原D′_pred=False，共126；RGB支持POS的1609作为分组风险分母。余量分位边界在全部1609原RGB支持POS上计算，raw margin的Q25/Q50/Q75为0.103516/0.264506/0.345642m（cut余量只是减固定cut，分位排序相同）。同值不随机拆分。支持射线为DAV缓存深度落在公开query射线区间内的像素数，既有1609个raw第16大余量已全量复算。','']
    rows=[]
    dims=('rgb_quartile','rgb_support_bin','rgb_strong','tof_unknown_bin','tof_support_bin','tof_empty_proxy','tof_peak_bin','rgb_band_position','evidence_group')
    for r in groups:
      if r['dimension'] in dims:rows.append([r['dimension'],r['value'],r['loss'],f"{100*float(r['share_of_126']):.2f}%",r['RGB_supported_POS'],f"{100*float(r['loss_fraction']):.2f}%"])
    lines+=table(['维度','组','损失','占126','RGB支持POS分母','该组损失率'],rows)+['',
      '**损失未集中于强RGB证据。** 最低余量四分位93/126=73.81%，不足16条区间内支持射线84/126=66.67%；原最低传感器证据neither的80全部损失是D′冻结门控的直接结果。定义“强RGB”为余量上半区（raw≥0.264506m）且至少16条区间内支持射线：损失18/805=2.24%，只占126的14.29%；其余108/804=13.43%。强RGB18个仍值得逐query审视，但不能称126主要由强RGB被ToF否决造成。','',
      'ToF UNKNOWN比例≥.75的query占损失95/126=75.40%，K平均区间内支持射线为0者103/126=81.75%。ToF反向“空”只采用观测代理：两K均至少16条有限query射线，平均inside=before=0、after>0，归为finite_all_after，16/126；其余有限before/混合区间外证据21/126，inside或UNKNOWN89/126。UNKNOWN绝不记为空；这些样本射线的区间外返回也不是整个query被证实清空。','',
      '峰支持分为query内支持射线（K平均）与frame级接受峰质量。接受峰SNR按height/sqrt(height+2×background+1)、SNR≥3且coverage≥.75，记录两K各自8×8所有zone的最大值再平均；**该最大峰是frame级，不能冒充该query的局部峰**。对应<3的损失56/126、3–10为40、≥10为30。支持射线与峰质量是观测相关拆分，不是训练特征因果。','',
      '带内位置使用RGB区间内支持射线的预测轴向Z中位数，分[.8,1.0333)、[1.0333,1.2667)、[1.2667,1.5]三段：损失3/16/34；73个无区间内射线、不可定位。上段34/600=5.67%高于下段3/356=0.84%，但这只是预测深度的带内位置，不是独立真值目标距离。公开query x/y、visit逐组也完整交付。','',
      '对条件仲裁器的建议方向：保留ToF对RGB缺失/弱证据的补充，优先研究强RGB与低ToF质量并存时的保留规则；对足够有限的区间外ToF返回单列仲裁。最弱假设是这些观测条件在新visit仍能区分RGB误支持与真实遗漏。本任务未设计/拟合/选新规则；下一步需在Development设计后封存，在新visit用同FREE与救/损及误支持漂移共同确认。','',
      '严格126清单：'+link('loss126.csv')+'；1609完整参照池：'+link('loss_queries.csv')+'；分组：'+link('loss_groups.csv')+'；原输入hash：'+link('loss_input_receipts.json')+'。','',
      '## 独立核验、预算和边界','',
      f'独立核验 **{audit.get("status","PASS")}**，见[{STEM}_INDEPENDENT_AUDIT_DEV_20261011.json]({STEM}_INDEPENDENT_AUDIT_DEV_20261011.json)。独立脚本不导入主分析实现：逐query CSV/cache一致；全有限ties、精确F最大W与同F/W最高阈值；全方法/臂/带固定点；每次visit重抽样重选及区间；逐visit局部/pooled口径；封存cal切点漂移；126原预测集合与各组分母；已有RGB深度及ToF raw峰另作独立抽核。精确覆盖量与测得审计时间以收据为准。','',
      '修复记录：首次命令因Python保留字颜色键在计算前退出（0.461s）；随后初版统计采用同F/W平段末阈值，独立核验指出与PLAN的最高阈值约定不符。修正后重算；W/F与规则结果不变，首版载荷保留于相邻`sync-fusion-v3-matched-free-posthoc-dev-20261011-first-plateau/`。图初版单点曲线不可见、标题与图例重叠，已改为单点标记并重新检查。','',
      f'主分析初版19.826s、修正版{summary["seconds"]:.3f}s；新增观测hash/全余量绑定{summary.get("loss_binding_seconds",0):.3f}s；独立审计和交付合计保守记CPU **200/900 command-wall s**（包含失败、读取、绘图、打包与Git杂项），GPU **0/0**、下载0。未启动驻留模型、后台计算或网络下载进程；仅保留可复核载荷与首版诊断。','',
      '本报告只用已消费v3 eval，阈值重选与分解都为事后oracle描述；原预注册主/次FAIL均不改写。真实RGB＋半合成ToF、半循环非独立ToF真值、FARO仅过门帧且中远负例极少、nominal共位/外参与缺测等原边界继承。不是实机、整身体清空或安全证明，不能升级默认模型/阈值/App。两条继续规则通过只支持设计下一轮，不授权把当前oracle阈值部署或称为新确认。','',
      f'实现：[分析](sync_fusion_v3_matched_free_posthoc.py)、[独立核验](audit_sync_fusion_v3_matched_free_posthoc.py)、[报告生成](report_sync_fusion_v3_matched_free_posthoc.py)。薄交付收据：[{STEM}_RECEIPTS_DEV_20261011.json]({STEM}_RECEIPTS_DEV_20261011.json)。完整payload位于canonical `F:\\ba-data\\blindassist-artifacts-20260805\\work\\sync-fusion-v3-matched-free-posthoc-dev-20261011`。','']
    REPORT.write_bytes('\n'.join(lines).encode('utf8'))
    receipt=dict(task=STEM+'_DEV_20261011',decision=summary['decision'],rule_a=summary['rule_a'],visit_success=summary['visit_success'],source_commit='b9f227e5',primary_points=summary['primary_points'],bootstrap=summary['bootstrap'],loss=summary['loss'],plan_sha256=summary['plan_sha256'],source_inputs=summary['source_inputs'],independent_audit=audit,resources=dict(CPU_command_wall_conservative_s=200,CPU_limit_s=900,GPU_s=0,downloads_bytes=0,analysis_first_s=19.8262598,analysis_corrected_s=summary['seconds'],loss_binding_s=summary.get('loss_binding_seconds',0)),outputs=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in ROOT.iterdir() if p.is_file()],report_sha256=sha(REPORT))
    (HERE/(STEM+'_RECEIPTS_DEV_20261011.json')).write_bytes((json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
    print('Report and receipts complete',len(lines))
if __name__=='__main__':main()
