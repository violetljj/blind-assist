# 前视障碍感知：当前状态

更新：2026-09-25

Status: `DTR_R2_DYNAMIC_RETAINED` (historical algorithm; no new promotion).

当前推进的是 **cane-complementary forward perception**，不是重新启动历史 DTR 动态研究。[中文总览](../../../docs/PROJECT_STATE.md)说明当前手机系统、代表性指标和研究范围。

## 系统与保留对照

- 手机 v10.15.1：原首页手动开始 A+LOCAL 实验模式；保留基础 ToF 切换。名义几何适配未完成物理标定。[硬件说明](../../../docs/HARDWARE_OBSTACLE_DEMO.md)
- 研究：保留 A 基线；LOCAL 的增量检出与误报一起报告。失败的 rescue gate 不进入当前手机组合。
- 历史 A* 是另一套四传感器主机研究/展示路径；其分数不与当前相机 + ToF 结果拼成“最好指标”。[A* 范围](nearfield/corridor_fusion_v1/ASTAR_EFFECT_USAGE_20260917.md)

## 固定对照与现存问题

[新实例受控仿真](nearfield/LOCAL_RESCUE_RESULTS_20260923.md)：576 帧、48 段、32 个正事件；224 正帧、352 负帧。

| 方案 | TP | FP | FN | 检出事件 | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 118 | 7 | 106 | 24/32 | 保留低误报对照 |
| A OR LOCAL | 159 | 22 | 65 | 29/32 | 多检出 5 事件，同时多 15 误报帧 |
| 固定 rescue gate | 159 | 18 | 65 | 29/32 | 只去掉 4/15 增量误报，未达门槛；不接入 |

UNKNOWN 均为 538/576，可与障碍提醒并存；无提醒不是无障碍。该数据受控、同渲染器，不是实机自然分布证据。旧稳定性失败仍有效。

当前瓶颈：如何从粗测距与图像中可靠判断近回波属于谁，以及接触位置能否迁移到未见布局。提醒存在、目标归属、接触定位必须分开检验。

## 最近结论，只留影响下一步的部分

- [范围与投影稳定性](nearfield/LEDGER_REPAIR_20260922.md)：账本回执修复不改变先前稳定性结论，不产生新的确认数据。
- [LOCAL 救援](nearfield/LOCAL_RESCUE_RESULTS_20260923.md)：保留增量信号，固定过滤方案失败，不再自动调阈值。
- [接触采样](nearfield/CONTACT_SAMPLING_RESULTS_20260923.md)：几何改善伴随召回下降，仅保留组件价值。
- [原始立体几何](nearfield/FOUNDATION_GEOMETRY_RESULTS_20260923.md)与 [VPP](nearfield/VPP_GEOMETRY_RESULTS_20260923.md)：查询深度提升未变成最终提醒提升；小型平面头部障碍仍失败。
- [立体适配](nearfield/STEREO_ADAPT_RESULTS_20260923.md)：普通适配只保留 challenger；固定平衡配方为负对照，不自动成为手机替代。

## 下一步及停止规则

CNH v3 的用户决定：目标/同类干扰物资产族隔离，普通背景资产允许共享并披露；
同一物理场地与布局不跨分区，参与通道侵入的建筑构件不能按背景豁免。
先完成共同写实几何/实例ID导出，再做Street200V7与City Sample各2布局工程比较；
用户已决定第一阶段仅室外街景：人行道、街口/交叉口、广场、窄道/巷道，室内推迟第二阶段。
各96布局（train45/dev10/test41）、试采各5；384总量、障碍族分配和既有预算不变，
规划计算不变不保证实际方差/功效不变。真实派生资产/可见ID单元通路已通过。
原两来源失败回执保留：Street200树木WPO前置拒绝；City两布局最大误差881/429mm及局部覆盖FAIL。
用户反馈后同场地工程修复：Street200临时关闭WPO/PDO、核实DynamicMesh默认材质，完成16端点，
按原门槛P95为6.422/6.036mm、max为18.471/50.812mm，每区漏/多采与可见细杆近层均PASS。
用户现授权Street200先采集、同步验证：两机各单UE会话完成6新布局，共12布局Development，
人行道/交叉口/广场各4，4段×40名义位姿共1920帧；两机各18帧抽样原门槛全部PASS。
七路异步实测约提速3.1倍。剩余能量、标签、隔离检查按布局记录，失败隔离不拖停其他布局。
3060跨机近场深度/插入物ID一致，但背景法线差异未解决，暂不准入全模态混用。
这些位姿仍属于同一Street200街区，不是独立训练/测试场地；完整20布局配额及384布局尚未完成。
用户明确可扩建自建Street200，已生图规划并在独立地图完成A区3×27m直巷与12×12m广场样板；
实际UE图已查看。1920帧已跑通H3合成、独立几何标签与CUDA线性基线，
完整布局960/960同场地调试，误报仍多；不作独立测试/算法收益结论，不替代能量和标签精度准入。
新增[参数化巷道生成器](nearfield/CNH_ALLEY_GENERATOR_20260924.md)：9个场地、72件杂物按干扰物隔离。
R2跨划分复用失败保留；R3a联合隔离杂物和插入族，实测最小净空175.334mm。
九图实际摆放和六份开发采集规格的[隔离复核](nearfield/CNH_ALLEY_ASSET_ISOLATION_AUDIT_20260924.md)确认
杂物/插入物语义族、源网格与场地跨区交集为0；旧插入材质三张 UE 实际使用纹理跨区共享，隔离 FAIL。
未保存派生材质关闭 MFPD 后，六份实际 train/dev 重采使用纹理跨区交集为0；与 test 的交集0仍仅有未保存材质探针一侧，test 未采。
3060六场采满960帧，18帧固定抽样原门槛全PASS、背景max最高0.615mm；test三场未采。
旧RGB欠曝不可用；train/dev 六场已按每布局近场测光固定 EV 重采960对双目，逐图数值门槛1920/1920通过，首/中/末双目抽查已记录。旧ToF、深度和标签按帧键及哈希绑定，未重写。
与旧Street合并2880帧、18布局的原ToF基线仍保留，冻结1440/1440和30epochs完成重训。
开发验证TP/FP/FN/TN=942/2249/358/5091、UNKNOWN=0，模型重载逐位一致，不作算法收益声明。
场地登记阻止换名跨区；只有墙/地面等背景共享，简化美术与Development范围保持披露。
City车辆移除开发诊断已跑通16帧并恢复：16近实例临时清零、48远实例不动、metadata保持。
City1 max428.617→28.341mm且原覆盖通过，City0仍881.078mm与5m边界FAIL，未宣布整体准入。
新增[路缘石逐实例派生替换反证](nearfield/CNH_CITY_CURB_SUBSTITUTION_20260924.md)：同一16帧旧布局中
Nanite关闭、LOD0锁定后City0 max仍880.927mm，四次重复射线未消失；zone53覆盖extra仍2.2214%>2%。
相邻射线指向停车收费表实例2边缘，但原生可见ID未获证；City0标记 phase 1 排除并停止追因。City1[新位姿同场地Development采集](nearfield/CNH_CITY1_FRESH_DEVELOPMENT_20260925.md)16/16帧中，一布局几何PASS但RGB近场亮度FAIL，另一布局几何FAIL；整批不准入当前融合训练，不算新物理场地确认。City整体正式准入仍未建立。
City按逐区/边缘/角度/预测物体归属诊断，未发现z/径向混用证据，不能直接归结为来源不合格；
5m缓冲带、轮廓容差是未启用的规则建议，不改旧FAIL。
背景像素ID聚合，
原生侵入物仍参与标签，背景距离检查不能代替回波覆盖检查。[工程回执](nearfield/CNH_SOURCE_ENGINEERING_20260924.md) ·
[CNH方案第4节](nearfield/CNH_ROUTE_COMPARISON_PLAN_20260924.md)。
布局生成先筛原生15cm净空，四段轨迹/全部查询共同检查；正例由插入派生资产承担，含墙/大型遮挡。
每布局最多16个候选，不通过则报告生成失败；不靠忽略原生侵入或背景角色改名绕过标签与资产隔离。

此前已完成选定证据的副机备份与恢复校验、短状态页和 Python/冻结文本回归接入。提醒策略保留为后续固定检测输出的比较问题；CNH现在推进Development批采与观测基线接口，不启动保护测试选模。
已实现[CNH+RGB局部视锥融合接口](nearfield/CNH_RGB_FUSION_INTERFACE_20260924.md)并完成几何与未训练前向检查；
一帧原始Street Development RGB/H3同帧身份烟测通过。六场巷道修复RGB上已完成[三seed同划分对照](nearfield/cnh_rgb_dev_comparison.py)：480 train/480 dev，ToF-only和CNH+RGB在固定logit 0阈值下每次均为TP0/FP0/FN408/TN2472；[保存预测的排序诊断](../../../artifacts.local/evidence/cnh-rgb-alley-rank-diagnostic-20260925-v1/result.json)显示AUROC约0.677、AP约0.269，两臂差异极小。此轮无RGB收益证据，亦未建立可用检测器；Street本地精简源仅有8/960张RGB，不能冒充四类环境对照。
独立冻结的[V2方位掩码与训练集类别平衡对照](nearfield/CNH_RGB_ALLEY_DEV_V2_PROTOCOL_20260925.md)在相同六场/三seed上解除了全负退化：ToF-only dev TP148–160/408、FP486–512/2472；CNH+RGB TP148、154、159，FP504、491、503。预声明结论为 `NONCOLLAPSED_DEVELOPMENT`、`NO_CONSISTENT_DEV_PARETO_SIGNAL`；误报仍高，不能声称融合获益。
新增[巷道瓶颈诊断](nearfield/CNH_ALLEY_BOTTLENECK_DIAGNOSIS_20260925.md)：V2两臂 dev AUPRC 均约0.22–0.23，跨帧打乱/置零 RGB 几乎不变；同预算全分辨率可见深度学习臂弱且不稳定，但预定义盒直接可见深度规则为 TP405/408、FP0/2472。可见几何大体对齐，先查模型表征、视锥汇聚和训练；不能由弱学习臂推断标签/查询不可观测。按本轮停止规则，预训练 RGB 与 CNH 对同响应标量控制均 `NOT_RUN`，City 与完整164布局test暂停，不再采集。
用户随后授权的[学习器与 H3 几何诊断](nearfield/CNH_LEARNING_DIAGNOSTIC_20260925.md)已完成：960帧深度与规则 EXR 逐元素一致，无全零事故；全深度 train AP 0.1489/0.2363/0.1542，32帧过拟合仍仅0.1576，ToF-only则达到0.9966。梯度非零，未发现使旧结果作废的实现 bug，故未重跑三seed。无噪声H3几何规则 dev AP0.8452，原模拟响应H3为0.2101；前者推翻把学习代理0.29当信息上限的解释。固定列掩码排除查询盒部分投影、查询前区内汇聚是优先修正的结构问题，尚未唯一归因于某一项。未来对象ID schema已定义，生产端未接入；不补标、不采集，City/test继续暂停。
用户明确授权后的[QG-1与响应分解](nearfield/CNH_QG1_RESULTS_20260925.md)已完成：全深度32帧AP0.9951，Gate0通过；三seed dev AP0.7205/0.7937/0.8070，未达到全部≥0.80的结构门槛。ToF约0.2314–0.2341，数值门槛虽过但最小提升仅0.000023；融合0.2548–0.2733且置乱均下降，只能证明重复Development中的RGB使用。响应0.8452→0.2101端点逐位复现；最大中间下降来自串扰造成的跨查询固定偏置/校准失配，不能称物理信息损失，随后噪声亦显著降低固定读出。旧数据仅作诊断，不恢复独立泛化资格；[新近场验收方案](nearfield/CNH_NEARFIELD_ACCEPTANCE_PLAN_20260925.md)保留待审，未采集，City/test继续暂停。
完整164布局test的[准备度审核](nearfield/CNH_FULL_TEST_READINESS_20260925.md)为真实场地和采集spec 0/164，受保护采集入口未实现；不以三张候选巷道图代替完整test。

新研究问题/预算/规则由用户确认；已授权问题内自主完成合理实现和验证。每个问题用一个能推翻假设的关键对照，保留成功、失败和不可评估的原始结果。具体规则见[研究工作方式](../../WORKFLOW.md)。

## 历史和恢复

[整理前完整正文](CURRENT_HISTORY_20260924.md)保留旧数字与历史 pending；[详细账本](README.md)保留结果与复现入口。历史文字不授予新执行权限。

ledger303 的回执与指定登记已[修复](nearfield/LEDGER_REPAIR_20260922.md)；未在修复范围内的历史缺失登记不能据此宣称全部解决。
