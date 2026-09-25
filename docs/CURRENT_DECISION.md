# 当前研究决定

更新：2026-09-25

Status: `L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`

## 范围

论文主体与唯一推进的算法主线为盲杖互补的前视障碍感知。保留 A 基线和 A+LOCAL 的收益与成本；当前 Android 实验适配不等于实机准确率获证。读者先看[中文项目状态](PROJECT_STATE.md)，再看[避障当前页](../research/active/dtr-r0/CURRENT.md)。

[L10](../research/active/l10-r0/CURRENT.md) 正式暂停：保留代码、冻结结果和复现入口，不再自动启动实验。TARO、SANPO、PanoLab 和语义锚点只作历史成果或有明确用途的素材，不另开并行研究。是否作为论文扩展章节取决于写作需要；不承诺新增一章。

## 工作顺序

CNH v3 已授权采用“障碍物/同类干扰物资产族隔离、普通背景资产共享并披露”的考场定义；
物理场地实例与布局仍分区，侵入查询通道的建筑构件不享受背景豁免。先打通写实来源导出，
再比较Street200V7与City Sample各2布局；用户已决定第一阶段只做室外街景：
人行道、街口/交叉口、广场、窄道/巷道；各96布局（train45/dev10/test41），试采各5。
总384布局、障碍族分配和原预算不变；室内移至第二阶段，实际功效仍依赖方差与布局独立性。
生成时先让原生几何避开四段轨迹的全部查询体积至少15cm，正例（含墙）由受控派生插入物承担；
净空与资产隔离分别检查，发现原生侵入不能忽略或改为负标签。
真实派生资产与可见ID单元通路已通过。原始Street200树木WPO前置拒绝、City径向最大误差/局部覆盖
失败的回执保留；不把City失败直接解释成来源精度不合格。用户授权的同场地工程修复中，
Street200关闭WPO/PDO并核实DynamicMesh默认材质后，完成16端点且通过原30/75mm、2%和细杆近层检查。
用户现授权先采集、同步检查：Street200两机共完成12个新布局、三类环境各4、1920帧Development数据，
每机共用一次UE会话，各18帧抽样原几何门槛全部PASS；七路异步实测约提速3.1倍。
能量收敛、精确标签与隔离按布局记录，失败只隔离受影响数据，City修复不阻塞Street。
跨机近场深度/插入物ID一致，但背景法线不一致，暂不混用全部模态。
Street200可清点人行道、交叉口、广场，未确认巷道；同街区候选不冒充独立训练/测试场地。
用户补充Street200为自建场景：优先按生图规划扩建巷道与不同物理场地；已建独立A区样板，
3×27m巷道接12×12m广场，实际UE图已查看。巷道现已参数化：独立保存9个不同几何场地，
各预分区均含直巷/L/T。72件杂物按干扰物而非共享背景处理；R2跨区重复判FAIL，
R3a已按分区联合隔离9杂物族和6插入物族，最小实测净空175.334mm，保留原150mm门槛。
实际九图的杂物/插入物家族和源网格跨区交集为0，但旧插入材质有三张跨区共用的 UE 实际使用纹理，隔离 FAIL。
未保存的派生材质关闭 MFPD 后，六份实际 train/dev 重采材质的使用纹理跨区交集为0；与 test 的零交集仍只由未保存材质探针支持，正式 test 未采。
[隔离复核](../research/active/dtr-r0/nearfield/CNH_ALLEY_ASSET_ISOLATION_AUDIT_20260924.md)。
巷道train/dev六场960帧已在3060完成，18帧固定抽样原门槛全PASS，背景max最高0.615mm；
三个test候选场地未采。旧RGB严重欠曝；六场train/dev已重采960对双目，按帧绑定旧ToF、深度和标签；1920张图全部通过亮度门槛，首/中/末双目抽查见[采集说明](../research/active/dtr-r0/nearfield/CNH_ALLEY_GENERATOR_20260924.md)。共享墙/地面与简化美术仍披露。
City已恢复车辆移除开发诊断：16近实例临时清零，48远实例与完整metadata保留，
旧布局1最大误差428.617→28.341mm且覆盖通过；布局0仍881.078mm及5m边界覆盖FAIL。
路缘石原位Nanite关闭/LOD0派生替换的同布局16帧反证中，布局0最大误差仍880.927mm，
同一射线重复四次超限，zone53覆盖extra仍2.2214%>2%；相邻证据提示停车收费表边缘，原生可见ID未证实。
不改原门槛；City0标记phase 1排除并停止追因。[City1同场地新位姿采集](../research/active/dtr-r0/nearfield/CNH_CITY1_FRESH_DEVELOPMENT_20260925.md)完成16帧，其中一布局几何PASS但RGB近场亮度FAIL，另一布局几何FAIL；整批不作可用融合数据或新物理场地确认。City整体正式准入未建立，完整近场车辆派生替换虽已实现但未运行。
[City反证](../research/active/dtr-r0/nearfield/CNH_CITY_CURB_SUBSTITUTION_20260924.md)。
Street1920+巷道960共2880帧、18布局已完成H3合成→几何监督→CUDA线性基线与逐位一致重载；
冻结1440/1440开发划分、30epochs，验证TP/FP/FN/TN为942/2249/358/5091，UNKNOWN为0。
保留原Street同场地划分，巷道按作者分区；不作算法收益或独立测试结论。
CNH+RGB局部视锥[融合接口](../research/active/dtr-r0/nearfield/CNH_RGB_FUSION_INTERFACE_20260924.md)已实现并做几何/未训练前向检查；
一帧原始Street Development RGB/H3身份烟测通过。已在六场巷道修复RGB上完成同划分三seed ToF-only对CNH+RGB训练：每臂每seed在480帧dev的2880个查询均为TP0、FP0、FN408、TN2472；排序AUROC约0.677、AP约0.269，RGB增量接近0。此Development结果未建立可用模型或RGB收益；Street本地精简源只有8/960张RGB，四类环境对照尚未完成。
另立冻结[V2方位约束与训练集类别平衡协议](../research/active/dtr-r0/nearfield/CNH_RGB_ALLEY_DEV_V2_PROTOCOL_20260925.md)后，同划分三seed的ToF-only在dev检出148–160/408个正例查询、误报486–512/2472个负例查询；CNH+RGB检出148、154、159，误报504、491、503。解除了全负退化但仍有高误报，预设规则判定无稳定RGB增益。
本轮[瓶颈定位](../research/active/dtr-r0/nearfield/CNH_ALLEY_BOTTLENECK_DIAGNOSIS_20260925.md)在现有数据上发现：三个V2融合checkpoint对dev RGB布局内打乱或置零几乎不变；同预算全分辨率可见深度学习臂弱，但直接用可见深度回投预定查询盒为TP405/408、FP0/2472。学习/表征与三训练布局覆盖是优先核查方向；该直接规则不独立证明标签精度。按“(b)弱则停下汇报”停止追加训练；预训练RGB和CNH对同响应标量均未运行。用户暂停City和完整164布局test，不新采集。
随后用户授权的[学习器诊断](../research/active/dtr-r0/nearfield/CNH_LEARNING_DIAGNOSTIC_20260925.md)已完成：960帧实际深度转换与规则读取一致，未重现全零；全深度32帧拟合AP0.1576，ToF-only为0.9966。无噪声H3直接几何dev AP0.8452，原模拟传感响应H3为0.2101；0.29不再有信息上限解释。固定列掩码与盒投影不一致、查询前汇聚是明确结构问题；未查出使旧结果作废的运行bug，未触发V2重跑。下一结构应显式查询盒→区/bin及图像区域；本轮只诊断并定义未来对象ID schema，不自动启动新模型实验或采集。
用户选择的完整164布局test当前真实布局和正式采集spec为0/164；[准备度记录](../research/active/dtr-r0/nearfield/CNH_FULL_TEST_READINESS_20260925.md)保留空槽计划，三张巷道候选图不可缩减替代。
能量、独立标签精度与正式隔离继续保持未完成，所有本轮采集/后处理进程已释放。
20布局完整配额与384布局尚未完成；不使用5m缓冲带、轮廓豁免或改门槛追认旧FAIL。
[工程回执](../research/active/dtr-r0/nearfield/CNH_SOURCE_ENGINEERING_20260924.md) ·
[方案与范围](../research/active/dtr-r0/nearfield/CNH_ROUTE_COMPARISON_PLAN_20260924.md)。

1. 已完成选定关键模型、基准与评估证据的副机备份及逐文件恢复校验；[范围与回执](operations/CRITICAL_EVIDENCE_BACKUP.md)明确未覆盖项，不能宣称全论文依赖已备份。
2. 已整理短中文状态入口并接入 Python CI；本地工具测试与 Windows 检出检查通过，Ubuntu 执行仍由 CI 覆盖。[测试与历史哈希差异](operations/PYTHON_CI.md)。
3. 提醒策略作为避障的有界工作包：固定检测输出，比较首次提醒延迟、重复提醒及关键事件被抑制次数；实际疲劳与 TalkBack 配合仍需交互验收。本次整理不启动新提醒实验。

## 决策和验证

新问题、额外预算、判定规则变化先交用户决定。用户已授权问题内，由执行者完成实现、必要机械修复、针对性验证和交付，不逐条请求许可。失败的冻结实验保留失败，不能事后换阈值/子集挽救；不同机制的新问题另行明确。

探索只保留问题、对照、判定、结果、下一步和必要身份信息。独立审计用于结论关键或存在具体完整性风险的工作，不是每次小实验的默认环节；自动断言总数不代表独立证据数量。正式论文数字和受保护数据继续遵守[正式治理](formal/RESEARCH_GOVERNANCE.md)。

`UNKNOWN`/`NOT_EVALUABLE` 不等于无障碍；Development 复用不恢复数据新鲜性。组件收益不自动成为整个系统收益。

## 已关闭的流程问题

ledger303 的原始回执已于 2026-09-22 恢复，并完成指定 operating-scope 实验登记。[修复记录](../research/active/dtr-r0/nearfield/LEDGER_REPAIR_20260922.md)只覆盖所述记录，不代表所有历史待登记项已补齐。历史快照内的 pending 是当时状态，不能作为今天的故障判断。

[整理前正文](operations/snapshots/CURRENT_DECISION_20260924.md)保存历史数字与决定；不作为新实验授权。
