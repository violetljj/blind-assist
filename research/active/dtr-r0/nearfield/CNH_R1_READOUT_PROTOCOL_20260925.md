# R1 ToF 规则读出：执行前冻结协议

2026-09-25，EXPLORE / Development only。用户授权不训练网络的读出诊断。新指标须在本协议及实现提交后计算。旧QG-1/Part B协议与结果不修改；近场采集验收方案仍待审，不采集、不访问City/test。

## 输入与边界

主集为原六巷道480 train /480 dev，沿用QG-1绑定collection、partition与原标签，dev408正/2472负、train432正/2448负。次要Street为已存在的1920帧Development，明确枚举的物化根，禁止递归发现其他划分；只描述、不参与门槛或配置选择。数据入口验证manifest/观测/标签/原深度与camera哈希和frame_key；重新合成原seed原参数128bin后聚合H3必须逐位复现已存float32观测，失败则停，不解释算法差异。

固定入口（均相对`artifacts.local/`）：alley collection=`evidence/cnh-alley-rgb-replay-six-20260925-v2/collection-overlay.json`，SHA256 `f72425577e4e06f9a95e39ba2ef0a9b3d6fc087157e74d31bcfce7927ba4840f`；partition=`work/cnh-route-comparison-20260924/plan/street-alley-merged-partitions-20260924-v1.json`，SHA256 `7ae115183bab26876eedc4d9bd3b11cb0b7613c2b56dddb3055266fe1a586a9b`。Street根=`evidence/cnh-street-e2e-main-20260924-v2`与`evidence/cnh-street-e2e-worker-20260924-v1`，manifest哈希分别`224522daa0f3d1357538ad453cfa7a26389d4d2ee6201301fc4d1bf3f6aa68c1`、`f093a5ed14ba9efb1002e642f33e2ed309c09f5fb0d2808cf2248079a8a5540e`。副机已有960帧source可恢复到`evidence/cnh-r1-street-worker-source-20260925-v1/capture`，须保持原raw-manifest与每帧camera/EXR/mask哈希，不重新采集。源不可恢复则保留主集完成结果并明确Street未完成，不缩分母冒充1920。

本轮R2/R3使用仿真内部128bin，比实际H3的16bin提供更细距离信息，不冒充仅依赖已传输H3的设备算法；8×8×128原始buffer/速率能力未在硬件验证。旧模拟器将串扰定义为残余量；精确已知常数扣除是特权标定对照，不假定实机残余可被完美扣除。空间步长不自动证明10Hz；实际时序以元数据审核记录为准，不能直接声称时间累积带来更早报警。

## 固定读出定义

所有signed值保留，不截负、不删除bin0。参数沿用SensorParameters原值。原生角采样16×16/zone，查询盒及固角定义不变。

- R0：原H3逐区/bin查询权重线性求和，保留原float32存储精度，复现dev AP0.2101159982。
- R1_MEDIAN：在全部且仅alley train帧上，逐zone×128bin中位数作为固定偏置；从原float32 H3观测扣除8bin聚合偏置，再按H3权重评分。它可能包含静态场景回波，不称纯串扰估计。
- R1_KNOWN：原参数signal_counts×crosstalk_fraction乘原pulse矩阵对应0.04m行，逐zone相同，精确已知残余扣除后按H3评分。仅对照，不替代主方案门槛。
- R2_ONLY：未扣偏置的原raw128与原pulse矩阵逐模板相关，每模板除其平方和，按128bin精确射线/盒重叠权重汇聚。不是反卷积，保留尾部/边缘模板形状。
- R2_RAW_CONTROL：raw128不匹配滤波，直接按128bin几何权重评分；仅描述性分离更细bin与滤波效应，不参与选K和门槛。
- R2_MEDIAN：先R1_MEDIAN，再上述匹配滤波。R3主方案以此为起点。
- R3主方案：对偏置扣除并匹配滤波后的128bin响应，K∈{1,2,4,8}因果平均；分别不补偿/补偿。过去帧信号向当前更近距离移动，delta=(当前序号−过去序号)×步长×该zone固角加权方向余弦；out[b]=past[b+delta/RAW_BIN_M]，线性插值、窗口外零、不跨zone重投影。这是固定zone小平移近似，忽略遮挡和角迁移，不用目标真值运动。
- R3_ONLY：原signed raw128直接作同样K与补偿/不补偿平均，再8bin聚合及H3评分，单独识别时间平均作用，不混入R1/R2。

K=1不插值，逐位返回当前响应，R3主方案分数须与R2_MEDIAN逐位一致。每个layout/clip重置历史，按真实序号排序，禁止未来帧；K_eff=min(K,当前clip可用前缀)。主表保留全部帧，另对每个K报告K_eff=K子集及分母，阈值仍由全alley train确定，不为子集另拟合。未提供可靠clip/顺序的Street不伪造时间实验，说明不可评估原因。

## 选择、阈值、敏感性

**最佳配置只用alley train选择**：R3主方案补偿步长0.1m，K∈{1,2,4}按train pooled AP最大选择，同分优先较小K；K=8及所有dev/Street指标不参与选择。先持久化train选择回执再报告dev。R1_KNOWN不参与最佳选择，不允许事后切换主方案。各臂仅用全部alley train最大pooled F1定一个阈值，同分高阈值，直接应用dev及Street。

步长误差仅对选定K：0.07/0.10/0.13m，报告相同因果配置和train-only阈值，K不重选。即便K=1导致补偿相同，也照实判定严格大于门槛失败，不改选K。

噪声敏感性仅R0与已选定配置：原参数新seed基准、ambient×0.5/2、signal×0.5/2共5种（一次只改一个参数，两个×1为同一基准）。新seed为SHA256("cnh-r1-readout-noise-v1|"+frame_key)前16hex转int，强制不等于原seed；各条件使用同一新seed族以减少无关变化。每条件仅按相应alley train重估相同median偏置及阈值；K/滤波器/几何不变，不看dev调参。此为可重标定条件鲁棒性，不声称冻结硬件偏置跨条件通用。Street不参与重标定，主Street表仅原条件。

## 冻结门槛

1. 串扰检查：在原累计NEIGHBOR+XTALK但无计数噪声阶段，对其alley train估计median偏置后评分，dev pooled AP≥0.85为主判定；KNOWN对照另报，不能用它挽救主判定。该检查只验证所声明读出，不改变完整响应主臂。
2. 读出突破：仅由train选定的R3主补偿K≤4，dev pooled AP≥0.42。
3. 补偿有效：同一选定K补偿AP严格大于不补偿。
4. 鲁棒性：ambient×2与signal×0.5两条件，选定配置dev AP均严格大于同条件R0。

分别报告每项及整体合取，失败不换阈值/子集/模型。没过不证明信息理论或硬件上限；过了也只支持重复Development、未标定仿真和raw128条件下的规则改进，不证明更早提醒/真实融合收益。

## 输出与执行

每臂train/dev pooled AP、AUROC、6查询AP、train阈值及dev TP/FP/FN/TN；时间臂完整历史子集同指标/分母，Street同类描述性表。保存源身份、偏置、配置选择、每帧分数、噪声参数/种子、运行日志。输出`artifacts.local/evidence/cnh-r1-readout-20260925-v1/`，不可覆盖；单次执行最多1800秒，失败保留，不扩大科学网格或重复挑结果。CPU NumPy小矩阵，无网络训练、GPU任务或新采集。

提交前合成单测：偏置估计隔离train；改变未来观测不改变当前分数；运动平移方向/幅度；K1与R2逐位一致；匹配脉冲模板与查询权重。执行后报告`CNH_R1_READOUT_RESULTS_20260925.md`，短更新两个CURRENT，相关文件非force推origin/master，释放进程，保留证据及失败回执。
