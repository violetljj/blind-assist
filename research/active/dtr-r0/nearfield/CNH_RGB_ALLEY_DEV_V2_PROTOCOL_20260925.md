# 巷道 CNH / RGB Development V2：方位查询约束与训练集类别平衡

状态：`FROZEN_BEFORE_V2_GPU_RUN`。这是独立的前瞻性 Development 机制检查，不改写 V1 失败结果，也不改变正式 CNH 路线的两 seed / 距离 / 事件门槛。V1 在三个 seed 的固定 logit 0 阈值下，两臂均为 0/408 查询召回；事后排名诊断只用于提出本机制，不能作为 V2 的独立确认依据。

## 问题与唯一改动

V1 的六个查询可以从全部 64 个 ToF 分区聚合，且普通 BCE 面对稀少正类。V2 同时固定以下两个机制，作为**一个**预声明组合假设，不把结果归因到其中单独一项：

1. 8×8 ToF 分区按行优先编号，列从图像左到右。`left_HEAD/BODY` 只看列 0–2（各 24 区），`centre_HEAD/BODY` 只看列 3–4（各 16 区），`right_HEAD/BODY` 只看列 5–7（各 24 区）。掩码在注意力 softmax 前固定为不可学习的布尔值，禁用分区权重严格为 0；HEAD/BODY 各自保留可学习查询，但共用其方位列。上下方向不额外施加先验。
2. 每查询仅从 **480 帧 train** 的已知标签计算 `pos_weight = 训练负例数 / 训练正例数`，对六查询 BCE 的正例施加权重。`UNKNOWN=-1` 不参与权重或损失。任一查询训练正例或负例为 0 时整轮拒绝，不能从 dev 借用统计量或临时调权。

ToF-only 和 CNH+RGB 两臂使用完全相同的 V2 模型、掩码、训练权重、初始化 seed、shuffle、batch、学习率、训练轮次和阈值。前者 RGB 张量精确置零，后者使用已修复的左相机 RGB；其余输入均为相同 H3 CNH、ambient、同响应标量/有效位和 age=0。RGB 固定缩为 128×72，4×4 patch 编码；视锥支持仅按精确 K、相机–ToF 位姿和图像/特征尺寸缓存，禁止使用场景深度、实例 ID 或标签构建支持。

## 冻结数据、预算与输出

- 六个巷道布局、每布局 160 帧：train straight/L/T 共 480 帧，dev straight/L/T 共 480 帧。沿用已冻结分区，不移动帧或布局。train/dev 的资产族隔离另见现有审计；各布局内帧及每帧六查询相关，不能当作 2,880 个独立样本。
- 六布局修复 RGB/原 ToF/标签绑定入口：`artifacts.local/evidence/cnh-alley-rgb-replay-six-20260925-v2/collection-overlay.json`，SHA256 `f72425577e4e06f9a95e39ba2ef0a9b3d6fc087157e74d31bcfce7927ba4840f`。脚本继续核验各 overlay、RGB、camera、物化 observations/targets 与 frame_key 哈希。
- 原冻结 train/dev 方案：`artifacts.local/work/cnh-route-comparison-20260924/plan/street-alley-merged-partitions-20260924-v1.json`，SHA256 `7ae115183bab26876eedc4d9bd3b11cb0b7613c2b56dddb3055266fe1a586a9b`。只读取这六个 Development 布局；Street worker RGB 不齐，Street 不纳入此轮。受保护 test 不读取。
- Seeds **20260924、20260925、20260926**；每臂每 seed **24 epochs**，batch **16**，AdamW 其余参数学习率 `3e-4`、RGB 编码器 `3e-5`、weight decay `1e-4`、24 epoch cosine。只训练六项占用 BCE；距离头无可用监督，不报告其数值。取固定最后一轮，无 checkpoint 或阈值选择；占用 logit **≥0** 报正。
- 每 seed/臂记录 train 损失轨迹、dev 总数及逐布局 TP/FP/FN/TN、召回分母 `TP+FN`、查询级误报分母 `FP+TN`、`UNKNOWN`、模型和预测哈希。保留原始预测，不额外用 dev 拟合校准。

## 预声明判定与停止

完整性要求：六个运行、全部输入身份与哈希、480/480 分区、相同 6 查询顺序、有限输出和全量分母。否则记 `INVALID`，只修复实现或输入错误，不从坏输出作效果解释。

固定阈值下，若任一 seed/臂 `TP=0` 或 `TN=0`，判为 `COLLAPSED_IN_AT_LEAST_ONE_ARM_SEED`，该 V2 组合没有解决双侧退化。若全部 seed/臂 `TP>0` 且 `TN>0`，只判 `NONCOLLAPSED_DEVELOPMENT`。RGB 的附加信号还须三个 seed **每个**满足 `RGB TP > ToF TP` 且 `RGB FP ≤ ToF FP`，才标 `CONSISTENT_DEV_PARETO_SIGNAL_ONLY`；否则标 `NO_CONSISTENT_DEV_PARETO_SIGNAL`。所有实际召回和误报分母仍需列出，不能用此布尔判定隐藏 FP 规模。该规则不触发 App、正式基线、独立测试或安全收益宣称。

第一轮完整输出后停止。不基于 dev 改掩码列、class weight 公式、阈值、训练轮次、种子或 checkpoint；任何后续机制另立 Development 协议。V2 既有两个联动改动，若表现变化，不能归因于 RGB、掩码或类别平衡中的某一个而不做新的隔离对照。

## 执行入口

`cnh_rgb_alley_v2.py` 为唯一训练入口；`test_cnh_rgb_alley_v2.py` 的形状、掩码和 train-only 权重测试必须先通过。City1 UE 占用 GPU 时只做 CPU 测试，不启动本轮 CUDA；得到 GPU 空闲通知后运行一次新输出目录，并保存代码、协议和输入哈希。
