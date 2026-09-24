# 六巷道可见深度上限诊断（Development）

状态：`COMPLETED_DEVELOPMENT / LEARNED_DEPTH_WEAK / DIRECT_GEOMETRY_STRONG`。[结果](CNH_RGB_VISIBLE_DEPTH_CEILING_RESULTS_20260925.md)。GPU 运行前冻结的原文保存在 [冻结快照](CNH_RGB_VISIBLE_DEPTH_CEILING_PROTOCOL_FROZEN_20260925.md)，SHA256 `90b47a1316075984cd9eb473ec49135434a9fb0a4f5c1d6ad8d86c4e554404a0`，与训练回执中的协议哈希一致。只用现有六布局 480 train / 480 dev、每帧六查询，不采集 test 或 City，不更动 V1/V2 结果。三 seed 20260924/25/26、24 epochs、batch 16、V2 方位掩码、train-only 正类权重、AdamW 学习率与 24 epoch cosine、最后一轮、logit 0 阈值均沿用 V2。每臂保存 checkpoint、逐帧 logit、总/逐布局计数和哈希；排名指标从原始 logit 另算，不在 dev 上选阈值。

输入只取 overlay 哈希绑定的原始 `depth_left.exr`（640×360、左相机首个可见表面的 axial Z）、有效掩码和公开 `K/T_camera_tof`。目标只用于损失与评价；标签使用的导出三角面、插入物 ID 和网格不进入任何输入。上限是**可见表面代理**，不是完整物理网格的无噪声同义输入。

1. `PERFECT_TOF_H3`：原来的 8×8 分区、每区 16×16 固定角采样；把可见深度依 K 换成径向距离，按 H3 的 16 个 300.2784mm bin 作固角加权直方图。不加反射率、光子/环境噪声、脉冲响应、串扰或检测门槛。范围外/无效射线保留在分母；最近可见采样距离及其有效位从同一射线导出。RGB 与 ambient 为零。输入尺度仍经过 V2 的有符号 `log1p` 变换，模型结构与 V2 ToF-only 相同。
2. `VISIBLE_DEPTH_FULL`：原生 640×360 float32 Z 保留，不缩图、不量化。第一层仍是 V2 的不重叠 4×4 patch 卷积，故随后输出为 160×90；三个通道为 `log1p(valid Z)/log(101)`、有效位、零。其后复用 V2 的每区公开视锥池化、64 维 token、固定方位掩码和六查询头；CNH/ambient/scalar/age 为零。仅第一层输入的物理意义不同，训练预算与优化器分组不变。这是匹配训练预算的深度学习臂，不能视为数学上限。

另保留一个无需学习的几何 oracle：逐像素以公开 K 将有效的首可见深度回投，若点落入任一事先定义的六个闭合查询盒就报该查询。它不调用目标计算函数，也不读取导出三角面。标签只用于事后列联；负例出现可见点或正例无可见点均逐项报告。这个规则与查询盒定义共享几何先验，所以只检验可见性与目标对齐，不是可部署模型或独立标签验证。

解释规则：若直接几何 oracle 很强而学习深度臂弱，优先怀疑表征、优化和三布局训练覆盖；不能据此宣称标签/查询本身不可学或 RGB 无用。若正例大量无可见点，才单列可观测性限制；完整标签网格比首可见深度含更多信息。全部结果只属 Development。
