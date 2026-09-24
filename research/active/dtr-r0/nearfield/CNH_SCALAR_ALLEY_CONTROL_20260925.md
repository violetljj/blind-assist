# 同响应标量对 CNH：巷道 Development 控制臂

状态：`NOT_RUN_STOPPED_BEFORE_GPU`。现有上限诊断的全分辨率可见深度训练臂仍弱且不稳定，触发用户“(b) 弱则停下汇报”的停止规则，因此本控制臂没有训练、checkpoint 或 dev 指标。以下仅保留已准备的设计与脚本，不构成 CNH 优于或不优于标量的证据。这是一项现有数据上的控制训练，既不新增采集，也不改变原 V2 回执或正式七组路线。问题是：在固定视锥查询网络里，直方图是否比**同一 ToF 响应**导出的标量提供额外任务信息。

输入沿用六份巷道修复 RGB/原 ToF overlay 及原 train/dev 分区：480/480 帧，train 与 dev 各直巷、L、T 三处已披露作者场地，受保护 test 不读取。标量臂 RGB 全零，保留与 V2 ToF-only 相同的 ambient、scalar/valid、age=0、查询方位掩码与相机标定支持；唯一任务输入差异是把 V2 的有符号 CNH bin token 改为同响应 `distance_m` scalar token。V2 的 CNH 臂本来已在公共 metadata 中持有这个标量，因此此控制测的是**直方图的增量**，不是有无距离读数。冻结 overlay SHA256 `f72425577e4e06f9a95e39ba2ef0a9b3d6fc087157e74d31bcfce7927ba4840f`、分区 SHA256 `7ae115183bab26876eedc4d9bd3b11cb0b7613c2b56dddb3055266fe1a586a9b`。

严格匹配 V2 的 seeds 20260924/25/26、每 seed 24 epochs、batch 16、train 已知标签逐查询负/正权重、AdamW 主干学习率 3e-4、RGB 编码器 3e-5、weight decay 1e-4、cosine；固定最后一轮及 logit≥0。标量臂为三次新训练，与已保存的 V2 ToF-only CNH 三次结果配对，V2 回执 SHA256 固定为 `719e224ce138f6bec92e7768a5e17dfe4bfebdc537507ca74fcf80e91cf0de85`；不重训或调阈值。脚本须验证原 overlay/分区/V2 回执哈希、训练标签正负俱有、六查询顺序和所有预测有限；保存模型/预测及其哈希。

主要比较为 dev 的查询级 AUPRC、AUROC 及其按帧 bootstrap 95% 区间，另列每 dev 布局和固定阈值 TP/FP/FN/TN、UNKNOWN。按帧重采只描述这六处 Development 场地内的波动，不能当作跨新物理场地置信区间。预声明读法：三 seed 的 CNH−scalar AUPRC 若同号且其配对按帧区间不跨 0，仅称 `CONSISTENT_DEV_INCREMENT_ONLY`；否则 `NO_STABLE_DEV_INCREMENT`。不以单个 seed 或固定阈值计数宣称正式 CNH 贡献。一次三 seed 完整输出后停止；不按 dev 结果更换训练权重、阈值、轮次或场地。
