# 巷道 V2 冻结预训练 RGB 编码器对照（Development）

状态：`NOT_RUN_STOPPED_BEFORE_GPU`。协议、代码和 CPU 结构检查已完成，但在任何预训练对照 GPU 训练前，现有上限臂发现可见深度几何规则可检出 405/408 且无误报，而同 V2 设置的全分辨率深度训练臂两 seed 弱、第三 seed 全报正。按用户“(b) 弱则停下汇报”的停止规则，此对照不执行；以下保留为未运行的冻结设计，不构成实验结果。这是独立于 V2 的诊断设计，不改 V2 冻结结果、阈值或受保护测试权限。只涉及原六个巷道 Development overlay、V2 分区与标签，不采集新帧，不运行 City 或 test。

## 固定输入与编码器

- Collection：`artifacts.local/evidence/cnh-alley-rgb-replay-six-20260925-v2/collection-overlay.json`，SHA256 `f72425577e4e06f9a95e39ba2ef0a9b3d6fc087157e74d31bcfce7927ba4840f`。
- Partition：`artifacts.local/work/cnh-route-comparison-20260924/plan/street-alley-merged-partitions-20260924-v1.json`，SHA256 `7ae115183bab26876eedc4d9bd3b11cb0b7613c2b56dddb3055266fe1a586a9b`。480 train 帧、480 dev 帧，三处 train 和三处 dev 巷道布局，六查询/帧。
- 本地 torchvision ResNet18 ImageNet 权重：`C:\Users\26442\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth`，SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。不联网。严格载入完整权重；仅使用 `conv1→bn1→relu→maxpool→layer1`，输出 `[64,18,32]`，与 V2 的 4×4 patch 网格 `[64,18,32]` 对齐。逐通道按 ImageNet mean `(0.485,0.456,0.406)`、std `(0.229,0.224,0.225)` 标准化。使用浅层是为了保留原网格，不做 dev 选择。
- 上述 conv/BN 全部 `requires_grad=False` 且始终 `eval()`。先对 960 张原 RGB 和一张全黑 RGB 逐位固定预计算特征；固定编码器无随机增强，缓存相当于每次前向运行。两臂用同一组权重、同一结构；ToF-only 为全黑 RGB 的冻结特征，融合臂为对应帧的 RGB 特征。特征缓存及哈希保留在本次输出目录。
- 每格仅加一个可训练 `64→64` 的 1×1 conv + GELU 适配器。随后沿用 V2 的 64 区视锥池化、CNH/ambient/同响应标量/有效位/age=0、左/中/右查询方位硬掩码、六占用头及未训练距离头。ResNet18 的卷积感受野跨格，故预训练特征可从池化支持区外吸收邻域像素；本实验检验这一具体编码器路线，不声称严格局部像素隔离。

## 固定训练与判读

- Seeds `20260924/20260925/20260926`，每 seed 成对训练 ToF-only 与 CNH+RGB。24 epochs，batch 16，训练帧顺序与 V2 相同，AdamW：适配器 `3e-5`、其他可训练参数 `3e-4`，weight decay `1e-4`，24 epoch cosine。仅对 train 中已知标签施加逐查询 `negative/positive` 的 weighted BCE；最后 epoch checkpoint，不选模、不校准。固定 logit `≥0` 计数只作描述。
- 主比较为 dev 查询级 AUPRC、AUROC；每 seed 保留逐布局结果、全量 logits、checkpoint 和 SHA256。95% 区间以每帧六查询为一单位，固定 seed `2026092501`，按三个固定 dev 布局分别重采 160 帧，1000 次 percentile bootstrap。这些区间条件于布局，不是新布局泛化置信区间。原 V2 双臂作为历史参照，预训练双臂自身成对比较；不把跨协议变化单独归因于预训练，除非同结构 ToF-only 对照稳定。
- 先检查完整性：权重/输入哈希、严格载入、固定编码器参数与 BN 状态、六个最终 checkpoint、同 seed 两臂初始可训练参数一致、480/480 和六查询身份、有限输出、已知/UNKNOWN 分母。任何失败记 `INVALID`，只修复机械错误。第一轮完整六组后停止，不用 dev 调编码器层级、学习率、阈值或 epoch。
- 判读为 Development 诊断：若 RGB 对同 seed ToF-only 的 AUPRC/AUROC 和固定阈值漏检/误报均无稳定改善，预训练浅层编码器未解当前瓶颈；若三 seed 的 AUPRC 均改善，才作为扩大训练布局的候选机制。无论结果如何，不推广为硬件、自然场景或安全收益。标签精度尚未独立准入。

执行脚本：`cnh_rgb_alley_pretrained_control.py`。`test_cnh_rgb_alley_pretrained_control.py` 的 CPU 结构/冻结/形状检查为 `2/2 PASS`；GPU 训练 `NOT_RUN`，无输出目录、模型或新 dev 指标。要恢复执行需另有明确决策，不能把此文件当作已完成的预训练对照。
