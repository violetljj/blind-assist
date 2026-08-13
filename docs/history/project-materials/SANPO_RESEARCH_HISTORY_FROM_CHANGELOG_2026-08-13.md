# SANPO 研究历史（从 CHANGELOG 归档）

状态：`archive / non-release / non-authoritative`

本文件于 2026-08-13 从根 `CHANGELOG.md` 原文迁出。它保留 v2/v3 公开连续序列扩展的
研究脉络和当时数值，不是发布说明，不拥有当前研究状态或晋级权限。当前 SANPO 状态见
[SANPO_CURRENT_STATUS.md](../../SANPO_CURRENT_STATUS.md)。

## SANPO v2 公开连续序列扩展验证

- 新增 real-only canonical v4：14 条互斥 SANPO 官方 session 组成 400 train + 200 session-held-out dev + 120 official-test blind，共 720 帧；四场景桶覆盖、官方 split、raw inventory、逐资产哈希与 blind policy 的 10 项总门全绿。新增 offline quality、INT8 fidelity、device event 三段独立晋级门，并把 input size / backbone alpha / decoder channels 的 canonical 配置哈希贯穿跨后端、导出和质量报告。
- 完成 step-budgeted、session-balanced、rare-class guided crop、CE+Dice+Focal、两阶段冻结/微调和三 seed 审计。最佳 384×384/alpha 1.0/decoder 96 单 seed 为 mIoU `0.4344`、boundary IoU `0.4506`，但其余两个 seed 为 `0.1804/0.1734` 与 `0.2498/0.1548`（mIoU/boundary IoU），稳定性不足。最佳候选离线门仍有三项 red，故停止在导出前，不生成 INT8、不运行设备事件门、不替换默认模型。
- SANPO v3 来源门禁升级为逐资产闭环：保存并绑定 Guide RGB/polygon、SANPO RGB/raw mask、assembly recipe 与逐样本 inventory；Guide receipt 对照远程 ZIP member/CRC/对象元数据，底层 raw asset 跨 train/dev/blind 复用会直接拒绝。修复发布后授权报告仍绑定 `.building` 的问题，新 evidence-v4 最终根门禁全绿。
- 使用修正后的 MobileNetV3 + LR-ASPP、RTX 5060、batch 64 重训；候选 dev mIoU `0.1711`、boundary/curb IoU `0.0000144`。新增 Torch↔TensorFlow 数值等价门；首轮暴露 CuDNN TF32 漂移，固化 TF32-off 精确执行契约后在原阈值下达到 `100%` argmax agreement（max abs `0.0000634`）。候选仍因质量门失败保持 `do_not_replace_default_model`，不导出 TFLite。
- 更正：此前 canonical green 与 GPU/TFLite 指标经最终安全复核降级为 audit-only；source inventory 和程序化原始证据闭合前不启动正式训练。训练入口已改为仅消费预生成门禁报告与 train/dev 哈希，不再读取 blind。
- 修复 LR-ASPP 误选浅层特征导致 MobileNetV3 后半段被裁掉的问题；模型参数量从 197,212 恢复到 670,588。新增不读取数据集的 GPU 吞吐工具；RTX 5060 上 batch 64 约 358 images/s、峰值约 2.88 GB，并设为训练默认值。

- SANPO v3 公开/程序化 source recipe 已构建 300 train/dev + 120 benchmark-only blind canonical 集并通过 SHA256 总门禁；程序化 dev/blind 只允许两份已 attested 来源 GT 的确定性组合，teacher/pseudo 仍禁止。
- 新增远程 ZIP64 Range 选择性下载、盲道占用确定性合成、backend-neutral Keras 模型定义和 Windows 原生 PyTorch CUDA 训练入口。RTX 5060 最佳全 INT8 候选 dev mIoU `0.3175`，但 boundary/curb IoU 约 `0.00038`，因此保持 `do_not_replace_default_model`，不进入 App assets。
- 新增三条 SANPO 官方 train session 的连续 50 帧 source package，共 150 对 RGB/panoptic mask；官方对象 MD5、逐文件 SHA256、唯一性与 manifest validation 均通过。素材仅保存在忽略的 `test-artifacts.local`，六场景未齐前不晋级 canonical。
- 公开数据 canonical builder 只接收 allow-list source adapter 和 SHA256 绑定的许可证、隐私、inventory 证据，直接传普通 manifest 不能进入 canonical 根；未知原生类别和未实现 mapper 一律拒绝。
- 新增标签权威分层：dev/blind 只允许 `source_ground_truth`；train 额外允许可验证的程序化标签和恰好两名独立 teacher 的共识伪标签，并绑定模型权重、逐 teacher 输出、共识 mask 哈希及 IoU/时序阈值。纯语义样本不得携带风险事件标签。
- 训练门禁：新增 SHA256-attested SANPO v3 总门禁。MobileNetV3 + LR-ASPP 训练入口只接受 canonical dataset root，先验证 300+120、四类掩码、hash、来源/隐私、train/dev/blind session 隔离和两个 `benchmark_only` blind session；报告非 green 时在 TensorFlow 导入前拒绝启动。该变更只影响本地 benchmark 训练工具，不改变默认 YOLO11n 或 app 版本。
- 新增公开 SANPO session 发现、review profile、不可变 sequence clone 与多序列合并工具；原始 RGB/mask 继续只保留在 `test-artifacts.local`。
- 新增连续稳定分割候选的单级晋升和反馈路径：仅适用于中心 `stairs` 或近场通用障碍，要求 `STABILITY_PROMOTED` 或 `MOTION_PROMOTED`，路沿仍不走普通障碍提醒路径。
- 最终 90 帧/3 序列真机结果：危险提醒召回 `88.9%`、主区域命中 `93.9%`、total P95 `58.405ms`，但错误提醒率 `25.9%`，未通过 `≤5.3%` 门槛；维持 `do_not_replace_default_model`、不训练模型。
