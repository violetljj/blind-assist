# BlindAssist 当前本地证据卡

## LOCAL-01：来源和程序化证据状态

- `evidence-v4` canonical 根门曾全绿，根报告 SHA256 为 `32968a7afa081f122cee463e6578feba6efea65172f81a9a0d4341dbf7af23d4`；随后模型实验转到更新的 real-only `sanpo-v4-real-canonical-r3-20260713`。
- 最新 real-only r3 包含 600 条 train/dev（400 train、200 dev）和 120 条 blind，reviewed manifest 共 720 条；train/dev 全部为 `source_ground_truth`。最终 10 项检查全绿，training gate SHA256 为 `4c68e43494012f0499d8f9f01a5160a80276682fcd2e78a6ac5ca4cf98a1d5e1`，assembly report SHA256 为 `f7f7b11e4ca0f733dd4c5ccfb9f01ccf30548014c406edd72f308bb1fd6967b5`。
- 重要时间边界：`SANPO_GPU_UTILIZATION_2026-07-13.md` 开头“训练仍关闭”描述的是较早阶段；后续验证段和 P0 审计记录了闭环后的真实重跑。报告不能继续把来源未闭环当当前阻塞。
- 支撑位置：`docs/SANPO_GPU_UTILIZATION_2026-07-13.md` 的“验证与限制”、`docs/SANPO_P0_SEED_FACTOR_AUDIT_2026-07-13.md` 及本地 r3 manifests。
- 可支持论点：当前阻塞已经从数据来源授权转移到模型质量稳定性；来源门闭合不等于模型可以晋级。

## LOCAL-02：旧训练失败与图结构修正

- 旧图错误地把 LR-ASPP 高层分支接到 1/16 浅层，后半段主干未进入 functional graph，参数量只有 197,212；修正后为 670,588。
- 修正模型的 batch 64 首轮真实重跑 dev mIoU 为 0.1711，boundary/curb IoU 为 0.0000144。该结果证明“修图 + GPU 吞吐”没有解决优化和细边界质量。
- 支撑位置：`docs/SANPO_GPU_UTILIZATION_2026-07-13.md` 的“根因与修正”“验证与限制”。
- 可支持论点：后续结构实验必须以固定优化步数、小 batch、多 seed 和边界专项指标比较，不能把吞吐配置当训练配置。

## LOCAL-03：seed 高方差的主因定位

- 384×384、head-only、固定 sampler、三个 model seed 的 selection score 范围为 0.1739–0.4424；固定 model seed、改变 sampler 的范围为 0.4312–0.4424。
- 两个描述性跨度相差约 24.1 倍；五组 worst scene 均为 `step_curb`。
- 当前证据只能定位到“初始化及与 model seed 绑定的 Torch 随机状态”，不能声称已分离纯初始化因果，也未估计交互项。
- 支撑位置：`docs/SANPO_P0_SEED_FACTOR_AUDIT_2026-07-13.md`。
- 可支持论点（P0 当时结论）：P1 必须优先验证 head 结构与数值稳定性；该实验现已完成，结果见 LOCAL-06。若完成 P2 后仍高方差，再进入 2×2 交互或更复杂正则。

## LOCAL-04：训练和评价合同

- 训练单位是 optimizer step，默认三 seed；checkpoint 使用 dev mIoU 与 boundary IoU 的调和评分。
- 数据采样按 source session 平衡并有 rare-class guided crop；损失为 capped weighted CE + Dice + focal。
- `unknown_nonwalkable` 是显式 abstain；报告 coverage、precision、recall、IoU 和 covered accuracy，不能用全 unknown 或从不 unknown 刷分。
- 候选依次通过 `offline_training_quality -> int8_fidelity -> device_event`；任何单门绿色都不授权替换 App。
- 支撑位置：`docs/SANPO_TRAINING_PROTOCOL.md`、`docs/SANPO_CANDIDATE_PROMOTION_GATES.md`。
- 可支持论点：所有论文机制必须嵌入既有三段门，不得创造旁路或读取 blind 调参。

## LOCAL-05：事件层和生产边界

- Oracle v2 在首个 30 帧否定集把错误提醒率从 90.0% 降到 3.3%，说明语义边界不能直接等同障碍事件。
- 扩展 90 帧连续序列中，危险提醒召回 88.9%，但错误提醒率 25.9%；主要问题是登阶后的重复提醒和边界区域被升级为 generic obstacle。
- 当前事件级晋级指标包括 event recall、critical miss、false alerts/min、post-event clearance、repeated alerts 和 P95 latency。
- 支撑位置：`docs/SANPO_TRAVERSABILITY_BASELINE.md`、`docs/SANPO_CANDIDATE_PROMOTION_GATES.md`、`docs/DETECTOR_BENCHMARK.md`。
- 可支持论点：时序研究必须优化事件生命周期和错误提醒，而不是只平滑 mask；逐帧一致性不能替代事件级成功。

## LOCAL-06：P1 LR-ASPP 对齐已完成，但没有关闭方差

- P1 已完成四组、每组五个 head-only 短跑。保留 `lraspp_sigmoid_no_pooled_bn_v1` 和 OS8/OS32 默认合同；OS4/OS32、OS4/OS16、OS8/OS16 均被拒绝。
- P1-A 最佳 mIoU/boundary 从 P0 的 `0.4344/0.4506` 提高到 `0.4642/0.5235`，但固定 sampler 的 model-seed selection range 从 `0.2685` 扩大到 `0.2951`；它提高上限，没有解决稳定性。
- OS4 detail 的两个 seed boundary IoU 坍塌至 `0.0271/0.0130`；OS4/OS16、OS8/OS16 的最佳 selection 仅 `0.0968/0.1549`。
- 保留权重通过跨后端等价：max abs `1.7524e-05`、argmax agreement `1.0`；但未生成 INT8、未运行设备门，所有候选仍为 `do_not_replace_default_model`。
- 支撑位置：`docs/SANPO_P1_LRASPP_ALIGNMENT_2026-07-13.md`。
- 可支持论点：近期不再重复 OS4/OS16，也不直接进入 Mobile-PID；先执行 P2 确定性 quota sampler。若方差仍高，再做 2×2 交互/初始化稳定性审计，结构扩展必须重新满足进入门。

## LOCAL-07：标注质量合同缺口

- 最新 real-only r3 的 `training_manifest.jsonl` 将 600 条像素标签统一记录为 `label_authority=source_ground_truth`，但未透传 SANPO 源帧的 HUMAN/MACHINE annotation quality。
- 一次性本地派生审计使用 `reviewed-source-manifest.jsonl.image_sha256` 回连 assembly recipe 所列 14 个 draft manifest 的 `source.sha256`，读取 `source_annotation_quality`；720/720 匹配：train 为 HUMAN 38/MACHINE 362，dev 为 HUMAN 39/MACHINE 161，blind 为 HUMAN 30/MACHINE 90。训练/开发集中 MACHINE 占 523/600，即 87.2%。这些值不适用于旧 evidence-v4，且回连尚未成为正式 schema、sidecar 或 gate，trainer 报告本身不可见。
- 因此目前无法在训练报告中诚实计算 HUMAN-only、MACHINE-only 质量或 teacher disagreement，也不能直接把 UPC/SWSEG 作为已满足前提的实验。
- 可支持论点：先扩展 canonical manifest、source inventory 和训练报告的 annotation-quality 字段，再进行噪声分层或半监督实验。
