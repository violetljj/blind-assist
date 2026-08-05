# DA V2 A2-392 时序 student P3 R0

日期：2026-08-05。协议在 P3 数据身份、A2 checkpoint 激活和新 clip P1 开启之前冻结。

终点：`P3_PROTOCOL_FROZEN_DATA_AND_A2_CHECKPOINT_ACTIVATION_PENDING_HOLDOUT_UNOPENED`

## 结论

P3 固定以 A2 的同一 DA V2 Metric ViT-S、同一 `392` 输入和唯一选中 checkpoint 为起点，
不再搜索尺寸、跳层、backbone 或量化分区。backbone 仍逐帧前向；训练单位固定为四个连续帧，
只增加一个非 recurrent 的轻量状态头和四类监督。A2 checkpoint 的真实文件哈希以及新数据
manifest 尚未绑定，因此本文件授权协议实现和数据准备，不授权启动训练。

## 四类监督

1. 单帧深度完全继承 A2：log-depth SmoothL1、log-depth gradient、median log-scale。
2. clearance 直接监督三条横向 band 的 `C_t - C_t-1`，不再期待深度 loss 间接产生稳定
   clearance；任一端 UNKNOWN/无有效 clearance 时该 delta 不计入回归。
3. 状态头对每个 band 的九类 `(previous, current)` 转换做交叉熵，状态词表只含
   `CLEAR / OCCUPIED / UNKNOWN_GROUND`。
4. UNKNOWN 头显式学习 abstention。teacher age 大于 `0.5 s`、ToF 无效、teacher 无效，或
   stop-gradient 的 teacher/student mean absolute log-depth disagreement 大于 `0.20` 时，目标
   一律改为 `UNKNOWN_GROUND`。

轻量头只读取逐帧 student depth 的 `4x4` pooled log-depth、相邻差和顺序固定的四个证据标量
`teacher_age_s / tof_valid / teacher_valid / mean_abs_log_depth_disagreement`；没有
ConvLSTM、视频 Transformer、recurrent head 或新视频 backbone。它直接输出 clearance delta、
state transition 和 UNKNOWN logits，不增加第五类 absolute-clearance loss。

## clip 与数据防火墙

- clip 长度固定为 4，不把 3/4/5 当作搜索空间。
- 每帧必须携带 source-native 的整数纳秒时间戳；相邻帧严格递增且间隔不超过 `0.5 s`。
- train、training-only validation、新 sealed holdout 三者按 video parent 完全隔离；同一帧不
  得重复进入多个 clip。
- 旧 120 帧 P1 及其所有 parent 只进入 exclusion ledger，不能再参与 loss、checkpoint、阈值
  或模型选择。
- 新 holdout 的公开文件只含 clip/frame identity 和 `sealed_target_id`。teacher depth、clearance、
  state、teacher/ToF validity 等目标必须放在独立密封 bundle，trainer 不可读取。

manifest 校验器会对 parent 泄漏、伪时间戳/倒序、超过时距、重复帧、旧 P1 parent 重用、
holdout 标签泄漏，以及无效证据却给确定状态等情况 fail closed。

## 新 clip-based P1

开启前至少需要 32 个完整可评价 clip、8 个 video parent，并且四类关键转换各至少 8 个：
`CLEAR->OCCUPIED`、`OCCUPIED->CLEAR`、known->`UNKNOWN_GROUND`、
`UNKNOWN_GROUND`->known。达不到就返回 `P3_CLIP_P1_NOT_EVALUABLE_NO_SUBSTITUTION`，不补帧、
不换 cohort、不降门。

统计以 clip 指标和 transition 行为单位；置信区间按 video parent 聚类 bootstrap，不能把连续
帧当独立样本。checkpoint 与候选 cache 哈希锁定后只允许开启一次。静态深度、clearance
delta、转换、abstention、false-clear 和有效性门全部通过才晋级，不做加权抵消。

## 串行晋级

P3 A2-392 temporal student 通过新 clip P1 后，才允许叠加已经冻结的 A5S selective W8A16；
量化后先在另一个预绑定质量门再次证明非劣，之后才做 QNN/HTP 转换。第一轮设备 canary
只固定 `YOLO/seg 15 Hz、student 10 Hz、teacher 2 Hz、ToF 硬件稳定频率`，只证明链路可运行，
不搜索 cadence。任何阶段失败都停在该层，不能并行调整下游层来解释或挽救结果。

当前证据上限仍是 Development 协议与实现准备，不是准确率、Android 性能、产品或安全结论。
