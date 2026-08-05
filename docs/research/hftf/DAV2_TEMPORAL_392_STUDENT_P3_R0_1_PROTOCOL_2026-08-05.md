# DA V2 A2-392 时序 student P3 R0.1 激活前修正

日期：2026-08-05。R0.1 是已提交 R0 的 successor；R0 保留为历史，不重写、不重标。

终点：`P3_R0_1_PRE_ACTIVATION_CORRECTION_COMPLETE_BINDINGS_PENDING_HOLDOUT_UNOPENED`

## 结论

R0.1 修复监督语义和 P1 判定完整度，但仍不授权加载模型或训练。唯一 evidence builder、
监督 mask、holdout 白名单、coverage receipt、九类权重公式和 clip-P1 evaluator 全部在数据、
A2 checkpoint 与 holdout 开启前冻结。

## 唯一证据链

`build_temporal_evidence(...)` 只接受真实 sample/teacher 时间戳、teacher/ToF validity 和由冻结
父 A2 checkpoint 预计算的 disagreement。它一次生成 head features、teacher depth frame mask、
clearance frame mask 和 external-abstention target；结果全部 detach。训练中的 P3 student 不参与
标签构造，调用方禁止自行拼第二套 evidence。

teacher 无效、age 大于 `0.5 s` 或 teacher pixel 无效时，log-depth、gradient、median scale
相应 frame/pixel 都被屏蔽。ToF 无效不屏蔽 teacher depth，但屏蔽依赖 ToF 的 clearance delta。
冻结 disagreement 只监督 abstention，不改写 geometry transition，也不移除 otherwise-valid
clearance delta，因此困难样本不会因当前 student 变差而自动退出时序监督。

## 几何状态与外部拒答解耦

geometry transition 辅助头只读取前后 pooled log-depth、差分和真实 `delta_seconds`，学习固定
九类 `CLEAR/OCCUPIED/UNKNOWN_GROUND` 几何转换；teacher stale、ToF validity 与 disagreement
不可进入该头，也不可改写它的 target。external abstention 头单独读取 evidence，logit `>=0`
时覆盖冻结 depth-to-geometry operator 的状态为 `UNKNOWN_GROUND`。

clearance-delta 与 transition 头都是训练辅助头：P1 的 absolute clearance 和 geometry state 仍由
既有冻结 depth-to-geometry operator 产生，避免辅助头凭空发明 frame-zero state 或第五类
absolute-clearance loss。

train/validation manifest 必须输出九类有效 transition 的精确分布。九类必须各有正支持；训练
权重只能由 train counts 使用 effective-number 公式 `beta=0.999` 生成并归一到均值 1，禁止手填、
禁止读取 validation/P1 结果调权。

## Holdout 防泄漏

sealed holdout frame 改为严格字段白名单：`frame_id/video_id/parent_id/timestamp_ns/`
`sealed_target_id/rgb_identity/rgb_sha256`。任何额外字段，包括换名的 `target_state` 或
`clearance_target`，都会拒绝。identity manifest 不内嵌自身 SHA 或 coverage receipt SHA，避免
形成不可解的循环哈希；两者只在外部 activation receipt 中互相绑定。

密封方另行生成不含逐条标签的 coverage receipt，绑定 identity manifest、sealed bundle 和
coverage producer SHA256，并只公开：可评价 clip 数、video-parent 数、四类关键转换计数及九类
geometry transition 分布。至少 `32 clips / 8 parents / 四类关键转换各 8`；不足时固定返回
`P3_CLIP_P1_NOT_EVALUABLE_NO_SUBSTITUTION`。

## 数值完整的 clip P1

所有 CI 以 video parent 为 cluster，确定性抽样 `10,000` 次；每次抽 parent 后带入该 parent 的
全部完整 clips。loss 指标使用 candidate-minus-canonical 差值的单侧 95% 上界，score 指标使用
差值的单侧 95% 下界，absolute ceiling 使用 candidate 的单侧 95% 上界。少于 95% draws 可定义
或任一 gated metric 非有限，一律失败。

继承旧 P1 的边界：pixel coverage drop `0.01`、AbsRel `+0.02`、scale-aligned AbsRel `+0.01`、
clearance MAE `+0.025 m`、clearance-delta MAE `+0.015 m`、ground recovery drop `0.01`、
false-clear `+0.01`。新增 transition Macro-F1/exact drop 各 `0.05`、delta-pair coverage drop
`0.01`、invalid-to-known `+0.01`、valid-to-unknown `+0.02`。

absolute ceilings 固定为 clearance MAE `0.25 m`、delta MAE `0.15 m`、false-clear `0.05`、
invalid-to-known `0.05`、valid-to-unknown `0.10`。false-clear 分母是所有 externally-valid、
truth-known band-frame decisions；candidate UNKNOWN 不退出分母，另由 valid-to-unknown 门约束。
17 个门全部 AND 通过才允许进入 A5S，不做加权抵消。

## 唯一下一步

下一步仍不是训练，而是生成 activation receipt：绑定 A2 checkpoint/训练 receipt、冻结 A2
disagreement cache、旧 P1 ancestry exclusion ledger、train/validation/public holdout manifest、
九类 counts/weights、sealed target bundle、coverage receipt，以及 trainer/module/evaluator/test hashes。
必须同时证明输出目录不存在、`model_loaded=false`、`holdout outcomes_opened=false`，才可第一次
加载模型。

R0.1 没有触碰量化分区、跳层、输入尺寸、clip 长度、cadence、QNN、Android 或旧/新 P1 结果。
