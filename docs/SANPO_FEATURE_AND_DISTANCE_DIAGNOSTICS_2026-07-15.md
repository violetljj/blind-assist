# SANPO 特征与距离场根因诊断（2026-07-15）

## 结论先行

当前瓶颈不是“随机 head 没优化好”。两个独立的冻结基础表征 probe 均无法达到预注册的边界可分门；而 boundary/step/curb 在 train 与 dev 的像素覆盖又严重错配。因此，在补齐事件级真值和边界覆盖前，继续堆叠 head、初始化或距离场损失不能构成可归因的解决方案。

本页所有试验仅打开 hash-attested canonical train/dev；blind holdout 从未被读取，且没有写出可部署模型权重。

## 确定性 feature probe 对照

固定闭式 float64 ridge、按 sample ID 排序和每类等额取样；两个重复的系数及 dev argmax 完全一致。

| 冻结特征 | feature dim | global mIoU | boundary IoU | 可分门 `mIoU >= .35 && boundary >= .20` | 解释 |
|---|---:|---:|---:|---|---|
| 既有 MobileNetV3 raw OS8+OS32 | 672 | .3308 | .0297 | 否 | 原始 backbone 对边界几乎不可分 |
| Depth Anything V2 Small DINO 最后层 | 384 | .4235 | .1312 | 否 | 通用几何表征改善场景/障碍，但未恢复 boundary recall |
| Depth Anything + MobileNet raw OS8+OS32 对齐拼接 | 1056 | .4144 | .1384 | 否 | 简单特征拼接没有产生可用的互补边界信息 |

后两份新报告：

- `artifacts.local/evidence/sanpo-depth-anything-linear-probe-20260715/probe_report.json`，SHA256 `77035bec8a92b50bee1a470abdf6684041b5106f97c11f80b8808d5bae304a76`
- `artifacts.local/evidence/sanpo-depth-mobile-hybrid-linear-probe-20260715/probe_report.json`，SHA256 `c743e0ae403e0da6e01cc7c8e8a08aff3e5f1007fc7b4c16ca1b41f6c22f2b67`

这不是对 Depth Anything 或 MobileNet 的生产比较：两者都只是冻结诊断特征，不能据此替换任何端侧模型。

## 距离场辅助监督的独立可行性检查

以 signed、truncate=16 px、384×384 的现有 distance-field 合同计算真实 SANPO train/dev 覆盖：

| split | frames | 含 boundary 帧 | boundary 像素占比 | `abs(distance) < 1` 近边界像素占比 | mean loss weight |
|---|---:|---:|---:|---:|---:|
| train | 400 | 293 | .8569% | 3.8388% | .0802 |
| dev | 200 | 145 | 16.9761% | 8.0587% | .4061 |

dev/train boundary 像素占比约为 **19.8×**。因此现在把 distance SmoothL1 接入训练，会把“数据覆盖错配”和“损失是否有效”混在一起；无论结果正负都不能归因为 auxiliary loss。独立诊断报告为 `artifacts.local/evidence/sanpo-boundary-distance-aux-20260715/report.json`，SHA256 `1ba1a6e7c92d56f5886a01082bd26ed44b06655b5823da81b81236d34e997b9c`。

### 覆盖匹配后的单因素距离场短跑

为避免上述 19.8× 的 train/dev 混杂，新增 `run_sanpo_balanced_distance_ablation.py`。它只使用 canonical **train** 中的八个 session：将 `9m1-lq…` 与 `SRHpBZ…` 两个完整 session 作为 train 内隔离评测（100 帧），其余六个 session 优化（300 帧）。在 384×384，评测/优化 boundary 像素覆盖比分别为 `.0079325/.0087812`，即 **0.9034**，满足预先固定的 `[0.80, 1.25]` 门槛；canonical dev、blind 资产均未读取，且脚本不保存权重。

五个固定 model/sampler seed 对各运行 100 step，baseline 与 treatment 的唯一差异是一个从 `lraspp_fuse` 分出的 signed 16 px distance head（weighted SmoothL1，权重 `.20`）。结果并不支持该辅助头：distance-minus-baseline 的平均 mIoU 为 `-.001918`，boundary IoU 为 `-.000610`，harmonic selection score 为 `-.001230`；五对中仅一对 boundary IoU 上升，且其 mIoU 仍下降。完整报告为 `artifacts.local/evidence/sanpo-balanced-distance-ablation-20260715/report.json`，SHA256 `f6482912c37e111e08d17214b5f3b15b30e7b99e20a58880ac672c79842fabef`。

这是一条对当前配方（冻结 backbone、100 step、distance weight `.20`）的**负向证据**，不是对所有距离学习的普遍否定；它足以排除“先把距离头接上”作为当前主线。因为该试验只使用 source pixel geometry，它不产生 risk/event/lifecycle 真值、校准或 benchmark 结论，并始终保持 `do_not_replace_default_model`。

## 已排除与保留的下一步

- 排除：再做 head-only bootstrap、直接拼接基础特征、或继续为当前距离场配方更换损失/权重；覆盖匹配后的五组短跑已给出负向证据。
- 保留：首先按 session/scene 重建 boundary coverage contract；新数据必须保留对应的非事件/平行边界对，才能把 geometry 与 alert 语义分开。
- 风险轮廓与生命周期头采用分层监督：隔离 GPT/Codex 共识为 `hash_bound_model_consensus`；许可、哈希绑定的公开 RGB/source mask/GPT-VLM 可生成 `hash_bound_model_silver_provisional` 暂定训练标签。像素/距离场仍为 `auxiliary_only`；单次模型输出不能冒充已审计共识，也不能单独承担标定、blind 评测或默认模型替换。

### 风险轮廓 / 生命周期原型已落地，但尚无可训练事件集

新增 `scripts/sanpo_risk_lifecycle_prototype.py` 将事件合同落实为一个带分层来源的时序接口：外部 frame feature 进入两个 causal temporal convolution；episode 级输出为四类 scene hazard、两类 corridor relation 和 `should_alert` logit，逐时刻输出 `non_alert / approach / alertable / post_event` 四态 logits。target adapter 接受 `hash_bound_model_consensus` 的完整双模型审查报告，或有 CC-BY 来源哈希、模型/提示词 attestation 的 `hash_bound_model_silver_provisional` 报告；两者可授权研究训练，但仍要求 `production_model_replacement_authorized=false` 和 `pixel_supervision_role=auxiliary_only`。

这只是结构与标签转换原型：它没有 trainer、图片加载器、伪标签路径、权重保存或阈值校准。对长度为 5 的两条 dummy feature 序列的 Keras torch smoke 已得到 `hazard=(2,4)`、`corridor=(2,2)`、`should_alert=(2,1)`、`lifecycle=(2,5,4)`；其运行时合同仍为 `do_not_replace_default_model`。由于真实双审 96-episode 矩阵仍不存在，不能把这个原型当作已开始的训练或效果结论。

### 冻结 DINO 的走廊熟悉度诊断：可作为 unknown 辅助，不是 alert

按 [temporal-human 研究笔记](research/frontier-upgrade-2026-07/notes/temporal-human.md) 的最小可证伪版本，新增 `run_sanpo_corridor_anomaly_probe.py`：只从 canonical **train** 的 source-semantic `walkable` patch feature 拟合 32 维 PCA 重建子空间，再在未参与拟合的 canonical **dev** 计算高重建误差的 source-class outlier AUROC。它不训练 DINO、不写权重、不读 blind，也不把 source class 转成 risk/event/lifecycle 标签。

在 25,600 个 train walkable patch 上拟合后，dev 中 `boundary_step_curb / obstacle / unknown_nonwalkable` 相对 walkable 的 AUROC 分别为 **`.8911 / .9417 / .8578`**，all-nonwalkable 为 **`.8981`**；预注册的 `unknown >= .80 && boundary >= .65` 解释性门通过。报告为 `artifacts.local/evidence/sanpo-corridor-anomaly-probe-20260715/report.json`，SHA256 `c7ad14b2e9ada2637f65b6a2a4db6f20298a1d0ff350bb4668292bf34c053a61`。

这条结果仅证明泛化表征能形成有效的“非熟悉表面/运动”**辅助候选**，其唯一允许解释是 `unknown_motion_or_surface`；它不能判断 path intrusion、可通行性、何时提醒或何时清除。尤其是 static curb、围栏与真实 cut-in 都可能是 outlier，因此不得接入 alert、阈值校准、benchmark 真值或默认模型替换；下一步若实现端侧蒸馏，也只能以 abstain/unknown 辅助头的离线原型形式进行，直到获得人审事件数据。

对应的端侧可行性检验也已完成且**不通过**：`run_sanpo_mobile_unknown_distill_probe.py` 用同一个冻结 DINO-PCA 分数作 teacher，在 25,600 个 train 和 12,800 个 dev 固定采样位置上，以既有冻结 MobileNetV3 raw OS8+OS32（672 维）做闭式 ridge 复现。独立 dev 得到 `R²=-.1015`、Spearman `.6014`，未达到预注册 `R² >= .50 && Spearman >= .70`。报告为 `artifacts.local/evidence/sanpo-mobile-unknown-distill-probe-20260715/report.json`，SHA256 `ab84880d5026db94fe30699dbc04cdf30f7f0231b74ffcbb4d4b61e34defde1`。因此不进入小型 unknown head、非线性 student、SAM/ASAM 或超参搜索；当前 MobileNet 表征既不能分离边界，也不能线性承载这个 DINO 辅助分数。

## 自动补齐 auxiliary-only 边界候选的进度

为避免把数据问题伪装成 loss 调参，新增的候选计划器会从完整 public discovery 记录中排除 canonical 的 12 个 session，并只按公开 source mask 的 step/curb 覆盖排序。第一批 8 个候选均明确标为 `auxiliary_pixel_geometry_only`；它们被禁止用于事件/risk/lifecycle 标签、校准、benchmark 真值及默认模型替换。

前三条候选已完成 remote mask-only 精筛，均为 50/50 帧 geometry candidate；这只证明像素几何存在，不证明真实风险事件。随后下载 qtty 的一条 50 帧公开 RGB+mask draft：文件完整性通过、官方 train split 与 CC-BY 4.0 记录齐全，所有风险字段均为 null，且 `benchmark_ready=false`。抽检发现该场景行人密集，故生成整人/车辆脱敏副本；50 帧共有 684 个模糊区域。该副本仍为 `privacy_audit_required=true`，并因遮挡范围可能损伤表征价值而尚未获准使用。

### wBP 低密度候选的隔离复核

`wBPxyouyX9_6p1hGzIf8sYB9UPqpoBex` 是另一条通过 remote mask geometry 筛选的 official-train 非 canonical 胸部左相机候选。其 50 帧 RGB、50 张 source mask 和 50 条 draft manifest 记录完全一致；所有 risk/event/lifecycle 字段保持 `null`，且源数据只是 official `train` split。可见抽样不见近距离行人或可辨识面部，但仍存在车辆和街景标识，因此不视为隐私放行。

机器脱敏后只保留了 50 帧 RGB 副本：YuNet 人脸、LPD-YuNet 车牌和 YOLOv8n 整个人/车辆保守模糊共命中 48 帧、186 个区域。回执 `machine_redaction_receipt.json` 的 SHA256 为 `d99b794c235e0f24a0656fe8da3f1719b283a795571e6de323e03997a9501129`，它显式声明 `source_mask_role=auxiliary_pixel_geometry_only`、`risk_or_event_truth_present=false`、`privacy_audit_required=true`、`training_execution_authorized=false`。因而该副本只能作为待独立隐私审核的像素/几何辅助候选，不可用于任何风险、事件、生命周期真值、校准、benchmark 或默认模型替换。

因此下一条候选只要通过普通公开渠道可下载即可先获取；非 canonical session、remote mask coverage、低遮挡和隐私检查决定它能承担的诊断/标签角色，但不再作为下载前置门。source mask 或模型输出仍不能冒充客观事件真值。
