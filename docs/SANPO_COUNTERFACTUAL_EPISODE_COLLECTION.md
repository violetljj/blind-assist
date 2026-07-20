# SANPO 交叉反事实事件采集协议 v1

## 目的与边界

本协议把数据单位定义为连续 `episode` 和物理 `risk_event`，而不是相互高度相关的独立帧。首轮冻结为 **6 个 session × 4 个 scene × 每格 2 个 positive + 2 个 matched negative**，合计 96 个 episode、48 个正负 matched pair。

本轮只定义可离线执行的采集、标注和评价合同，不下载数据，不训练模型，不读取既有 blind holdout，也不授权导出、设备接入或替换 App 默认模型。像素 mask 可作为少量辅助证据，但不再是事件晋级的唯一真值。

机器可读合同位于 `configs/sanpo_counterfactual_episode_collection_v1.json`。配置中的 session 是待采集槽位，不代表数据已经存在或通过复核。

执行入口为 `scripts/validate_sanpo_counterfactual_episodes.py`。它只验证人工复核后的本地 episode manifest，强制来源许可/隐私证据与文件 SHA256、配对上下文、正负事件锚点和完整 session-scene 矩阵；缺任一项均 fail closed。只有带 `--require-complete` 且完整矩阵通过时才会报告 `training_eligible=true`，并且它始终输出 `production_model_replacement_authorized=false`。它不下载数据、不生成标签，也不替代人工的 `should_alert` 判断。

以 `configs/sanpo_counterfactual_episode_manifest_template_v1.json` 为字段模板，在 `artifacts.local/evidence/` 创建每批本地 manifest；模板本身是空的 `in_review` 文件，不能被当作已采集数据。正式复核后执行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts\validate_sanpo_counterfactual_episodes.py `
  --config configs\sanpo_counterfactual_episode_collection_v1.json `
  --manifest artifacts.local\evidence\counterfactual\<batch>\episode_manifest.json `
  --require-complete
```

缺真实人工复核矩阵时不要加 `--require-complete`，更不得以候选 mask 或模型推断绕过该门。

开始采集前，可先生成不含个人数据、文件路径或标签的 96-slot 清单：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts\generate_sanpo_counterfactual_capture_plan.py `
  --config configs\sanpo_counterfactual_episode_collection_v1.json `
  --output artifacts.local\evidence\counterfactual\capture_plan_20260714.json
```

该计划只列出待采集的正/负配对、时长、共享上下文、风险轮廓和生命周期模板。每个 slot 初始都是 `not_captured`，不能作为 manifest、receipt、标签或训练输入。

## 采集矩阵

六个 session 均须在同一设备、相机配置和连续采集上下文内覆盖以下四类 scene：

1. `parallel_boundary`：平行路沿/边界不应提醒；匹配的正例为边界进入或横断中心走廊。
2. `step_curb`：接近台阶或路沿为正例；已经通过、正在远离或仅位于远场为匹配负例。
3. `center_obstacle`：障碍侵入中心近场走廊为正例；外观相似但未侵入或仍处远场为匹配负例。
4. `lateral_pedestrian_or_ebike`：真实切入路径为正例；仅侧向经过、不进入路径为匹配负例。

每个 session-scene 单元必须包含两个 matched pair。每对由一个 positive 和一个 negative 组成，尽量保持地点、光照、相机姿态、物体类别和拍摄时段一致，只改变与提醒相关的运动或空间关系。不得用不同地点、昼夜或设备来制造容易识别的“伪反事实”。

每个 episode 必须持续 **10–20 秒**，保留采集前后上下文，不把同一连续事件裁成多个训练样本。validator 会拒绝短于 10 秒或长于 20 秒的 “complete” episode。原始视频只归属于一个 session；衍生帧、光流、mask 或 clip 不得跨 session/fold 复用。

## 事件标注

positive episode 至少包含：

- `risk_event_id`：同一物理事件稳定不变；
- `first_visible_ms`：风险对象首次可审计地出现；
- `alertable_start_ms`：第一次允许系统提醒的时刻；
- `passed_or_cleared_ms`：事件已通过、远离或清除的时刻；
- `expected_should_alert=true`。

三个时间点必须满足 `first_visible <= alertable_start < passed_or_cleared`。建议双人独立复核至少 20% 的 positive；`alertable_start` 或 `passed_or_cleared` 分歧超过 500 ms 时须仲裁。

negative episode 使用 `expected_should_alert=false`，三个正事件时间点必须为 `null`，并提供 `negative_reason`。matched pair 必须共享 `matched_pair_id`，同时记录允许保持不同的风险几何字段，避免把负例伪装成与正例完全相同的事件。

## 风险轮廓与生命周期目标

每条 episode 还必须记录 `risk_profile` 和 `lifecycle_intervals_ms`，它们是未来风险轮廓/生命周期头的主监督，像素 mask 仅作为辅助监督：

- positive：`primary_hazard_type=scene_id`、`corridor_relation=enters_or_blocks`、`lifecycle=approach_alertable_clear`；区间必须严格写为 `approach=[first_visible, alertable_start]`、`alertable=[alertable_start, passed_or_cleared]`、`post_event=[passed_or_cleared, duration_ms]`。
- matched negative：`primary_hazard_type=scene_id`、`corridor_relation=outside_or_nonblocking`、`lifecycle=no_alert`；区间必须为 `non_alert=[0, duration_ms]`。

这些整数毫秒区间把“看到了什么像素”与“何时应提醒、何时清除”拆开。静态围栏、长椅、广场边界或固定结构即便在 mask 中进入保守走廊，只要人工核定为不阻断目标路线，就必须落入 `outside_or_nonblocking/no_alert` 的反事实负例；不能被像素几何升级为 alert 正例。

已完成的人工矩阵可用下列命令生成风险轮廓/生命周期 head 的可复现主监督 target。该脚本会再次执行完整 manifest gate，输出标注为 `pixel_supervision_role=auxiliary_only`；它不训练模型，也不授权训练执行或生产替换：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts\build_sanpo_risk_lifecycle_targets.py `
  --config configs\sanpo_counterfactual_episode_collection_v1.json `
  --manifest artifacts.local\evidence\counterfactual\<batch>\episode_manifest.json `
  --output-dir artifacts.local\evidence\counterfactual\<batch>\risk_lifecycle_targets
```

人工标注的是“是否应提醒和何时应提醒”，不是证明系统已经安全。模型输出、伪标签或已有提醒日志不得直接充当事件真值。

## 风险轮廓 / 生命周期头原型

`scripts/sanpo_risk_lifecycle_prototype.py` 已将这份真值合同落实为一个**无训练入口**的时序头原型：输入只能是外部产生的每帧特征；输出为 episode 级 `hazard`、`corridor_relation`、`episode_should_alert`，以及逐时刻 `non_alert / approach / alertable / post_event` logits。时间标签只由已双审、带哈希 attestation 的 target 中的半开毫秒区间确定性生成。

它显式拒绝哈希不匹配的 report、风险词表外类别，以及任何将像素监督提升为主监督的输入。它接受两条明确分层的路径：双审 target 为 `attested_human_reviewed`；许可、哈希绑定的 public RGB/source mask/GPT-VLM 标签可作为 `hash_bound_model_silver_provisional` 暂定监督。后者可用于训练，但始终带来源/提示词 attestation，且不得改称人工事件真值、用于标定或直接替换默认模型；像素分割仍可作为 `auxiliary_only`。当前原型仍没有 trainer、权重保存或阈值校准，未来训练还必须满足 session 隔离和独立安全评测门槛。

## 独立复核证据

每条 episode 必须随本地、带 SHA256 的 `annotation_evidence_path` 保存
`blindassist_sanpo_counterfactual_annotation_evidence_v1` 记录。记录中至少有两名不同的
`reviewer_id`，且 `reviewer_type=human`；模型输出或模型复核不能作为此证据。每位复核者都要独立填写 `should_alert`；正例还要独立填写三个事件锚点。

validator 会拒绝复核者名单与 episode 不一致、复核者少于两人、正负判断不一致，或任一正例锚点的两人差异超过 500 ms 的批次。通过仲裁后的 episode 锚点仍须满足本合同的生命周期区间规则；这使“人工已复核”成为可审计文件，而非 manifest 中的口头声明。

## 来源、许可、隐私与哈希

每个 session 必须具有本地 `source_receipt`，至少记录：

- 数据拥有者/来源、采集日期和许可状态；
- 隐私复核状态、人物可识别信息处理方式和复核者；
- 原始视频相对路径、字节数和 SHA256；
- 设备、相机参数和任何重编码/裁剪派生关系；
- episode manifest 与标注文件 SHA256。

配置和 manifest 不依赖网络 URL 才能验证。远程出处可以作为说明字段，但本地证据、许可文本、隐私记录与哈希 inventory 必须足以独立完成审计。缺许可、隐私状态不是 green、SHA256 缺失或哈希不匹配时，该 session 整体不得进入训练或评价。

## 切分与随机审计

采用固定 leave-one-session-out 六折。每折完整留出一个 session；其原始视频、episode、衍生资产、标签和 matched pair 均不得出现在该折训练、校准、阈值选择或主动选样中。不得按帧随机切分，也不得把一个 matched pair 拆到不同数据权限区域。

主动失败队列必须保留至少 20% 的固定随机审计样本。随机样本从尚未进入候选队列的有效长序列中抽取，并与模型分歧、高置信误提醒和事件再生候选分开报告。holdout session 不参与主动选样。

## 评价与门槛

每折以及六折汇总必须报告：

- `event_recall` 与 `critical_miss_rate`；
- `false_alerts_per_minute`；
- `delivered_alerts_per_event` 与 `delivered_repeated_alert_rate`；
- `post_event_clearance_rate`、`clearance_latency_ms`；
- `event_regeneration_rate`；
- worst-session、worst-scene 和 matched-negative false-alert breakdown。

首轮成功门冻结为：每个 fold 的 event recall ≥0.90、critical miss rate ≤0.05、false alerts/min ≤0.50、delivered repeated alert rate ≤0.10、post-event clearance rate ≥0.90。逐帧 recall、mIoU、boundary IoU 和 session identity probe 只作诊断；它们不能单独授权晋级。

若关键漏报增加、真实切入事件被反事实约束压制、事件指标不随像素指标改善、session identity 显著可预测、随机审计持续发现主动队列漏掉的系统性失败，或事件时间点无法稳定复核，则停止扩训并回到采集/标注合同修订。

## 采集安全

采集应在静态或受控路线中进行，由健视安全员负责环境控制和停止权。不得要求盲人参与者依赖未验证候选完成风险路线，不得把采集结果描述为临床、人因或真实助盲安全验证。本项目始终是辅助提醒原型，不替代盲杖、导盲犬、人工判断或专业辅助设备。
