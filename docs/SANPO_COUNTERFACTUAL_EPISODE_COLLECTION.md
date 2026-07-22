# SANPO 交叉反事实事件采集协议 v1

## 目的与边界

本协议把数据单位定义为连续 `episode` 和物理 `risk_event`，而不是相互高度相关的独立帧。首轮冻结为 **6 个 session × 4 个 scene × 每格 2 个 positive + 2 个 matched negative**，合计 96 个 episode、48 个正负 matched pair。

本轮只定义可离线执行的采集、标注和评价合同，不下载数据，不训练模型，不读取既有 blind holdout，也不授权导出、设备接入或替换 App 默认模型。像素 mask 可作为少量辅助证据，但不再是事件晋级的唯一真值。

机器可读合同位于 `configs/sanpo_counterfactual_episode_collection_v1.json`。配置中的 session 是待自动获取/生成的槽位，不代表数据已经存在或通过模型复核。

执行入口为 `scripts/validate_sanpo_counterfactual_episodes.py`。它只验证经 GPT/Codex 隔离复核并完成模型共识/仲裁的本地 episode manifest，强制来源许可/隐私证据与文件 SHA256、配对上下文、正负事件锚点和完整 session-scene 矩阵；缺任一项均 fail closed。只有带 `--require-complete` 且完整矩阵通过时才会报告 `training_eligible=true`，并且它始终输出 `production_model_replacement_authorized=false`。它不下载数据；模型复核流程负责 `should_alert`、criticality 和事件锚点判断。

以 `configs/sanpo_counterfactual_episode_manifest_template_v1.json` 为字段模板，在 `artifacts.local/evidence/` 创建每批本地 manifest；模板本身是空的 `in_review` 文件，不能被当作已采集数据。正式复核后执行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts\validate_sanpo_counterfactual_episodes.py `
  --config configs\sanpo_counterfactual_episode_collection_v1.json `
  --manifest artifacts.local\evidence\counterfactual\<batch>\episode_manifest.json `
  --require-complete
```

缺完整、哈希绑定的 GPT/Codex 共识矩阵时不要加 `--require-complete`，更不得以单次候选 mask 或未审计的模型输出绕过该门。

启动自主采集 Agent 前，可先生成不含个人数据、文件路径或标签的 96-slot 清单：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts\generate_sanpo_counterfactual_capture_plan.py `
  --config configs\sanpo_counterfactual_episode_collection_v1.json `
  --output artifacts.local\evidence\counterfactual\capture_plan_20260714.json
```

该计划只列出待自动获取/生成的正负配对、时长、共享上下文、风险轮廓和生命周期模板。每个 slot 初始都是 `awaiting_autonomous_acquisition`，不能作为 manifest、receipt、标签或训练输入；来源 Agent 应依次尝试许可公开数据、自动设备采集、仿真/合成和带 provenance 的模型生成，不创建人工拍摄待办。

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

三个时间点必须满足 `first_visible <= alertable_start < passed_or_cleared`。每条 episode 都要由互不可见的 GPT 多模态角色与 Codex 证据角色独立复核；标签不一致或锚点分歧超过 500 ms 时，必须由全新上下文的第三模型仲裁。

negative episode 使用 `expected_should_alert=false`，三个正事件时间点必须为 `null`，并提供 `negative_reason`。matched pair 必须共享 `matched_pair_id`，同时记录允许保持不同的风险几何字段，避免把负例伪装成与正例完全相同的事件。

## 风险轮廓与生命周期目标

每条 episode 还必须记录 `risk_profile` 和 `lifecycle_intervals_ms`，它们是未来风险轮廓/生命周期头的主监督，像素 mask 仅作为辅助监督：

- positive：`primary_hazard_type=scene_id`、`corridor_relation=enters_or_blocks`、`lifecycle=approach_alertable_clear`；区间必须严格写为 `approach=[first_visible, alertable_start]`、`alertable=[alertable_start, passed_or_cleared]`、`post_event=[passed_or_cleared, duration_ms]`。
- matched negative：`primary_hazard_type=scene_id`、`corridor_relation=outside_or_nonblocking`、`lifecycle=no_alert`；区间必须为 `non_alert=[0, duration_ms]`。

这些整数毫秒区间把“看到了什么像素”与“何时应提醒、何时清除”拆开。静态围栏、长椅、广场边界或固定结构即便在 mask 中进入保守走廊，只要隔离的 GPT/Codex 共识核定为不阻断目标路线，就必须落入 `outside_or_nonblocking/no_alert` 的反事实负例；不能被像素几何升级为 alert 正例。

已完成的自动多模型共识矩阵可用下列命令生成风险轮廓/生命周期 head 的可复现主监督 target。该脚本会再次执行完整 manifest gate，输出标注为 `pixel_supervision_role=auxiliary_only`；它不训练模型，也不授权生产替换：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts\build_sanpo_risk_lifecycle_targets.py `
  --config configs\sanpo_counterfactual_episode_collection_v1.json `
  --manifest artifacts.local\evidence\counterfactual\<batch>\episode_manifest.json `
  --output-dir artifacts.local\evidence\counterfactual\<batch>\risk_lifecycle_targets
```

自动多模型证据回答“是否应提醒和何时应提醒”，但不证明系统已经安全，也不是客观传感器事实或真人用户效果。候选模型自身输出、未隔离伪标签或已有提醒日志不得为自己充当评测参考；只有冻结输入、互盲双模型复核、必要时第三模型裁决并通过哈希合同的 evidence 才能进入研究训练或冻结评测。

## 风险轮廓 / 生命周期头原型

`scripts/sanpo_risk_lifecycle_prototype.py` 已将这份真值合同落实为一个时序头原型：输入只能是外部产生的每帧特征；输出为 episode 级 `hazard`、`corridor_relation`、`episode_should_alert`，以及逐时刻 `non_alert / approach / alertable / post_event` logits。时间标签只由已完成双模型复核、带哈希 attestation 的 target 中的半开毫秒区间确定性生成。

它显式拒绝哈希不匹配的 report、风险词表外类别，以及任何将像素监督提升为主监督的输入。完整事件矩阵使用 `hash_bound_model_consensus`，可自动授权研究训练；许可、哈希绑定的 public RGB/source mask/GPT-VLM 标签使用 `hash_bound_model_silver_provisional`，只能作为暂定训练监督。两条路径都必须保留来源、prompt、模型版本和输入哈希，不能单独用于标定、blind 结论或默认模型替换；像素分割仍为 `auxiliary_only`。默认模型替换另需独立发布模型复核收据和 Android/设备门。

## 独立复核证据

每条 episode 必须保存两个本地、带 SHA256 的 `blindassist_independent_ai_event_review_v1` 记录，角色分别为 `gpt_multimodal_reviewer` 和 `codex_evidence_reviewer`。每个记录必须绑定相同输入哈希、独立上下文、模型/版本、prompt 哈希、置信度和 verdict；正例还要独立填写三个事件锚点。

validator 会拒绝 reviewer run 与 episode 不一致、角色不完整、输入哈希不同、低置信度或 abstain。两路一致时写入 `model_consensus`；标签不一致或任一正例锚点差异超过 500 ms 时，必须绑定第三个独立模型的 `independent_ai_adjudicator` 收据。通过后的锚点仍须满足生命周期区间规则，使模型复核成为可审计文件，而非 manifest 中的口头声明。

## 来源、许可、隐私与哈希

每个 session 必须具有本地 `source_receipt`，至少记录：

- 数据拥有者/来源、采集日期和许可状态；
- 自动隐私审计状态、人物可识别信息处理方式和审计 Agent；
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

若关键漏报增加、真实切入事件被反事实约束压制、事件指标不随像素指标改善、session identity 显著可预测、自动审计持续发现候选发现器漏掉的系统性失败，或事件时间点无法由隔离模型稳定复核，则停止扩训并修订自动获取/生成与标注合同。

## 采集安全

默认不开展依赖人员现场布置、拍摄或审核的采集。优先使用合法公开连续数据、仿真/合成数据和无人值守设备脚本；任何自动设备采集都必须采用不会要求参与者依赖未验证候选的静态安全设置，并保留自动停止条件。不得把所得证据描述为临床、人因或真实助盲安全验证。本项目始终是辅助提醒原型，不替代盲杖、导盲犬、人的安全判断或专业辅助设备。
