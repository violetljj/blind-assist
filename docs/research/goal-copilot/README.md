# Goal-Driven Visual Copilot

状态：`current / PRODUCT_AND_RESEARCH_MAINLINE / BA_ADT_REAL_EVIDENCE_ACTIVE / ADT_0_SAMPLE_MINED_PARTIAL_EVENT_COVERAGE / FULL_SEQUENCE_SELECTION_NEXT / RGB_ONLY_SYSTEM_INPUT / GT_EVALUATOR_ONLY / SKY_DISABLED / DEFAULT_APP_UNCHANGED`

## 当前主线：BA-ADT-REAL-EVIDENCE

BlindAssist 的当前唯一 Goal Copilot successor 是 `BA-ADT-REAL-EVIDENCE`。它从最终产品目标倒推，
先补齐真实第一视角视觉证据，再接回冻结的 Goal Copilot。路线依次为 ADT-0 episode mining、ADT-1
RGB-only Observation、ADT-2 prerecorded Offline Goal Copilot，以及只在真实失败明确归因到 policy 层时
才允许设计的 ADT-3/Sky task。

系统侧输入只允许真实 RGB；ADT bbox、object/device trajectory、depth、segmentation 与 visibility GT
只允许进入隔离的 mining/evaluator。ADT 是录好轨迹，因此本路线最多证明真实 RGB 能否恢复 target
visibility、bearing、tracking、reacquisition、relative nearness 与 approach evidence，以及这些 evidence
能否支撑合理的 prerecorded guidance timeline；它不能证明引导改变了用户动作或完成 closed-loop
navigation。

ADT-0 的稳定实现入口见
[`scripts/research/ba_adt_real_evidence/README.md`](../../../scripts/research/ba_adt_real_evidence/README.md)。
当前只激活 sample acquisition 与 GT-only episode mining；Sky、GC2-C、held-out、Android/default-App
接线、产品和安全主张均关闭。

Sample 已完成，结果见
[`BA_ADT_REAL_EVIDENCE_ADT0_SAMPLE_RESULT.md`](BA_ADT_REAL_EVIDENCE_ADT0_SAMPLE_RESULT.md)：300 帧中
102 个目标满足持续可见候选门，事件候选覆盖 search/acquire/track/lost/reacquire/approach，但没有单一
目标覆盖完整六阶段。下一步保持门槛不变，先对少量完整 sequence 做 GT-only mining。

## 上位产品定义

BlindAssist is a goal-driven visual copilot for visually impaired users. Instead of only
reporting objects or hazards, the system maintains a user goal across time, searches for
relevant visual evidence, tracks progress, recovers from target loss, guides the user
toward completion, and verifies that the requested task has actually been completed.

BlindAssist 是一个目标驱动的视觉副驾。用户提出“找入口、找门、找座位、找到某个物品”等目标后，
系统持续观察环境、维护任务状态、追踪目标、判断进展、在目标丢失后重新搜索，并持续引导直到任务完成
或安全地声明无法继续。

这一定义不废弃现有模块。object detection、depth、optical flow/motion、OCR、VLM、tracking、
traversability、risk estimation 和 semantic perception 统一作为 `Evidence Providers`。`Goal Copilot
Brain` 拥有 task belief、temporal memory、target tracking state、progress estimation、action
proposal/selection、recovery、termination 与 goal-completion verification。

## GOAL-COPILOT-1

全名：`BlindAssist Goal-Copilot-1: Completion-Capable Closed-Loop Policy Synthesis`。

研究问题限定为：在冻结的 symbolic/oracle-style perception evidence 已给定时，外部搜索能否提出具有
完整 goal-completion chain 的 closed-loop policy，而不仅是局部动作 patch。V0 只完成零模型 mock
roundtrip；未启动 Sky 模型搜索、EvoX、多臂实验、真实 perception、摄像头或训练。

三个 task family：

- `FIND_AND_REACH`：search → acquisition → alignment → approach → completion；
- `TRACK_AND_REACQUIRE`：track → loss → recovery → reacquisition → continued progress；
- `FIND_ALIGN_INTERACT`：search → acquisition → fine alignment → approach → interaction readiness。

typed candidate observation 见
[`task_api.py`](../../../scripts/research/goal_copilot_bridge/task_api.py)。它只暴露 target bearing/scale/
confidence、可通行方向、relative nearness/approach rate、tracking/observation quality 和 interaction
readiness；不暴露 scenario graph、正确动作、completion truth 或 hidden labels。

## Authority 与 hard gates

BlindAssist 独占 task definition、scenario truth、evaluator、safety contract、score vector、validation
gate 和 `ACCEPT / REJECT / NOT_EVALUABLE` authority。SkyDiscover 只有 candidate proposal/search
authority；其 score 永远只是 provenance。

evaluator 输出 `goal_completion`、`normalized_progress`、`reacquisition_success`、
`tracking_continuity`、`wrong_way_actions`、`unsafe_guidance`、`premature_completion`、
`recovery_steps`、`total_actions`、`timeout`、`semantic_validity`。`unsafe_guidance > 0`、
`premature_completion > 0` 或 semantic invalid 立即硬拒绝；未完成目标时，secondary metrics 只能描述
partial progress，不能改写成 completion。V0 不用单一 scalar 产生科学 verdict。

## Bridge V0

稳定入口是 `python scripts/run_research_tool.py goal-copilot sky_bridge.py`。导出的 public
`SearchTaskBundle` 只有 typed API、初始 policy、protocol、public scenario descriptions、README、
manifest 与 checksums；BA evaluator 和 `sealed_scenarios.json` 永不导出。CandidateBundle 只允许
`candidate/policy.py` 作为候选源码，并用精确成员 allowlist、受限 AST、protocol/source digest 和逐文件
SHA-256 fail closed。完整外部合同见
[`SKYDISCOVER_INTEGRATION_CONTRACT.md`](SKYDISCOVER_INTEGRATION_CONTRACT.md)。

## 与 L10M 的边界

Existing L10M work is precursor evidence for goal-conditioned temporal control and search
behavior. GOAL-COPILOT-1 is a new protocol lineage. No previous L10M result is
retroactively reclassified, recomputed, or claimed as GOAL-COPILOT-1 evidence.

L10M 中的 progress memory、tracking、recovery、safety contract、termination 与 balanced
exploration 可以解释为 Goal Copilot Brain 的先导机制，但其封存 protocol、terminal、receipt、run root、
claim ceiling 与 evidence role 均保持原样。

## 当前结论

- 当前结论：`GOAL_COPILOT_1_SKY_SEARCH_SIGNAL_ESTABLISHED_ON_SEALED_PILOT`；
- 证据范围：small deterministic symbolic closed-loop Pilot；
- 默认 App、真实用户、安全效果和产品可用性：无新权限；

## 唯一 successor

`ADT0_FULL_SEQUENCE_SELECTION`：保持 sample miner、事件定义和阈值不变，对少量完整 ADT sequence
先做 GT-only mining；只为自然多阶段目标下载对应真实 RGB，随后才能另立 ADT-1 RGB adapter。
Sky、GC2-C、held-out、Android/default-App 与交互式导航均不在此 successor 内。

`GOAL-COPILOT-1-SKY-PILOT` 已按独立冻结协议完成并封存；协议见
[`GOAL_COPILOT_1_SKY_PILOT_PROTOCOL.md`](GOAL_COPILOT_1_SKY_PILOT_PROTOCOL.md)，结果与严格 claim
ceiling 见 [`GOAL_COPILOT_1_SKY_PILOT_RESULT.md`](GOAL_COPILOT_1_SKY_PILOT_RESULT.md)。若继续，
应另立 `GOAL-COPILOT-2 / NOISY-OBSERVATION ROBUSTNESS` 协议，而不是恢复本 Pilot 或复用其 fresh。

GOAL-COPILOT-1 现已永久关闭，不再授权任何 GC1 模型搜索。下一条已冻结路线是
[`GOAL_COPILOT_2A_PROTOCOL.md`](GOAL_COPILOT_2A_PROTOCOL.md)：在不调用模型、不接触 GC1
fresh cohort 的前提下，对冻结 GC1 baseline 与 winner 做 perception-uncertainty
characterization。GC2-A 只使用已消费 dev 场景语义，因此其结果不得称为 fresh evidence。

GC2-A 已完成并确定性 replay 通过；结果见
[`GOAL_COPILOT_2A_RESULT.md`](GOAL_COPILOT_2A_RESULT.md)。冻结 winner 在 primary
`COMBINED_MODERATE` 下 completion `0/12`、eligible reacquisition `0/3`，并有 `1` 次
premature completion，因此只准入 GC2-B noise-robust Sky search 的独立协议设计。GC2-B
模型调用、正式 Sky/EvoX 搜索和多臂实验仍未授权。

GC2-B 的双仓协议设计已冻结，见
[`GOAL_COPILOT_2B_PROTOCOL.md`](GOAL_COPILOT_2B_PROTOCOL.md)。冻结设计规定 BA 独占
task/noise/evaluator/hidden/acceptance authority，Sky 仅有 proposal/search authority；其当前
claim ceiling 仅为 protocol design。正式 bundle、held-out envelope、provider/run seal 与模型调用
仍需后续单独授权。

GC2-B 随后已按独立 formal seal 完整执行并关闭；见
[`GOAL_COPILOT_2B_RESULT.md`](GOAL_COPILOT_2B_RESULT.md)。32/32 generation calls 完成，
但 BA 锁定的公开 dev winner 在 `COMBINED_MODERATE` 仍为 `0/12`，未达到 held-out
admission，因此 encrypted held-out 未进入 winner-lock 后的正式开启，也从未用于候选评估。当前结论是
`GC2B_NOISE_ROBUST_SEARCH_SIGNAL_NOT_ESTABLISHED`，不授权 rescue rerun、扩预算或同一候选面的
GC2-C。

随后完成的零模型
[`GOAL-COPILOT-2 observability and reality audit`](GOAL_COPILOT_2_OBSERVABILITY_AUDIT_RESULT.md)
只使用 consumed scenarios、冻结 simulator、candidate traces 和既有 device evidence。12 个 moderate
episode 的首次偏离分散在 stale evidence、tracking collapse、dropout 与方向/动作错误；逐项关闭任一
corruption 最多只恢复到 `1/12`。Hidden oracle、完整历史 lookup 和六函数 surface memorization 均可在
这 12 条 consumed 轨迹达到 `12/12`，但 lookup 明确是 simulator leakage diagnostic，不是可迁移
policy evidence。现有 Android trace 是公开/已消费真实世界 RGB 的 device replay，不是 real-phone
capture，且缺少 target identity、tracking、bearing、nearness 与时间映射，所以真实手机噪声校准为
`NOT_EVALUABLE`。

该审计的历史决策仍保持：停止 synthetic moderate optimization 并保持 policy search 关闭。新的
`BA-ADT-REAL-EVIDENCE` 不是 GC2 rescue；它使用独立 ADT 真实 RGB/GT lineage，先执行 ADT-0
数据适配性与 episode mining。GC2-C、held-out opening、新模型/Sky 调用、扩预算和 consumed
representation ladder 均继续禁止。
