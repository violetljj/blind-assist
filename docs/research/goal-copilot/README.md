# Goal-Driven Visual Copilot

状态：`current / PRODUCT_AND_RESEARCH_MAINLINE / GOAL_COPILOT_SKY_BRIDGE_V0_MECHANICS_READY / GOAL_COPILOT_1_SKY_PILOT_PROTOCOL_IMPLEMENTED_MODEL_SEARCH_NOT_STARTED / DEFAULT_APP_UNCHANGED`

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

- 当前结论：`GOAL_COPILOT_SKY_BRIDGE_V0_MECHANICS_READY`；
- 模型状态：`GOAL_COPILOT_1_MODEL_SEARCH_NOT_STARTED`；
- 默认 App、真实用户、安全效果和产品可用性：无新权限；

## 当前 successor

`GOAL-COPILOT-1-SKY-PILOT` 已形成独立冻结协议；正式模型调用只能由 hash-bound formal seal
授权。协议、候选隔离、两 replicate 的 2×16 预算、dev-only winner selection、加密 fresh cohort 和
PASS gate 见
[`GOAL_COPILOT_1_SKY_PILOT_PROTOCOL.md`](GOAL_COPILOT_1_SKY_PILOT_PROTOCOL.md)。在 formal seal
出现前，模型状态仍为 `NOT_STARTED`。
