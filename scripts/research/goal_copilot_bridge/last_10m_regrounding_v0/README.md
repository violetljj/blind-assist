# BLINDASSIST_LAST_10M_REGROUNDING_V0 runner

状态：`MECHANICAL_EXECUTION_COMPLETE / NETWORK_SCENE_3X5 / MILESTONE_CLOSED / NO_P1 / NO_REFERENT_PERSISTENCE / NO_SUCCESSOR`

本目录只实现“入口寻找—引导—重新观测—确认”的当前帧机械闭环。`provider_adapter.py` 直接复用现有
Grounding DINO `run_inference` 与现有 single-Brain baseline 的 render/prompt/schema/output 函数；不复制或修改
provider 决策逻辑，也不读取 evaluator truth。冻结身份保持为 Grounding DINO Tiny revision `a2bb814...`、权重
SHA-256 `1a2412...`、原 prompt/threshold/NMS，以及 Codex CLI `0.149.0`、`gpt-5.6-terra / medium` 和 baseline
policy。任何 identity/config/hash 漂移均 fail closed，不自动换模型。

每次 `observe` 只允许新的 `frame_id` 和不同的 image SHA-256。P0 `GROUNDED` 必须把 selected candidate、provider run、supporting
evidence、请求的 `goal_name` 和 handoff 全部绑定到该帧；handoff 校验后立即丢弃。持久 state 不含 candidate id、bbox、图像、特征、
score、handoff 或 identity。旧 observation 仅存在于 `events.jsonl` 审计日志，决策代码不会读取它。

机械策略固定为：当前 bbox 中心偏左/偏右时小幅转向；居中且高度不足时前进一步；居中且 normalized height
达到 `0.55` 时停下并要求新的当前帧再确认。三次连续无可靠 P0 grounding 或 12 条指令仍未完成时进入
`ABSTAIN`，停止自动引导并给出真人协助出口。任何 non-`GROUNDED`、stale、跨帧或 contract-breaking 输出均
作为 observation 记录并 fail closed。指令不声称
“前方安全”。这些 bbox 规则只用于机械任务，不是距离、安全或科学有效性模型。

## 网络场景 3x5 执行

按用户更新，不再需要真实设备。`online_scene_runner.py` 从既有 Silver-B Development 素材中选择 3 个已审阅为
`UNIQUE` 且至少有两个不同 frame 的真实世界 Mapillary 地点，每处生成 5 个固定循环起点，共 15 episodes。
公开 playlist 与 evaluator-only truth sidecar 以 hash 绑定；provider 不读取 truth、playlist index、坐标或历史结果。

准备与执行命令如下；payload 和报告写入 ignored `artifacts.local/`：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.online_scene_runner prepare `
  --brain-cohort artifacts.local/evidence/p0-s0/2026-08-21-silver-b-dev-cohort-v1/brain-cohort.json `
  --output-dir artifacts.local/evidence/last-10m-regrounding-v0/online-scenes-v0

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.online_scene_runner run `
  --scene-dir artifacts.local/evidence/last-10m-regrounding-v0/online-scenes-v0 `
  --run-dir artifacts.local/evidence/last-10m-regrounding-v0/network-scene-run-v0
```

每个 playlist frame 都重新完成现有 P0 调用和 state transition，provider receipt 成功后才能进入下一 observation。
已 adjudicate episode 会跳过，完整 observation receipt 可恢复但绝不重放 provider call。固定网络序列不会响应方向
指令，所以不能冒充真实用户控制闭环；序列耗尽仍未到达时 fail closed 并归因交互/控制瓶颈。

也可用通用 `cli.py` 读取外部按同一 P0 V1 合同生成的 envelope：

Observation envelope 只有八个顶层键；`p0_output` 必须是现有 P0 V1 输出原文：

```json
{
  "schema_version": 1,
  "episode_id": "site-a-e01",
  "observation_id": "site-a-e01-o001",
  "frame_id": "fresh-frame-001",
  "frame_sha256": "<64 lowercase hex characters>",
  "captured_at_ms": 1787335200000,
  "processed_at_ms": 1787335220000,
  "p0_output": {}
}
```

只有恰好 `3 locations x 5 adjudicated episodes` 才输出 `MECHANICAL_EXECUTION_COMPLETE`。报告首先单列错误入口确认数，
再报告完成率、完成时间、首次发现时间、指令数、重扫数，并且只使用三种归因：

- `CURRENT_FRAME_GROUNDING_BOTTLENECK`
- `INTERACTION_OR_CONTROL_BOTTLENECK`
- `REGROUNDING_LOOP_MECHANICALLY_USEFUL`

本次冻结结果见
[`BLINDASSIST_LAST_10M_REGROUNDING_V0 result`](../../../../docs/research/goal-copilot/BLINDASSIST_LAST_10M_REGROUNDING_V0_RESULT_2026-08-22.md)。

专项检查：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/last_10m_regrounding_v0/test_core.py
```
