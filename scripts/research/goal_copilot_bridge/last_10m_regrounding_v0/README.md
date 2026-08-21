# BLINDASSIST_LAST_10M_REGROUNDING_V0 runner

状态：`ENGINEERING_READY / FIELD_EXECUTION_REQUIRED / NO_P1 / NO_REFERENT_PERSISTENCE`

本目录只实现“入口寻找—引导—重新观测—确认”的当前帧机械闭环。它读取现有 P0
grounding/provider 已生成的 `p0_output_schema.json` 输出，不调用或修改 provider，也不读取 evaluator truth。

每次 `observe` 只允许一个新的 `frame_id`。P0 `GROUNDED` 必须把 selected candidate、provider run、supporting
evidence、请求的 `goal_name` 和 handoff 全部绑定到该帧；handoff 校验后立即丢弃。持久 state 不含 candidate id、bbox、图像、特征、
score、handoff 或 identity。旧 observation 仅存在于 `events.jsonl` 审计日志，决策代码不会读取它。

机械策略固定为：当前 bbox 中心偏左/偏右时小幅转向；居中且高度不足时前进一步；居中且 normalized height
达到 `0.55` 时停下并要求新的当前帧再确认。三次连续无可靠 P0 grounding 或 12 条指令仍未完成时进入
`ABSTAIN`，停止自动引导并给出真人协助出口。任何 non-`GROUNDED`、stale、跨帧或 contract-breaking 输出均
作为 observation 记录并 fail closed。指令不声称
“前方安全”。这些 bbox 规则只用于机械任务，不是距离、安全或科学有效性模型。

## 真实现场执行

先选三个真实地点，各自只使用清晰、相对唯一的建筑入口；不得用 prerecorded replay 冒充用户动作闭环。
现场 payload 和报告写入 ignored `artifacts.local/`：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.cli init-roster `
  --run-dir artifacts.local/evidence/last-10m-regrounding-v0 `
  --site "site-a=建筑 A" --site "site-b=建筑 B" --site "site-c=建筑 C"

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.cli start `
  --run-dir artifacts.local/evidence/last-10m-regrounding-v0 `
  --episode-id site-a-e01

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.cli observe `
  --run-dir artifacts.local/evidence/last-10m-regrounding-v0 `
  --episode-id site-a-e01 --observation artifacts.local/work/current-observation.json
```

每次转向、前进或重扫后，先用现有 P0 provider 对新 frame 生成一个新的 observation envelope，再次运行
`observe`。系统到 `COMPLETE` 或 `ABSTAIN` 后必须现场 adjudicate 是否错误确认入口；错误确认始终归入
`CURRENT_FRAME_GROUNDING_BOTTLENECK`。主动停止可用 `stop`，并在两个 bottleneck 中选择一个。最后运行：

Observation envelope 只有六个顶层键；`p0_output` 必须是现有 P0 V1 输出原文：

```json
{
  "schema_version": 1,
  "episode_id": "site-a-e01",
  "observation_id": "site-a-e01-o001",
  "frame_id": "fresh-frame-001",
  "captured_at_ms": 1787335200000,
  "p0_output": {}
}
```

然后运行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.last_10m_regrounding_v0.cli summarize `
  --run-dir artifacts.local/evidence/last-10m-regrounding-v0
```

只有恰好 `3 locations x 5 adjudicated episodes` 才输出 `FIELD_EXECUTION_COMPLETE`。报告首先单列错误入口确认数，
再报告完成率、完成时间、首次发现时间、指令数、重扫数，并且只使用三种归因：

- `CURRENT_FRAME_GROUNDING_BOTTLENECK`
- `INTERACTION_OR_CONTROL_BOTTLENECK`
- `REGROUNDING_LOOP_MECHANICALLY_USEFUL`

专项检查：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/last_10m_regrounding_v0/test_core.py
```
