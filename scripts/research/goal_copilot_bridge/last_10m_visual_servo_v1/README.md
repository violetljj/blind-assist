# Last-10m current-frame visual servo V1

状态：`S0V11_BBOX_HEIGHT_REJECTED / COMPLETION_NEARNESS_S1_SEALED / DEPTH_REDUCES_FALSE_COMPLETION_BUT_TRADES_OFF_TRUE_COMPLETION`

本模块把已经建立的 goal-semantic proposal 与 leftmost relation selection 接入一个完全当前帧的 pan/zoom/复扫
visual-servo simulator。它不恢复 P1、tracker、referent memory 或跨帧 candidate identity。

每个 observation 都从同一公开真实 facade RGB 的预冻结 view state 独立渲染，并重新运行固定
YOLOE-26n-seg。候选先按 `LEFTMOST_CANDIDATE_X_CENTER` 选择；当前 bbox 未覆盖中央 corridor 时执行左右 pan，
覆盖中央但高度不足时进入下一 zoom，达到高度 cue 后必须用新的 jittered frame 再检测一次才允许完成。

private evaluator 使用人工 door bbox，检查每一步的真实 target availability/selection，并把完成要求绑定到：连续两帧
selected proposal 都命中 target、target 至少 80% 可见、中央 ray 位于 target 内且 target 高度达到 0.55。provider 与
controller不读取这些 truth。

S0v11 在未消费的 FacadeElements path-hash roster（skip 48, take 24）上得到 13 个 visible case。74 次 provider call
一次完成、无 replay：target candidate availability `13/13`，target selected at least once `11/13`，但 true completion
只有 `1/13`，false completion `9/13`。因此 bbox height 作为 nearness/completion signal 被否决；facade still-image
与 synthetic zoom 也不能回答真实 physical approach。

运行入口：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.visual_servo authorize ...
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.visual_servo run ...
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.visual_servo evaluate ...
```

Focused mechanics check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.test_visual_servo
```

结果真源：
[`S0v11 result`](../../../../docs/research/goal-copilot/BLINDASSIST_LAST_10M_VISUAL_SERVO_S0V11_RESULT_2026-08-22.md)。

## Independent completion-nearness S0/S1

`completion_nearness.py` 和两个 dataset materializer 自动建立 public RGB/private truth 边界。NYUv2 S0 使用
预冻结 unique leftmost-door contract；fresh SUNRGBD S1 使用 set-valued visible-door contract，并排除所有 NYUv2
ancestry。S1 的 48 帧一次运行表明 depth gate 把 false completion `8 -> 1`，但 true completion `2 -> 1`；
因此独立 metric depth 是有信息但不充分的 completion cue，尚不能授权 completion control。

Focused mechanics check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.test_completion_nearness
```

结果真源：
[`completion-nearness S0/S1`](../../../../docs/research/goal-copilot/BLINDASSIST_LAST_10M_COMPLETION_NEARNESS_S0_S1_RESULT_2026-08-22.md)。
