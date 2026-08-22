# Last-10m current-frame visual servo V1

状态：`S2_S5_CURRENT_FRAME_COMPLETION_NOT_ESTABLISHED / D3_SAM3_FUNCTIONAL_REGION_CONFIRMATION_FAILED / PROCTHOR_VULKAN_RUNTIME_NOT_EVALUABLE`

最新全自动 TartanGround 差速驱动确认见
[`TartanGround public-goal route servo result`](../../../../docs/research/goal-copilot/BLINDASSIST_TARTANGROUND_ROUTE_SERVO_RESULT_2026-08-23.md)：
Development 18-state action Recall@3 为 `83.3%`，但 untouched 15-state confirmation 只有 `60.0%`，far `55.6%`、
STOP `66.7%`，因此 `GOAL_RGBD_SERVO_ACTION_AVAILABILITY_NOT_ESTABLISHED`。后验显示 Top-10 target coverage 为
`86.7%`，但 object proposal miss 的目标包含入口阈值/通行带；下一合法接口必须携带 provider-public destination waypoint，
不能再把 exact-door component proximity 当 arrival truth。

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

## Fully automated public RGB-D S2-S5

S2-S5 从公开 TartanAir RGB/depth/segmentation 自动下载、自动分层取样并机械生成 private truth；不要求用户
选图、标帧、运行命令或判断中间结果。四个各 48-case 的 fresh cohort 均一次运行、禁止 replay，结果依次为：

- S2: `4/11` correct opportunities，`0` false completion；
- S3: `4/5` correct opportunities，`1` false completion；
- S4: `1/24` correct opportunities，`4` false completion；
- S5: `8/24` correct opportunities，`6` false completion。

四个 terminal 都是 `FRESH_CURRENT_FRAME_COMPLETION_NOT_ESTABLISHED`。S4 的 23 个 missed opportunities 中，
23 个都已经有正确 YOLOE proposal，因此当前主要失败层已从 proposal coverage 收窄到 current-frame
functional/range selection。S4 上看似有效的 depth-aperture 规则在 untouched S5 上没有复现；固定深度结构
Random Forest 在 S5 development split 上达到 `16/23`，但产生 `5` 个 false completion，同样是
`NOT_PROMISING`，不授权 fresh successor。

结果真源：
[`current-frame completion S2-S5`](../../../../docs/research/goal-copilot/BLINDASSIST_LAST_10M_CURRENT_FRAME_COMPLETION_S2_S5_RESULT_2026-08-22.md)。

## Functional-region development D1C

后续审计收窄了 S2-S5 truth：它只能机械确认 exact-door bbox 与 metric depth，不能确认 open aperture、
房间连通性或可通行性。新开发线因此只声称 synthetic exact-door ground-connected approachability proxy。

全自动环境资格筛选先后得到：Supermarket 远程 ZIP 传输不可评估；DesertGasStation 在 RGB/provider 前因
`near=4, far=408` 失败；HongKong D1C 以 `near=112, far=1654` 通过并自动冻结 24 near + 24 far。
约束 RANSAC ground plane 在 Office development truth 上达到 floor+carpet precision `0.9411`、recall
`0.7821`、IoU `0.7456`。YOLOE functional mask、RGB/CLIP/context verifier 都未过零误报与 50% coverage 门。

官方 SAM 3 exact text prompt `door` 在 D1C 的固定 0.50 首轮为 `6/24 correct, 0 false`；开发 proposal
floor 0.10 为 `13/24 correct, 1 false`。固定阈值网格中最优零误报配置只有 `10/24`，仍未授权 fresh/formal
successor。D1C 只用于开发，不可回标成 independent confirmation。

独立确认自动筛选了四个新环境：AbandonedFactory2、OldBrickHouseNight、CoalMine 均在 RGB/provider 前因
near denominator 不足退出；HQWesternSaloon 通过但因一帧 ground-plane exception 在 evaluator 前成为
`NOT_EVALUABLE_RUNTIME`。补齐 fail-closed 后，untouched JapaneseAlley D3 的 48/48 ground preflight 通过，
唯一一次冻结 SAM 3 运行得到 `1/24 correct, 0 false, coverage=0.0417`，确认失败且禁止 replay。

随后建立的 ProcTHOR/AI2-THOR 5.0 Docker CloudRendering 通道证明 CUDA GPU 可见，但 Docker Desktop Vulkan
仅暴露 llvmpipe；Unity 以 `-11` 退出。因此 topology/reachable-position 评估是
`NOT_EVALUABLE_VULKAN_RUNTIME`，不是算法负例。

结果真源：
[`functional-region D1C`](../../../../docs/research/goal-copilot/BLINDASSIST_LAST_10M_FUNCTIONAL_REGION_D1C_RESULT_2026-08-22.md)。
[`functional-region D3 confirmation`](../../../../docs/research/goal-copilot/BLINDASSIST_LAST_10M_FUNCTIONAL_REGION_D3_RESULT_2026-08-22.md)。
