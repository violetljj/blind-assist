# Goal-Driven Visual Copilot

状态：`current / PRODUCT_AND_RESEARCH_MAINLINE / SYNTHETIC_CURRENT_FRAME_ARRIVAL_AUTHORITY_CLOSED / SELECTIVE_GUIDANCE_V0_CONTRACT_IMPLEMENTED / REAL_EPISODE_PILOT_TOOLCHAIN_NEXT / NO_P1 / DEFAULT_APP_UNCHANGED`

完整系统蓝图见 [`V2 路线图`](BLINDASSIST_GOAL_DRIVEN_VISUAL_COPILOT_V2_ROADMAP_2026-08-21.md)。本页是
Goal Copilot 动态执行状态唯一真源；日期化 protocol/result、archive、旧 handoff 与历史 successor 不产生当前权限。

## 当前结论

[`Synthetic current-frame arrival closeout`](BLINDASSIST_SYNTHETIC_CURRENT_FRAME_ARRIVAL_CLOSEOUT_2026-08-23.md)
前向关闭 synthetic current-frame completion / arrival 的产品证明责任：

```text
SYNTHETIC_CURRENT_FRAME_ARRIVAL_AUTHORITY_CLOSED
DESTINATION_RELATIVE_ARRIVAL_NOT_ESTABLISHED
REAL_EPISODE_SELECTIVE_GUIDANCE_PILOT_IS_CURRENT_SUCCESSOR
```

已建立的 bounded 事实是：合法 public Goal Contract 驱动的 semantic proposal 在受限公开/合成条件下通常能把目标放入
Top-K；它不等于 referent authority、destination arrival、入口可用性或用户完成。

已否决或未建立的责任链是：

- detector bbox extent 不能承担 nearness/completion；
- exact-door component proximity 与 ground-connected proxy 不能承担 functional aperture 或 arrival；
- route-bearing replay 没有稳定 public destination，不能承担 destination-relative arrival；
- `STOP_FOR_SAFETY != ARRIVED`，`HANDOFF_READY != COMPLETED`；
- episode completion 只接受显式用户确认，或独立、受合同约束的可信交互 receipt。

TartanAir/TartanGround 只保留为 regression、controller mechanics、geometry sanity 与 leak check。禁止 D2、S6 或新的
synthetic completion confirmation cohort；不得通过 detector、SAM、depth model、threshold 或 provider sweep 恢复该权限。

## 当前唯一 successor

唯一 successor 是 `REAL_EPISODE_SELECTIVE_GUIDANCE_PILOT_V0`，属于 `REVERSIBLE_EXPLORATION`：

1. 实现 current-frame-only selective guidance、争议/弃权、handoff 与用户确认责任合同；
2. 复用现有 [`goal-capture-app`](../../../apps/demos/goal-capture-app/README.md) 与
   [`prospective capture`](../../../scripts/research/goal_copilot_bridge/p1_prospective_capture/README.md)，完成真实 goal-driven
   episode 的 capture plan、schema、annotation、materializer、baseline、evaluator、report 与设备 bundle；
3. 只有发现合法 goal-before-truth 真实第一视角 episode 时才运行一次冻结 baseline；否则 terminal 为
   `PHYSICAL_EPISODES_NOT_CAPTURED`，不得用 synthetic、room-door、Mapillary 或 replay 填补。

Pilot 按 speech/action decision 时刻和 episode 评估 visibility、proposal Recall@K、referent selection、confident wrong
guidance、abstention/contested/lost/stale、range、handoff、用户确认/否认、完成时间、指令与纠正。所有条件指标必须保留
conditioned denominator；provider/evaluator 分离、goal-before-truth、provenance、fail-closed 与 claim ceiling 保持有效。

Claim ceiling：`EXPLORATORY_MECHANICS_AND_FAILURE_ATTRIBUTION_ONLY`。

## V0 与 P1 边界

`CONTESTED / ABSTAIN / LOST / STALE / HANDOFF_READY / COMPLETED_BY_USER` 是 V0 的合法 current-frame 决策/交互状态，
不构成 persistence。当前实现不得导入 tracker、cross-frame identity、re-ID、gallery growth、world memory/anchor、VIO/SLAM
或 scene graph，也不得静默修改默认 App。

只有真实 pilot 后同时满足以下证据，才可另行提出而不能在本轮实现 P1 successor：

1. 连续可见片段内 referent selection 已足够可靠；
2. episode failure 仍显著；
3. 主导失败明确是出画/遮挡后错锁同类实例或无法恢复；
4. camera pointing、proposal miss、range 与 interaction 均不是主导层。

## 关键历史证据入口

- [`Goal-semantic proposal + RGB-D servo`](BLINDASSIST_GOAL_RGBD_SERVO_RESULT_2026-08-23.md)：proposal availability
  established；fresh action availability not established。
- [`TartanGround route servo`](BLINDASSIST_TARTANGROUND_ROUTE_SERVO_RESULT_2026-08-23.md)：Top-10 target 多数存在，
  但 route/functional selection 与 STOP 未建立，waypoint 不是 destination truth。
- [`S0v11 visual servo`](BLINDASSIST_LAST_10M_VISUAL_SERVO_S0V11_RESULT_2026-08-22.md)：`9/13` false completion，
  bbox extent responsibility rejected。
- [`S2–S5 current-frame completion`](BLINDASSIST_LAST_10M_CURRENT_FRAME_COMPLETION_S2_S5_RESULT_2026-08-22.md) 与
  [`D1C`](BLINDASSIST_LAST_10M_FUNCTIONAL_REGION_D1C_RESULT_2026-08-22.md) / [`D3`](BLINDASSIST_LAST_10M_FUNCTIONAL_REGION_D3_RESULT_2026-08-22.md)：
  synthetic exact-door/ground-connected proxy 的边界。
- [`P0 grounding contract`](P0_GROUNDING_PROTOCOL_V1.md)：`UNIQUE / SET_VALUED / AMBIGUOUS`、provider、selection 与
  evaluator 分离；既有 provider/threshold 不因本 successor 改变。
- [`Prospective recorder result`](P1_PROSPECTIVE_DEVICE_RECORDER_IMPLEMENTATION_RESULT_2026-08-22.md)：独立 CameraX
  recorder 已实现且不进入默认 App。
- P1 A1–A4、W1/W2、AMRM0 历史终态保持原样；旧文件只用于追溯，不恢复执行权限。

## 默认 App 与声明边界

默认 App 保持当前 YOLO/risk 正式路径不变。本 successor 是 research/experimental integration；任何正结果都不自动授权
Android/default-App 接线、模型晋级、导航/安全、用户有效性或产品成功声明。
