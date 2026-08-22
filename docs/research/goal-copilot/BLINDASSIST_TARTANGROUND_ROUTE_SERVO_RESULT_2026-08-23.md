# TartanGround public-goal route servo result — 2026-08-23

## 结论

本轮没有要求用户采集、选图、标帧、运行命令或判断中间结果。数据由公开
[TartanGround](https://huggingface.co/datasets/theairlabcmu/TartanGround) 自动下载；它提供差速驱动轨迹、同步
RGB、metric depth、semantic segmentation 和 camera pose。Goal Contract、provider 权重、prompt、阈值、merge 与
controller 都在 private confirmation truth 前冻结。

冻结控制器在 AbandonedCable Development 的 10 次独立 door approach、18 个 RGB-D 状态上达到：

- target-action Recall@3 `83.3%`；
- far `80.0%`；
- STOP `87.5%`。

这只授权一次新鲜确认。随后按 exact `door`、下载成本和 positive-only future-approach 规则自动筛选未用于开发的环境；
SeasideTown、CoalMine、CarWelding 为零机会，Supermarket、AbandonedFactory2、Sewerage、Rome、GothicIsland 最终形成
9 次独立 approach、15 个状态。RGB 只在 roster 封闭后下载，四路 provider 只读 public 输入并各运行一次。

独立确认结果：

| metric | result |
|---|---:|
| target region Recall@1 | 26.7% |
| target region Recall@3 | 60.0% |
| target action Recall@1 | 26.7% |
| target action Recall@3 | 60.0% |
| far target action Recall@3 | 55.6% |
| STOP target action Recall@3 | 66.7% |

预声明 gate 要求 `case_count >= 12` 且 far、STOP 均不低于 `80%`，因此 terminal 为：

```text
GOAL_RGBD_SERVO_ACTION_AVAILABILITY_NOT_ESTABLISHED
```

不得把 Development 成功、Top-10 coverage 或后验诊断改写成最后十米控制成功。

## 合法 public Goal Contract

Provider 可见：

- `帮我找一扇沿当前路线可以接近的门`；
- 全局冻结 `ROUTE_APPROACHABLE_DOOR / SET_VALUED / door`；
- 当前 RGB、metric depth；
- 从 pose-only replay waypoint 计算的 route bearing。

Private evaluator 才见：future-demonstrated exact-door component、bbox/mask、阶段和 desired action。未被未来轨迹证明的
其他对象不是 negative。全局 goal text 与 canonical prompt 的 pre-truth predecessor commit 是 `58643839`；TartanGround
confirmation 控制器冻结于 `76504bf4`，多环境 label adapter 与 confirmation receipt 修复分别冻结于 `425674c4`、
`90045552` 和 `02b5e000`。这些修复均发生在对应 provider/evaluator 输出前。

## 失败层归因

后验只读诊断显示 future-demonstrated target 在 merged pool 的 functional-region Recall@1/3/5/10 为：

```text
53.3% / 66.7% / 73.3% / 86.7%
```

所以目标通常仍在 Top-10；主要损失是 Top-10 到可执行 Top-3 的 route/functional-region selection。但 far 有 2/9 个
target 不在 Top-10，使现有 object-proposal pool 的 far 理论上限只有 `7/9 = 77.8%`，单纯 rerank 已不可能通过 80% gate。

图像审计进一步显示这些 miss 的 private `door` component 是入口阈值、横向通行带或遮挡构件，而不是完整门实例。
这验证了 `Proposal–Identity Responsibility Mismatch` 只是部分解释：产品需要的是 route-conditioned functional aperture，
而 exact-door object proposal/evaluator 仍在要求对象区域。

一个只读 public waypoint 的 route-aperture Development diagnostic 把已消费 confirmation 的 far action Recall@3 提高到
`88.9%`，但 STOP 只有 `33.3%`。原因是原 replay waypoint 每个 frame 都重新取 `current+30`，并不是稳定的用户目标；
越过入口后它会转向下一段路线。episode-stable waypoint adapter 已实现并通过 focused tests，但不能把任意 30-frame pose
冒充 destination/arrival truth。

## 当前接口结论与下一合法 cohort

已经建立：

```text
public goal semantics + current RGB-D + route bearing
-> bounded semantic/functional candidate actions
```

尚未建立：

```text
provider-public destination waypoint
-> route-conditioned traversable aperture
-> goal-relative remaining distance
-> STOP / ARRIVAL
```

现有自动轨迹中，只有 2 次 door approach 落在采集前已存在的 trajectory final-goal pose 2 米内，不足以形成确认 cohort。
因此下一 cohort 必须先带真实 public destination/route endpoint，再生成 private functional-aperture 与 arrival truth；不能继续
把“接近某个 door semantic component”偷换成“到达用户目标”。

Claim ceiling：

```text
PUBLIC_GOAL_SEMANTIC_PROPOSAL_AVAILABILITY_CONFIRMED;
TARTANGROUND_ROUTE_RGBD_SERVO_CONFIRMATION_FAILED;
DESTINATION_RELATIVE_FUNCTIONAL_APERTURE_AND_ARRIVAL_NOT_ESTABLISHED
```

## 实现与证据入口

- `tartanground_future_door_approach.py`：TartanGround exact-door future-approach 自动筛选。
- `materialize_tartanground_servo_cohort.py`：public/private RGB-D、pose waypoint 与多环境 label pairing。
- `run_goal_rgbd_servo.py`：冻结 Top-3 route/action/range controller。
- `evaluate_goal_rgbd_servo.py`：positive-only far/STOP evaluator。
- ignored evidence：`artifacts.local/evidence/last-10m-tartanground-development-v2`、
  `artifacts.local/evidence/last-10m-tartanground-confirmation-v1`。
