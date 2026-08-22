# Synthetic current-frame arrival authority closeout — 2026-08-23

状态：`SUPERSEDING_CLOSEOUT / REVERSIBLE_EXPLORATION / DEFAULT_APP_UNCHANGED`

## Terminal

```text
SYNTHETIC_CURRENT_FRAME_ARRIVAL_AUTHORITY_CLOSED
DESTINATION_RELATIVE_ARRIVAL_NOT_ESTABLISHED
REAL_EPISODE_SELECTIVE_GUIDANCE_PILOT_IS_CURRENT_SUCCESSOR
```

本 closeout 前向收口 synthetic current-frame completion / arrival 的产品证明责任，不修改、重评分或删除任何历史
protocol、result、receipt 或 terminal。既有 TartanAir/TartanGround 资产继续用于 regression、controller mechanics、
geometry sanity 与 provider/evaluator leak check；它们不再授权新的 D2、S6 或同形状 completion confirmation cohort，
也不再支撑产品 completion、arrival、入口可用性、安全或默认 App 主张。

## 已建立与未建立的边界

- 合法 public Goal Contract 驱动的 goal-semantic proposal availability 已在受限公开/合成条件下建立；这只说明
  bounded candidate availability。
- S0v11 的 `9/13` false completion 否决了 detector bbox extent → nearness/completion。
- S2–S5、D1C/D3 表明 exact-door component proximity 与 ground-connected proxy 不能建立可通行入口或任务完成。
- Goal RGB-D servo 与 TartanGround route-bearing replay 中，目标通常仍在 Top-10，但 Top-3 action 与 STOP 未建立；
  replay waypoint 也不是稳定 public destination。
- 因此 destination-relative arrival authority 未建立；继续改变 detector、SAM、depth model、threshold、provider 或
  synthetic cohort 不会修复缺失的任务构念。

`STOP_FOR_SAFETY` 只是“先停”的控制动作，不等于 `ARRIVED`。`HANDOFF_READY` 只表示目标已缩小到适合手、盲杖或
用户确认的局部搜索范围，也不等于完成。Episode completion 仍是合法终态，但只能来自显式用户确认，或另一个独立、
受合同约束的可信交互 receipt；perception、provider 和 controller 均无权直接产生完成事实。

## 只读历史错误分解

本节只读取既有 sealed/consumed evaluator 输出，不建立 protocol、gate 或 confirmatory claim。分类采用既有 primary
Top-3 functional-region/action 判定；各 partition 不重叠。

| 历史 partition | 描述性分类 | 数量 | 解释 |
|---|---:|---:|---|
| S0v11 false completion | `CORRECT_REFERENT_WRONG_RANGE` | 9 | 9/9 曾选中合法目标，但 bbox extent 被误用为 nearness/completion |
| Goal RGB-D servo confirmation far wrong action | `WRONG_REFERENT` | 10 | 10/10 Top-3 内无 legal target region；不能归因于动作映射 |
| TartanGround route servo confirmation far wrong action | `WRONG_REFERENT` | 4 | 4/4 Top-3 内无 legal target region；不能归因于动作映射 |
| 两个 confirmation 的 near/STOP states | `INSUFFICIENT_OR_INVALID_TRUTH` | 19 | exact-door proximity/phase 不能表达 destination arrival |
| 已证明 legal region 命中但 action 错误 | `CORRECT_REFERENT_WRONG_ACTION` | 0 | 当前只读证据没有这种可分离错误 |
| 无法由现有 evaluator 解释 | `UNCLASSIFIABLE` | 0 | 未把 UNKNOWN 填成 negative |

来源：S0v11 `evaluation.json`、Goal RGB-D servo `goal-rgbd-servo-evaluation-v1.json`、TartanGround
`goal-rgbd-servo-evaluation.json`。它们均位于 ignored `artifacts.local/evidence/`；本表只固化描述性计数和归因规则，
不把本地 payload 提交为新的证据版本。

## 当前唯一 successor

当前 successor 是 `REAL_EPISODE_SELECTIVE_GUIDANCE_PILOT_V0`，顺序固定为：

1. 实现 current-frame-only 的 selective guidance、争议/弃权、handoff 与用户确认责任合同；
2. 复用现有 goal-capture-app / prospective recorder，建立真实 goal-driven episode 的轻量 materializer、annotation、
   baseline、evaluator 与 report；
3. 仅在存在合法 goal-before-truth 真实第一视角 episode 时运行冻结 baseline；没有物理视频时只保留
   `PHYSICAL_EPISODES_NOT_CAPTURED`，不得用 synthetic、room-door 或 replay 填补。

本 successor 明确不包含 P1 persistence/reacquisition、tracker、re-ID、world memory/anchor、gallery growth、VIO/SLAM、
scene graph、模型/阈值/provider sweep 或默认 App promotion。只有真实 pilot 后证据把主导失败定位为出画/遮挡后同类实例
错锁，且排除 camera pointing、proposal、range 与 interaction 主导，才允许另行提出 P1 successor；本轮不实现。

Claim ceiling：`EXPLORATORY_MECHANICS_AND_FAILURE_ATTRIBUTION_ONLY`。

