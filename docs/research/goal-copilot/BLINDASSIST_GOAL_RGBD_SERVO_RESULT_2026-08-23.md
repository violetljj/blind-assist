# Goal-semantic proposal and RGB-D servo result — 2026-08-23

## 结论

本轮完全自动从公开 TartanAir V2 下载、筛选、物化和评估数据；没有要求人工采集、挑图、标帧或运行命令。

两个结论必须分开：

1. `GOAL_SEMANTIC_PROPOSAL_AVAILABILITY_ESTABLISHED`：合法 public goal `帮我找一扇沿当前路线可以接近的门` 经全局 prompt `door` 驱动四路 public proposal provider，在独立 14-case cohort 上普通 IoU Recall@1/3/5/10 为 `85.7% / 100% / 100% / 100%`，命中候选的 bearing action agreement 为 `100%`。
2. `GOAL_RGBD_SERVO_ACTION_AVAILABILITY_NOT_ESTABLISHED`：冻结的 Top-3 RGB-D controller 在 fresh Office 35-case cohort 上 target-action Recall@3 只有 `57.1%`；远距引导 `54.5%`，近距 STOP `61.5%`。不得升级为完整最后十米控制成功。

## 自动数据合同

Provider 只见：

- 原始用户目标与 `ROUTE_APPROACHABLE_DOOR / SET_VALUED / door` Goal Contract；
- 当前 RGB 与 metric depth；
- public proposal stream；
- 后续诊断臂中独立于 segmentation/door truth 计算的 replay waypoint proxy。

Private evaluator 才见：未来 30 帧中持续跟踪、深度下降和面积增长共同证明的 door positive、当前 target bbox/mask、阶段与目标动作。未被未来轨迹证明的其他门始终不是 negative。

连续轨迹筛选要求：当前门深 `2.5–8.0 m`、未来最小深度不高于 `2.0 m`、至少下降 `0.75 m`、depth ratio 不高于 `0.70`、面积增长至少 `1.30x`、至少跟踪 8 帧。STOP truth 只在最接近帧 target depth 不高于 `1.50 m` 时形成。

公开数据来自 [TartanAir V2](https://huggingface.co/datasets/theairlabcmu/tartanair2)。官方文档说明其 RGB、depth、seg、camera pose 与 IMU 同步采集，并给出 640x640、90 度 FoV 的相机合同：[modalities](https://tartanair.org/modalities.html)、[examples](https://tartanair.org/examples.html)。

## 冻结 proposal 结果

Development 使用 JapaneseAlley + HQWesternSaloon 的 15 个独立 future-approach event。四路 provider 为：

- SAM 3 exact text prompt `door`, confidence `0.10`；
- Grounding DINO Tiny exact prompt `door`；
- frozen base functional-door detector；
- frozen domain-adapted functional-door detector。

Round-robin、NMS `0.70`、最多 10 candidates 的 Development ordinary IoU Recall@10 为 `73.3%`。诊断发现四个 miss 的 target coverage 都为 `1.00`，但 component-vs-region IoU 仅 `0.09–0.21`。预先冻结的 bounded-containment 指标拒绝候选面积超过画面 `40%` 的近全屏框，Development functional Recall@10 为 `93.3%`。

之后在 AmericanDiner、CountryHouse、House、RetroOffice 的 14 个独立 event 上一次确认：ordinary IoU 与 functional-region Recall@3/5/10 均为 `100%`，Recall@1 为 `85.7%`。因此 proposal 层已经建立 bounded target availability；这不等于 route identity selection 或 task completion。

## 冻结 RGB-D controller 与 fresh failure

Controller 对最多 3 个候选计算：

- horizontal bearing：`TURN_LEFT / ADVANCE / TURN_RIGHT`；
- interior metric-depth median；
- SAM 3 mask depth p20 仅作为有 interior-depth 上界约束的 hint；
- `1.60 m` controller stop threshold。

在 20 个 Development 状态（14 far + 6 near）上，target-action Recall@3 为 `90.0%`，far 为 `85.7%`，STOP 为 `100%`。这些结果只授权 fresh confirmation，不是最终结论。

网络自动筛选先下载 OldBrickHouseNight 的 seg+depth；其 qualifying event 为 `0`，因此未运行 RGB provider。随后 Office 自动得到 148 个重叠窗口，聚成 22 个独立事件，其中 13 个含 near-stop，共 35 个 fresh 状态；只在 roster 冻结后下载 RGB 并运行模型。

Fresh 结果：

| metric | result |
|---|---:|
| target region Recall@1 | 40.0% |
| target region Recall@3 | 62.9% |
| target action Recall@1 | 37.1% |
| target action Recall@3 | 57.1% |
| far target action Recall@3 | 54.5% |
| stop target action Recall@3 | 61.5% |

Merged Top-10 中 future-demonstrated target 为 `30/35 = 85.7%`，说明主要损失发生在 Top-10 到可执行 Top-3 的 route/identity selection，而不是动作映射本身。

## Outcome 后诊断，不构成确认

Office 已消费为 Development diagnostic。两项自动 successor 均未过原 gate：

- cross-provider geometric consensus；
- 从同步 IMU position + Euler attitude 计算、且不读取 semantic/target truth 的 3 秒 replay waypoint proxy。

后者是“若产品已有路线规划器，公开 waypoint 应如何进入接口”的 mechanics proxy，不是合法 prospective route-plan confirmation。它没有救回 Office，因此不允许把 future executed trajectory 当作成功的 public intent substitute。

## 当前责任边界

已建立：

```text
public Goal Contract
-> semantic door proposal pool
-> bounded target availability
-> candidate-wise bearing/range actions
```

未建立：

```text
generic “沿当前路线”
-> 在多个同类门中唯一选出 future executed route 对应的门
-> closed-loop arrival completion
```

下一 cohort 必须在 target truth 前真正携带 public route waypoint / destination semantics；否则 verifier 缺少区分同类门的合法输入。不得从 private future target center、segmentation 或 adjudication 反推 public route intent，也不得用 Office outcome 继续调 Top-K、阈值或 provider order。

## 实现入口

- `future_door_approach_dev.py`：连续未来帧自动 positive materialization。
- `materialize_future_servo_cohort.py`：far/near RGB-D 状态与 private truth 分离。
- `merge_future_approach_proposals_dev.py` / `merge_consensus_future_proposals.py`：calibration-free proposal merge。
- `evaluate_functional_region_availability.py`：ordinary IoU 与 bounded containment 分离。
- `run_goal_rgbd_servo.py` / `evaluate_goal_rgbd_servo.py`：public RGB-D candidate action 与 positive-only evaluator。

Claim ceiling：`PUBLIC_GOAL_SEMANTIC_PROPOSAL_AVAILABILITY_CONFIRMED; ROUTE_IDENTITY_AND_CLOSED_LOOP_SERVO_NOT_ESTABLISHED`。

