# BlindAssist Goal-Driven Visual Copilot V2 路线图

状态：`CURRENT_SYSTEM_BLUEPRINT / DESTINATION_GOAL_GROUNDING_P0 / P0_CONTRACT_V1_FROZEN / MOCK_MECHANICS_PASS / EVIDENCE_GATED / ASYNC_FAST_MID_SLOW / NO_COHORT_BASELINE / R5_PERMANENTLY_CLOSED / DEFAULT_APP_UNCHANGED`

本文是 BlindAssist Goal-Driven Visual Copilot V2 的系统设计蓝图。它把原始完整产品方案与 GC、ADT
已经得到的证据重新合并，定义能力层次、运行时协作、研究顺序与停止边界。它不是新的 formal protocol，
不授权数据消费、模型调用、Sky 搜索、Android/default-App 接线或产品/安全主张。

## 1. 产品目标

BlindAssist 不是逐帧播报检测框的模型，而是维护用户目标直到完成或安全退出的视觉副驾：

```text
用户目标
   ↓
Goal Copilot Brain
   ↓
主动寻找相关视觉证据
   ↓
锁定现实世界中的正确目标
   ↓
持续跟踪、判断进展与恢复
   ↓
接近、对准、确认任务完成
   ↓
连续且可撤回的视觉引导
```

系统必须区分四件事：

1. `Goal Grounding`：用户说的目标对应哪个现实实体或区域；
2. `Target Persistence`：目标锁定后能否保持身份与连续性；
3. `Approach / Completion`：是否在安全接近，何时真正到达或可交互；
4. `Active Perception`：证据不足时，下一眼看哪里最有判别价值。

任何单个 provider 的高分都不能直接宣告目标完成。`completion`、`unsafe guidance`、`abstain` 与
`termination` authority 始终属于 Goal Copilot Brain 的冻结合同和独立 evaluator。

## 2. 系统组成

### Evidence Providers

Evidence Providers 只提供带 provenance、时效和不确定性的观察，不独占最终目标身份或动作决策。

| Provider | 主要 evidence | 典型作用 |
|---|---|---|
| Detection / open-vocabulary grounding | object/region candidates、类别或文本相似度 | 产生候选，不单独决定目标 |
| OCR / logo | 招牌文字、机构名、标识 | 将用户 goal 绑定到建筑、店铺或设施 |
| VLM / relational reasoning | facade、入口、功能与空间关系 | 判断 `entrance_of(target_building)` 等关系 |
| Tracking / optical flow | 短时运动、bearing continuity、track confidence | 锁定后保持目标连续性 |
| TargetMemory / redetection | appearance memory、LOST search、identity evidence | 有条件恢复目标，允许保守拒绝 |
| Depth / relative nearness | 距离趋势、scale change、approach rate | 判断是否接近，不直接证明入口归属 |
| Traversability / doorway geometry | 可通行区域、门洞、clearance、alignment | 支撑 approach 与 interaction readiness |
| Risk / safety | hazard、TTC、路径风险、UNKNOWN | 约束引导；安全 veto 高于任务进展 |
| POI / map prior | 目标的大致位置、建筑或入口先验 | 作为 coarse prior，不能覆盖现场视觉反证 |

### Goal Copilot Brain

Brain 维护跨帧、跨 provider 的任务状态：

```text
Task belief
Destination / target identity
Evidence ledger and provenance
Temporal memory
Target state and LOST state
Progress / nearness / alignment
Recovery policy
Uncertainty and active-view request
Termination / completion verification
```

最小状态流为：

```text
GOAL_UNGROUNDED
  → DESTINATION_CANDIDATES
  → TARGET_LOCKED
  → TRACKING
  → LOST / RECOVERY
  → APPROACH
  → COMPLETION_PENDING
  → COMPLETED | ABSTAIN | BLOCKED
```

状态只能由满足相应 evidence contract 的事件推进；低频 semantic belief 不能伪造高频 tracking，短时
tracking continuity 也不能反向证明目标语义正确。

## 3. 快 / 中 / 慢异步视觉架构

V2 不要求每帧运行一个大模型。不同 evidence 按任务需要和时效异步更新：

| 层 | 目标频率 | 典型能力 | 输出责任 |
|---|---:|---|---|
| Fast loop | 30–60 Hz | IMU、optical flow、tracking、TTC、短时 risk | 运动连续性、安全约束、短期 bearing；不得重写目标语义 |
| Mid loop | 5–15 Hz | detector、open-vocab grounding、depth、entrance candidates | 目标位置、障碍、接近趋势、候选更新 |
| Slow loop | 按需 | OCR、VLM、logo/facade reasoning、POI/map association | 目标建筑身份、入口归属、歧义解释、主动取证请求 |

Brain 以 event-time 而不是“同一帧全部齐备”为假设融合这些 evidence。每条观察至少绑定 provider、
timestamp/frame、target/candidate identity、confidence/uncertainty、source/config identity 与有效期。过期的
slow-loop 语义只能作为 prior；一旦现场 evidence 冲突，必须降置信或重新 grounding。

推荐的数据流是：

```text
Slow: “招牌显示这是 XX 医院” ─────┐
                                    ├→ Destination belief
Mid:  facade 上的入口候选 A/B/C ───┤
                                    ├→ relation ranking → target lock
Fast: bearing / flow / TTC / risk ──┘
                                            ↓
                                  continuous guidance
```

## 4. “找到 XX 医院入口”的端到端参考路径

| 阶段 | Brain 问题 | 主要 evidence | 失败时动作 |
|---|---|---|---|
| Goal parse | 用户要找哪个实体和完成条件 | utterance、POI prior | 请求澄清，不猜目标 |
| Destination grounding | 哪栋是 XX 医院 | OCR、logo、facade、map prior | 保留多个候选或请求转头 |
| Entrance grounding | 哪个门属于目标医院 | entrance candidates、facade relation、VLM | 排序、解释 evidence、允许 abstain |
| Target lock | 当前锁定的入口 identity 是否充分 | 多源一致性、candidate margin | 不足则不进入强引导 |
| Persistence | 入口是否仍在视野及哪个方向 | tracking、flow、redetection | LOST 后保守恢复或重新 grounding |
| Approach | 是否越来越近且路径可行 | bearing、depth/nearness、traversability、risk | 停止、改向或请求新视角 |
| Completion | 是否真的到达可进入位置 | doorway geometry、nearness、alignment、多帧确认 | `COMPLETION_PENDING` 或 abstain |

初始 P0 只覆盖 destination/entrance grounding 和排序；`public / accessible entrance`、连续引导、
closed-loop completion 与真实用户效果属于后续能力，不能由 P0 排名结果提前声称。

## 5. 现有证据放回系统中的位置

| 证据 | 已建立的事实 | 在 V2 中的位置 | 不允许外推 |
|---|---|---|---|
| GC1 sealed symbolic pilot | 在冻结 symbolic/oracle-style evidence 上建立过 bounded search signal | Brain policy 与 completion-chain 先导机制 | 真实 perception、真实用户或产品闭环 |
| GC2-A / GC2-B | moderate perception uncertainty 下 winner 失效，Sky search signal 未建立 | Brain 对 noisy evidence 的失败证据 | Sky 无价值或完整 Copilot 不可行 |
| ADT-0 / ADT-1 | 真实 recorded RGB 上 flow 改善 target temporal evidence，但长 dropout 保留 | Persistence 基线与 evaluator 基础 | closed-loop navigation 或用户引导效果 |
| TargetMemory R1 | recall/IoU 有限提升，13 次实例重检测无 wrong-instance，但长时恢复未解决 | Conservative persistence mechanism | arbitrary tiny-object 长时重捕获已解决 |
| YOLOE visual prompt | 固定 canary 上 candidate recall 更差 | 被否掉的单-provider proposal 路线 | open-vocabulary grounding 整体无用 |
| R3 observability audit | 失败主要集中于不可见/重遮挡和极小目标 | Persistence failure anatomy | 所有大尺度 destination target 同样失败 |
| R4 resolution / tiling | 已测试的尺度 arms 在固定 tiny windows 为 0/3 | 关闭同窗继续扫尺度 | 任意高分辨率 grounding 都无效 |
| R5 DINOv-SwinL | 仅 W4 20 px 目标 1/3 弱命中，正确分数低于错误候选，1,974 wrong-instance candidates | appearance-only tiny-target capability upper bound | 纯 RGB 一定无法重捕获 |
| SAM 3.1 audit | 冻结 cross-image image-only arm 无受支持接口，`NOT_EVALUABLE_INTERFACE` | 接口边界 | 记为 0/3 或模型能力负结论 |

R1–R5 只是把 `Goal Persistence` 的一个极端角落探深。它们支持停止把“几像素任意小实例的
appearance-only 长时重捕获”作为 P0，不构成 Goal-Driven Visual Copilot 的系统失败。

## 6. V2 能力路线

### P0 — Goal Grounding（当前唯一主线）

核心问题：用户目标到底对应现实世界里的哪个实体或区域？

首个任务固定为：给定“找到目标建筑的入口”，在真实多建筑、多门图像或视频中，将目标建筑所属入口
排在其他门之前，并输出逐候选 evidence。

最小接口：

```text
Input:
  user_goal
  RGB image/video
  optional coarse POI prior

Output:
  destination candidates
  entrance candidates
  goal-conditioned ranking
  entrance_of relation evidence
  uncertainty / abstain reason
```

首个 evaluator 至少区分：candidate coverage、目标建筑识别、入口归属排序、top-1/top-k、evidence
faithfulness、无足够证据时的 abstention，以及 distractor building/door 错锁。具体 dataset、denominator、
threshold 与 fresh/Development 角色必须在任何 baseline outcome 前由独立协议冻结；本蓝图不预注册数值门。

### P1 — Target Persistence

在 P0 锁定医院入口、服务台等大尺度目标后，复用 flow、tracking、TargetMemory、conservative LOST、
redetection 与 ADT failure evaluator，回答“找到以后能否别跟丢”。P1 优先检验产品尺度目标，不恢复
carrot tiny-object teacher zoo，也不把 R5 同窗改名重跑。

### P2 — Approach / Completion

将 bearing、relative nearness、depth、motion、doorway geometry、traversability 与 risk 接入同一 target
identity，回答“方向是否正确、是否越来越近、路径是否可行、是否真的到门前”。Completion 必须多源、
多帧且 fail closed；语义识别正确不等于已到达，可见门洞也不等于安全可进入。

### P3 — Active Perception

当 destination belief 或 entrance ranking 不可判定时，Brain 选择最有信息价值且安全的下一视角，例如请求
用户小幅向左转头以看到完整 facade，或查看两个近分入口中的另一侧。P3 的目标是减少歧义和无效等待，
不是为了得到答案而无限要求用户转动，也不能绕过安全 veto。

## 7. Sky 的位置

Sky 不决定研究方向，只在 BlindAssist 已有真实任务、冻结 evaluator 和明确可测 headroom 后搜索候选机制：

```text
BlindAssist 定义任务 / evaluator / hard gates
        ↓
baseline 暴露稳定 failure anatomy
        ↓
仅把有界 policy surface 交给 Sky
        ↓
BlindAssist 独立验证与决定 claim ceiling
```

Goal Grounding 的潜在 Sky surface 可以是 `rank_candidates`、`update_destination_belief` 或
`select_semantic_evidence`，但只有在 baseline 证明错误来自这些组合策略而不是 candidate coverage、OCR
缺失、数据不合格或 evaluator 歧义后才可授权。Sky 不接触 hidden truth、hard gates、completion authority
或最终科学 verdict。

## 8. 推荐执行顺序

1. 冻结 P0 的真实场景、goal、candidate/truth schema、Development/fresh 角色与 ranking evaluator；
2. 建立最小可解释 baseline，先得到 candidate coverage 与 relation-ranking failure anatomy；
3. 只有明确 signal 后才选择补 OCR、facade association、VLM relation 或 POI prior 中的一个缺口；
4. P0 达到预注册门后，把锁定的大尺度入口交给 P1 persistence；
5. P1 稳定后接 P2 approach/completion；证据不足时再开启 P3 active perception；
6. 仅在具体模块有可测 headroom 时建立独立 Sky task。

第 1 步的 P0 goal、schema 和 evaluator mechanics 已由
[`P0 Goal Grounding Protocol V1`](P0_GROUNDING_PROTOCOL_V1.md) 冻结并通过 mock unit tests。当前下一动作
只允许设计第 3 步所需的 cohort materialization、去重、strata denominator 与 Development/fresh 角色规则；
没有授权下载、采集、物化 cohort、冻结 baseline 数值门、运行 baseline、调用模型、启用 Sky、恢复
VIO/SLAM、接 Android/default App 或提出安全/产品成功主张。
