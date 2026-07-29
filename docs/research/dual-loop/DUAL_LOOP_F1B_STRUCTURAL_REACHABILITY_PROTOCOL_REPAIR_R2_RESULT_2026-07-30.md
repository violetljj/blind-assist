# F-1B 结构可达性协议修复 R2 结果

状态：`COMPLETE / NO_INCREMENT / SCIENCE_PROTOCOL_VALID / MAINLINE_STOPPED`

执行者：`viojjet`

## 结论

在当前 hash-bound 的五通道 Sparse LK 候选、生产 YOLO 风险语义和“不改目标、方向、
距离带、风险级别、状态机、冷却与反馈”的薄融合边界内，B 分支没有任何能改变实际
可交付提醒或内部状态历史的合法转移：

```text
fresh semantic states derived = 19
fusion action reachable = 0
per-step internal and delivery transition equal = true
history equivalent by induction = true
EARLY_RESPONSE reachable = false
RISK_DISCRIMINATION reachable = false
RISK_CONTINUITY reachable = false
MULTIPLE_INCREMENT reachable = false
first-deliverable-alert lead upper bound = 0 frame
```

因此：

```text
SCIENCE_STATUS = NO_INCREMENT
SCIENCE_PROTOCOL_STATUS = VALID
F-1C_AUTHORIZED = false
PAPER_DUAL_LOOP_CLAIM = STOPPED
CLAIM_CEILING = DEVELOPMENT_ROUTE_REJECTION_ONLY
```

这是现有候选的信息语义缺口，不是数据量、残差阈值、调度或手机性能问题。增加 decision
帧、调 threshold 或优化队列都不能补出候选从未提供的目标身份、LEFT/CENTER/RIGHT
区域或接近方向。按阶段−1合同，双环主线在 F-1B 合法终止，不进入 F-1C。

## 为什么不运行 decision A/B

现有 `SparseLkGeometryProbe` 只输出：

```text
success
inlierRatio
validCorridorFraction
corridorResidual
lowerCorridorResidual
```

它没有目标身份、左右区域、接近方向、径向扩张或 TTC。若不把“外观残差”伪装成
“接近”，最宽仍可辩护的动作只有：

> 几何与同一区域可归因，且当前 pair 已具 planner 提醒资格时，为现有 `MEDIUM`
> 语义候选替代一次时序确认。

生产提醒状态空间使该动作的可达集合为空：

- CENTER `NEAR/CRITICAL` 已是 `HIGH`，A 一帧立即确认，B 无提前空间；
- LEFT/RIGHT `NEAR` 是唯一需两帧确认且 planner 可提醒的 `MEDIUM`，但全局中心走廊
  几何不能归因到左右目标，必须 abstain 并退化为 A；
- 侧向 approaching 的 motion promotion 仍被生产 policy 封顶为 `MEDIUM`，所以同样
  需要两帧、同样不能由中心几何归因；
- CENTER/MID 可经 temporal tracker 成为 `MEDIUM`，但默认 planner 不提醒；R2 明确
  禁止确认替代作用于非 planner-eligible pair，因此不会改变 stabilizer 内部历史；
- `FAR` 和无候选状态不能由几何创建、升级或维持提醒。

R2 validator 从绑定规则派生 19 个 fresh state。所有状态的 fusion action、advance、
suppress 与 continuity transition 均不可达。A/B 从同一 temporal、stabilizer、
side-person、event、cooldown 和 fatigue 状态开始；每一步内部状态和 planner/effect
输入相同，所以 held risk、cooldown、fatigue 和实际 delivery history按归纳保持相同。
由此四个合同端点全部为零可达，无需用 sealed decision 输出重复证明结构恒等式。

## R0、R1 缺陷与 R2 修复

R0 直接相信 spec 行内预填的 `b_permitted_difference=false`，未推导完整状态，也漏绑
下游实现；协议状态为 `INVALID`。

R1 绑定 13 个生产实现并派生 19 个状态，但：

1. 错把侧向 temporal `NEAR/MEDIUM` 列为 `HIGH`；
2. 未阻止确认替代作用于非 planner-eligible 的 CENTER/MID/MEDIUM，因此不能证明
   stabilizer history 相同。

R0、R1 spec 与 validation 原样保留，未覆盖、未回写。R2 是第二次也是最终
protocol-only successor：

- 正确把侧向 temporal NEAR 保持为 `MEDIUM / 2-frame`；
- 冻结 `confirmation_substitution_requires_planner_eligible_pair=true`；
- 逐状态派生 `fusion_action_reachable`；
- `per_step_transition_equal` 检查任何内部或交付变化，不只检查即时提醒；
- 精确冻结 13 个实现 path、2 个前置 validation 与继承合同；
- mutation tests 证明：移除 planner 前置会使 CENTER/MID 内部动作可达；放开 LEFT
  归因会使 side-near 提前可达并转入 `REQUIRES_EMPIRICAL_AB`。

未重跑 F-1A 或 F-1B0，未运行 F-1B decision 候选，也未修改 Android 生产链。

## Decision 隔离边界

```text
decision output status = DECLARED_NOT_ACCESSED_NOT_MACHINE_VERIFIED
decision sessions consumed = 0
```

“未访问”是协议声明和执行记录，不伪装成可由程序证明的文件访问事实。执行者与两名独立
复核者均被明确要求不得访问 decision YOLO、Sparse LK 或 A/B 输出；R2 只读取合同、
前置 validation、实现源码和前序协议凭据。sealed decision 数据仍可用于未来其他全新、
预冻结命题，但不得被本轮双环规则回调消费。

## 独立复核

两条互相独立的只读复核均为 `PASS`：

- 生产语义复核确认 13 个 implementation identity、侧向 temporal MEDIUM cap、
  planner-eligible 前置、19-state closure、历史归纳和四端点均正确；
- 几何/端点复核确认 Sparse LK 信息边界未被夸大，CENTER/MID 与 LEFT/RIGHT/NEAR
  两个潜在交集均为空，`NO_INCREMENT` 无需 decision 执行。

当前 validator 对正式 validation 的重算与 exact-binding recheck 字节完全一致；
R2 mutation tests 为 `9/9 PASS`。

## 可复算凭据

```text
R0 spec sha256:
e215f4d525d71dc71d55c224c65daa5a0b8e86e97d21765219bc4a6b9ef16b50

R0 validation sha256:
04dfbcb7682bfb9a79fba32d063926f48e6983cb1b454e7bd9a1f8ab9fb36dbf

R1 spec sha256:
d30af4f73882675555c895347f4e8493313433a4b212d342548dc64a7a1c6b68

R1 validation sha256:
607534f6a1ead5d25eaa5f12621b6ac76bc95a9b4264963af7f9d859d17fae1f

R2 spec sha256:
cff88b1207a508392b6bfec3205f1add4aa9a827d4b344b36cfe0caa7bfac739

R2 validator sha256:
18d7fad1d5b1f8bfc173e0173bd8e31da16d0183aedac94216b9ca250d24b596

R2 validation and exact-binding recheck sha256:
aef5c931569430b932ec21fa26e1d172ecd0490da37635b9957c91e5ae1216c8
```

正式本地凭据位于：

`artifacts.local/evidence/dual-loop/f1b-structural-reachability-r0/`

`artifacts.local/evidence/dual-loop/f1b-structural-reachability-protocol-repair-r1/`

`artifacts.local/evidence/dual-loop/f1b-structural-reachability-protocol-repair-r2/`

## 主线终点

阶段−1顺序主线已完整走到科学生死门：

```text
F-1A  READY / VALID
F-1B0 READY / VALID
F-1B  NO_INCREMENT / VALID
F-1C  STOPPED_BY_F-1B / NOT_RUN
```

现有 Sparse LK 候选不进入论文双环、正式融合器或生产 CameraX。未来若重新提出双环，
必须先冻结一个确实输出可归因区域级接近证据的新几何源和全新确认合同；不能把本轮
timing-only 凭据、sealed decision 数据或残差阈值调节改写成双环已有效。
