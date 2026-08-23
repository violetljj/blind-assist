# SUN3D native-door approach V0 result (2026-08-24)

状态：`SEALED / NATIVE_OBJECT_45_TRUTH_15_OF_15 / PUBLIC_REFERENT_AMBIGUOUS / SELECTION_NOT_EVALUABLE / NO_ALGORITHM_SUCCESSOR / NO_P1`

## 结论

封存运行对 private `object 45` 的原始计数保持不变：15 个 observations 中 object 45 在当前帧不可见 `11/15`；
可见的 4 帧中 proposal 可用 `3/4`，private-object 计分为 `0/3`。但后续
[`Referent Identifiability Audit`](BLINDASSIST_SUN3D_REFERENT_IDENTIFIABILITY_AUDIT_V0_RESULT_2026-08-24.md)
证明公开目标 `the door` 无法唯一绑定 object 45：同一 sequence 还有 `object 57 = door: bathroom`，且三个可用
proposal 帧都至少有两扇视觉上合理的门。因此 `0/3` 与 `4/15 wrong` 均改为 public-goal `NOT_EVALUABLE`，不能
作为 referent selection 或 active-search successor 的授权证据。

这是单个预录制室内 generic-door episode 的 private-object visibility/proposal 描述。它不建立 public-goal
acquisition/selection、functional aperture、命名目的地、闭环控制、到达完成、用户安全或产品性能，也不授权新的
算法 successor。

## 冻结与防火墙

数据来自 [SUN3D 官方主页](https://sun3d.cs.princeton.edu/)所列 fully annotated sequences with pose
correction。冻结规则在下载 pixels 和调用 provider 前确定：按官方列表顺序选取第一个具有精确 `door` object、
至少 5 个有效 polygon、首个可见距离为 4--10 m 且随后存在首个可见距离小于 2 m 的 sequence，再在固定起止帧间
均匀取 15 个 observation。

规则选中 `hotel_umd/maryland_hotel3` 的 object 45，固定 annotation frame 94--200：

- 15 observations：`VISIBLE=4 / NOT_VISIBLE=11`；不得补抽可见帧；
- map-derived range：4.45 m 到 1.53 m；两个 observation 小于 2 m；
- roster 冻结前：pixels/provider/teacher calls 均为 0；
- roster SHA-256：`84299ed82b816757c55e29f76acf077d39e9b0de8965e6475d841645be5262fe`；
- pixels manifest SHA-256：`fed47caeec1ca14c5ad6ca2de5ece2ca33a46def7cbbeddd702ebcfaa24d7487`。

Baseline/provider 只收到固定 goal `the door`、当前图像和 Grounding DINO 编号候选。原生 polygon、可见性、
world target、range/bearing 和 arrival truth 只进入 private evaluator。Teacher 未运行；`LOST=11` 仅由 episode
FSM 在首帧 `VISIBLE` 后的 `NOT_VISIBLE` 派生，不代表 tracking 或 persistence。

## 冻结结果

| 指标 | 结果 |
|---|---:|
| Strong truth coverage | 15/15 |
| Visible / not visible | 4 / 11 |
| Proposal availability on visible | 3/4 |
| Private-object selection score given usable proposal | 0/3, public-goal `NOT_EVALUABLE` |
| Private-object wrong-guidance score, all observations | 4/15, public-goal `NOT_EVALUABLE` |
| Correct grounding | 0 |
| Proposal miss | 1 |
| Abstained/ambiguous with usable proposal | 2 |
| Wrong confident on visible | 1 |
| Wrong confident on not visible | 3 |
| Correct non-commitment on not visible | 8 |

`provider_calls/attempts/in_doubt/teacher_calls/baseline_reruns = 2/2/0/0/0`。Final report content SHA-256 为
`36f2eee5ee9a183fcd4a80d506fa71e76d17d9f591fedd8642135b8a34690f36`，文件 SHA-256 为
`a21136af01957c1d9909272c35a9e2fa62403165e87f8025e7dca46a309226c8`。

## Failure attribution 与边界

这一 episode 有完整的 private object-45 polygon/range denominator，但没有完整的 public referent contract。
`NOT_VISIBLE=11/15` 只描述 object 45 是否在画面中，不等于字面目标 `the door` 是否在画面中；`0/3` 也不能把
公开上 equally-valid 的其他 door 判为错。唯一仍可保留的分层事实是：object 45 的 current-frame visibility 与其
proposal availability 可机械计算；public-goal acquisition、selection 与 wrong commitment 均不可评价。

不允许用本 episode 改 prompt、阈值、provider、goal、候选或抽帧来救结果；不从派生 `LOST` 启动 P1。后续若继续，
必须先冻结 independently public-identifiable 的 `UNIQUE / SET_VALUED / AMBIGUOUS` goal contract；在此前不得增加
SUN3D cohort 来复现这条无效 failure ordering，也不得在本 episode 上调参。

Claim ceiling：
`PRERECORDED_REAL_RGBD_PRIVATE_OBJECT_VISIBILITY_AND_PROPOSAL_DESCRIPTOR_ONLY_NO_PUBLIC_REFERENT_SELECTION_ACTIVE_SEARCH_CONTROL_USER_SAFETY_OR_PRODUCT_CLAIM`。
