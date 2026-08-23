# SUN3D native-door approach V0 result (2026-08-24)

状态：`SEALED / NATIVE_STRONG_TRUTH_15_OF_15 / CURRENT_FRAME_GROUNDING_FAILURE / NO_ALGORITHM_SUCCESSOR / NO_P1`

## 结论

在一个冻结的真实 RGB-D 门接近 episode 上，V0 首先失败在 current-frame grounding / selective
commitment，而不是缺少可评价 truth。15/15 observations 都有官方原生 polygon 加 pose-corrected trajectory
支持的强 truth；4 个目标可见帧中 proposal 可用 `3/4`，但给定可用 proposal 的正确选择为 `0/3`。全 15 帧有
4 次 wrong confident guidance，其中 3 次发生在目标不在当前帧时。

这是单个预录制室内 generic-door episode 的机制证据。它不建立 functional aperture、命名目的地、闭环控制、
到达完成、用户安全或产品性能，也不足以授权新的算法 successor。

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
| Selection accuracy given usable proposal | 0/3 |
| Wrong confident guidance, all observations | 4/15 |
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

这一 episode 排除了此前 8x89 的 `TRUTH_OR_CONTRACT_INSUFFICIENT` 主导解释：强 truth denominator 是完整的。
可见帧里已有 3 个可用 proposal，却没有一次正确 commit；同时不可见帧仍出现 3 次错误 commit。因此本次最直接的
信号是 referent selection / selective commitment，proposal miss 为次要但仍存在的失败层。11/15 当前帧不可见也说明
camera pointing / acquisition 是真实的 episode 条件，但这个预录制 sequence 不能评价 active camera policy。

不允许用本 episode 改 prompt、阈值、provider、goal、候选或抽帧来救结果；不从派生 `LOST` 启动 P1。后续若继续，
应增加预冻结的独立 SUN3D approach episodes 来检验这一 failure ordering 是否复现，而不是在本 episode 上调参。

Claim ceiling：
`PRERECORDED_REAL_RGBD_DOOR_APPROACH_CURRENT_FRAME_GROUNDING_ONLY_NO_CLOSED_LOOP_CONTROL_USER_SAFETY_OR_PRODUCT_CLAIM`。
