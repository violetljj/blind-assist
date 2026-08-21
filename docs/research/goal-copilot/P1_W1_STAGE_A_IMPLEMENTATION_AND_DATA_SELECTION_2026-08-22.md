# P1-W1 Stage A outcome-blind implementation and data selection

状态：`IMPLEMENTATION_FROZEN / ROSTER_FROZEN / PERFORMANCE_NOT_RUN / STAGE_A_EXECUTION_NOT_AUTHORIZED / STAGE_B_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

Claim ceiling：`MECHANICS_AND_DATA_SUPPORT_ONLY / NO_EMPIRICAL_CAPABILITY / NO_PRODUCT_OR_SAFETY_AUTHORITY`

## 1. 授权与问题

本 successor 只完成冻结协议指定的
`P1_W1_STAGE_A_OUTCOME_BLIND_IMPLEMENTATION_AND_DATA_SELECTION`。它实现 C0/W1-T0 的共同 observation、identity、
referent fusion 与两个 spatial evidence path，冻结 exact roster、deadline、tolerance 和资源预算；不运行 17 个
真实 episode 的 performance，不读取 arm outcome，不选择 T1 pose source，也不接 Android/default-App。

Stage A 仍只问：W1-T0 是否能在有证据时保留比 C0 更多合法 persistence，并在 keyframe geometry 失效或
translation 越界时及时 stale，而不是靠延续 bbox 或提高 abstention 获胜。

## 2. 最小 matched implementation

稳定实现位于 `scripts/research/goal_copilot_bridge/p1_w1_stage_a/`：

- `FrozenRgbProvider`：固定 source keyframe、source region 与 1,000-point ORB scene geometry；无 online update、
  future frame、detector、tracker、pose/depth truth 或 global database。
- 共同 candidate path：source-region ORB descriptor 到当前帧的 local matches；少于 6 个 match 时 observation
  为 `NONE`，不能只把预测 region 当 observation。
- 独立 identity path：固定 source-region HSV `24×16` histogram；Bhattacharyya similarity `>=0.70` 才为
  `SUPPORTED`。它不参与 spatial transform。
- C0：只接受 camera-relative、rotation-compatible homography；scale 必须在 `[0.92, 1.08]`，shear
  `<=0.08`，perspective `<=0.0015`，且 fundamental inlier ratio 不得比 homography 高 `0.15`。
- W1-T0：使用 current-to-source-keyframe homography；scene match `>=12`、Lowe ratio `0.75`、RANSAC
  `3 px`、homography inlier ratio `>=0.50`、二维 keypoint spread `>=0.10`。Fundamental advantage `>=0.15`
  视为 parallax/translation overreach，必须 stale。

两 arm 共享 candidate、identity threshold 和 `ReferentSnapshot` fusion；只切换 `CAMERA_RELATIVE` 与
`KEYFRAME_RELATIVE` spatial evidence。`REACQUIRED` 仍严格要求 spatial compatibility 与独立 identity 同时
`SUPPORTED`。Provider 无法观察 motion、geometry degenerate 或 translation overreach 时，同一 frame 内立即
`SPATIAL_ANCHOR_STALE`，清空 bearing、禁止 directional guidance，但不删除仍有效 identity memory。

这只是一个刻意弱、可审计的因果 baseline，不是 tracker/SLAM 选择。ORB、HSV 或这些 threshold 不因未来结果
搜索或更新。

## 3. outcome-blind roster

Selector 不读 C0/T0 输出，只读已消费 P1-D0 的 temporal/instance truth 与 ADT camera trajectory。旧 15 个 episode
全部纳入，避免按未来结果挑 clip；它们缺少 pure-rotation support，因此只在同两条已下载 source 内机械补入 2 个
`translation <=0.10 m / rotation >=15° / return rotation <=10°` 的 90-frame turn-away/return 窗口。没有新增来源、
下载或模型调用。

公开 roster 只含 opaque `w1a-*` ID；source episode、physical target、timestamp、初始 bbox 和 motion summary
留在独立 evaluator-private truth map。冻结 support 为：

```text
real episodes                         17
real frames                         1904
ROTATION_DOMINANT                       2
SMALL_TRANSLATION                       7
TRANSLATION_BEYOND_TIER0                7
OCCLUSION_OR_REAPPEARANCE              11
IDENTITY_CONFUSER                       8
OBSERVATION_LOSS                       11
GEOMETRY_DEGENERATE mechanics fixture   1
```

两个 rotation cases 的最大平移均为 `0.091718 m`、最大旋转 `22.900242°`、回到起始朝向误差
`1.882504°`。Geometry-degenerate 由固定的“valid textured keyframe → blank current frame” mechanics fixture
验证 fail-stale；它不进入真实 utility denominator。

机器 receipt 位于 ignored `artifacts.local/evidence/p1_w1_stage_a_selection_v3/`：

```text
public_roster.json                 sha256 1969560ba8a3863ad4aef16fca9141602144a4b4555ee38c38ff49b6f62bef70
evaluator_private_truth_map.json   sha256 fd23bb01d928fdf97d65fa0f1d67868b85c0050108a9c632f8296d660f75aad8
```

首次只复用旧 15 集的 selection receipt 保留在 `p1_w1_stage_a_selection_v1/`，其 rotation support 为 0、终态为
`NOT_EVALUABLE_DATA_SUPPORT`；它没有 arm output。补入 rotation 窗口后才形成当前 v3 roster。

## 4. evaluator constants 与 verdict

- timely-stale deadline：guard 首次触发的同一 frame；下一帧才 stale 计失败。
- supported-anchor bearing tolerance：`10°`；只在 arm 声称 anchor 可用且 evaluator bearing truth 可用时进入分母。
- event matching：同一 frame exact match；不设 post-hoc grace window。
- aggregate 与每个 support bucket 同时报 denominator；synthetic fixture 不计 real utility。
- 统计 margin：无软 margin；沿用冻结协议的 strict count/rate comparison 与所有 zero hard gates。

`adjudicate_stage_a` 已按冻结顺序实现 `NOT_EVALUABLE`、hard-gate failure、
`HONESTY_GAIN_ONLY_BY_ABSTENTION`、`NOT_SUPPORTED` 与 `WORLD_REFERENT_SIGNAL_ESTABLISHED`。任何 fabricated
observation、single-channel reacquisition、stale guidance、truth leakage 或 future access 都不能被 utility 抵消。

## 5. 预算、验证与停止

未来若用户另行授权 Stage A execution，只允许对该 v3 roster 做 C0/W1-T0 各一次 deterministic pass；不得重选
episode、改 threshold、添加模型、按 outcome 重跑或搜索 policy。资源上限为单机 CPU/OpenCV、17 个真实
episode、1 个 mechanics fixture、0 external/model calls。

本次只运行 9 个 synthetic/contract tests，全部通过；测试覆盖 observation honesty、双条件 reacquisition、
translation/degeneracy stale、reference-frame isolation、共享 RGB provider、selector motion/confuser 分桶和 verdict
hard gate。没有执行真实 arm，因此当前没有 C0/T0 指标或科学结论。

唯一 successor：`P1_W1_STAGE_A_SINGLE_EXECUTION`，状态 `NOT_AUTHORIZED`。只有 Stage A 先建立信号且 translation
support 非零、用户再次授权，才允许 outcome-blind 选择 W1-T1 pose/anchor implementation；不得由本文件自动进入
Stage B。
