# USTRF observability program real-world authority terminal R0 结果（2026-07-25）

状态：`EVIDENCE_PROGRAM_BLOCKED_BY_REAL_WORLD_AUTHORITY / VALID`

权限：`RESEARCH_TERMINAL / ALGORITHM_EFFECT_UNTESTED / G1-G7_CLOSED / ANDROID_CLOSED / HUMAN_CLOSED / PRODUCTION_CLOSED`

## 总结论

本轮持续研究已经走到总目标允许的最终研究结局：

`EVIDENCE_PROGRAM_BLOCKED_BY_REAL_WORLD_AUTHORITY`

核心算法没有在权威 route/event truth 和完整对照下被执行，因此不能写成 `USTRF_CORE_HYPOTHESIS_NOT_SUPPORTED`；所需权威输入也没有全部存在，因此不能写成 `TASK_NOT_OBSERVABLE_WITH_CURRENT_SENSOR_OR_ROUTE_STACK`。

准确结论是：source transport、离线工程链和若干局部机制可以复算，但 G1–G7 所需的 canonical authority、fresh metric geometry、intended-route truth 与独立 event lifecycle truth 尚未同时存在，正式实验不能合法执行。

## Authority gap matrix

| 输入/证据族 | 已证明 | 阻塞项 | 终态影响 |
| --- | --- | --- | --- |
| 当前 41-sequence canonical pack | 41/41、62,229/62,229 frame 可复算 | canonical transform 全 unknown；authoritative truncation absent | `SOURCE_AUTHORITY_ABSENT / VALID`，G1 关闭 |
| JRDB source transport | 同帧 label、RGB、capture time、calibration 闭合 | 无 intended-route/event authority | 证明“并非所有 source 都不可得” |
| JRDB background affine | RGB/time/features/track/residual 可用 | 仅 11/31 pair 达 inlier ratio 门 | `EGOMOTION_QUALITY_AVAILABILITY_INSUFFICIENT / VALID` |
| SM-S9280 ARCore | 861 raw-depth candidate、TRACKING 可观察 | fresh current-frame depth 仅 1/861；pose `EPHEMERAL_PER_FRAME` | metric geometry/VIO gate 关闭 |
| 三源 RGB-D+pose R2 | 3/3 geometry transport 通过 | route/event truth unavailable | `DO_NOT_SELECT_HARDWARE` |
| 动态 RGB-D R3 | 独立 pose/route ledger/evaluator 已实现 | 3/3 source route-event admission 拒绝；event truth false | `DO_NOT_SELECT_HARDWARE` |
| 人体/真实采集 | continuous goal 未授权协调参与者、consent 或实地人体采集 | 无新 route/event authority | 不能自动补采回救 |

## 机器验证

producer 与 validator 分别重新读取并 hash-bound：

- G0 validation；
- JRDB single-frame validation；
- JRDB ego-motion validation；
- ARCore freshness receipt；
- sensor replay R2/R3 final reports；
- `SANPO_CURRENT_STATUS.md`。

结果：

- config SHA：`52fa6e40327b86f2afce814ae1132cb70b844b5553bdfed35f4a13706ced9dd3`
- producer PID：`40796`
- receipt SHA：`25962cd46a0f9bd9b689a2e32871baabcdb7a6f49a29736a20ae3587aed84be1`
- validator PID：`36308`
- validation SHA：`4b587df5836ad4434570ca3cfc7b3a9d1e31c0c1e856bae4dd5a41ca67865c3d`
- checks：八项 authority fact、deterministic recomputation、PID isolation、terminal、not-algorithm-rejection、not-unobservability 与 production closed 全部通过

## 保留成果

- source-only / aggregate-only / independent-validator 的 G0 审计模式；
- 41/41 canonical input 与既有 replay/profile；
- JRDB ZIP64 bounded range materializer；
- 同帧 label/RGB/time/calibration source-authority canary；
- 背景 sparse-LK/global-affine 的明确负 availability 结果；
- RGB-D+pose replay、route/truth 分离、fail-closed safety kernel 与 adapter seams；
- 全部历史 blocked/reject terminal 保持不可变。

## 恢复条件

只有至少一项新的外部事实出现，才应恢复本总目标：

1. 同帧 fresh metric geometry + `INTER_FRAME_STABLE` pose；
2. hash-bound、因果可用的 intended-route provider truth；
3. 独立 route-event onset/alertable/passed/cleared lifecycle truth；
4. 若需要新参与者或真实采集，取得明确协调、consent 与采集授权。

恢复后必须另立版本化 goal，从对应 authority admission 开始；不得直接跳到 G4–G7，不得复用本轮 discovery 数据作为 acceptance。
