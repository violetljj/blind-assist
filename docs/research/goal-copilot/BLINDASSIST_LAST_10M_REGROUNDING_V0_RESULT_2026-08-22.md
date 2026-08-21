# BLINDASSIST_LAST_10M_REGROUNDING_V0 result

状态：`MECHANICAL_EXECUTION_COMPLETE / NETWORK_SCENE_REPLAY / MILESTONE_CLOSED / NO_SUCCESSOR`

终态：`INTERACTION_OR_CONTROL_BOTTLENECK`

Claim ceiling：`NETWORK_SCENE_MECHANICAL_REPLAY_ONLY_NO_REAL_USER_OR_SCIENTIFIC_CONFIRMATION`

## 1. 执行边界

按用户更新，本里程碑不使用真实设备。执行输入是 3 个真实世界 Mapillary 地点的 9 张既有公开场景帧，每个地点
固定生成 5 个循环起点，共 15 个机械 episodes。场景来自既有 Silver-B Development 素材中已审阅为
`UNIQUE`、且至少有两张不同图像的目标；它们只承担 operational playlist，不建立新 scientific cohort。

每个 observation 都重新运行未修改的 P0：冻结 Grounding DINO Tiny proposal provider，再运行冻结的
`gpt-5.6-terra / medium` single-Brain baseline。控制 state 只保留计数、终态以及上一张图像的 SHA-256 防重放，
不保留 candidate、bbox、特征、身份、handoff 或世界位置。方向指令之后的判断只接受下一张不同图像哈希；
evaluator truth 独立存放，只在终局裁决读取，provider 不可见。

网络 playlist 是预先固定的视角序列，不响应左转、右转或前进一步，因此这里只能检验 regrounding loop mechanics，
不能冒充真实用户控制闭环、导航效果或安全验证。

## 2. 结果

首要安全指标单独报告：

```text
错误入口确认数                         0
实际入口完成确认数                     0
```

这里的 `0` 不是“确认安全”：15 次均未进入 `COMPLETE`，所以系统没有作出任何入口到达确认，也就没有产生错误
确认。全部 episodes 在公开视角耗尽后 fail closed 为 `ABSTAIN` 并提供真人协助出口。

| 指标 | 结果 |
|---|---:|
| 任务完成率 | `0 / 15 = 0%` |
| 完成时间 | `N/A`（没有完成 episode） |
| 首次可靠发现时间 | median `9,745 ms`; min `8,102`; max `20,472` |
| 方向指令数 | total `40`; per-episode median `2`; min `2`; max `4` |
| 重新扫描数 | total `5`; per-episode median `0`; min `0`; max `2` |
| 当前帧 observations | `45` |
| 可靠当前帧 candidates | `40` |

首次发现时间按 episode start 到首个成功 provider observation receipt 的墙钟时间计算。早期 V0 envelope 的
`captured_at_ms` 位于 provider 调用之前，最终汇总因此明确使用 receipt mtime 修正；provider receipt、attempt 和输出
均未重写或重跑。实现随后增加 `processed_at_ms`，避免后续再依赖该兼容口径。

## 3. 仅允许的失败归因

```text
CURRENT_FRAME_GROUNDING_BOTTLENECK          0 / 15
INTERACTION_OR_CONTROL_BOTTLENECK          15 / 15
REGROUNDING_LOOP_MECHANICALLY_USEFUL        0 / 15
```

P0 在 `40 / 45` 个当前帧 observation 给出可靠 candidate，但所有 fixed playlists 都没有提供满足“当前帧近距 cue，
再用新帧确认”的控制结果。因此主要瓶颈不是本批次的 current-frame candidate availability，而是网络回放缺少会响应
指令的交互/控制路径。该结果不支持把 regrounding loop 声称为 mechanically useful；它只证明实现能够持续重新调用
P0、记录指令/重扫并在不能确认时停止。

## 4. 证据与封口

ignored 本地证据根：
`artifacts.local/evidence/last-10m-regrounding-v0/network-scene-run-v0/`。其中包括 provider lock、run manifest、
逐 observation dispatch/completion/原始输出、append-only events、15 个 episode summaries 和最终
`field_report.json`。公开 manifest 保留 Mapillary image key 与 source URL；evaluator truth sidecar 与 public manifest
hash 绑定。

最终审计为 `45` 个 observation receipts、`45` 个 dispatch journals、`45/45 RUN_SUCCESS`、`15` 个 episode
summaries，没有 `IN_DOUBT` 调用；`field_report.json` SHA-256 为
`0bc443cac390db1654b09289631ee987f3f6fc066d2dd99d115f7809ce4cf294`。

本里程碑到此关闭。P1 保持关闭；不建立 P1-W3，不重开 referent persistence，不创建新模型、训练、数据 cohort、
多臂比较、Android 集成或后继研究协议。
