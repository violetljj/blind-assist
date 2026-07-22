# USTRF-SENSOR-REPLAY-R2 多来源 RGB-D+pose replay 结果（2026-07-22）

状态：阶段完成；`DO_NOT_SELECT_HARDWARE` / benchmark-only / production-isolated

## 结论

R2 已回答第一半问题：当输入确实包含同步 RGB、metric depth 与 `INTER_FRAME_STABLE` camera pose 时，统一运输、哈希、同步和时序深度重投影链可在三种许可清晰来源上自动重放。三源各冻结连续 120 帧，source-aligned fraction 均为 `1.0`，geometry transport 均通过。

R2 尚未回答“USTRF 算法闭环是否成立”：三份公开序列都没有独立 pose estimate、body-bound route truth 或因果事件 lifecycle truth。两次互不可见且不看 candidate alert 的模型审核一致拒绝 route/event admission。因此 pose drift、route projection error、event recall、false alerts/min 与 alert clearance 必须保持 `not_evaluable`，不能伪记为 0；最终 verdict 合法地保持 `DO_NOT_SELECT_HARDWARE`。120 episode、U0、重复静置 ARCore、硬件选择、Android runtime 与生产权限继续关闭。

## 冻结来源与许可

| 来源 | 性质 | 许可 | 自动获取与哈希 |
| --- | --- | --- | --- |
| ETH3D `cables_1` | 真实 RGB-D + mocap pose | CC BY-NC-SA 4.0 | mono `682857a3…fcc651`；rgbd `7770bab4…74a31` |
| ICL-NUIM `living_room_traj0` | 合成精确 RGB-D + pose | CC BY 3.0 | PNG 包 `4eca8c2e…1f1ad`；pose `658bfae1…67d61` |
| TartanAir `JapaneseAlley/Hard/P000` | 合成同步 RGB-D + pose | CC BY 4.0 | Agent 已获取的预处理 archive `7a30f648…1cfcf`，397 帧中冻结前 120 帧 |

Bonn Dynamic 虽已有 590 帧本地真实审计，但官方页面未找到明确数据许可证，本轮不计入“许可清晰三来源”。来源、URL、单位、坐标和 archive hash 冻结于 `configs/ustrf_sensor_replay_r2_sources_v1.json`；门槛冻结于 `configs/ustrf_sensor_replay_r2_prereg_v1.json`。

## Replay 量化结果

| 来源 | aligned | 有效 depth | temporal reprojection median / p95 | clearance geometry proxy | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| ETH3D | 1.000 | 0.699 | 0.60 / 4.13 mm | 0.616 m | transport pass |
| ICL-NUIM | 1.000 | 1.000 | 1.66 / 9.15 mm | 2.374 m | transport pass |
| TartanAir | 1.000 | 1.000 | 4.73 / 288.83 mm | 4.262 m | transport pass；worst source |

这里的 p95 是 12 对抽样相邻帧各自 p95 残差的中位数；TartanAir 夜景/强旋转是本轮最差来源。clearance 只是下半画面中央 20% depth 的 p05 中位数，不是提醒生命周期 clearance。

统一 harness 已实现未来可选 evaluation input：有独立 estimated pose 时报告 translation/rotation drift；有独立 route truth/prediction 时报告 pixel median/p95 与 unknown rate；有 hash-bound event truth 和 candidate alert trace 时报告 event recall、false alerts/min 与 clearance rate/p95。缺任一输入，相应闭环门 unavailable，不从宏平均中消失。

## 隔离路线与事件审核

三张 contact sheet 各绑定 12 个固定帧及 SHA-256。Reviewer A 与 B 在互不可见的新上下文运行，均明确看不到 candidate alert：

- ETH3D：桌面线缆近景，没有可走地面、身体朝向或行走走廊；route invalid。
- ICL-NUIM：沙发/墙面加大幅 roll，没有稳定 forward corridor；reject/abstain。
- TartanAir：早期虽见巷道与自行车，但随后大幅 roll/pitch，无法绑定 intended route、TTC 或 clearance；reject/abstain。

两模型没有形成语义分歧，故不启动第三模型；共识为 `fail_closed`，event truth authority 为 false。审核清单、原始输出、prompt 和共识哈希位于 `artifacts.local/evidence/ustrf-sensor-replay-r2/review-inputs-v1/`。

## 实现与证据

- 独立 Module：`scripts/research/ustrf_sensor_replay/`。
- 稳定入口：`scripts/run_research_tool.py ustrf-sensor-replay ...`。
- 规范化收据：`artifacts.local/evidence/ustrf-sensor-replay-r2/normalized-v1/normalization_report.json`。
- 最终报告：`artifacts.local/evidence/ustrf-sensor-replay-r2/replay-report-v2.json`。

## 下一步边界

下一轮仍应留在 R2，而不是转硬件：自动取得具备可审计行走路线/事件条件的来源序列，产出独立 estimated-pose trace、route truth/prediction 与 model-consensus event truth，再让实际 USTRF candidate 生成 alert trace。只有三来源的五项必需闭环指标均可评且 worst-source 通过，才允许比较能自动导出同步 RGB-D/pose 的眼镜或深度硬件；新硬件仍必须独立通过 `>=100 pair / >=0.95 source-aligned / INTER_FRAME_STABLE`。
