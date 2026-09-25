# Track A 放量 v1 失败记录 — 2026-09-26

**状态：STOP_GEOMETRY_GATE_FAIL（在失败已成定局后提前终止生成）。未运行任何读出，数据不用于正式结论。** 协议：[v1](CNH_TRACK_A_SCALE_PROTOCOL_20260926.md)，登记 `cnh-track-a-scale-20260926`。

| 项 | 实际 |
|---|---|
| 已完成单位 | 160 / 192 |
| INCOMPLETE（配置在 4 条轨迹 × 32 候选后仍不可行） | 2：unit 59（train，配置 22）、unit 115（calib，配置 9） |
| 单位级 G2 失败 | 3：unit 66、76（方向 2）、72（方向 0）的 HEAD/BODY 不一致率低于 20% |
| 终止时未完成 | 30 个单位（均为 calib/audit），在硬门槛失败已确定后停止以释放计算 |

v1 把“每单位 G2 通过”与“无 INCOMPLETE”列为硬门槛且不允许替补；12 单位试采全部通过不足以说明 192 单位时每单位都能通过，这是执行者的协议设计失误。按项目规则不事后放宽，改为新种子族的 [v2 协议](CNH_TRACK_A_SCALE_V2_PROTOCOL_20260926.md)。

第 150 个单位之后启动的进程读取了已加入默认关闭选项（快速余量、关闭 10 Hz、强制 5 Hz）的同名源文件；默认行为不变，但清单中的源哈希会不同。已完成的 160 个单位保留在 `artifacts.local/evidence/cnh-track-a-scale-20260926-v1/geometry/`，只作已披露的 Development 数据（计划用于“确认畅通距离”方法开发），不作正式检验。
