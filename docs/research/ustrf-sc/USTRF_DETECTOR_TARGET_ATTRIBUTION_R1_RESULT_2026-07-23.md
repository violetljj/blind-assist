# detector target attribution R1 与 association-only H1 结果（2026-07-23）

状态：`G1B_SEMANTIC_PARITY_PASS / TARGET_BASELINE_HARD_GATE_PASS / STOP_DETECTOR_CHANGES / T0-T3_COMPLETED / SHADOW_GATE_FAIL / H2_CLOSED`

## 结论

冻结 App YOLO11n FP16-320 不再只有“画面中出现过 person”的 pooled 证据：在 detector 输出对真值审核隐藏后，15 个事件均冻结了事件级唯一 target person、alertable→clear 可见状态与 bbox；15 个负窗的 2,297 帧均冻结为逐帧 all-person boxes 或 confirmed-absent。随后解封同一 Android canonical baseline，两来源分别达到 target event coverage `3/3` 与 `12/12`，critical miss 均为 `0`，所以 detector 硬门通过并按协议停止换模型。

T0–T3 association-only 随即在同一 detection、route 与 event kernel 上完整重跑。四臂都只召回 `14/15` 个路线目标事件，并产生 `22` 次负窗提醒（`8.620/min`）；因此没有方案取得 Android shadow 入场资格。按停止门，production-isolated shadow 未执行，H1 尚未形成稳定 D0 基线，H2 的 D0/D1/D2 时序深度与 route-risk flip 继续关闭。

## G1b Android canonical parity

- Android Canvas 导出的 320×320 RGB canonical tensor：`4,594/4,594` 与 host 重建逐字节一致。
- person `.35` 状态、pre-NMS 数量、post-NMS 数量与 detection identity：均为 `4,594/4,594` 一致。
- 跨 ARM/Windows XNNPACK 的逐元素 raw fallback 仍为 `4,575/4,594`，旧 raw gate 保持 `fail`；本轮没有放宽 `1e-5/1e-4` 容差，也没有用 G1b 语义一致性覆盖旧门。
- 最大 matched detection confidence 差 `5.72e-6`，最大 bbox 坐标差约 `0.00103 px`。

## 冻结 target 与负例真值

- target truth：15 个 event-scoped unique target person；只计 alertable→clear 生命周期。
- 视觉确认离场的 clear 状态显式冻结为 `not_visible_cleared`，没有把共现者或空墙插值框接管为目标。
- 负窗：2,297 帧，其中 1,255 帧 confirmed absent；其余帧保存经双 annotation proposal、时序清洗与全帧 contact-sheet 复核后的 all-person boxes。
- 旧 first-fit negative 只提供窗口位置，不再充当 person-absent 或 detector FP 真值。
- frozen truth SHA-256：`564c3c1986176a53c42297e6114f1e5ea83f7fc21da12072ab94c8dff5240bb8`。

## baseline target attribution

| 来源 | target event coverage | critical miss | matched target frames | `<.35` score miss | localization miss | taxonomy confusion | 负例 detector FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamics_0 | 3/3 | 0/3 | 81 | 3 | 2 | 0 | 1 |
| lt_changes_dynamics_0 | 12/12 | 0/11 | 1,006 | 18 | 10 | 6 | 32 |

LT 的 6 帧 taxonomy confusion 为 `microwave×3 / refrigerator×2 / chair×1`。这些是帧级残差，不影响“每个事件至少一次正确命中目标”的硬门，但会继续保留为 worst-case 诊断。负例 FP 按 detector person box 对逐帧 all-person truth 的 `.30 IoU` 计算；共现行人命中不再误记为 FP。

硬门逐来源独立判定：两来源 coverage 均为 `1.0` 且 critical miss 均为 `0`。结论是 `STOP_DETECTOR_CHANGES_AND_REOPEN_T0_T3`；候选 roster 保持空，没有运行第二个 detector，也没有修改 `.35`、NMS、route 或事件门。

## T0–T3 association-only

| Arm | event recall | critical miss | false alerts/min | 首次正确提醒 P95 | clearance | clearance P95 | repeats | target fragmentation | evidence age P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 | 14/15 | 0 | 8.620 | 2,800 ms | 13/15 | 0 ms | 12 | 16 | 0 ms |
| T1 | 14/15 | 0 | 8.620 | 2,800 ms | 13/15 | 0 ms | 12 | 16 | 66.7 ms |
| T2 | 14/15 | 0 | 8.620 | 2,800 ms | 13/15 | 0 ms | 12 | 12 | 66.7 ms |
| T3 | 14/15 | 0 | 8.620 | 2,800 ms | 13/15 | 0 ms | 12 | 15 | 66.7 ms |

按预注册词典序，T2 仅因 target fragmentation 较低成为相对胜者；它不是合格方案。worst source 为 `lt_changes_dynamics_0`：event recall `11/12`、false alerts `7.582/min`、clearance `10/12`、repeat alerts `12`。`dynamics_0` 虽为 `3/3`，负窗误提醒仍达 `16.129/min`。唯一漏召回是非 critical 的 `pedestrian_route_intersection_006`：四臂均无 route-conditioned alert，说明 association 不能恢复固定 detection+route kernel 没有形成的连续风险证据。

路线 unknown 帧一律立即弃权，四臂均为 `0` 个 unknown-route active alert；但这一项通过不足以抵消 event recall、负例误提醒、repeat 与 clearance 的失败。

## 阶段决定

1. detector baseline 通过，停止 detector 候选与阈值调节。
2. T0–T3 已完成，不以 MOT/fragmentation 单项选择 T2 进入设备。
3. shadow 入场门失败，故不写 App、不运行 production-isolated Android shadow，也不声称已形成连续可用事件链。
4. H1 D0 未在两来源、负例、clearance 与 worst-source 上同时稳定，H2 继续关闭。
5. 若另开后续研究，应新建预注册，仅研究固定 detector 下的 route-target evidence/事件抑制；不得用本窗口回调 `.35`、NMS、route 或事件门。

本报告及全部 ledger 仅有 benchmark research authority，不提供训练、App、生产或真实辅助安全授权。
