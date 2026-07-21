# USTRF Bangkok 替换与 R1.2c v2 结果（2026-07-22）

## 结论

Bangkok 已前向替换被仲裁排除的 Japan，R1.2c v2 六正例 truth—路线 oracle 达到 `6/6`，因此唯一预注册的 London FP16-768 GPU 候选获准执行。机械 canary 通过，但 SM-S9280 的完整 12 事件门只达到正例 `5/6`：Bangkok、Edmonton、Thailand、Bridge、Roadwork 命中，London 连续 `22` 帧仍未建立目标关联。事件门失败后按预注册顺序停止，未运行 600 秒 soak，R1.3、训练、App 默认模型和生产替换权限全部保持 false。

## 前向物化与 oracle

- 连续清单：`configs/ustrf_crosscam_continuous_events_r12c_seen_v2.json`，SHA-256 `35fd28f37353c9bcc3625febf752417739f296e6ba8b19803fc805088e9bd9c9`；12 个事件中只用 Bangkok 替换 Japan，其余事件逐字段继承。
- 协议：`configs/ustrf_crosscam_truth_geometry_r12c_prereg_v2.json`，SHA-256 `10f245b316fa86a55b5bae5d275939d668f7d34ba12a51287ad38ce961ba5e0a`。
- oracle 收据：`artifacts.local/evidence/ustrf-crosscam-codex/truth-geometry-r12c-seen-diagnostic-v2/truth_geometry_consistency.json`，SHA-256 `b21600fda31bd1376503fe794714b92fa2014e2f80b81ade2d6438524b91688c`。
- 结果：六个正事件均至少有一个 `.01/.02/.03` 三档 robust-inside alertable anchor；未决 truth—geometry conflict 为 `0`。该结果只授权唯一 768 候选，不直接授权连续门、soak 或 R1.3。

## 768 真机事件门

同一 YOLOE-11s 静态三类权重只把导出输入由 640 改为 768；TFLite SHA-256 为 `aa274c986ec6e360717b07efda06eb3ebe045cdd73c0ff71e1a1329bec1fc407`。机械 canary 在 GPU delegate 上通过，27 次测量 0 失败；London 的两个 canary 帧仍均未匹配目标，因此 canary 只证明 parser/backend 可运行。

随后把 12 段视频物化成逐帧 SHA-256 绑定的 PNG 输入。Bangkok 新增 25 帧，其余 11 段复用既有精确帧；输入收据 SHA-256 为 `de002d2b9a684cd6e929e24859c7fc59449c3418420b14477c33dff028be89f5`。SM-S9280 事件门结果如下：

| 指标 | 冻结门 | 结果 |
| --- | ---: | ---: |
| 正事件召回 | `6/6` | `5/6`，失败 |
| 负例假告警 | `0` | `0` |
| 重复交付 / 共现接管 / 身份切换 | `0 / 0 / 0` | `0 / 0 / 0` |
| 关联歧义帧率 | `<=10%` | `3.65%` |
| 首次正确提醒 | `<=5,000ms` | 已命中事件全部通过 |
| 出画清除 | `<=1,000ms` | London 未建轨迹，按 `10,000ms` fail-closed |
| 事件阶段性能（非 soak） | 仅报告 | inference p50/p95 `50/52ms`；total detect `110/119ms`；236 次、0 失败 |

设备收据位于 `artifacts.local/evidence/ustrf-crosscam-codex/mobile-r12c-seen-diagnostic-v2/android-arm/continuous-r12c-gpu768-event-gate.json`，SHA-256 `005f8316530a3171aa478b623559d7ea211608212377543c408fdc040c4dbcfd`。收据显式记录 `device_gate_evaluated=false`、`device_gate_passed=null`，不把未执行的 soak 伪记为通过。

## 成熟度推进与下一门

768 已证伪“仅提高同权重导出分辨率即可恢复 London”的假设，故分辨率搜索关闭。下一轮已在 `configs/ustrf_crosscam_small_target_detector_r12d_prereg_v1.json` 前瞻冻结为 stride-4/P2 小目标检测头假设；当前候选数为 `0`，必须先冻结唯一模型权重、训练 manifest 与审查/许可/精确几何收据，不能用阈值、tracker、更多同权重分辨率或合成/provisional 标签回救。

即便未来 R1.2d seen 门通过，也只允许考虑 R1.3 来源发现。正式成熟仍受两项更高层硬门约束：人类 route-conditioned event truth 与同设备米制 route/pose/depth 几何。当前公开视频/model review 只能作为 benchmark-only 代理证据。

## 验证

- cross-camera Python：`30 tests OK`；
- JDK 17：`:app:assembleDebug :device-benchmark:assembleDebug` 通过；
- SM-S9280：机械 canary 通过；12 事件 instrumentation 按预期写出收据后以 frozen event gate failure 结束；
- 未改正式 App 默认模型、反馈路径、R1.3 inventory 或生产授权。
