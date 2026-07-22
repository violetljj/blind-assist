# USTRF tracker/TTC 固定协议消融 R1 结果（2026-07-22）

状态：`STOP_T1_T3_NO_PERSON_DETECTIONS_AND_ID_TRUTH_UNAVAILABLE / BENCHMARK_ONLY / R3_EVALUATOR_NOT_RUN`

> 2026-07-22 后续诊断：本报告冻结并保留当时的 host 结果，但其 detector coverage 解释已被 [detector taxonomy coverage v1](USTRF_DETECTOR_TAXONOMY_COVERAGE_V1_RESULT_2026-07-22.md) 取代。旧脚本把 `[1,84,2100]` 的 channel 轴误读为 prediction；正确 host/SM-S9280 实际有 2639/2617 个 person proposal 帧。T1–T3 仍未自动重开，因为 PIL/Android Canvas exact parity 与目标实例 truth 尚未闭合。

## 结论

按 `tracker_ttc_ablation_v1` 预注册完成 T0 输入与基线后，冻结的 App YOLO11n FP16-320 模型在 host PIL 复现的 Android letterbox 几何上，对 15 个动态行人正事件窗口和 15 个同源等长负窗口、共 4,594 帧产生 `0` 个阈值内 `person` 框。阈值前 person 最高分仅为 `.300942/.329009`，均低于冻结 `.35`。T0 host 基线因此为 event recall `0.0`、critical miss rate `1.0`、negative false alerts/min `0.0`；clearance、对象 ID switch 和 TTC proxy 均不可评。

T1–T3 没有运行。alpha-beta、ByteTrack-style 或 OC-SORT-style 关联只能关联 detector 已输出的目标，不能恢复从未出现的 person observation；继续运行会得到机械相同结果，违反“只改善轨迹观感而无事件收益即停止”的冻结门。本结果关闭当前 H1 实验，不改变 detector、confidence/NMS、route、事件阈值或 App。

## 冻结输入

- 预注册：`configs/ustrf_tracker_ttc_ablation_v1.json`，SHA-256 `7cc9878fb6f50f3ef51a75383c6a0d15509111a06b37226def203245866df7f6`。
- Detector：`app/src/main/assets/yolo11n_fp16_320.tflite`，SHA-256 `00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2`；输入 `320×320 float32`，confidence `.35`，class-wise NMS IoU `.45`。
- 来源：R3 已准入的 `lilocbench_dynamics_0_front` 与 `lilocbench_lt_changes_dynamics_0_front`；R3 source count 仍为 `2/3`。
- Host 预处理：PIL 黑底等比缩放，复现 Android letterbox 尺寸、scale 与 padding 几何；没有逐像素 Android Canvas parity 权限，因此本结果不能外推为真机 detector 测量。
- Truth partition：完整序列 consensus 中 3 + 12 个动态行人事件；每个正窗口加前后 30 帧，并确定性生成一个不与任何正事件重叠的同源等长负窗口。窗口账本 SHA-256 `383c1a8aabfbe3bb2d5ccbf53c67a091d012b06c31a02fd44b93cea6ce1445b5`。
- Detector ledger 在推理时看不到 event truth、track truth、candidate alerts 或其他臂输出；route/event truth 只由独立评价阶段读取。

## Detector ledger

| 来源 | 选中帧 | 任意类别有输出的帧 | 任意类别框 | person 帧/框 | 阈值前 person 最高分 | 阈值前任意类别最高分 | ledger SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| `dynamics_0` | 558 | 551 | 551 | `0 / 0` | `.300942` | `.986740` | `e9273cdd3739462ef553f8cbc94c78e5a1d3b984fb276d99104e8be4d141a253` |
| `lt_changes_dynamics_0` | 4036 | 3998 | 3998 | `0 / 0` | `.329009` | `.986985` | `06900447307db7eb85d4444ddcc50137328ff262b281b0b991c5a1f7f3ba5c45` |

任意类别输出覆盖 `4549/4594` 帧，说明 host TFLite 解释器、RGB 解码、letterbox、推理与 NMS 链路在工作；本轮失败点是冻结 host pipeline 的 person coverage，而不是空文件、模型未加载或 evaluator 把缺失写成零。由于没有 Android Canvas 逐像素 parity，本报告不声称真机也必然为零。

## T0 结果

| 来源 | 正/负窗口 | event recall | critical miss | negative false alerts/min | clearance | ID switch / TTC |
|---|---:|---:|---:|---:|---|---|
| `dynamics_0` | `3 / 3` | `0.0` | `1.0` | `0.0` | `not_evaluable` | `not_evaluable` |
| `lt_changes_dynamics_0` | `12 / 12` | `0.0` | `1.0` | `0.0` | `not_evaluable` | `not_evaluable` |

两个来源在 event recall 和 critical miss 上并列最差；JSON 为确定性 tie-break 显示 `dynamics_0`，不能解释为另一来源通过。结果收据：`artifacts.local/evidence/ustrf-tracker-ttc-ablation-v1/result-v2.json`，SHA-256 `f0892513051a51dcda3e7f0ed2a36ec45a3872c4a53b6c1569c41892f2e22f21`。

## 权限与停止决定

1. `T1/T2/T3 = not_run`，不是零分，也没有机会宣称优于 T0。
2. 没有对象级 ground-truth track IDs，identity switch/id precision/id recall 不得推断；没有 person track，TTC proxy 不得生成。
3. Bonn 没有进入本次正/负窗口，也没有补第三来源。
4. 本次不是 R3 evaluator；R3 五项/worst-source 仍受 `3/3` 前提约束且未运行。
5. 不降低 `.35` confidence、不改 NMS、类别映射或事件路线回救 H1。若继续研究 person coverage，必须作为新的 detector-taxonomy/coverage 假设独立预注册，不能事后改写本结果。
6. App、默认模型、U0、设备、训练、硬件选择和生产权限全部不变。
