# BlindAssist Assistive Geometry B0 truth reader 与 registration lock 结果

状态：`B0_TRUTH_READER_AND_REGISTRATION_LOCK_PASS / B1_TRAINING_NOT_AUTHORIZED`

## 结论

B0 truth reader 与 registration lock 通过。冻结实现已经能把 ARKitScenes 的毫米深度、
confidence、pinhole intrinsics 和 trajectory 统一到逐帧 upright metric frame，并派生
gravity-bound ground、Left/Center/Right body-swept clearance、占用和 UNKNOWN。

这个 PASS 只证明 TRAIN-only source geometry reader 可用于下一步训练协议设计；它不是
human safety truth，也不授权 B1 训练、DEVELOPMENT/CONFIRMATION 消费、HTP 部署、默认 App、
production 或 safety claim。

机器结果见
[result JSON](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TRUTH_READER_AND_REGISTRATION_LOCK_RESULT_2026-08-09.json)，
完整逐帧 receipt 位于
`artifacts.local/evidence/hftf/assistive-geometry-b0-truth-reader-validation-20260809/receipt.json`
（SHA-256 `32F680B4...04EA`）。

## 证据

| 检查 | 结果 |
|---|---:|
| TRAIN-only FARO/AppleDepth 对照 | 6 videos / 157 frames |
| 逐帧尺度、配准、朝向组合门 | 94.27% |
| AppleDepth/FARO frame-median absolute error | median 0.0190 m / p90 0.0290 m |
| 双源 ground-height absolute difference | 136 pairs / median 0.0164 m / p90 0.1978 m |
| 双源 clearance absolute difference | 382 pairs / median 0.0127 m / p90 0.0647 m |
| Occupied decision agreement | 95.48% / 1,151 pairs |
| 主 TRAIN ground opportunity | 71.04% / 480 sampled frames |
| 主 TRAIN all-band known | 70.83% / 480 sampled frames |
| 有 ground opportunity 的主 TRAIN 视频 | 16/16 |
| 最大 pose bracketing gap | 99.96 ms，门限 250 ms |
| UNKNOWN clearance leakage | 0 |

冻结的 16 项 gate 全部通过。所有阈值只用 TRAIN 诊断校准；没有打开
DEVELOPMENT/CONFIRMATION 内容或 outcome，也没有读取模型输出。

## 修复的残留冲突

物化前捕获并关闭了两个会污染结论的冲突：

1. upsampling ZIP 的四个模态复用同一帧 stem，旧通用扫描器会误报重复；现在按官方模态
   目录独立索引，并用 synthetic ZIP regression 固定行为；
2. 初始 9 个候选中 3 个实际属于冻结 DEVELOPMENT role。下载在媒体写入前 fail-closed，
   最终 authorization、protocol 和 materialization 只保留 6 个真正 TRAIN 视频。

同时修正 `ground_valid` 的语义：它现在只标记拟合平面支持点；无 ground plane 时保持全
false，不再把所有有效 depth 误标为 ground。

## 数据能力边界

本结果按“signal 能支持什么 claim”判定，而不是按文件数量判定。中位数尺度与 clearance
一致性很强，但 ground-height 和 clearance 的最大差异仍分别达到 0.754 m 与 2.205 m。
这些 tail disagreement 被保留为下一阶段 confidence/UNKNOWN 设计的负证据，不能因 B0 PASS
而抹掉，更不能直接解释为产品安全容差。

静态 `sky_direction` 与逐帧 pose 朝向一致率为 94.27%。9 个不一致帧仍由逐帧 gravity pose
决定 upright rotation；静态 session metadata 只作为独立交叉检查，不覆盖动态设备姿态。

## 冻结实现

- truth reader：`scripts/research/assistive_geometry/arkitscenes_truth_reader.py`
  （SHA-256 `F5018766...18C1`）；
- validator：`scripts/research/assistive_geometry/validate_b0_arkitscenes_truth_reader.py`
  （SHA-256 `2CD1EF0F...F079`）；
- validation protocol：
  [JSON](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TRUTH_READER_VALIDATION_PROTOCOL_2026-08-09.json)
  （SHA-256 `905A92AC...28D`）；
- upsampling materialization protocol：
  [JSON](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_ARKIT_UPSAMPLING_TRAIN_MATERIALIZATION_PROTOCOL_2026-08-09.json)
  （SHA-256 `9AAF96C0...B2CE`）。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_B1_CONFIDENCE_THRESHOLD_AND_TRAINING_PROTOCOL_LOCK`

下一步只冻结 B1 的 target tensor/schema、confidence/UNKNOWN threshold、near-field weighting、
clearance 与 asymmetric false-clear loss、训练/验证角色和停止条件。协议关闭前不启动 student
训练，也不读取 DEVELOPMENT/CONFIRMATION outcome。
