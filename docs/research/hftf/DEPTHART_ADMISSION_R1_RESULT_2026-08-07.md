# DEPTHART_ADMISSION_R1 结果

终态：`DIAGNOSTIC_COMPLETE / G3-C_PRIMITIVE_REFERENCE_CONVERTIBLE / HTP_REFERENCE_SOURCE_READY / HEXAGON_SDK_AUTH_BLOCKED / HTP_NOT_EVALUATED`

研发结论：DepthART 保持 `preferred experimental backbone / research mainline`；DA2 保持 frozen baseline、teacher、regression reference 与 fallback。正式 admission 尚未评价，因为没有新的 session/parent-disjoint holdout，R1 不改变 R0 的 `FAIL`。

| 工作包 | 结果 | 读法 |
| --- | --- | --- |
| A0 内参/预处理 | PASS | 官方 lower-bound resize 对 `K` 的缩放一致，无 crop/padding 平移项 |
| A1 metric false-block | 3.10%，33 decisions / 31 frames | 集中在 center/right、1.5–2.0 m horizon、1.5–3 m truth clearance，不像均匀噪声 |
| A2 relative truth-aligned | false-block 2.06%，false-clear 8.33%，MAE 0.1638 m，temporal 0.1219 m | 比 metric 的 false-block 好，但仍有边界残差；支持 Camera Adapter/metric 分支贡献部分问题，不支持“问题全在 adapter” |
| A3 ONNX/QNN | G3-A PASS；G3-B PARTIAL PASS；G3-C primitive reference technically convertible；HTP source ready/toolchain blocked；G4 NOT_EVALUATED | exact unrolled graph parity/QAIRT 通过但 QNN IR 膨胀到 21,440 ops；scalar HTP reference kernel 源码已就绪，编译等待需 Qualcomm 登录的 v73 SDK/toolchain；没有 context 或设备执行证据 |

## 决策

这轮没有产生 `ADMIT` 或 `REJECT`。最强可签署结论是：

1. A0 没发现当前 TUM 链路的内参变换错误，不能把 3.10% 简单归因于 `K'`。
2. relative control 将 false-block 从 3.10% 降至 2.06%，说明 metric/Camera Adapter 有贡献；但 2.06% 未归零且 false-clear 上升到 8.33%，backbone/边界几何与下游 clearance 仍需共同检查。
3. 下一算法动作应是冻结一个不接触新 holdout 的开发调试包，针对右侧/1.5 m 边界做 raw-depth 与 clearance-stage attribution；不得在 R0 120 帧上搜索新 admission 阈值。
4. 下一部署动作是完成 canonical numerical parity；用户完成 QPM3 登录后安装 Hexagon SDK 5.5.5 + Tools 8.7.06，编译已落盘的 scalar HTP kernel 并以 exact primitive 图作算子 oracle。primitive 图和未编译源码均不授权 quantization、context、partition 性能或真机 latency。

建议性的 `false-clear <= 8% / false-block <= 2%` 仍只是新 holdout 形状，不得回套本轮已消费数据。

机器结果见 [R1 JSON](DEPTHART_ADMISSION_R1_RESULT_2026-08-07.json) 与 [A3 result](DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md)。
