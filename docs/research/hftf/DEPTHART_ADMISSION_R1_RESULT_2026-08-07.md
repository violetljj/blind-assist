# DEPTHART_ADMISSION_R1 结果

终态：`DIAGNOSTIC_COMPLETE / DEPLOYMENT_NOT_EVALUABLE`

研发结论：DepthART 保持 `preferred experimental backbone / research mainline`；DA2 保持 frozen baseline、teacher、regression reference 与 fallback。正式 admission 尚未评价，因为没有新的 session/parent-disjoint holdout，A3 也没有形成 ONNX/QNN/HTP 证据。R0 的 `FAIL` 永久不变。

| 工作包 | 结果 | 读法 |
| --- | --- | --- |
| A0 内参/预处理 | PASS | 官方 lower-bound resize 对 `K` 的缩放一致，无 crop/padding 平移项 |
| A1 metric false-block | 3.10%，33 decisions / 31 frames | 集中在 center/right、1.5–2.0 m horizon、1.5–3 m truth clearance，不像均匀噪声 |
| A2 relative truth-aligned | false-block 2.06%，false-clear 8.33%，MAE 0.1638 m，temporal 0.1219 m | 比 metric 的 false-block 好，但仍有边界残差；支持 Camera Adapter/metric 分支贡献部分问题，不支持“问题全在 adapter” |
| A3 ONNX/QNN | NOT_EVALUABLE | custom Selective Scan 导出链阻塞，且本机无 QAIRT/QNN 工具；HTP 支持未知 |

## 决策

这轮没有产生 `ADMIT` 或 `REJECT`。最强可签署结论是：

1. A0 没发现当前 TUM 链路的内参变换错误，不能把 3.10% 简单归因于 `K'`。
2. relative control 将 false-block 从 3.10% 降至 2.06%，说明 metric/Camera Adapter 有贡献；但 2.06% 未归零且 false-clear 上升到 8.33%，backbone/边界几何与下游 clearance 仍需共同检查。
3. 下一算法动作应是冻结一个不接触新 holdout 的开发调试包，针对右侧/1.5 m 边界做 raw-depth 与 clearance-stage attribution；不得在 R0 120 帧上搜索新 admission 阈值。
4. 下一部署动作是取得官方 Selective Scan CUDA extension 与 QAIRT/QNN SDK，先完成有效 ONNX graph、数值 parity、converter inventory，再谈真机 HTP。

建议性的 `false-clear <= 8% / false-block <= 2%` 仍只是新 holdout 形状，不得回套本轮已消费数据。

机器结果见 [R1 JSON](DEPTHART_ADMISSION_R1_RESULT_2026-08-07.json) 与 [A3 result](DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md)。
