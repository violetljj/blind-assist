# AtomS3R Android NMS 低分配 R14 结果（2026-08-06）

## 结论

将 YOLO NMS 从 `removeAt(0)` 逐项搬移改为“置信度排序 + `BooleanArray` 抑制”，保持现有同类 IoU 规则和输出顺序。代码通过视觉/设备单元测试与 APK 编译，并完成 10 秒、1 分钟真机验证。

该改动降低了 NMS 的数组搬移和临时列表压力，但没有形成可确认的端到端延迟下降；保留作为低风险实现优化，不宣称为主要性能收益。

## 1 分钟验证

- 1631 帧，0 error，0 reconnect，0 gap，0 overwrite
- postprocess：P50 `2.18 ms`，P95 `6.12 ms`，P99 `6.96 ms`
- detector total：P50 `6.38 ms`，P95 `14.26 ms`，P99 `15.71 ms`
- capture→risk：P50 `67.45 ms`，P95 `83.92 ms`，P99 `103.62 ms`
- frame age at first byte：P50 `42.07 ms`，P95 `47.90 ms`，P99 `78.32 ms`

对照 R12 的 capture→risk `67.49/83.22/104.45 ms`，差异在测量噪声范围内。当前主要瓶颈仍是设备相机/JPEG/网络前段，而不是 NMS。

## 边界

结果为 development-only 性能证据，不代表检测准确率、安全或物理反馈起点。设备端采集—发送解耦仍需具备 ESP32 编译链后单独制作固件实验。
