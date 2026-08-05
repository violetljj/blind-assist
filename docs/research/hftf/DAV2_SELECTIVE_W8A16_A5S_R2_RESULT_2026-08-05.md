# DA V2 选择性 W8A16 A5S-R2 结果

日期：2026-08-05

终点：`MODEL_VARIANT_R1_ENGINEERING_NONINFERIORITY_FAIL`

A5S-R2 只将 12 个 Transformer block 中 48 个静态 `qkv/proj/fc1/fc2`
权重做 signed symmetric per-output-channel INT8；activation、动态 attention MatMul、
Softmax、LayerNorm、residual、patch embed、depth head 与图输入输出保持浮点语义。
标准 ONNX QDQ 模型经完整 checker 通过后，在不读取 P1 真值的物化阶段生成唯一 120 帧
候选缓存，再按冻结的 P1-R1 协议只评价一次。

## 判定

- 14 个 R1 门通过 13 个；唯一失败项是 `temporal_clearance_delta`。
- 时序 clearance delta MAE 从基线 `0.11313 m` 上升到 `0.13511 m`，越过冻结非劣界。
- raw AbsRel 为 `29.03%`，scale-aligned AbsRel 为 `8.38%`，clearance MAE 为
  `0.38337 m`，ground recovery 为 `100%`。
- false-clear 从基线 `24.253%` 略降到 `24.061%`，false-block 为 `0.469%`；这些改善
  不能抵消已冻结的时序门失败。
- 候选产生 5 个 beneficial change、2 个 harmful change，净改善 3 个决定，但终态仍为失败。

主机 ORT QDQ 物化的 P95 为 `654.51 ms`，仅用于确认质量路径可执行，不代表 Android、
QNN 或 App 性能。按 P2 顺序规则，本路线停止，不进行 Android 转换、设备 profile 或以速度
挽救质量失败；Windows QAIRT 的既有序列化崩溃也不再需要为本候选解决。

证据绑定：QDQ ONNX `93E6A390...CCDA`，候选缓存 `6DEF8338...EF67`，P1-R1 结果
`6927A969...506E`。
