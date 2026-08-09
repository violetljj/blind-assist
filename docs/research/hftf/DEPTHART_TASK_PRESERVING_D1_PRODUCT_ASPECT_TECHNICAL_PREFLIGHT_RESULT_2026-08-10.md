# DepthART task-preserving D1 product-aspect technical preflight result

状态：`HOST_PREFLIGHT_PASS / SINGLE_CANDIDATE_LOCKED / DEVICE_GATE_PENDING / NO_OUTCOME`

产品 portrait full-FOV `1×3×608×448` 图已由冻结 checkpoint 重建。PyTorch 输出为
finite `1×608×448`，camera prompt 外置与原路径 bit-exact（`max_abs=0`）；ONNX
经过 Einsum、graph hygiene、fixed-shape 与 23 个 float32 LayerNorm custom mapping 后，
锁为 SHA-256 `5A65CFFB...7086`。QAIRT `2.47.0.260601114230` host converter
成功写出 31,998,468-byte DLC（SHA-256 `755138FA...D42`），DLC-info 复核 image、
四级 prompt 与 depth 输出 shape，保留 5 个 SelectiveScan 和 23 个 DepthArtLayerNorm。

同名机器结果已把唯一 candidate ONNX/DLC、canonical reference checkpoint、冻结
Development roster 与 task postprocess SHA 一并锁定，锁后禁止修改 operator recipe。
该结果只证明 synthetic shape、图结构和 host conversion；DLC 的 9.601G MACs 与
891.1 MiB memory estimate 仅为 diagnostic。

执行时 `adb devices -l` 无设备。因此没有生成 SM8650/v75 context，没有运行 HTP
full-graph parity，也没有激活或读取 D1 task outcome。唯一下一门是连接冻结的
`SM-S9280 / SM8650 / HTP v75`，对 exact DLC 建 context 并完成无 outcome activation
preflight；设备门通过前不得开始 Development quality screen，更不得访问 R2 cohort。
