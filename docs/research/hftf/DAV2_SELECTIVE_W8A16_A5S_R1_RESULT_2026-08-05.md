# DA V2 选择性 W8A16 A5S R1 转换结果

日期：2026-08-05

终点：`A5S_R1_WINDOWS_QAIRT_QUANTIZE_SERIALIZATION_NOT_SUPPORTED_NO_DLC_CREATED`

generic HTP converter 读取 ONNX，处理了 96 条内部 encoding，并明确把所有未编码 tensor
设为 FP16；随后 Windows 原生 `ir_to_dlc.quantize_cpp_graph` 在序列化时 access violation。
没有生成 DLC、模型输出或 P1 缓存。

本机没有 WSL/Linux QAIRT host，不能把 Windows crash 当作模型精度失败，也不能安装系统级
运行时来绕过授权边界。R2 保持完全相同的 48 个 W8/FP16 activation 合同，先以标准 ONNX
per-axis DequantizeLinear 表示离线量化权重并运行 P1 R1；只有质量通过才值得继续解决 QAIRT
Linux/DLC 部署问题。
