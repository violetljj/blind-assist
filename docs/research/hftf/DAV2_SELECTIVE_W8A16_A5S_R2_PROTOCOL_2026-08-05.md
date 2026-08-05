# DA V2 选择性 W8A16 A5S R2 执行协议

日期：2026-08-05（P1 候选缓存前冻结）

QDQ ONNX 已通过完整 checker：48 个静态 Transformer 权重为 axis-1 symmetric INT8，48 个
DequantizeLinear，activation quantizer 为 0；最大权重重建误差 `0.0022231`。

唯一一次质量执行使用 frozen official `image2tensor(518)`，ONNX Runtime CPU 只负责生成
质量缓存，输出按 `align_corners=true` 对齐至 `480x640`。在缓存 SHA 锁定前不读取 P1 depth
或几何真值。随后只运行一次 P1 R1，必须 14/14 通过。

ORT latency 不代表 Android。质量通过后才能补充 QAIRT Linux converter 与 SM8650 cached
context；质量失败则 A5S 永久停止，不改 axis、scale、舍入或权重集合。
