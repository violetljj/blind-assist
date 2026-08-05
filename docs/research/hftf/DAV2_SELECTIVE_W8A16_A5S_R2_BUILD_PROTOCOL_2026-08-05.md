# DA V2 选择性 W8A16 A5S R2 build freeze

日期：2026-08-05

Windows QAIRT 在 DLC 序列化时原生崩溃且没有产物。R2 不改变模型方案，只把相同的 48 个
axis-1 symmetric INT8 静态权重写入标准 ONNX，并在各原始 MatMul 前插入 DequantizeLinear。
不添加 activation quantizer，不搜索 axis、scale、层集合或 bitwidth。

QDQ 模型生成后先锁哈希，再冻结执行协议并物化 P1 RGB 输出。只有 P1 R1 14/14 通过，才
值得为 QAIRT Linux host 或真机 cached context 补齐部署环境。
