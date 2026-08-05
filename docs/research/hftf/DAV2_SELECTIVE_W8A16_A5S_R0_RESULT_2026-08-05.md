# DA V2 选择性 W8A16 A5S R0 转换结果

日期：2026-08-05

终点：`A5S_R0_HOST_SOC_NAME_PREFLIGHT_INVALID_NO_DLC_CREATED`

QAIRT 2.47 Windows converter 在读取 ONNX 前拒绝 `--target_soc_model SM8650`，错误为
`SOC model SM8650 is not supported`。没有生成 DLC、量化日志、模型输出或 P1 缓存。

该 SDK 的既有成功路径是先生成 generic HTP DLC，再在 SM8650 真机生成 target cached context。
因此 R1 只删除 host converter 不支持的 SoC 参数；48 个 INT8 权重、所有 FP16 边界、哈希、
质量门和性能门不变。R0 保留为 preflight invalid，不重写。
