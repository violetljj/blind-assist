# TARO O1R R6 prospective factor → reducer integration protocol lock

状态：`FROZEN / SOURCE_ONLY / IMPLEMENTATION_AUTHORIZED / NO_SCIENTIFIC_OR_PRODUCT_CLAIM`

机器合同：[JSON](TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_PROTOCOL_LOCK_2026-08-11.json)

本锁把已经完成的 R6 prospective factor bundle 接入一个唯一的、确定性的 9-query interval reducer。输入只包括 sealed R6 bundle、绑定的 candidate depth、confidence、Apple intrinsics 和已经由 8 个 `ADAPTER_FIT` parent / 211 帧拟合并封存的不确定性模型；公开入口不接受 FARO、truth、outcome 或 task metric。

最终值沿用 R6 冻结的 `QUERY_CLEARANCE = R1_BASELINE`，support/boundary 只提供有效性与不确定性组成。每个 query 从 candidate 在 Apple pixel centers 的 source-only corridor 中确定 confidence/range lookup，再从冻结模型解析 scale/support/boundary Q95。任何 query frame、factor、support 或 uncertainty 缺失都保留 `UNKNOWN`。

interval reducer 是唯一可产生 `CLEAR_OBSERVED / OCCUPIED_OBSERVED / UNKNOWN` 的组件；R6 factor builder 仍无最终状态权限。结构或 hash 绑定错误整包终止，不生成结果；普通不可观测性按 query fail-closed 为 `UNKNOWN`。

本锁只授权实现和聚焦 mechanics 验证，不授权在 16 个 eval parent 上调阈值，也不构成最终效果、设备、部署、产品或安全结论。

唯一后继：`TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_IMPLEMENTATION_LOCK`
