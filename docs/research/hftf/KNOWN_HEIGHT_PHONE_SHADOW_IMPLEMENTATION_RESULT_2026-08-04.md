# Known-height phone shadow implementation result

日期：2026-08-04

结论：`IMPLEMENTATION_READY / DEVICE_CAPABILITY_SUPPORTED / PHYSICAL_P0_NOT_EVALUATED`

## 已完成

- 最终 student 已封存为 `CAMERA_CONDITIONED_SCALE_STUDENT_R0_FINAL_5P`，没有 fit/update
  入口；封存发生在已消费 external confirmation 之后，因此证据上限仍是 consumed synthetic
  mechanism candidate。
- Python 与 Kotlin 使用同一十维特征顺序、mean/std、intercept/weights 和 `[0.25, 4.0]`
  fail-closed 范围。模型收据 SHA-256 为
  `5417A24EFCAD59713F35F0AA13B7C56A182D72C9670B5B12D418DC06C9793BDD`，golden 收据为
  `A473D429BB6971E86B717936E96182C997BED58576E787E61517615A12A126EC`。
- P0/R2 协议已在手机 outcome 前冻结；host preflight 会核验真实量高、量高不确定度、
  mount/camera/intrinsics 身份、逐帧 SHA、严格递增 capture timestamp 和独立参考清单。
- 隔离 Android benchmark 包新增 120 帧 RGBA->PNG 采集入口。参数不全时不打开相机，写
  `HOLD_MISSING_PHYSICAL_RECEIPTS`；采集不会触发 App 提示或状态。

## 当前真机证据

设备：Samsung SM-S9280，Android 16，ADB serial `R5CX10M8Y8X`。

- 既有 CameraX capability audit：4/4 通过；640x480 RGBA、30 帧 timestamp 单调；
  rotation-vector bracket coverage 1.0，最大 bracket 19.756 ms；camera 0 的观测内参
  `(fx, fy, cx, cy)=(2766.1165, 2771.1763, 2041.3307, 1530.0737)`。
- 封存 head device canary：3/3 golden vectors 通过；100,000 次调用的 head-only mean
  latency 为 1.27171615 us。
- 不带物理参数运行 P0 采集入口：测试通过并按合同产生
  `HOLD_MISSING_PHYSICAL_RECEIPTS`，缺少 session、phase、serial、量高/不确定度、mount 和
  independent reference receipt。

这些结果只支持采集/计算能力。既有 CameraX audit 明确没有验证真实重投影精度、App
runtime 或 production；本轮也没有 3 个现场固定支架 session，不能计算 P0/R2 效果。

## 未伪造的终态

- P0：`READY_FOR_PHYSICAL_CAPTURE / NOT_EVALUATED_MISSING_MEASURED_HEIGHT_AND_REFERENCE`
- R2：`BLOCKED_BY_P0 / NOT_STARTED`
- 完整设备 latency：`NOT_EVALUABLE_NO_EQUIVALENT_ANDROID_EXPORT`

精确 depth checkpoint 是 PyTorch Depth Anything V2 Metric Hypersim ViT-S，SHA-256
`B782898D8A3E8BE1F639DE33837ED85E9B4B73E40F8F5E5CD99067588D722545`。仓库当前没有经
三臂 golden parity 证明等价的 TFLite/ONNX，因此不能用另一模型的 latency 替代。

## 下一次唯一有效动作

用固定支架现场测相机光心高度和不确定度，准备独立卷尺/激光参考清单，再按冻结参数采
3 个 P0 session。3/3 经 preflight 入场后才运行 host 三臂；只有 P0 可评价才采 8 个 R2
session。任何缺失 receipt 均保持 HOLD，不估填。
