# BlindAssist Assistive Geometry B1 dual-orientation implementation lock

终态：`B1_DUAL_ORIENTATION_TARGET_AND_MODEL_IMPLEMENTATION_LOCK_PASS`

本阶段关闭了双方向 TRAIN target、模型/head/loss 与完整 checkpoint 前向/反向的实现问题；
没有启动正式训练，也没有打开 DEVELOPMENT/CONFIRMATION outcome。

## TRAIN target 完整性

- 只消费冻结的 16 个 TRAIN video/visit，共 4,800 帧；
- portrait `2,724` 帧，landscape `2,076` 帧，与 Attempt 02 完全一致；
- 逐一验证 4,800 个 NPZ 的大小、SHA-256、字段、dtype、shape、方向、K、gravity、ground、
  clearance、occupancy 与 UNKNOWN mask；
- `3,424` 帧存在 ground plane，已知 clearance band `6,991` 个，已知 occupancy cell
  `21,060` 个，可形成 confidence truth 的 band `6,990` 个；
- 未知 clearance 没有被填成 clear，prediction-dependent confidence target 没有提前物化；
- 未读取 Development/Confirmation 内容或任何模型输出。

K 复核曾在第二帧触发 bit-exact 失败。差值为 `3.0517578e-05`，恰好是该数值处一个
FP32 ULP：producer 由原始高精度 K 缩放后转 FP32，validator 则由已保存的 FP32 source K
重算。最终冻结的门是“有限且不超过一次 FP32 舍入”，其余语义门没有放宽。

## 模型与训练算子

实现复用 DepthART-S metric encoder/depth decoder 的 stride-4、48-channel shared feature，新增：

- dense Ground head；
- Left/Center/Right clearance head；
- 三 band × `1.0/1.5/2.0 m` body-swept occupancy head；
- band confidence head，并按冻结 interface 扩展到三 horizon；
- A0–A4 additive losses、近场权重、censored-clear/UNKNOWN mask、K/左右 band flip 与 AdamW
  exact-resume 合同。

首次完整 checkpoint smoke 的数值和梯度虽有限，但部署包外层 `torch.library` operator 触发
“未注册 Autograd key”警告，因此保留为
`HOLD_MISSING_AUTOGRAD_KEY_REGISTRATION_WARNING`，不作为 PASS。

Attempt 02 仅对训练路径绕过该外层 dispatcher，直接进入包内显式
`_SelectiveScanAutograd`。注册路径与 eager 路径的 CUDA forward bit-exact，`max_abs=0`；
portrait `1×3×608×448` 与 landscape `1×3×448×608` 的完整模型 loss/gradient 全 finite，
每个方向均有 616 个 encoder/depth 参数和 12 个 assistive-head 参数取得非零梯度，且不再出现
缺失 Autograd 注册警告。

## 权限边界

这只证明实现可进入下一份训练执行合同。它不证明模型质量，不授权读取 Development 或
Confirmation outcome，不运行 DA3/Metric3D teacher，不改变 HTP/default App，也没有产品或
safety authority。

唯一 successor：

`BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEPTH_ONLY_THREE_SEED_TRAIN_EXECUTION_LOCK`

在该 successor 冻结 optimizer/scheduler、数据加载、checkpoint/resume、三 seed 与 receipt
之前，正式 student training 仍未授权。
