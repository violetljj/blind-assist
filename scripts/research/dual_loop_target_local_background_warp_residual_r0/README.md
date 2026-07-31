# TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0

状态：`DEVELOPMENT / OFFLINE_ONLY / TRUTH_BLIND_PRODUCER`

本 Module 实现已获授权的 B Development 离线 producer、truth-late evaluator、R1–R4
确定性选择和实现锁。它只对应
[设计合同](../../../docs/research/dual-loop/TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_DESIGN_CONTRACT.md)
的 B 阶段，允许主张上限为 `DEVELOPMENT_SIGNAL_DIAGNOSTIC_ONLY`。

## 稳定 Interface

从仓库根目录调用：

```powershell
E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts\run_dual_loop_target_local_background_warp_residual_r0.py produce `
  --input <input.jsonl> `
  --output <artifacts.local/.../producer.jsonl> `
  --receipt <artifacts.local/.../producer_receipt.json>

E:\codex-tools\tools\venvs\blindassist-venv-export312\Scripts\python.exe `
  scripts\run_dual_loop_target_local_background_warp_residual_r0.py evaluate `
  --producer-output <producer.jsonl> `
  --producer-receipt <producer_receipt.json> `
  --truth <truth-late.jsonl> `
  --output <artifacts.local/.../evaluation.json>
```

实现锁通过 `create-lock` 和 `validate-implementation` 子命令生成/校验；`test` 只运行
本 Module 的 synthetic fixtures。`prepare-burned input` 只冻结 burned REveL frame/track/image
身份；`prepare-burned truth` 是独立的 truth-late join，必须在 producer receipt 完成后再执行。

## 输入

producer input 是每行一个严格相邻 frame pair 的 truth-blind JSONL，至少包含：

```text
source_id, session_id, sequence_id, parent_event_id
previous_source_frame_id, current_source_frame_id
previous_frame_index, current_frame_index
previous_image, current_image
previous_image_sha256, current_image_sha256
previous_frame_shape, current_frame_shape
captured_at_ns_previous, captured_at_ns_current
target_id, track_epoch, previous_bbox, current_bbox
previous_dynamic_bboxes, current_dynamic_bboxes
```

dynamic box 只允许 `{"bbox": [x0,y0,x1,y1], "dynamic": true}`。producer 会在打开
图像前拒绝 truth、event label、pose/Vicon、oracle、旧 decision 和后验输出字段。
truth 只能由 evaluator 的独立 truth-late JSONL 读取；truth 文件至少绑定上述 pair
identity、`parent_event_id`、`truth_eligible=true` 和 canonical `truth_state`。

producer 对每个输入 pair 输出 R1–R4 四行；所有失败仍写入固定分母和唯一
`abstention_reason`。输出、receipt、evaluation 和 lock 只能写入显式的
`artifacts.local/` 路径，拒绝覆盖已有文件。

## 输出

producer JSONL、truth-late evaluation JSON、receipt 和 implementation lock 只写入
显式的 `artifacts.local/` 目录，并拒绝覆盖已有文件。

## 安全边界

本 Module 不读取既有 R1/D0 output，不接 Android、CameraX、QNN、shadow、active APK，
不修改默认行为。它不执行 C1/C2 metadata admission，也不产生 Confirmation、产品或
安全证据。truth、identity、contract hash、producer output hash 或 implementation lock
不一致时立即 fail closed；四个 ring 均无最低可评价 event 时终点为
`SIMILARITY_CANARY_NOT_SUPPORTED`，否则按合同输出 `NO_DEVELOPMENT_INCREMENT` 或
`B_DEVELOPMENT_SIGNAL_DIAGNOSTIC_ONLY`。

## 停止条件

任一 truth firewall、identity、shape、hash、schema、quality gate 或 output namespace
失败均停止当前 B evidence version；不缩小分母、不换 ring/model、不重跑既有失败结果，
也不自动进入 C1/C2。

## 失败资产复用

synthetic fixtures、弃权 ledger、truth firewall 和 shape/identity 反例只可作为
`DIAGNOSTIC`、`REGRESSION_FIXTURE`、`COUNTEREXAMPLE` 或 `QUALITY_GATE`；B 结果不能
包装为 C1/C2 unseen Confirmation，也不能授权 runtime 或产品行为。
