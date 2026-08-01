# Model matrix runner

状态：`MODEL_MATRIX_R0_IMPLEMENTED / DEVELOPMENT_OFFLINE_ONLY / TRACE_SCHEMA_V1 / RESUME_PREFIX_SAFE / HISTORICAL_REPLAY_NO_RERUN`

## 稳定 Interface

从 `E:\linnan\linnan` 执行：

```powershell
python scripts/run_model_matrix.py --validate-only
python scripts/run_model_matrix.py --manifest scripts/research/model_matrix/matrix_manifest.json
```

`matrix_manifest.json` 是运行选择层。换模型、分辨率、数据或输出根时，优先改
manifest 的 job；模型资产和适配器入口只在 `model_registry.json` 中声明，不在
runner 代码内硬编码。`model_registry.json` 的 `legacy_trace_replay` 条目优先复用
已完成的 RISKSEG-R0 trace，避免把相同设备实验重跑一遍。

支持的适配器：

- `legacy_trace_replay`：把历史 JSONL trace 转为统一逐帧 schema；旧 trace 缺失的
  raw boxes/logits/mask 会明确标成 `partial` 或 `not_provided`。
- `truth_mask`：仅对显式 oracle job 开放 mask 路径，并在 trace 留下
  `truth_fields_read` 和 `evidence_role=oracle_reference`。
- `fixed_rule`、`fixture`：固定规则与 synthetic canary。
- `tflite`、`depth_anything_v2`：可选 host 依赖存在时运行；缺依赖只产生
  `NOT_EVALUABLE` receipt。
- `python_callable`：由 registry/manifest 指定未来薄适配器，接口为
  `factory(context) -> object.infer(frame_input)`。

## 输出

每个 job 写入 `artifacts.local/evidence/.../jobs/<job_id>/`：

- `trace.jsonl`：每个 source frame 一行，含 detections、segmentation logits、mask、
  depth、risk output、known/UNKNOWN、clearance、latency、model/config hash 和
  source/frame identity；大型 tensor 以 artifact 引用写出。
- `receipt.json`：job 级状态、资产清单、trace hash、错误行数和执行身份。
- 顶层 `run_receipt.json`、`progress.jsonl`、`resume_state.json`：矩阵汇总、外部
  可监控进度和可恢复状态。

输出状态区分 `COMPLETE`、`PARTIAL_ERROR`、`NOT_EVALUABLE` 与
`PRECHECK_ONLY`；缺失模型输入、旧实现没有逐帧合同或依赖不存在都不能伪装成空预测。

## 安全边界

- 这是 host/offline Development runner，不接 Android 默认 App、提醒、安全或生产
  promotion authority。
- 数据 registry 的 `truth_fields` 默认从普通 adapter 输入中剥离；只有 truth-mask
  adapter 显式读取 oracle 字段。
- `model_hash` 来自真实资产时标为 `asset`；固定规则/adapter 标为 `logical`；资产
  缺失时为 `missing`，不会生成伪造的 checkpoint hash。
- 既有 RISKSEG-R0 设备 trace 只按 replay 读取；runner 不改写原 evidence root。

## 停止条件

manifest、registry、dataset manifest、trace schema 或 runtime state 的 hash 发生漂移
时，runner fail closed；已有 trace 必须是预期 frame identity 的连续前缀，不能通过跳过、
重排或重复行“恢复”。需要改变输入合同时，创建新的 `run_id`/output root 或新的 job
identity。

临时数据、模型和批处理输出必须留在 `artifacts.local/`；本模块不自动下载依赖、不自动
启动设备实验、不把 `Development` 结果写成模型晋级结论。
