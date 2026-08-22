# P1-PA2 target representation observability audit protocol

日期：2026-08-22（Asia/Hong_Kong）

模式：`REVERSIBLE_EXPLORATION / CONSUMED_DEVELOPMENT_ORACLE_DIAGNOSTIC`

## 问题与固定设计

只在 PA0/PA1 已消费的 7 个 target-visible first-poison frame 上回答：拿掉全图搜索难度后，冻结的
YOLOE-26n-seg visual-prompt representation 是否仍有足够的 target-conditioned proposal signal。

执行前固定三个 arm，不在结果上调参：

1. `exact_target_target_only`：GT target 的整数外接 crop，原 frame-0 target bbox 作为 visual prompt；
2. `oracle_roi_target_only`：以 GT target 为中心、宽高各 3x、图像边界裁切的 ROI，同一 target-only prompt；
3. `oracle_roi_target_plus_context`：完全相同的 3x ROI，只把 exemplar prompt bbox 固定扩为 2x immediate context。

三臂共享冻结 checkpoint、`imgsz=640`、`conf=0.001`、`max_det=100`。主观察为完整 provider-postprocessed rank
上的 `IoU >= 0.30` recall，同时报告 IoU 0.10/0.50、Recall@10、first-correct rank、best IoU、proposal count 和
latency。YOLOE 公共接口不暴露 prompt embedding similarity 或 pre-NMS proposal，因此 Test A 使用“直接看到真目标
crop 后能否定位”的 operational recognizability，不伪造内部 similarity。

## 分叉与停止条件

- exact/ROI/context 在 IoU 0.30 都为零：`A_CURRENT_VISUAL_PROMPT_REPRESENTATION_SIGNAL_NOT_OBSERVED`；
- target-only ROI 非零且 context 不更高：`B_TARGET_SIGNAL_EXISTS_SEARCH_OR_LOCALIZATION_REMAINS_PLAUSIBLE`；
- context ROI 高于 target-only ROI：`C_CONTEXT_CONDITIONED_PROPOSAL_SIGNAL_OBSERVED`；
- 其余结果：`MIXED_OR_INSUFFICIENT_REPRESENTATION_OBSERVABILITY_SIGNAL`。

执行一次后停止；不搜索 ROI scale、context scale、threshold、resolution、K、NMS 或 checkpoint，也不自动启动
parent-first 或新模型。

## 权限与 claim ceiling

GT bbox 明确进入 query crop construction 与 evaluator，所以结果不得进入正式 provider input、Confirmation、模型选择、
泛化、产品、安全或默认 App claim。AMRM、memory、reacquisition、verifier、VLM、VIO/SLAM、geometry 与 App 均不参与。

Claim ceiling：`FAILURE_COHORT_ORACLE_REPRESENTATION_OBSERVABILITY_ONLY_NO_MODEL_SELECTION_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM`。
