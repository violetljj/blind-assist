# DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1

状态：`MODEL_SELECTION_NOT_EVALUABLE`

## 结论

路线 A 的模型训练、统一转换和开发集比较已经完成，但本轮不能产生正式模型选择结论。fresh formal 执行在读取第一行 source-native mask 时触发输入契约错误：冻结的 fresh manifest 直接绑定 SANPO 原生 mask，而 R1 evaluator 要求 canonical `0..3` 四类 mask。

这不是候选模型的通过或淘汰证据。正式 holdout truth 已在失败尝试中被访问，因此不能在同一 R1 身份下补做固定映射后重新宣称 fresh formal 结果；失败执行保持不可变，后续若继续必须另立 protocol、重新冻结映射和一份未消费的 official-test holdout。

## 已完成且仍有效的 Development 证据

- DDRNet-23-Slim 与 SegFormer-B0 均完成相同 400/200 train/dev、三 seed、1200 optimizer-step 训练。
- 两者均完成统一四类 full-int8 TFLite 契约；SegFormer 的 native TF SavedModel 与 PyTorch 最大绝对误差约 `1.7e-6`，argmax 一致率为 `1.0`。
- canonical dev 上共享 YOLO trace 的比较：
  - DDRNet：`C-A delta recall=0.248641`、candidate component recall `0.776559`、false components/frame `7.885`。
  - SegFormer：`C-A delta recall=0.198221`、candidate component recall `0.625086`、false components/frame `3.645`。
- 独立 host runtime benchmark（不含 truth 和 file I/O）：DDRNet 总增量 P95 `20.504 ms`；SegFormer `74.139 ms`。这些是 Development 结果，不是 device/QNN 结果。

开发结果只说明两候选都能补到 YOLO 漏检像素，但当前固定候选算子仍有误组件/误报和成本问题；不能据此授权提醒或生产模型升级。

## 失败边界与下一步

canonical 数据构建器使用仓库内固定的 `SANPO_MAP` 将 SANPO 原生类映射为项目四类。R1 fresh freezer 保存了原生 source mask，却没有在正式协议中冻结一个独立的映射视图，导致 formal evaluator 将原生类值误判为非法 canonical 类值。

因此：

1. R1 不选择 DDRNet 或 SegFormer；
2. 不重跑当前 fresh holdout，不用当前失败执行产生 gate 结论；
3. 不同时推进路线 B，也不接入 Android、QNN、risk/event 或主动提醒；
4. 若要继续，先另立 R2：显式冻结完整 SANPO 映射、生成映射后 mask 的哈希闭合视图、重新选择未消费 official-test holdout，再重复一次完整 formal/independent-validation 链路。

相关身份收据保存在 `artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/`，包括 formal freeze receipt、两候选训练/转换/runtime receipts、共享 dev/fresh YOLO traces 以及失败前未写入正式报告的状态。
