# dual_loop_segmentation_model_selection

状态：archive

## 研究问题与版本

历史 R1 比较 DDRNet-23-Slim 与 SegFormer-B0 的 source-native
pixel/component utility 和 host runtime。R1 已固定为
`SEGMENTATION_MODEL_SELECTION_R1_BLOCKED / MODEL_SELECTION_NOT_EVALUABLE`。

## 稳定 Interface

本 Module 仅保留历史实现、失败复核和 regression fixture。不得调用任何 runner 修复或
重跑 R1；当前状态与后继权限只以 `docs/research/dual-loop/README.md` 为准。

## 输出

历史输出位于
`artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/`，不得覆盖。

## 安全边界

四个 R1 fresh session 已消费，永久只能用于 regression/rehearsal/validator。不得恢复
fresh/unseen 身份，不得启动 Android、QNN、device、risk/event 或主动提醒。

## 停止条件

R1 已关闭且禁止修复、重跑和结果升级。任何后继只能使用新的协议、候选身份与权限。

## 假设与规则质疑

R1 的 decoder contract failure 使正式结果不可评价；该失败不否定语义分割双环问题，
但不能通过事后解释、改 decoder 或复用 consumed truth 恢复 R1。

## 失败资产复用

历史代码、数据与输出可作 negative evidence、diagnostic、regression、rehearsal、
validator 或 canary，不得重新包装为 unseen confirmation。
