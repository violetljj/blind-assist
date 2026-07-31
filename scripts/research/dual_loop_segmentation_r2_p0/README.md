# dual_loop_segmentation_r2_p0

状态：closed-development-readiness

## 研究问题与版本

`DUAL_LOOP_SEGMENTATION_R2_P0` 只判断一个新 DDRNet pipeline identity 是否值得进入
未来 R2 formal。当前 evidence instance 为 P0；允许 claim 仅为
`READY_FOR_R2_FORMAL` 或 `R2_NOT_WORTH_BURNING_FRESH_HOLDOUT`。

## 稳定 Interface

入口均使用 `python -m scripts.research.dual_loop_segmentation_r2_p0.<module>`。
materializer 只接受冻结 source config，输出 SHA-closed canonical view；rehearsal evaluator
只能读取该 view。所有 identity、零行、未知 ID、路径越界和覆盖写入均 fail closed。

稳定入口：

- `generate_synthetic_canaries`
- `materialize_canonical_view`
- `validate_canonical_view`
- `run_rehearsal` / `validate_rehearsal`
- `benchmark_runtime_rows` / `validate_runtime_rows`
- `run_ddrnet_refinement`
- `build_candidate_gate_matrix`
- `audit_holdout_metadata`
- `build_readiness_lock`
- `validate_readiness_closeout`
- `build_artifact_inventory`

## 输出

仅写入 `artifacts.local/evidence/dual-loop-segmentation-r2-p0/`。

## 安全边界

仅使用 train/dev、consumed old blind、R1 consumed fresh 和 synthetic canary。不得选择、
下载或读取新 formal mask truth，不运行新 holdout candidate output，不接 Android、QNN、
device、risk/event、主动提醒或生产模型。

## 停止条件

一次冻结的 36 点 DDRNet 后处理搜索完成后停止。若没有候选以 readiness margin 通过全部门，
或 rehearsal/runtime validator 非 `VALID`，终态为
`R2_NOT_WORTH_BURNING_FRESH_HOLDOUT`。当前已按该终态关闭，R2 未授权。

## 假设与规则质疑

唯一 causal difference 是 component-level 面积、置信度、margin 与空间下界过滤。它预期
降低 false activation；falsifier 是 recall/component recall 或一致 session 下降，或总链路
成本不满足 margin。SegFormer-B0 保留为 runtime-failed comparator，不原样进入 R2。

## 失败资产复用

R1 consumed fresh 永久只作 regression/rehearsal/validator/canary；任何复制、重命名、映射
或新 manifest 都不能恢复 unseen 身份。失败输出可保留为诊断和回归，不升级证据。
