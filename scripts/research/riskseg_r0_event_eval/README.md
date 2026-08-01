# riskseg_r0_event_eval

状态：development，`EVENT_EVAL_NOT_YET_FROZEN`

## 研究问题与版本

`RISKSEG_R0_EVENT_EVAL_V1` 在任何 PIDNet、YOLO 或 truth-mask oracle 输出打开前，
从 SANPO 原始 RGB 与 source masks 冻结一个 source-session-disjoint 的事件级评价集。
它只回答四类可通行性事件是否具备可评价真值，不把 source-mask 几何候选当成事件真值。

## 稳定 Interface

先从完整 draft 生成逐帧 RGB 盲审包：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.riskseg_r0_event_eval.prepare_review_bundle `
  --contract-ledger docs/research/dual-loop/RISKSEG_R0_DATA_ROLE_LEDGER_2026-08-01.json `
  --draft-root artifacts.local/evidence/datasets/example-a `
  --draft-root artifacts.local/evidence/datasets/example-b `
  --output artifacts.local/evidence/riskseg-r0/event-eval/review-bundle-v1
```

两路隔离 RGB review 与必要的第三路裁决完成后，冻结 cohort：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.riskseg_r0_event_eval.validate_and_freeze_cohort `
  --contract-ledger docs/research/dual-loop/RISKSEG_R0_DATA_ROLE_LEDGER_2026-08-01.json `
  --candidate-index artifacts.local/evidence/riskseg-r0/event-eval/review-bundle-v1/candidate_index.json `
  --review-a artifacts.local/evidence/riskseg-r0/event-eval/reviews/review-a.json `
  --review-b artifacts.local/evidence/riskseg-r0/event-eval/reviews/review-b.json `
  --adjudication artifacts.local/evidence/riskseg-r0/event-eval/reviews/adjudication.json `
  --output artifacts.local/evidence/riskseg-r0/event-eval/frozen-v1
```

已通过 output-blind mask 几何筛选、但尚未值得下载完整窗口的 session，可以先生成
稀疏 RGB 成本控制审阅包：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/acquire_sanpo_rgb_timeline_candidate.py `
  --session-id=<session> --start-frame 120 --target-fps 1 --frame-count 10 `
  --purpose event-eval-screening `
  --output-root artifacts.local/evidence/riskseg-r0/event-eval/sparse/<session>

E:\codex-tools\bin\blindassist-python.cmd `
  -m scripts.research.riskseg_r0_event_eval.prepare_sparse_review_bundle `
  --input-root artifacts.local/evidence/riskseg-r0/event-eval/sparse `
  --output-root artifacts.local/evidence/riskseg-r0/event-eval/sparse-review
```

稀疏包只决定“是否值得物化完整连续窗口”，不得产生 event truth、进入训练或查看
PIDNet/YOLO/oracle 输出。

当一轮双审证明正例配额不足时，`extend_review_bundle` 会保留原 candidate IDs、
逐项绑定和 contact-sheet hash，只追加新的完整窗口。两位原 reviewer 仍需各自隔离
审阅新增项，并把自己的旧 review 与新增项重绑定到扩展 index；不得把一方判断透露给
另一方。

若旧版 acquisition 已生成 v3 event-eval spec、但逐帧 manifest 仍残留
`model_assisted_candidate_screening_only`，使用
`repair_sparse_role_metadata` 做一次显式迁移，然后重新生成 review bundle 以刷新 hash。

## 输出

全部输出只写入显式的 `artifacts.local/`：

- `candidate_index.json` 与 `review_bundle_receipt.json`：source-session、连续窗口、
  RGB/source-mask hash 和全窗口（30--120 帧）contact sheet 绑定；
- `truth_ledger.jsonl`：冻结的 parent events、四桶身份以及正例
  alertable/passed intervals；
- `cohort_freeze_receipt.json`：两路 review、裁决、合同、实现与 truth ledger 的 SHA-256。

## 安全边界

- 候选选择和 RGB truth review 禁止访问 PIDNet、YOLO、oracle 或其派生输出。
- train、dev 与固定 90 帧回归集的 source sessions 必须取并集排除；camera head/chest
  仍属于同一 native session。
- source-mask 几何只负责成本控制和 shortlist。目录名、`center_obstacle`、
  `step_curb` 等 profile 都不是冻结 truth。
- 四桶固定为 `blocking_obstacle_positive`、`boundary_level_change_positive`、
  `parallel_curb_negative`、`normal_walkable_negative`。

## 停止条件

不足 30 个 parent events、任一桶未达 `8/8/7/7`、少于 8 个 source sessions、
任一桶少于 2 个 sessions、单 session 超过总事件 25% 或单桶 50%、正例缺少
alertable/passed interval、窗口重叠、review 绑定/隔离失败，统一终止为
`HOLD_EVENT_EVAL_DATA`。不得因此开始训练或改用像素 IoU 作为替代晋级证据。

## 假设与规则质疑

假设是：四类 risk/traversability mask 能在自然事件上优于 YOLO-only，而不是更擅长
物体命名。反证包括新增 false-alert event、passed clearance 下降、共同命中普遍变晚
或设备门失败。成本是一次盲态事件真值冻结和一个候选的预检/训练；不得用新增手工 gate、
FP sampler 或组件分类器补救。

## 失败资产复用

被拒绝窗口保留为候选筛选 counterexample；桶配额不足的完整 RGB/source-mask 素材可继续
作为 Development 数据盘点，但不能包装成独立 event-eval。冻结后 cohort 不得参与训练、
阈值选择、seed 选择或 adapter/rule 选择。
