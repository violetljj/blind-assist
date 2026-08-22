# P1-PA0 target candidate availability result

日期：2026-08-22（Asia/Hong_Kong）

终态：`P1_PA0_TOP1_COLLAPSE_SIGNAL_ON_FAILURE_COHORT`

证据角色：`POST_OUTCOME_SELECTED_CONSUMED_DEVELOPMENT_MECHANISM_DIAGNOSTIC_ONLY`

## 最窄答案

在 P1-AMRM0 first-poison autopsy 的 7 个 target-visible/zero-IoU 帧上，冻结的 YOLOE-26n-seg visual-prompt
provider 以当前帧和 frame-0 target exemplar 为唯一输入，输出按 provider score 排序、最多 10 个 candidate。所有
memory、reacquisition、identity selection 和 verifier 均被移除。

| 指标 | IoU >= 0.10 | IoU >= 0.30 | IoU >= 0.50 |
|---|---:|---:|---:|
| Recall@1 | 0/7 | 0/7 | 0/7 |
| Recall@3 | 0/7 | 0/7 | 0/7 |
| Recall@5 | 0/7 | 0/7 | 0/7 |
| Recall@10 | 2/7 | 0/7 | 0/7 |

两个 IoU >= 0.10 candidate 分别首次出现在 rank 9（wall clock，best IoU `0.2179`）和 rank 10（wine rack，
best IoU `0.1397`）。其余 5/7 在 bounded pool 内没有合格 candidate；background-only frame rate 为 `5/7`。

因此本 failure cohort 支持一个很弱但真实的 `top-1 collapse` 信号：A4 的单候选结构确实提前丢掉了两个低排名、
低空间质量的 target-overlapping proposal。但它不支持“扩大 K 已解决 proposal availability”：在更严格的
IoU >= 0.30 下仍为 `0/7`，且 pot、smart-home display 等中等尺寸目标仍完全 absent。

## 分层与代价

IoU >= 0.10 的 Recall@10：

- shortest side `<16 px`：`0/2`；
- `16-31 px`：`0/1`；
- `32-63 px`：`1/3`；
- `>=64 px`：`1/1`。

六帧达到 K=10 cap，一帧有 5 个 candidate；平均 candidate count `9.29`。端到端 provider latency median
`109.1 ms`、P95 `1555.7 ms`；P95 含首帧 cold-start mechanics，不是稳态设备性能结论。峰值 CUDA allocated/reserved
约 `360/608 MiB`。

## 失败层边界

YOLOE 公共接口没有暴露 pre-NMS/raw proposals，因此 5 个 absent case 只能标为
`GENERATION_OR_PROVIDER_POSTPROCESS_NOT_SEPARABLE`。不得把它们伪分成 detector 根本未激活、score filter 删除或
NMS 压制。当前结果只说明正确 candidate 没有进入冻结的 bounded K=10 pool。

ADT category、instance、visibility 和未来 bbox 全部留在 private evaluator。Provider 只读取 current frame、
frame-0 exemplar/bbox 和 public referent handoff；没有使用 GT category prompt。

## 决策

- AMRM1/2/3、memory、reacquisition、verifier、VLM、VIO/SLAM、geometry 和 App 继续冻结；
- contrastive verifier 仍为 `NOT_EVALUATED`，不能从两个低质量 candidate 推导可修复性；
- 单纯把 A4 从 top-1 改成 top-10 不成立，因为 A4 本身没有 ranked proposal surface；
- 下一项若另行启动，应改变 target-conditioned proposal generation，并在预测前保留 provider postprocessed/full-rank
  trace；tiny/partial target search 与 parent-to-child 只能作为独立预冻结 arm，不能从本结果直接宣布答案。

## Evidence

- `artifacts.local/evidence/p1_pa0_target_candidate_availability_v1/public_input.json`
  SHA-256 `845beedc98dd62500d5c5b72e7dc4385ce38e56f9f0d2844c41286edf695ddef`
- `private_eval_input.json` SHA-256 `c2056d60447419b45a6d4c380f84874f9d911bc6f66f34a62477cc492fdfe7b0`
- `yoloe_prediction.json` SHA-256 `75db2cb390c6e7d13ac54f4598a0a185dd506cfb2bb3ad41c9fbf2e8d270c947`
- `yoloe_evaluation.json` SHA-256 `6059f3bbc22f817467163c562e4d168522970bc10ab9f571a48d3539d49f819a`

默认 App：不变。
