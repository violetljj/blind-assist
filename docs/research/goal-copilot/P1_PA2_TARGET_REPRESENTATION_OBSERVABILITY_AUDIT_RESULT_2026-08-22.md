# P1-PA2 target representation observability audit result

日期：2026-08-22（Asia/Hong_Kong）

终态：`P1_PA2_WEAK_CONTEXT_CONDITIONED_SIGNAL_ONE_OF_SEVEN_REPRESENTATION_MISMATCH_REMAINS_PRIMARY`

证据角色：`POST_OUTCOME_SELECTED_CONSUMED_DEVELOPMENT_ORACLE_MECHANISM_DIAGNOSTIC_ONLY`

## 最窄答案

在 PA0/PA1 完全相同的 7 个 target-visible first-poison frame 上，PA2 用 GT 把全图搜索难度拿掉，并在执行前固定
exact target crop、3x oracle ROI target-only、同一 ROI 的 2x exemplar-context 三臂。IoU >= 0.30 的完整
provider-postprocessed rank recall 分别为：

| arm | IoU >= 0.10 | IoU >= 0.30 | IoU >= 0.50 | proposals |
|---|---:|---:|---:|---:|
| exact target crop + target-only | 0/7 | 0/7 | 0/7 | 0 |
| 3x oracle ROI + target-only | 1/7 | 0/7 | 0/7 | 12 |
| 3x oracle ROI + target+context | 1/7 | 1/7 | 0/7 | 43 |

因此，预声明分叉机械上落入 `C_CONTEXT_CONDITIONED_PROPOSAL_SIGNAL_OBSERVED`，但信号只来自 wine rack 一例：
target-only ROI 的 best IoU 为 `0.1379`；context arm 在 rank 4 首次达到 IoU 0.30（`0.3346`），best IoU
`0.3769` 位于 rank 6。其余 6/7 在 context arm 的 IoU 0.30 仍为零；其中 pot 虽产生 4 个 proposal，但 best
IoU 仍为 0，其余五例没有 proposal。

这不是“context 已解决问题”。它说明 immediate context 可以改变 provider 的 proposal field，并在一个结构区域型
referent 上恢复弱 localization signal；同时，oracle 已知真目标位置仍不能让 target-only representation 在 7 例中
产生一个 IoU 0.30 proposal。主归因因此继续落在 `target representation / target-conditioned grounding mismatch`，
而不是全图 search、K 截断或固定尺度。parent-first 尚未获得充分授权。

## Test A / B / C 解释

- Test A：真目标 crop 直接作为 query 时 7/7 均为 0 proposal。YOLOE 公共接口不暴露 prompt embedding
  similarity，所以这只是不支持 operational recognizability，不能声称内部 embedding 数值为低。
- Test B：3x oracle ROI target-only 在主阈值仍为 0/7，排除了“只要 oracle 告诉大概位置即可定位”这一简单解释。
- Test C：target+context 相对 target-only 只救回 wine rack 的 IoU 0.30 candidate；这是 case-local interaction，
  不是 cohort-level context rescue，也不足以把 hierarchy/context 提升为唯一 successor。

## 决策

- 关闭 PA2，不在已打开结果上搜索 ROI scale、context scale、threshold、resolution、K、NMS 或 checkpoint；
- 不进入 adaptive search 或 parent-first；固定 tile 与 oracle ROI target-only 已分别给出负结果；
- 下一刀若另行授权，应直接审计或改变 target representation / prompt interface，并用 wine rack 作为 context-interaction
  positive diagnostic、其余六例作为 representation-negative counterexamples；不做无归因的 model-zoo sweep；
- AMRM、reacquisition、verifier、VLM、VIO/SLAM、geometry 与默认 App 继续冻结。

Claim ceiling：`FAILURE_COHORT_ORACLE_REPRESENTATION_OBSERVABILITY_ONLY_NO_MODEL_SELECTION_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM`。

## 执行与 Evidence

命令：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/goal_copilot_bridge/p1_proposal_availability/run_pa2_oracle_representation_audit.py --public artifacts.local/evidence/p1_pa0_target_candidate_availability_v1/public_input.json --private artifacts.local/evidence/p1_pa0_target_candidate_availability_v1/private_eval_input.json --model artifacts.local/models/yoloe-26n-seg.pt --output artifacts.local/evidence/p1_pa2_target_representation_observability_v1/audit.json --device cuda:0
```

- `audit.json` SHA-256 `785878afc411d4a0572b369ae42aa6cabbeb572e5ff9f2d10ece9af4573ae8ec`；
- public/private input SHA-256：`845beedc98dd62500d5c5b72e7dc4385ce38e56f9f0d2844c41286edf695ddef` /
  `c2056d60447419b45a6d4c380f84874f9d911bc6f66f34a62477cc492fdfe7b0`；
- implementation SHA-256 `0634da6af50fec9e5b2bd464be00b70c107c5cd579ef78796e39b841ea4e5ded`；
- model SHA-256 `1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b`；
- Ultralytics `8.4.52`，CUDA，peak allocated/reserved 约 `315.2/340.0 MiB`。

默认 App：不变。
