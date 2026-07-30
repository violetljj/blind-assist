# LITE R1 shape-guard design review

日期：2026-07-30（Asia/Hong_Kong）

```text
TERMINAL: DESIGN_REVIEW_PASS
DESIGN_LOCK_SHA256: a4418a298e534b84e63003c598b7df3a17db0b15b84d7c35cb3288685d6780be
SOURCE_AUDIT_SHA256: 38802bacd9ec08de95556986445fe59c2c8ea815745f0f8a89147ffd2ec704bd
R0_TERMINAL_PRESERVED: true
IMPLEMENTATION: NOT_CREATED
FORMAL_REPLAY: NOT_AUTHORIZED
TRUTH_JOIN: NOT_AUTHORIZED
OLD_F1B_DECISION: SEALED
```

独立复核确认 R1 是新的 burned-Development evidence version，不是 R0 重跑或
Confirmation。native decoded `(H,W)` guard 位于 history/epoch/delta-t 后且位于
任何 bbox、ROI、feature 或 LK 运算前；reason precedence 固定为：

```text
INSUFFICIENT_HISTORY
→ HISTORY_GAP
→ FRAME_SHAPE_CHANGE
→ arm-specific reasons
```

32 个 mismatch opportunities 必须产生 64 条双臂共同 abstention，完整 producer
仍为 26,028 行；current observation 替换该 target 的 previous，禁止跨 mismatch
保留任何更早状态。resize、letterbox、pad、crop、bridge、删行或 common-success
评价均被禁止。469 个 primary events 和全部 R0 科学门保持不变。

本 PASS 仅授权实现、fixtures、stable root adapter、no-truth pilot、实现 identity
及其独立评审。正式 producer 仍要求 guarded-host preflight 和单独 activation
review；producer 或 evaluator 的首次正式失败即关闭 R1，不调参、不重跑。
