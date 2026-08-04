# scale_free_traversability_r0

状态：development / frozen operator / external replication evaluation

## 研究问题与版本

R0 tests whether one fixed scale-invariant, three-band relative-depth operator
executes stably. R1 and R2 evaluate that unchanged operator against public RGB-D
sensor truth. Authority is Development only and follows
`docs/RESEARCH_GOVERNANCE.md`.

## 稳定 Interface

- `core.py`: frozen score and five-frame causal decision mechanics;
- `evaluate_phone_consumed.py`: consumed fixed-phone mechanics diagnostic;
- `evaluate_bonn_rgbd_consumed.py`: Bonn RGB-D R1 evaluator;
- `evaluate_arkitscenes_rgbd_consumed.py`: ARKitScenes RGB-D R2 evaluator;
- `validate_bonn_rgbd_result.py`: evaluator-independent ledger/result validator.

All evaluators require explicit input and new output roots and fail if an output
root already exists. Checkpoint, protocol, roster, and/or source archive hashes
are bound before inference. Run focused tests with:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest test_core.py test_bonn_rgbd_evaluation.py
```

## 输出

Machine evidence is written only to a caller-selected new directory under
`artifacts.local/evidence/hftf/`. Durable protocol and result summaries live in
`docs/research/hftf/`.

## 安全边界

Outputs mean only relative image-band agreement. They do not mean clear, safe,
blocked, distance, collision risk, an alert, App integration, or production
fitness. Consumed public data must keep its disclosed Development role.

## 停止条件

Do not change the R0 checkpoint, ROI, bands, valid-coverage rule, percentiles,
causal window, margin, winner count, source cohort, truth reconstruction, or
gates after reading the corresponding round's outputs. A failed source-support
precondition remains `NOT_EVALUABLE`; a failed accuracy gate remains
`NOT_SUPPORTED`.

## 失败资产复用

Failed or not-evaluable ledgers remain valid source-characterization,
regression, validator, and counterexample fixtures. They may be reused with the
consumed role disclosed, but cannot be repackaged as unseen Confirmation.
