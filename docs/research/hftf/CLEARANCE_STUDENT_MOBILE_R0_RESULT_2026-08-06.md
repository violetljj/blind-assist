# Clearance-Student Mobile R0 开发结果

状态：`NOT_SUPPORTED / DEVELOPMENT_ONLY / NO_QNN_PROFILE / NO_PRODUCTION_AUTHORITY`

## 执行

- S0：MobileNetV3-Small fallback encoder + compact multi-head decoder，固定 `384x384`，
  单帧 FP16 autocast，参数 `1,271,434`。
- 数据：沿用 A2 teacher cache 绑定的 3000-frame parent/session-disjoint development
  stream；按 A4 R1 eligibility 在模型初始化前排除 36 个无 confidence-2 metric truth
  帧，实际 train/validation 为 `2374/590`。
- teacher：Canonical DA V2 518 offline，训练和 cache materialization 均保持
  `truth_inputs_opened=false` / `p1_truth_opened_during_materialization=false`。
- 训练：5 epochs，seed `20260806`，AdamW，batch `16`；第 5 epoch validation total
  loss `0.7347607403`。

## 120 帧 development gate

机器结果：[S0 gate receipt](../../../artifacts.local/evidence/hftf/clearance-student-mobile-r0/s0-development-gate-20260806.json)

| 指标 | Canonical C0 | S0 | 结论 |
|---|---:|---:|---|
| raw AbsRel median | 0.2943 | 0.3048 | 小幅变差 |
| scale-aligned AbsRel median | 0.0833 | 0.1190 | 失败 |
| ground recovery | 1.0000 | 0.9748 | 失败 |
| camera-height MAE | 0.1843 m | 0.3472 m | 失败 |
| clearance MAE | 0.3804 m | 0.8484 m | 失败 |
| collision agreement | 0.7527 | 0.4913 | 失败 |
| false-clear | 24.25% | 0.19% | 改善，但不能补偿其它失败 |
| temporal clearance delta | 0.1131 m | 0.1439 m | 失败 |
| geometry state agreement | 1.0000 | 0.0000 | 失败 |
| transition agreement | 1.0000 | 0.4655 | 失败 |

终态是 `MODEL_VARIANT_ENGINEERING_NONINFERIORITY_FAIL`。因此 S0 未达到“质量接近
Canonical + 真机明显更快”的双条件，不进入 QNN FP16 profile、W8A8 QAT、独立确认集
或任何 Android/默认 App 流程。false-clear 的改善只作为机制诊断，不能转成安全证据。

## 资产与后续

- checkpoint/cache/receipt 保存在 `artifacts.local/evidence/hftf/clearance-student-mobile-r0/`，
  仅作 development negative evidence/regression fixture。
- B0 MiDaS Small 256 仍是 roster 中的速度下界占位；本轮未下载/未运行，避免把外部
  benchmark 数字冒充本机同口径结果。
- 不允许在 consumed 120 帧上调 loss、阈值、seed、checkpoint 或重新标注来挽救 S0。
  下一轮若继续，必须版本化新数据/新合同或明确改变研究问题。
