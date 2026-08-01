# information_ceiling_three_arm_d0

状态：`development / complete / valid / protocol-frozen / no-training`

## 研究问题与版本

实现 `INFORMATION_CEILING_THREE_ARM_D0`：在同一 90-frame / 3-parent-event SANPO
Development cohort 和同一 Kotlin `AssistDecisionKernel` 上比较 current YOLO、
source-region truth boxes 与 source-mask oracle。

Arm B 是从同一 source mask ancestry 派生的风险框，不是独立 detector-native GT。
Arm C 的 dense mask 先经现有 adapter 过滤并收敛为最多一个 segmentation-source
`Detection`；由于 B/C source policy 也不同，C 的增量最多支持
`CURRENT_MASK_ADAPTER_AND_SOURCE_POLICY_GAIN_SUPPORTED`，不能单独确认 bbox 几何上限。

协议：
`docs/research/dual-loop/INFORMATION_CEILING_THREE_ARM_D0_PROTOCOL_2026-08-01.md`。

## 稳定 Interface

设备执行使用现有稳定 Adapter：

```powershell
pwsh -NoProfile -File scripts/run_detector_ab_device_benchmark.ps1 `
  -DatasetKind BlindAssistEvalSet `
  -ComparisonMode InformationCeilingThreeArm `
  -DatasetRoot artifacts.local/evidence/datasets/blindassist-sanpo-v2-event-labeled-20260711 `
  -ImageLimit 90 -AppRunsPerImage 3 -RiskConfig current -SkipDefaultRegression
```

设备报告拉回后运行独立复算：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/validate_information_ceiling_three_arm_d0.py `
  --benchmark-json <device benchmark.json> `
  --manifest artifacts.local/evidence/datasets/blindassist-sanpo-v2-event-labeled-20260711/manifest.jsonl `
  --output-dir artifacts.local/evidence/information-ceiling-three-arm-d0/<run-id>
```

输入缺三臂、90 帧共同 membership、mask/region/event truth、current risk config 或同一
decision-kernel contract 时 fail closed。

## 输出

只写入 `artifacts.local/evidence/information-ceiling-three-arm-d0/<run-id>/`：

- `event_ledger.jsonl`
- `summary.json`
- `result.md`
- `validation.json`

冻结设备 outcome：

- runner root：`artifacts.local/evidence/detector-ab-device-benchmark/20260801-104938`
- validator root：`artifacts.local/evidence/information-ceiling-three-arm-d0/20260801-105134`
- terminal：`MIXED_DETECTOR_AND_REPRESENTATION_GAPS`
- tracked result：
  `docs/research/dual-loop/INFORMATION_CEILING_THREE_ARM_D0_RESULT_2026-08-01.md`

## 安全边界

两条 oracle arm 是真值信息上限，不是可部署模型。结果为 consumed Development
mechanism evidence；不训练模型、不改 App、不驱动提醒、不产生产品或安全结论。

## 停止条件

独立 validator 得到协议定义的一个有效终态后停止。不在同一 outcome 上改 taxonomy、
阈值、事件规则或 cohort。`NOT_EVALUABLE` 只允许修复 pre-metric schema/runner 问题并
生成新的 evidence version。

## 假设与规则质疑

本审计直接质疑“继续在 YOLO 输出后增加规则即可突破”的假设。falsifier 是 oracle boxes
或 oracle masks 在相同决策链上仍不产生 parent-event 增量。

## 失败资产复用

输出可作为 detector/mask 表征边界、事件策略诊断、回归 fixture 和后继 Development
选题依据；不得重包装为 fresh Confirmation。
