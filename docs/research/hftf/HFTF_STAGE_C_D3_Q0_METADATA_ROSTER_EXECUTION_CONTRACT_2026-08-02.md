# HFTF Stage C D3-Q0 metadata roster 执行合同

## 结论

本合同只授权一次 metadata-only roster scan：在绑定的 SANPO-Synthetic official
train split 中，排除既有 78 个冻结 parents 与完整 D2 六源 cohort 后，按
`session_id` 升序锁定前 40 个 metadata-eligible slots。

它不读取 RGB、mask、depth bytes 或 pose 内容，不计算 support、future truth、
qualification、effect 或 student。成功只允许另冻 qualifier + effect skeleton
execution contract。

## 固定排除与选择

排除集合必须从已封存的 D2 metadata qualification 机械派生：

- 原 78 个 historical/Development/closed/reserved parents 全部继承；
- 增加 D2 已消费的 6 个 parents；
- 5 个类别与新增 D2 类别必须互斥，总数精确为 84；
- 不允许手工增删，也不允许按 D2 成败、fps、scene、motion 或 risk deficit 排序。

metadata eligibility 只检查 synthetic description、chest-left intrinsics、5/20 Hz
timebase、pose object receipt，以及 exact 13-frame RGB/mask/depth object receipts。
metadata-ineligible candidate 尚未成为 slot，可按固定字典序继续；一旦锁定为 slot，
后续 acquisition/authority failure 将消耗该 slot，不允许替换。

## 一次性执行门

正式 CLI 只有在 contract/planner/test/D2 helper 均 tracked、clean、hash-bound，
且 `HEAD == origin/master` 时才运行。canonical root 必须原先不存在；在首个网络请求
前必须 exclusive create、`flush + fsync` durable attempt。

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/plan_stage_c_d3_q0_metadata_roster.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_D3_Q0_METADATA_ROSTER_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3 `
  --output artifacts.local/evidence/hftf/stage-c-d3-q0-metadata-roster-20260802/roster.json
```

锁定 40 slots 后立即停止。执行失败或 eligible universe 不足均不得重跑、追加或替换；
attempt/failure 必须保留。该扫描不等于 40-slot truth-screening：此时所有 slot 的
media、pose content、support 与 truth 仍未打开。

## 权限边界

成功终态只能是 `D3_Q0_METADATA_ROSTER_40_SLOTS_LOCKED`，并只授权另冻完整
reference-and-support qualifier、sealed-truth firewall 与 outcome 前 effect skeleton。

D3 media/pose、support/truth、effect、RGB student、reserved official-test、研究主线、
默认 App、Android、生产与 safety 权限全部保持关闭。
