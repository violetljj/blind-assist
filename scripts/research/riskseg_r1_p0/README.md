# RISKSEG-R1 P0 soft dense adapter audit

This package is a benchmark-only audit. It does not modify the frozen R0
adapter, event chain, risk rules, or the default Android application.

The runner keeps the complete four-channel INT8 output through dequantization
and softmax, records a SHA-256 for every raw output tensor, and computes frozen
dense corridor statistics. It never creates a fake `Detection`, never replaces
uncertainty with `confidence=1`, and never truncates connected components with
`take(1)`.

The consumed 30-event/30-session cohort is used only as nested Development:
five deterministic bucket-stratified outer folds produce one out-of-fold result
per parent event. The decision seed selects adapter settings on each inner
split; the same frozen search space is evaluated independently for the other
seeds and the truth-mask reference. This is mechanism evidence, not fresh
confirmation or App promotion evidence.

Run tests:

```powershell
E:\codex-tools\venvs\riskseg-r0-py311\Scripts\python.exe -m unittest scripts.research.riskseg_r1_p0.test_core
```

Run and validate:

```powershell
E:\codex-tools\venvs\riskseg-r0-py311\Scripts\python.exe -m scripts.research.riskseg_r1_p0.run_audit `
  --contract docs/research/dual-loop/RISKSEG_R1_P0_SOFT_DENSE_ADAPTER_AUDIT_CONTRACT_2026-08-01.json `
  --output artifacts.local/evidence/riskseg-r1/p0-soft-dense-v1

E:\codex-tools\venvs\riskseg-r0-py311\Scripts\python.exe -m scripts.research.riskseg_r1_p0.validate_audit `
  --contract docs/research/dual-loop/RISKSEG_R1_P0_SOFT_DENSE_ADAPTER_AUDIT_CONTRACT_2026-08-01.json `
  --report artifacts.local/evidence/riskseg-r1/p0-soft-dense-v1/report.json `
  --output artifacts.local/evidence/riskseg-r1/p0-soft-dense-v1/validation.json
```


状态：`development`

## 稳定 Interface

公开入口、输入不变量和失败模式以本目录脚本帮助和专项协议为准；跨域调用不得依赖私有 Implementation。

## 输出

只写入 artifacts.local/ 下的明确证据目录；不写仓库根目录或正式 App 资产。

## 安全边界

本模块不产生默认 App、生产、安全或 unseen confirmation authority；结果按当前协议声明的 Development/diagnostic 角色使用。

## 停止条件

最小判别实验完成、输入权威缺失、预算耗尽或重复失败时停止当前 evidence version，并保持最小 failure scope。
