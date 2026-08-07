# Dual-rate Metric Depth Observer R1

Frozen protocol:
[DUAL_RATE_METRIC_DEPTH_OBSERVER_R1_PROTOCOL_2026-08-03.md](../../../docs/research/hftf/DUAL_RATE_METRIC_DEPTH_OBSERVER_R1_PROTOCOL_2026-08-03.md)

Run the focused tests:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover `
  -s scripts/research/dual_rate_metric_depth_observer_r1 `
  -p "test_*.py" -v
```

Run the hash-bound consumed Development replay once to a new ignored output:

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/dual_rate_metric_depth_observer_r1/evaluate_r1.py `
  --output artifacts.local/evidence/hftf/dual-rate-metric-depth-observer-r1/result.json `
  --trace-output artifacts.local/evidence/hftf/dual-rate-metric-depth-observer-r1/trace.json
```

The evaluator refuses to overwrite either output. A/B/C are read from their frozen A0 reports.
D is a causal async replay with a fixed robust affine fit and source-age `UNKNOWN` policy. The
phone section is a scheduling/resource audit only because the measured phone DA asset is not the
same metric checkpoint used by the PC quality arm.

状态：`development`

## 稳定 Interface

公开入口、输入不变量和失败模式以本目录脚本帮助和专项协议为准；跨域调用不得依赖私有 Implementation。

## 输出

只写入 artifacts.local/ 下的明确证据目录；不写仓库根目录或正式 App 资产。

## 安全边界

本模块不产生默认 App、生产、安全或 unseen confirmation authority；结果按当前协议声明的 Development/diagnostic 角色使用。

## 停止条件

最小判别实验完成、输入权威缺失、预算耗尽或重复失败时停止当前 evidence version，并保持最小 failure scope。
