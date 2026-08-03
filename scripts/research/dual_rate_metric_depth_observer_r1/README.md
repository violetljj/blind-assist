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
