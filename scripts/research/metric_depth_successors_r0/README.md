# Metric-depth successors R0

This directory implements the two successors frozen after the asynchronous affine R1:

- dense Metric3D residual propagation with bidirectional RAFT consistency and DA new-region fill;
- offline Metric3D teacher distillation into a 770-parameter DA layer-11 CLS calibration head.

Both use the hash-bound consumed TUM cache and remain Development-only. Run focused tests with:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover `
  -s scripts/research/metric_depth_successors_r0 -p "test_*.py" -v
```

The materializer and evaluators refuse to overwrite outputs. Exact commands and evidence paths are
recorded in the result documents after execution.
