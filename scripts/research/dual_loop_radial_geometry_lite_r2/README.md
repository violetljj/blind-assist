# Dual-loop causal radial geometry LITE R2

状态：execution-envelope successor implementation；formal 尚未授权

## 稳定 Interface

仅通过 `scripts/run_dual_loop_radial_geometry_lite_r2.py` 的 `produce`、
`evaluate` 和 `validate-implementation` 子命令调用。

## 输出

开发和正式证据只写入
`artifacts.local/evidence/dual-loop/target-track-causal-radial-geometry-lite-r2/`。

## 安全边界

R2 producer 只读取冻结 replay 与 RGB，不读取 R1 output、truth、events、Vicon
或旧 F-1B decision。REveL 仍仅为 single-capture Development。

## 停止条件

任何 identity、guard、progress、13,014 / 26,028 / 32 / 64、atomic publication
或 pre-truth firewall 漂移均停止。失败不重跑，不扩展到 Confirmation、Android、
产品、运行时或安全结论。
