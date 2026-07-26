# Real-data geometry canary R0

状态：`canary / implementation-review`

## 研究问题与版本

`RCLE-PHASE-B-REAL-DATA-GEOMETRY-CANARY-R0` 只检验 burned TUM
`fr2/rpy` 四窗上的 pair identity、统一 schema、弃权语义与 float64 parity。
允许声明只有 `IMPLEMENTATION_DEBUGGED` 或 `NOT_EVALUABLE`。

## 稳定 Interface

`producer.py` 只读取绑定合同与 TUM archive，可导入冻结的 PB-H1 geometry
primitive；`validator.py` 独立复算 association、pose、depth、geometry 和 window
summary，不导入 producer、PB-H1 或旧 TUM audit。根 runner 在 activation lock
缺失时 fail closed。正式 activation 后、任何 archive decode 前，runner 必须以
`O_EXCL` 原子创建不可覆盖的外置 claim；成功、异常或中断都永久消费 R0 attempt，
异常另保留 failure receipt。

## 输出

唯一正式目录是
`artifacts.local/evidence/rcle_phase_b_real_data_geometry_canary_r0/formal_run_r0/`，
结构固定为 `pair_ledger.jsonl`、`window_summary.json`、`receipt.json` 和
`validation.json`。implementation review 不创建该目录。

## 安全边界

只 inventory `rgb.txt`；绝不解码 RGB，不运行 RCLE、LK、optical flow、
local affine 或 score。fixture 使用临时 synthetic depth/pose archive，不构成
真实 canary 或算法 outcome。

## 停止条件

正式执行前任何绑定 hash 不符即停止；任何 identity、schema、abstention、
float64 或 branch gate mismatch 关闭 R0 implementation/evidence version，不换窗、
不换源、不原地修复。

## 假设与规则质疑

最小区分实验只覆盖三个成功窗和唯一 depth-coverage abstention 窗。若该 cohort
不能稳定暴露 interface mismatch，应以新版本修改 canary，而不是扩大 R0。

## 失败资产复用

失败 fixture、validation 与正式输出可保留为 regression/counterexample；
不得包装成 unseen confirmation、算法效果或产品证据。
