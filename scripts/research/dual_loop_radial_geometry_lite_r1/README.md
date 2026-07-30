# Dual-loop causal radial geometry LITE R1

状态：design review passed；implementation candidate 与 no-truth pilot 已完成，
formal replay 尚未授权或执行

## 研究问题与版本

`DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R1` 是 R0 的前向新 evidence
version。它只研究如何在 source RGB 解码尺寸变化时保持两臂公平、因果和固定分母，
再回答原有 Development 比较问题。R0 的
`EXECUTION_INVALID_STOP_NO_RERUN / NOT_EVALUABLE` 不被改写。

## 稳定 Interface

稳定入口为：

```powershell
E:\codex-tools\venvs\dual-loop-radial-geometry-lite-r0\Scripts\python.exe `
  scripts/run_dual_loop_radial_geometry_lite_r1.py audit-shapes `
  --replay-input artifacts.local/evidence/dual-loop/target-track-causal-radial-geometry-lite-r0/input-freeze/replay_input.jsonl `
  --image-root artifacts.local/evidence/datasets/revel-dynamic-images-labels-v1-20260720/extracted/images/images `
  --output artifacts.local/evidence/dual-loop/target-track-causal-radial-geometry-lite-r1/source-shape-audit/shape_audit.json
```

同一 Adapter 的 `produce` 只接收冻结 replay 与 RGB；`evaluate` 只允许在 producer
完整终态和 pre-truth keyset/identity 校验通过后接触 Development truth/event。
正式命令必须另由 one-shot activation 和 guarded-host preflight 精确绑定。

## 输出

只写入 `artifacts.local/evidence/dual-loop/target-track-causal-radial-geometry-lite-r1/`。
source audit 使用确定性 JSON，并记录 replay identity、解码 shape 和 transition 分母。
producer 以流式临时文件写入，成功后原子发布 output、receipt 与 progress；失败仅发布
failure receipt，不保留部分 output。

## 安全边界

- audit/producer 不读取 truth、natural events、Vicon 或旧 F-1B decision；
- REveL 只承担已烧入的 single-capture Development；
- R1 不产生 Confirmation、Android、产品或安全 authority。

## 停止条件

decode failure、replay identity 漂移、固定 `13,014 × 2 = 26,028` keyset 不完整、
`32` 个 shape-change opportunity / `64` 个 arm row 不成立，或尺寸语义不能同时
适用于两臂时停止。正式执行前另行冻结 implementation、one-shot、progress 与 terminal
合同；任何正式失败仅关闭 R1 evidence version，不扩大为算法或研究问题失败。

## 假设与规则质疑

候选假设是：对前后 decoded shape 不同的 pair，两臂共同 abstain 并把当前帧作为
下一 pair 的唯一历史，可避免 resize/pad 制造的伪尺度，同时只损失极少固定分母
coverage。falsifier 是固定事件覆盖或 readiness 因该语义不可评价，或 producer
仍不能完整运行。它是低成本、单一语义修复，不进行参数搜索。

## 失败资产复用

R0 failure receipt 和跨尺寸帧作为 regression fixture/source characterization；
不得改名为 R1 成功或 unseen confirmation evidence。
