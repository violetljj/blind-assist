# RCLE periodic self-motion counterfactual R2

状态：`frozen design / execution not authorized`

## 研究问题与版本

`RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2` 是 DEVELOPMENT 阶段的受控
2×3 反事实：在非平面 3D 场景中独立操纵 endpoint-closed 周期性 6DoF camera
self-motion 与 clean/blur/low-texture，判断 unchanged R3 高触发密度来自 motion、
quality 或 interaction。

当前只包含设计合同的静态 validator。这里没有 generator、renderer、RCLE runner、
formal claim、analysis producer 或 activation lock。

## 稳定 Interface

从仓库根目录只读验证当前三份冻结 JSON：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts\research\egomotion_compensated_looming\periodic_self_motion_counterfactual_r2\validate_freeze.py
```

失败模式：

- JSON、依赖文件或 SHA-256 不匹配；
- 2×3 arm、四 block、20 seed/block、480 sequence 或 80 cluster 漂移；
- R3 `0.01/s`、三 pair、reset 或 implementation identity 漂移；
- geometry required gate、统计支持/排除规则、terminal precedence 或 budget 漂移；
- 任一当前文档把 `formal_execution_authorized` 设为 true。

## 输出

validator 只向 stdout 输出一个 compact JSON，并且不写文件。未来生成器和正式证据
只能位于：

```text
artifacts.local/datasets/rcle_periodic_self_motion_counterfactual_r2/
artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/
```

## 安全边界

- 不读取或运行 RCLE output；
- 不访问 ADVIO sequence16；
- 不修改 R3、Sparse LK、support manager、`0.01/s`、三-pair 或 PairState；
- 不运行 CoTracker、RGB formal experiment、Android 或 realtime；
- synthetic mechanism evidence 不是自然视频 false-alert、gait、obstacle、risk、
  product 或 safety evidence。

## 停止条件

静态 bundle 或独立设计审查不通过时停在
`EXECUTION_NOT_AUTHORIZED`。未来任何 geometry、response-blind calibration、R3
transport equivalence、analysis lock 或 guarded-host preflight 失败，都不得靠换
seed、降门、减 arm 或继续切 ADVIO 回救。

## 假设与规则质疑

480 条只是 80 个配对 cluster。20 seed/block 是当前固定最小预算，不是 power
保证；若 response-blind precision 预检认为不足，只能在任何 formal output 前另立
版本。五个 confirmatory contrast 使用 familywise max-t interval，避免多重比较和
逐帧伪样本。

## 失败资产复用

失败的 geometry fixture、calibration panel、pairing manifest 和 validator
mutation 可保留为 regression/counterexample。它们不得被重新包装成 unseen natural
evaluation 或 confirmation。
