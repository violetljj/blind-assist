# Objective image-space candidate increment D0 protocol

状态：

```text
FROZEN_PRE_OUTPUT /
CONSUMED_SESSION_DISJOINT_THESIS_DEVELOPMENT /
TIMING_NOT_EVALUABLE_ONSET_INCOMPLETE
```

机器合同：
[OBJECTIVE_IMAGE_SPACE_CANDIDATE_INCREMENT_D0_PROTOCOL_2026-08-02.json](OBJECTIVE_IMAGE_SPACE_CANDIDATE_INCREMENT_D0_PROTOCOL_2026-08-02.json)。

## 结论边界

本实验只回答一个问题：在固定 PIDNet-S `512x288 / INT8 / seed-20260801`
语义预算下，把 PIDNet 的 `{blocking_obstacle, boundary_level_change}` 原生 argmax
区域作为 YOLO 框外候选，能否增加 source-mask 可复算的客观像素/组件覆盖，同时控制
额外 false area、碎片组件和低成本图像空间算子开销。

它不训练模型，不恢复 actionability/中央阻塞标签，不运行旧 mask adapter、风险规则、
事件链或反馈链，也不修改默认 App。

## 数据角色

主集为 RISKSEG-R0 的 30-session / 1,920-frame 设备视图。它与 520-frame PIDNet
train/dev 按 source session 隔离，但已经被 RISKSEG-R0、R1-P0 和 ACT-A0 消费，因此
永久只能提供 `THESIS_DEVELOPMENT` 机制证据，不能恢复为 fresh 或 Confirmation。

执行前已生成 objective-only manifest，仅保留 session、frame/time、RGB、source-mask
身份与哈希；`positive/bucket/alertable/passed/event_candidate_id/parent_event_id`
全部剥离。旧 520 帧和旧 120 帧不进入主分母，因为它们分别属于 train/dev，且旧
120 帧与 PIDNet train 精确重叠。

## 四类与未知区

四类顺序保持：

```text
0 walkable
1 blocking_obstacle
2 boundary_level_change
3 unknown_nonwalkable
```

客观 truth 是 `{1,2}`。class 3 从 truth 和 false-positive 分母中排除，不能被解释为
安全或可通行。这里的 `blocking_obstacle` 只表示 source-mask 实体像素，不表示值得提醒、
需要绕行或真实路线阻塞。

## 三臂和连续量

```text
A: D_t = frozen YOLO post-NMS box union
B: H_t = PIDNet argmax in {1,2}
C: D_t union H_t
R: H_t minus D_t
U: objective truth minus D_t
```

不加 confidence、面积、top-k、类别特例或时序 latch。每帧记录候选面积比例、图像中间
三分之一占用、冻结梯形 ROI 占用、bottommost-y、组件数、相邻面积变化与 raw adjacent
IoU。这些都是图像空间连续证据，不是风险或行动标签。

主门沿用旧 candidate-utility 尺度：`C-A recall >= .05`、residual component recall
`>= .50`、added FP area `<= .05`、false components/frame `<= 3`；另要求 blocker 与
boundary 各 `>= .02`、session median gain `>= .05`、session P90 added FP `<= .05`。
host objective operator P95 必须 `<=30 ms`；既有 SM-S9280 证据必须保持总链路 P95
`<=100 ms`、10 分钟 final/initial `<=1.20x`、无 failure、无 severe thermal、完整
QNN 委派。

## Timing 不可评价

冻结 onset 定义为：ROI 内先有至少 5 个连续零 truth observation，再有至少 3 个连续
非零 truth observation。主集只有 4 个 session 满足，而 timing 最低要求是 12 个；
因此无论 coverage 结果如何，本 D0 都不得声称“更早”。若 coverage/FP/cost 全部通过，
最高终态也只是：

```text
OBJECTIVE_IMAGE_SPACE_INCREMENT_SUPPORTED_BUT_ONSET_COHORT_REQUIRED
```

需要新建 onset-complete、session-disjoint 自然 cohort 后，才能运行 timing 门。

## Stop rules

- 身份、mask、坐标或 YOLO 配对失败：
  `HOLD_OBJECTIVE_PIXEL_TRUTH_OR_PAIRING_NOT_EVALUABLE`；
- coverage、类别、FP 或 component 任一门失败：
  `STOP_FIXED_PIDNET_OBJECTIVE_CANDIDATE_NO_ROBUST_INCREMENT`；
- utility 通过但固定预算失败：
  `STOP_FIXED_PIDNET_OBJECTIVE_CANDIDATE_BUDGET_FAIL`；
- utility 与预算通过但 timing 数据不足：
  `OBJECTIVE_IMAGE_SPACE_INCREMENT_SUPPORTED_BUT_ONSET_COHORT_REQUIRED`。

任何负终态都关闭 exact seed+operator，不允许增加规则救援；任何正终态也不自动授权
提醒、默认 App 或安全主张。

