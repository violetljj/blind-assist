# F-1B 结构可达性协议修复 R1 结果

状态：`PROTOCOL_INVALID / SUPERSEDED_BY_R2 / DECISION_NOT_CONSUMED`

执行者：`viojjet`

## 结论

R1 修复了 R0“相信自报 truth-table boolean”的主要缺陷，绑定了 13 个生产实现 identity，
派生 19 个 fresh state，并加入 temporal、hold、side-person gate、event、cooldown、
fatigue 和四个科学端点。

但独立复核发现两个阻断问题：

1. R1 把侧向 approaching `NEAR/MEDIUM` 错列为 `HIGH`；生产
   `ConservativeRiskFusionPolicy` 会把侧向 promotion 封顶回 `MEDIUM`。
2. R1 允许确认替代作用于不具 planner 提醒资格的 CENTER/MID/MEDIUM。即使当前帧不
   交付提醒，也可能改变 stabilizer 的 pending/confirmed/hold 内部历史，因此 R1 的
   history induction 不成立。

所以 R1 的 `NO_INCREMENT` 科学方向保持待重算，`SCIENCE_PROTOCOL_STATUS=INVALID`。
R1 spec 与 validation 原样保留，未覆盖；未访问或运行 decision YOLO、Sparse LK 或
A/B 输出。最终协议状态以 R2 successor 为准。

```text
R1 spec sha256:
d30af4f73882675555c895347f4e8493313433a4b212d342548dc64a7a1c6b68

R1 validation sha256:
607534f6a1ead5d25eaa5f12621b6ac76bc95a9b4264963af7f9d859d17fae1f
```
