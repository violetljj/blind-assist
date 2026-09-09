# RCLE Synthetic Stress Diagnostic R1 preregistration

状态：`FROZEN_BEFORE_STRESS_OUTCOME_ACCESS / DIAGNOSTIC_OPEN`

父终态为 `SYNTHETIC_MECHANISM_SEALED_REVISE`，唯一失败是未见
`offset_corridor_sealed` 旋转接近序列的 R1 trigger retention
`0.8620689655 < 0.90`。R0 claim 已消费，禁止重跑、换 seed、降低门槛或用本 R1
回写父终态。

R1 使用两个全新 scene identities、四个核心 motion，并对 matched clean control
分别施加六种 one-factor-at-a-time 扰动：低曝光、低纹理、9 px 运动模糊、35%×45%
局部遮挡、2% deterministic Gaussian depth noise、relative-yaw pose error。共冻结
56 个 stress sequence、5656 帧、5600 pair。

RCLE estimator、`>0.01/s` old trigger、三连续 pair R1 和全部 R0 gates 不变。每个
scene × motion × condition 保留固定 100-pair 分母与 abstention；condition 只有在
两个 scene 的四个 motion 全部通过时才记为 pass，禁止 pooled rescue。

首次 build 前必须逐字节复核父 R0 activation、已消费 claim、sealed terminal、
R0 renderer/dataset/truth/evaluator/validator、底层 RCLE 实现及三 pair confirmation
实现。`>0.01/s` 和连续 3 pair 还须与运行时常量相等；任一 hash、claim identity、
父终态或 retry 边界漂移即 fail closed。build 必须在创建输出目录前满足 10 GiB
剩余空间，并在逐帧写入时执行 8 GiB 磁盘与 3600 秒运行预算检查。

depth noise 不进入 RGB algorithm；primary truth 仍为 latent exact geometry，另报告
noisy observed-depth truth diagnostic，并要求 depth-noise 与 matched-control 的
algorithm ledger 在去除 condition identity 后完全相同。pose stress 只扰动 algorithm
收到的相邻旋转，truth pose 保持精确。

receipt 绑定 frozen parent chain、全部实现 hash、stress builder/evaluator/validator/QA、
三个 manifest、base R0 receipt、资源实测与 authority。validator 强制检查完整
`condition × scene × motion × frame` 笛卡尔积、精确字段 allowlist、路径 containment、
逐文件 hash、observed-depth 唯一性与 101 帧索引。科学求值只允许在候选数据通过该
preflight 且科学输出尚不存在时开始。

整体终态只称 `STRESS_R1_CHARACTERIZATION_COMPLETE`，无论多少 condition 通过都不
构成 R0 rescue、真实 external confirmation、性能资格、Android、产品、人因或安全证据。
