# TARO O1R R7 positive occupancy + censored clear coverage task lock

状态：`FROZEN / FIT_ONLY_CANARY / CPU_ONLY / NO_PROMOTION`

机器合同：[JSON](TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_TASK_LOCK_2026-08-12.json)

R6/O1R 已经证明，继续把同一 clearance interval 接到同一 uncertainty envelope 不会产生最终状态：有效 clearance 的下界是 `-0.30 m`，而最小 uncertainty 已是 `0.4595 m`，occupied 判定结构性不可达；大多数 clear 场景又先被 source coverage/knownness 拦截。

R7 将问题拆为两个可证伪通道：

- `OCCUPIED_POSITIVE`：由 capsule 内 confidence-2 AppleDepth 连通障碍组件提供正证据，不再依赖饱和在 `-capsule_radius` 的 clearance 值；
- `CLEAR_CENSORED_FAR`：把有限且显著超过 2 m horizon 的 candidate depth 作为 line-of-sight 的 far-censored coverage，只增加 clear knownness，绝不生成近处 support/obstacle 点。

所有候选阈值网格已冻结。只允许在 8 个 `ADAPTER_FIT` parent 上做 leave-one-parent-out CPU canary；每折只用 7 个 parent 选阈值，再应用于 held parent。已经观察过 R6 morphology 的 16 个 eval parent 只能作诊断，不能 promotion。任何 R7 promotion 都必须使用新的 parent-disjoint untouched cohort 和新授权。

唯一后继：`TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_CANARY_IMPLEMENTATION_LOCK`。
