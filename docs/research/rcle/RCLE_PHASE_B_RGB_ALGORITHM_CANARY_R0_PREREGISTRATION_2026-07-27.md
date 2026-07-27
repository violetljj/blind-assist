# RCLE Phase B RGB algorithm canary R0 预注册

状态：`F1_PREREGISTERED_DESIGN / HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / EXECUTION_NOT_AUTHORIZED`

日期：2026-07-27

机器合同：
[RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_CONTRACT_2026-07-27.json](RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_CONTRACT_2026-07-27.json)

数据角色：
[RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_DATA_ROLE_MANIFEST_2026-07-27.json](RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_DATA_ROLE_MANIFEST_2026-07-27.json)

性能规范：
[RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_PERFORMANCE_PREFLIGHT_SPEC_2026-07-27.md](RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_PERFORMANCE_PREFLIGHT_SPEC_2026-07-27.md)

## 结论与科学价值

值得预注册的问题只有一个：在已经烧掉的 TUM `fr2/rpy` rotation windows `0/3/6` 上，rotation-compensated local expansion 相对同 support 的 raw-flow baseline 是否降低 rotation leakage。这个问题能把“真实 RGB 上是否保留 Phase A 所指向的旋转补偿方向”与 geometry interface 是否调通分开，失败也能定位到真实纹理、跟踪、局部支持或补偿机制，因此具有 canary 阶段的信息增益。

geometry canary 的 `VALID_IMPLEMENTATION_DEBUGGED_GEOMETRY_INTERFACE_ONLY` 只证明 pair identity、schema、弃权、branch 和 float64 parity。它没有读取或评价 RGB algorithm outcome，不能升级为算法有效性。

当前没有 independently admitted real positive approach role。TUM source audit 明确未建立 approach role；Bonn 已烧掉且只能作 regression/counterexample；Phase A 是 synthetic calibration/fixture。因此本预注册不得设置或预写 approach/closing-retention PASS。设计包即使通过独立审查，当前终态仍是 `HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`，最大权限仍为 `EXECUTION_NOT_AUTHORIZED`。

## Hash binding

| 对象 | SHA-256 |
| --- | --- |
| Phase B Progressive Protocol | `43d0e6eb637d56262856bbb27198856f85ccd03684abd2f6544333dd1a4ca742` |
| geometry canary formal result | `9f068ec046ffdea23257ac0f3aa4663ecc531b0354e3a822c31e86266e4ef0cf` |
| geometry canary receipt | `b55417cebe7188cdbee40db02c36b06e58294acd8a9906edd6aba2ad00f211cd` |
| geometry canary validation | `c28c65accb9d19ee3cb409b3a391245cb29cf082ce6b3d46a0c8d39cce557154` |
| geometry implementation lock | `0d833b835d242468fe8c466414882044c3717e8f0b16d6d79a6b5f112e1e2387` |

上述文件任一漂移都关闭本 evidence version。implementation lock 的 hash 只绑定已经完成的 geometry canary，不授权复用它作为 algorithm implementation lock。

## 数据角色与 firewall

| 内容 | 角色 | 当前 access | 后续复用 |
| --- | --- | --- | --- |
| TUM `fr2/rpy` windows `0/3/6` | `CANARY / BURNED / GEOMETRY_SELECTED / ROTATION_STRESS` | `GEOMETRY_ONLY` | 本命题永不回到 confirmation |
| TUM `fr2/rpy` window `4` | `ABSTENTION_OR_INTERFACE_STRESS_ONLY` | `GEOMETRY_ONLY` | 科学权重固定为零 |
| Bonn frozen cohort | `REGRESSION_OR_COUNTEREXAMPLE_ONLY` | `FULL` historical | 不得承担 canary/confirmation 科学角色 |
| Phase A synthetic R1 | `CALIBRATION_OR_FIXTURE_ONLY` | `FULL` | 不得冒充真实数据 |
| ICL/ETH3D 等 | `NOT_ADMITTED / RESERVED` | `NONE` | R0 禁止读取；准入需另立版本 |

三个位于同一 TUM sequence 的科学窗口共享一个 independence group。window
`0/3/4/6` 的 exact Unix-second boundaries、各 `299` 个 pair 的 canonical
identity hash 和 geometry pair ledger SHA 已写入 manifest；身份不再依赖
`#window-N` 别名。`897` 个计划 scientific pair 以及 geometry interface 的
`1196` 个 pair 都是时间相关记录，不是独立科学样本。未来 confirmation partition
目前不分配任何 content identity，并明确排除 TUM、Bonn 与 Phase A 的 ancestry。

当前设计任务禁止创建或读取 algorithm claim、output、failure receipt、activation 或 implementation lock。firewall 只允许读取文件名库存和本设计包，不得为了证明“没有 outcome”而打开疑似 outcome 文件；一旦出现疑似路径即 fail closed。

## Comparator、单位与分母

候选实现唯一冻结为
`TUM_POSE_ROTATION_COMPENSATED_OBSERVABLE_THREE_FRAME_LOCAL_EXPANSION_R0`。
rotation 只来自 TUM source-native mocap pose：RGB timestamp 内做不超过
`0.050 s` bracket 的 quaternion SLERP，使用 `R_current_from_previous =
R_world_current^T R_world_previous` 与固定 `fx=fy=525, cx=319.5, cy=239.5`
构造 homography；禁止 visual homography、IMU 或融合替代。current grayscale
以 inverse homography、OpenCV linear interpolation warp 回 previous coordinates，
validity mask 使用 nearest interpolation。

raw arm 和 compensated arm 使用相同 RGB pair、固定 `3×3` image grid、相同
initial support 和相同共同可评价 cell support。Sparse LK 参数、Shi-Tomasi、
forward-backward 门、唯一 `OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0`、7×7
photometric patch、4×4 deterministic supplement、RANSAC affine、support/hull/
condition/residual/common-cell 门均由机器合同和六个 normative source SHA 冻结；
implementation task 不得再选择实现或参数。每个 cell 的 local affine expansion
为 `e = 0.5 × trace(A)`，除以 pair `dt` 后单位为 `s^-1`。pair rotation leakage
是共同可评价 cells 上 `abs(e)` 的中位数；pair score 为：

```text
raw_pair_rotation_leakage - compensated_pair_rotation_leakage
```

score 越大越好，正值表示补偿后 rotation leakage 更低。window score 是该窗所有共同可评价 pair score 的中位数。科学单位是预先选择的 window，不对 pair 做 IID 推断，不给出把时间相关 pairs 当独立样本的 p-value 或 CI。

windows `0/3/6` 各有固定 `299` candidate pairs，覆盖率分母不因弃权缩小。
first pair 没有三帧 history 时运行 unchanged R1 baseline，仍留在原分母。pair
至少需要 `5/9` common evaluable cells 才进入 score；单臂删除、补零、插值、
换窗和 pooled rescue 均禁止。window `4` 仍保留 `299` candidate pairs，但只
验证 abstention/interface contract，不参与科学分数。

## F1 gates

| Gate | 判据 | 解释 |
| --- | --- | --- |
| upstream binding | mismatch `= 0` | 证据基础不能漂移 |
| outcome firewall | forbidden artifact/field `= 0` | 设计必须先于 outcome |
| role/identity closure | violation `= 0` | content identity、ancestry、independence 和 reuse policy 必须闭合 |
| real positive approach role | admitted independence group `>= 1` | 当前为 `0`，所以保持 HOLD |
| per-window coverage | 每个 `>= 0.80` | 低于门则该窗 `NOT_EVALUABLE`，不得替换 |
| rotation direction | `0/3/6` 三窗 window score 均 `> 0 s^-1` | 零是自然 no-improvement comparator，不设 outcome-derived effect-size 门 |
| validator recomputation | mismatch `= 0` | summary/hash 不能替代独立复算 |
| performance qualification | violation `= 0` | 不合格即 `PERFORMANCE_NOT_QUALIFIED`，claim 不得创建 |

每个数值门的 unit、rationale、calibration source、sensitivity plan 和 revision
policy 已写入机器合同。`0.80` coverage 是继承 Phase A paired-denominator
comparability 的 canary guardrail，不是由真实 algorithm outcome 标定的效果阈值；
其证据强度有限，因此同时报 `0.70/0.80/0.90` 诊断，但本 R0 只能使用 `0.80`。
window aggregate 同时报 median 与 20% trimmed mean，leave-one-window-out 只作
敏感性；这些诊断都不能改变冻结 gate。

## 两轴终态

execution validity 与 scientific outcome 分开：

| 条件 | execution validity | scientific terminal | scope |
| --- | --- | --- | --- |
| identity、cache、schema、numeric、summary 或 validator mismatch | `INVALID` | `NOT_EVALUABLE_DUE_TO_EXECUTION` | 最小 `EVIDENCE_VERSION` |
| 任一 scientific window coverage `<0.80` | `VALID` | `NOT_EVALUABLE` | `WINDOW` 传播到本 branch |
| 三窗不全为正方向 | `VALID` | `MECHANISM_DIRECTION_NOT_SUPPORTED` | `BRANCH` |
| 三窗方向成立但 approach role 仍缺失 | `VALID` | rotation mechanism direction only；overall `HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE` | `BRANCH` |
| performance pilot 不合格 | `NOT_RUN` | `PERFORMANCE_NOT_QUALIFIED` | `IMPLEMENTATION_VERSION` |

本预注册不把任何未来 outcome 写成 PASS。即使 rotation direction 得到支持，也只形成 burned single-sequence canary 的 mechanism direction，不形成 closing retention、confirmation 或 Kill Gate B 结论。

## Immutable cache 与性能准入

未来 implementation task 必须在 claim 前按 TGZ archive header order 做一次顺序
物化，逐 member 记录 archive ordinal、unique normalized path、size、source hash、
member hash 和 canonical manifest hash；不为了 path 排序回扫 gzip。producer 与
独立 validator可以共享已经全量验证的 immutable bytes，但不得共享 pair ledger、
aggregate、summary 或 scientific computation。

正式 claim 前必须用相同真实访问机制做 bounded pilot，至少比较 `1` 与 `8` worker；`12` worker可替代或补充 `8`。两种配置的有序 identity、schema、abstention 和 float hex 必须严格等价，并记录 wall time、core-equivalent、I/O、RAM、throughput、projected/max wall time和至少两个真实 progress samples。progress sidecar 必须包含 phase、completed/total、throughput、ETA、last progress time、PID、input hash、implementation hash 和 status。后续长跑只能通过 `scripts/run_guarded_host_research.ps1`。

## Independent validator

未来 validator 必须位于独立 package，不 import producer，不信任 producer summary 或 hash。它从 frozen contract 和 verified immutable cache 全量独立复算 cache identity、pair/window identity、schema、顺序、弃权、float64 数值、aggregate、terminal 与 progress contract。

恶意反例必须覆盖数据角色重叠、content identity/ancestry/independence/access/reuse
漂移、contract-manifest 冲突、algorithm outcome field/filename 泄漏、字段缺失、
顺序变化、数值漂移、summary 伪造、cache member 篡改、progress contract 缺失，
以及 timestamp/phase/status/PID/ETA/freshness 伪造。任一反例未被拒绝都表示
validator independence 未闭合，implementation task 不得建立 lock。

## Freeze 与合法后继

F1 字段从本设计包通过独立 design review 后冻结。任何 algorithm outcome access 后只能新建版本，不能原地修改 comparator、窗口、分母、coverage、方向门、弃权或 failure scope。

另立 implementation task 前仍缺：

1. 一个真实、positive approach、identity/ancestry 闭合且不侵占 confirmation 的独立数据角色；
2. producer、独立 validator、cache materializer、progress sidecar 与 guarded launcher 的实现计划；
3. 相同真实访问机制上的 1 与 8/12 worker 性能资格与等价性 receipt；
4. 仅在新任务内创建的 implementation lock 和独立 implementation review；
5. algorithm claim、output 与 failure receipt 仍不存在的新鲜证明。

design review PASS 只允许把这份设计标为 `VALID`。它不会自动授权实现或执行。
