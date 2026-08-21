# P1-W1 Stage A single-execution result

状态：`CONSUMED / W1_T0_NOT_EVALUABLE_DATA_OR_INTERFACE / INTERFACE_INITIALIZATION_SUPPORT_FAIL / NO_STAGE_B / DEFAULT_APP_UNCHANGED`

Claim ceiling：`CONSUMED_ADT_DEVELOPMENT_MECHANICS_ONLY / NO_C0_T0_VERDICT / NO_WORLD_MEMORY / NO_PRODUCT_OR_SAFETY_AUTHORITY`

## 1. 一次性执行

用户授权唯一 successor `P1_W1_STAGE_A_SINGLE_EXECUTION` 后，冻结 commit
`08aa765c8821d41a735519ffcede35ae51451b17` 依次完成：

```text
prepare: v3 public/private roster hash binding
run:     public RGB + P0 initialization only, 17/17 cases, C0/T0 each once
evaluate: private truth once, frozen adjudication
```

执行没有新增来源、重选 episode、调整 ORB/HSV/threshold、换 tracker、重跑 arm 或读取 future/GT。Formal run
位于 ignored `artifacts.local/evidence/p1_w1_stage_a_single_execution_v1/`；manifest 最终为
`performance_outcome_accessed=true`，predictions/result 均 write-once。

## 2. 正式终态

```text
W1_T0_NOT_EVALUABLE_DATA_OR_INTERFACE
```

17 个 case 中只有 3 个通过冻结的 source-region initialization；14 个在 arm 运行前即失败：

```text
source region has insufficient target-local ORB support
```

失败来自冻结 implementation 对 P0 source region 要求至少 6 个 target-local ORB descriptors，而不是 C0/T0
outcome。它使实际可评估 support 退化为：

| stratum | selected | evaluable |
|---|---:|---:|
| rotation dominant | 2 | 0 |
| small translation | 7 | 1 |
| translation beyond Tier-0 | 7 | 2 |
| identity confuser | 8 | 1 |
| loss / reappearance | 11 | 2 |

因此 Stage A 的核心 rotation denominator 为 0，identity/reacquisition 与 translation strata 也不足以形成冻结比较。
按协议这不是 `W1_T0_NOT_SUPPORTED`，更不是正信号；不能用 3 个可运行 case 代替 17-case verdict。

## 3. 仅诊断指标

以下来自 3 个可运行 case，只用于说明 fail-closed mechanics，不能判定 C0/T0：

| endpoint | C0 | W1-T0 |
|---|---:|---:|
| fabricated observation | 0 | 0 |
| single-channel reacquisition | 0 | 0 |
| stale-anchor guidance use | 0 | 0 |
| false continuity | 0 | 0 |
| false reacquisition | 0 | 0 |
| identity-confirmed reacquisition | 0 | 0 |
| usable-anchor frames | 5 | 19 |
| honest NONE | 468 | 468 |
| visible-frame abstention | 133 | 133 |
| supported bearing frames | 5 | 5 |

两 arm 均通过 translation-overreach same-frame stale 与 geometry-degenerate fail-closed mechanics。W1-T0 的
`19 vs 5` usable-anchor frames 不能写成信号：rotation support 为 0、reacquisition 为 0，且 formal terminal 已先
被 interface gate 截止。

## 4. 证据绑定与停止

```text
public roster       1969560ba8a3863ad4aef16fca9141602144a4b4555ee38c38ff49b6f62bef70
private map         fd23bb01d928fdf97d65fa0f1d67868b85c0050108a9c632f8296d660f75aad8
public input        ac8e818f00226c3c0f9d75a4739010d75e5dfddaa15cf4b9076d04d516107c07
private truth       1e6952e512e6ffc6592901f76d8c1a3c0e881c8f57032f66732e35a72decf595
predictions         432d3c0b551ff90847dcf0b62ccea9cf2e89e248656525039cc698f995cc143f
result              9d336103a43da0515f27ec81fdfd2964032a3307097891930fbdffade324c6591
```

Stage A v1 已消费并封口，不得放宽 `>=6`、在这 17 个 episode 上 tuning 后重宣称验证，或把 initialization miss
计为算法负例。W1-T1 / Stage B、SLAM、tracker zoo 与 App 均不授权。

当前没有自动 successor。若未来重开，必须先另立 outcome-blind interface/data-adequacy contract，解决 P0 source
region 与 selected identity provider 的可初始化支持；不得覆盖或续跑本 sealed v1。
