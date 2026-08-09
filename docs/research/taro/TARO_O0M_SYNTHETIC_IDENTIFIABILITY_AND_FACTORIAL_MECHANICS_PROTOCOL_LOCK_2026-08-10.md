# TARO O0M synthetic mechanics 协议锁

状态：`O0M_PROTOCOL_FROZEN / SCIENTIFIC_STATUS_NOT_RUN / IMPLEMENTATION_NOT_AUTHORIZED / EXECUTION_NOT_AUTHORIZED`

日期：2026-08-10

机器合同：[JSON](TARO_O0M_SYNTHETIC_IDENTIFIABILITY_AND_FACTORIAL_MECHANICS_PROTOCOL_LOCK_2026-08-10.json)

执行族 fixture：[JSON](TARO_O0M_EXECUTION_FIXTURE_SPEC_2026-08-10.json)

## 结论

本节点只冻结未来 O0M 的输入与判据，没有实现或运行 canary。执行族固定为：

- 10 个带新 ID 的 identifiability case；
- 5 个新 factorial scene × 8 arms × 2 oracle modes = 80 条逐臂记录；
- 2 个 camera-only/body-motion action filter；
- seed `1729`，但 `rng_used=false`；
- CPU-only、30 秒、256 MiB peak RSS、1 MiB 输出上限；无 network/GPU/device/real data。

`R_weak` 只进入 2 cm identifiability gate，`H_meas` 只扩宽 decision interval。VALUE_ONLY
保持 validity、sigma、provenance 与 common-support；FULL_BLOCK 才允许同时替换它们，且只能作为
diagnostic。Factorial 的 `halfwidth_m` 固定为
`1.0 × sqrt(sigma_measurement_m² + Σ sigma_factor_m²)`；它是确定性的 budget halfwidth，既不是
Gaussian `1σ` coverage，也不是 95% coverage claim，不能与 identifiability 的 95% measurement
interval 混称。每条 arm 冻结 payload、output、common-support 与 intervention-guard SHA-256。
每个 scene 另冻结 `observed_base_mean_m`：solver 只消费该 baseline 与声明的 patch delta；
`truth_clearance_m` 只供 verifier 对照，不能进入当前输入、base posterior 或 arm choice，从而使 truth
mutation 只能改变核验结果而不能倒灌算法输入。

Implementation lock 还必须冻结非轴对齐正交重参数化 projector 测试、future-frame/truth、oracle
outcome 与 B1 consumed identity 泄漏突变、uncertainty 单调性以及两次 byte-identical replay；这些测试
未就绪前不得打开 one-shot execution。

## 三步授权链

1. 当前协议锁：只冻结 fixture、truth、十门、预算与路径；
2. 唯一 successor `TARO_O0M_IMPLEMENTATION_LOCK`：可在独立 runtime Module 创建纯解析实现和测试，
   但 execution authority 仍为 false；
3. 只有再提交 exact code/test/fixture hash-bound one-shot execution lock，才可运行一次 synthetic canary。

真实 O0R 继续是 `TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE`。任何未来 O0M PASS 也只证明
synthetic mechanics，不建立真实 factor causal headroom、主动观察价值、设备、产品或安全证据。
