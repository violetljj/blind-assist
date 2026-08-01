# HFTF Stage C D3-Q0 reference-and-support-only challenge qualification

## 结论

D2 保持
`D2_NOT_EVALUABLE_OPPORTUNITY_INADEQUATE_NO_SOURCE_REPLACEMENT`，不重跑、不换源、
不降低门。唯一允许的后继是一个新数据角色：

> 在任何新 source outcome 前冻结的
> `REFERENCE_AND_SUPPORT_ONLY` opportunity-qualified conditional challenge cohort。

D3-Q0 只改变评价人口，不改变 D2 的 signed-clearance field、恒速因果 SE(2) transport、
两臂、estimand、opportunity gates 或 effect gates。当前只授权冻结并审计后续
metadata roster、qualifier 与 effect skeleton execution contract；不授权扫描、媒体、
truth、effect 或 student。

## 为什么不是纯 reference-only

D2 的 opportunity 分母不是 future truth 单方 known，而是
`truth ∩ persistence support ∩ advected support` 的三方 common-known。若只按
reference truth 选 source，不能保证正式比较可判定。

因此资格器可计算 exact D2 predicted basis、三方 9-probe known/support masks，以及
common-known 上的 future truth risk/safe counts；但不得计算或落盘 persistence/
advected clearance values，不得产生 MAE、F1、confusion、delta、improvement 或按
“离门多远”排序。这是 support/opportunity selection，不是 arm effect selection。

## 固定 source pool 与预算

后续 implementation contract 必须在任何 D3 media/pose content 或 truth 打开前：

1. 绑定 official SANPO-Synthetic train split generation/SHA；
2. 构造全部历史 HFTF burned/consumed/closed、完整 D1/D2 cohort 与 reserved
   official-test 的 exact exclusion union；
3. 按 `session_id` 字典序锁定 metadata-only roster；
4. 绑定 exact qualifier、D2 mechanics primitives、effect skeleton、tests 与 transport
   dependencies，并提交推送、确认 clean remote parity。

最多允许 40 个 truth-screened slots，首 6 个合格 source 即停止。slot 锁定后的
authority/acquisition failure 也消耗该 slot；不得替换、扩到第 41 个、人工跳序或在
D2 的 parent 成败、fps、场景、运动、风险缺口上定向选源。未打开的剩余 roster
不得为本轮读取媒体或 truth。

## Per-source qualification

每个 source 固定 7 anchors、`body/head × 0.4/0.8 s` 四个 strata；每个 stratum
分母固定为 `7 × 6 × 6 = 252`。四个 strata 必须全部满足：

- 三方 common-known coverage `>=0.10`；
- common-known 上 future truth signed clearance `<0` 的 risk cells `>=5`；
- common-known 上 future truth signed clearance `>=0` 的 safe cells `>=20`；
- UNKNOWN→SAFE violation `=0`。

selector receipt 只能暴露 authority/hash、slot/source ID、上述 counts、gate booleans
和 qualified boolean。per-cell truth magnitude 必须写入与 selector 隔离的 sealed
payload；不得暴露 motion/yaw/scene/semantic ranking fields。source 一旦打开
media/pose support 或 truth 就 burned；ineligible source 也不能复用。

预算内不足 6 个即
`D3_REFERENCE_SUPPORT_OPPORTUNITY_COHORT_NOT_EVALUABLE_BUDGET_EXHAUSTED_NO_EXPANSION`，
不得继续扫描或降门。freeze/hash/order/firewall 漂移即
`D3_QUALIFICATION_INVALID_STOP`。

## 预冻结 effect skeleton

为了避免资格结果反向修改比较，后续 execution contract 必须在第一个 D3 truth 前同时
锁定完整 effect skeleton。资格成功后只能填入 6 个 selected IDs、authority 与
qualification hashes，不能改 mechanics 或数值门。

formal predictor 仍只能读取 selected source 的 current/history inputs，并在 effect
evaluator 读取 sealed truth payload 前 durable。qualification support 与 formal
prediction support 必须逐 cell 一致，opportunity 四门必须复算；不一致直接
`D3_NOT_EVALUABLE_QUALIFICATION_RECOMPUTE_MISMATCH_NO_REPLACEMENT`。

正式 effect 完整继承 D2：

- 6 parents × 7 anchors × 2 horizons，parent session 是独立单位；
- `CURRENT_FIELD_PERSISTENCE` 对
  `HISTORY_CAUSAL_ADVECTED_CURRENT_FIELD`；
- macro MAE 相对下降至少 10% 且绝对下降至少 0.03 m；
- body、head、0.4 s、0.8 s 均不劣；
- 至少 5/6 parents 严格改善；
- parent-macro risk-sign F1 增量至少 0.03；
- UNKNOWN→SAFE 为 0。

任一 effect gate 失败仍是
`CAUSAL_SIGNED_CLEARANCE_TRANSPORT_NOT_SUPPORTED_STOP`。全部通过也只授权另冻 RGB
student Development protocol。

## Claim ceiling

D3 即使成功，也只能写成“在 reference-and-support-opportunity-qualified conditional
challenge cohort 上，exact D2 transport 获得/未获得冻结 effect”。它不代表自然
prevalence、fresh generalization、人类事件、安全或产品效果。

研究主线、默认 App、Android、reserved official-test、RGB student training/execution、
生产与 safety 权限全部保持关闭。
