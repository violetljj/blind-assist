# HFTF Stage B reference-only opportunity qualification result R3.1

日期：2026-08-01

终态：`R3_1_REFERENCE_OPPORTUNITY_COHORT_NOT_EVALUABLE`

## 1. 结论

R3.1 已按冻结顺序用完 SANPO-Synthetic train 的 40-session inventory-eligible
screening budget，得到 `0/4` qualified sessions。因此不得构造 conditional challenge
cohort，不得运行 candidate/baseline arm comparison，也不得降低门槛或继续扩大同一
队列。

这次结果把 R3 的混合 opportunity blocker 定位得更清楚：

- 40 个 source 中 3 个 authority 失败、3 个缺完整 pose/local-ground binding；
- 其余 34 个完成 dense reference ground 计算；
- 34 个 ground reports 的 reference risk cells 合计为 **0**，没有任何非零会话；
- 34 个可计算 source 中，29 个通过全部四项 obstacle opportunity checks。

因此当前失败不是“人体 swept envelope 没有 obstacle 增益”的正式反证。R3 的正向
obstacle diagnostics 和本次 29/34 obstacle readiness 只能继续作为机制证据，不能
越过 ground 顺序门。当前 SANPO evidence version 的 semantic-ground-only continuity
reader 对台阶/落差机会结构性不足，无法支撑原 Stage B 的 foot-ground effect claim。

## 2. 绑定报告

聚合报告：

`artifacts.local/evidence/hftf/r3-1-reference-opportunity-cohort-20260801/cohort_result.json`

SHA-256：

`6c61d8c333cc6bad59f37e2f0c3bc34c8baabfa138958ec14a484d56510979e7`

绑定：

- qualification protocol SHA-256：
  `4f53bb18e563a5cbd8f0420618177abbea18509432f288d5d7f42ee54698fb0e`
- burn ledger SHA-256：
  `932e1aeb767c98fb1a8e533a5b64a1618e4ee78bcc856af4628a7bb6a28e5ab2`
- inventory plan SHA-256：
  `de42952c99236f7d1775732055076042ea2ca4986bb667ece47bd7f92cb3a599`
- qualifier SHA-256：
  `b827d6b5f0b7332b51e3e6c29af94e522aab982dca849086a71dbb9c80d5eb99`
- cohort aggregator SHA-256：
  `92f43f65ded85007cc493baa0e91934e8946d8dd2d6d8b0b494db4fc2e655353`

聚合器逐项验证 40 个 report 的 protocol/ledger/plan hashes、inventory rank 与 session
映射、字典序连续前缀、唯一性和 reference-only firewall。candidate grid、angular
baseline、arm metric/delta 均未计算。

## 3. Gate 统计

| 项目 | 结果 |
| --- | ---: |
| screened / maximum budget | `40 / 40` |
| qualified / required | `0 / 4` |
| authority failures | `3` |
| incomplete geometry bindings | `3` |
| reference-ground reports | `34` |
| nonzero ground-risk sessions | `0 / 34` |
| total reference ground-risk cells | `0` |
| obstacle all checks pass | `29 / 34` |

失败 check 计数：

- `ground_reference_risk_cells/frames/directions`: `34/34/34`
- `ground_known_coverage`: `1`
- `obstacle_primary_positive_each_height`: `5`
- `obstacle_all_sensitivity_thresholds_have_micro_opportunity`: `1`

## 4. 科学边界

本终态支持：

- 冻结的 SANPO R3.1 source representation 无法组成同时包含 obstacle 与 ground
  opportunity 的 4-session cohort；
- 继续扫描同一 frozen queue 或降低 ground gate 没有授权；
- source/reference insufficiency 与 candidate effect failure 必须分开。

本终态不支持：

- swept human envelope 对 obstacle 没有增益；
- foot/body/head 表示应被删除；
- ground step/drop 可由零机会结果判成“表现良好”；
- Stage C、student/H2、研究主线、Android、提醒、默认 App、生产或安全 claim。

## 5. Governed successor

下一条 Development successor 必须拆分 **source role**，而不是删掉 ground：

1. obstacle-envelope effect 使用新的 SANPO reference-only obstacle-qualified cohort，
   保持 R3 的 candidate/baseline/reference grids 和 effect gates；
2. foot-ground effect 使用独立的 metric terrain source，必须有解析或 source-native
   surface elevation truth，覆盖 flat negative、rise、drop、ramp/roughness
   counterexamples 与 unknown/occlusion；
3. ground reference 不得继续由 semantic-ground class membership 自证；
4. 两个 source role 都通过后，才可说 Stage B teacher mechanics 获得 split-source
   Development support；这仍不等于自然 prevalence、人类事件效用或 Stage C execution
   授权。

R3.1 的 40 个 source reports 均已 outcome-open，后续只能作 burned diagnostics，
不能作为新 formal obstacle comparison cohort。
