# HFTF Stage B swept-envelope reference comparison result R3

日期：2026-08-01

终态：`R3_SOURCE_OR_REFERENCE_NOT_EVALUABLE`

## 1. 结论

正式 R3 在第一顺序门停止：4/4 authority、exact set、obstacle known coverage 与 ground
known coverage 均通过，但 session `043db91a…` 在全部 reference thresholds 上没有
任何 obstacle positive。primary threshold 2 下该 session 为 0 positive / 883
negative，candidate 与 baseline F1 均不可定义。因此预冻结的 4/4
positive-and-negative opportunity gate 失败，后续 obstacle gain 与 ground 不得形成
正式支持终态。

这不是 swept-envelope gain 的负结果。它是 source opportunity 不足造成的
`NOT_EVALUABLE`。

## 2. 绑定报告

报告：

`artifacts.local/evidence/hftf/stage-b-swept-envelope-reference-comparison-r3-20260801/stage_b_r3.json`

SHA-256：

`512a5dda7e84148820e398af39eab4d5841f4a2ac6c94871cfb6754b374cb5af`

protocol SHA-256：

`17f4c7a9c2ca2755cc228b5b80cbda813847973ffdec1cf1bde09ed5fdfedf15`

finalizing implementation commit：`7855f24`

runner SHA-256：

`e833ad923447084165670c25118237b4a2b36c282c12ce40acb2675967923340`

dependency hashes：

- reference metric helper：
  `46859f8aed7c21440fe9e2f4c415c6173a3f6babf3594aff15f700cd393b476f`
- swept-envelope mechanics：
  `a41395ae0eafaa5d4a35b65236f25cbf269293c7c1e82a891b99b9e8e4a94735`

首次正式调用在 fresh metrics 完成后因无 predicted positive 时 F1 helper 返回
undefined 而 fail closed，没有创建报告。四个 sessions 从该时起已 consumed。随后只
把 F1 修正为标准 `2TP/(2TP+FP+FN)`，增加 per-height opportunity readiness 与
dependency hashes；没有换 source、读取数值调门或修改 frozen gates。最终报告使用
同一 consumed inputs。

## 3. Ordered checks

| check | result |
| --- | --- |
| authority + exact source set | pass |
| obstacle known coverage each height/session `>=.10` | pass |
| positive + negative reference opportunity each session/threshold | **fail** |
| primary height reference opportunity at cohort level | pass |
| ground candidate/reference/shared known coverage | pass |
| source and reference ready | **fail** |

每个 session 的 obstacle known coverage 均为 foot `.176–.229`、body
`.294–.426`、head `.332–.503`。失败不是可见性不足，而是一个 session 没有 obstacle
positive case。

ground candidate/reference/shared known coverage 4/4 均越过 `.10/.10/.08`，但
651 个 shared-known cells 全部为 no-risk；fresh cohort 没有 step/drop opportunity。

## 4. 后序 diagnostics，不升级为正式支持

由于 opportunity gate 已失败，下列数值只能用于机制定位：

- 其余三个 session 的 primary F1 delta：
  `+.1670 / +.2831 / +.1455`；
- cohort diagnostic：F1 delta `+.1915`，precision delta `+.3273`，
  recall delta `-.0038`；
- threshold `1/2/4/8` 的 cohort F1 delta：
  `+.1871 / +.1915 / +.1952 / +.2025`；
- primary foot/body/head F1 delta：
  `+.2327 / +.1320 / +.2114`；
- paired candidate-only / baseline-only 在四 thresholds 为
  `254/7, 256/5, 257/4, 261/0`。

这些 diagnostics 与 D1 方向一致，但不能越过 4/4 source gate。

## 5. Governed successor

唯一合理 successor 是 outcome 前冻结的 opportunity-qualified R3.1 challenge
cohort：

1. qualification 只能读取 dense reference opportunity/known，不读取 candidate 或
   baseline output；
2. session eligibility 必须同时包含 obstacle positive/negative opportunity 与
   ground step/drop opportunity；
3. source selection、qualification thresholds 与 claim ceiling 必须在扫描前冻结；
4. 正式 arm comparison 保持 R3 的 stride、reference、primary threshold 与所有 effect
   gates，不降低 4/4，也不删除 ground；
5. 结论只外推到 opportunity-qualified challenge cohort，不冒充自然 prevalence。

R3 四个 sessions 永久 burned。当前不授权 Stage C、student/H2、研究主线、Android、
提醒、默认 App、生产或安全 claim。
