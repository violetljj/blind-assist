# HFTF Stage B split-source validation result R4

日期：2026-08-01

终态：`R4_STAGE_B_SPLIT_SOURCE_TEACHER_MECHANICS_SUPPORTED`

## 1. 结论

R4 的两个冻结 source roles 均通过：

1. 新 SANPO-Synthetic obstacle-qualified challenge cohort 上，swept human-envelope
   candidate 相对同点 angular-support baseline 通过全部 R3 effect gates；
2. deterministic analytic metric terrain 上，five-section ground continuity 对
   rise/drop/localized bump/pit 与 flat/ramp/subthreshold/UNKNOWN 通过全部 controlled
   mechanics gates。

因此 Stage B 现在获得的是
`SPLIT_SOURCE_DEVELOPMENT_TEACHER_MECHANICS_ONLY` 支持。它回答“人体包络标签 mechanics
是否值得继续”，不回答自然台阶 prevalence、真实 RGB student 效果、助盲事件效用或
安全性。

按照冻结协议，唯一新权限是：

`FREEZE_STAGE_C_SOURCE_FEASIBILITY_CONTRACT_ONLY`

Stage C execution、student training、研究主线、Android、提醒、默认 App、生产与安全
claim 仍未授权。

## 2. 报告绑定

| report | SHA-256 |
| --- | --- |
| terrain component | `3994b2df73978f7053e2e809eecd3645501689821d0b791818a8729848c11d8d` |
| obstacle inventory plan | `a3a0c3e7fe132397ae18623041672adcc353df54075cf1f841a1230538f58059` |
| obstacle cohort lock | `1e25b06256279d065af072c760b660f148bac6a70298600a4ee5c463ab75c48e` |
| obstacle comparison | `58c2b3d7784e66e9d0ae1be530e71dbcc3e8a05eddc076501e06e454d1b044f4` |
| joint R4 result | `cc7adb2b08ceb1ef4542a0c0c86957e4bb20fc6f50f1d01e31b22f66f1177453` |

joint report：

`artifacts.local/evidence/hftf/r4-split-source-result-20260801/r4_result.json`

实现先于 arm outcome 提交为 `7c91849`。实现 hashes：

- analytic terrain runner：
  `76f29509c6a3595c61bbef7358500e67c1f46fffb6c839505c3b032e1146ac51`
- obstacle comparison runner：
  `dc634af3877c071ccd1451e013d077ad11582cab3ec3bca78a1b39150e494fc5`
- joint aggregator：
  `b5b3dfbb4fe2a3c949faa9c42790d2a1c51be16d73c99a74e9eaa5a498506bf0`

## 3. Obstacle role

R4 排除 R0–R3.1 共 56 个 outcome-open sessions 后，按 official train 完整 session
ID 字典序冻结最多 12 个 inventory candidates。前四位全部通过 reference-only
obstacle opportunity qualification，因此按合同立即停止，没有读取第 5–12 位。

四个 selected sessions：

`11363093… / 11838cbb… / 11c4b307… / 1278aa62…`

primary threshold 2 的 cohort：

| metric | candidate | baseline | delta |
| --- | ---: | ---: | ---: |
| precision | `.99749` | `.61957` | `+.37792` |
| recall | `.97783` | `.98276` | `-.00493` |
| F1 | `.98756` | `.76000` | `+.22756` |

4/4 session F1 delta：

`+.16437 / +.17683 / +.26190 / +.36842`

foot/body/head F1 delta：

`+.30103 / +.14230 / +.27215`

四个 sensitivity thresholds 的 F1 与 paired-correctness directions 均通过；
primary candidate-only/baseline-only correct 为 `249/7`。这支持 conditional
challenge cohort 上的 obstacle-envelope geometry-proxy gain，不代表自然 prevalence
或用户效果。

## 4. Terrain role

42 个 exact profiles：

- 20 risk：hazardous rise/drop、localized bump/pit；
- 16 no-risk：flat、traversable ramp、subthreshold rise/drop；
- 6 occluded UNKNOWN。

candidate 在 36 个 known cases 上：

`precision=recall=F1=specificity=1.0`

6/6 occluded cases 全部 abstain，UNKNOWN→SAFE 为 0。最佳 baseline 是
`endpoint_elevation_delta`，F1 `.75`；candidate F1 delta `+.25`。

这是由解析 profile 与相同物理阈值构成的 controlled mechanics benchmark，不能证明
真实深度噪声、真实台阶/坑洼分布或真实行走事件性能。

## 5. 下一门

下一步只能冻结 Stage C source-feasibility contract，先回答：

- 是否存在 history RGB + metric geometry + pose/future frames 的 source；
- 是否能在不读取 future outcome 选样的前提下生成 current 与短期 future layered
  teacher labels；
- train/dev/held-out source 是否 parent-session disjoint；
- current-only、single-frame 与 temporal student 的比较能否在同一标签/UNKNOWN
  denominator 下执行。

在该来源门通过和正式 Stage C protocol 冻结前，不训练时序 student。
