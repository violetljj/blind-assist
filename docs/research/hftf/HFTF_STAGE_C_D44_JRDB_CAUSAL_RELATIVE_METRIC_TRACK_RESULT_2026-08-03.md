# HFTF Stage C D44 JRDB causal relative metric-track result

日期：2026-08-03（Asia/Hong_Kong）

## 结论

终态：

`D44_JRDB_CAUSAL_RELATIVE_METRIC_TRACK_SUPPORTED_DEVELOPMENT_ONLY`

冻结 7/7 support gates 全部通过。仅用 same-target 7-frame relative metric center
history 做 OLS 外推，相对 current-static：

- mean horizontal error：`0.80935 -> 0.35324 m`，降低 `56.36%`
- median horizontal error：`0.74938 -> 0.13948 m`，降低 `81.39%`
- horizontal-error better fraction：`79.787%`
- mean absolute range error：`0.56035 -> 0.11714 m`，降低 `79.10%`
- mean absolute bearing error：`5.30386° -> 2.42908°`，降低 `54.20%`
- 四个 sequences 全部改善

D44 建立 causal relative metric-track information ceiling。它不证明当前手机已经
拥有可靠 metric depth。

## 与 D42 的关键比较

| candidate | mean horizontal error | median horizontal error |
|---|---:|---:|
| D42 ego+person world teacher | 0.34757 m | 0.14080 m |
| D44 relative metric track | 0.35324 m | 0.13948 m |

D44 在不显式预测 ego pose、不转换到 world frame的情况下，几乎达到 D42 完整
teacher。这说明对当前一秒 horizon：

> same-target relative metric history 已吸收完成 future geometry 所需的大部分
> ego+object combined motion。

因此下一工程瓶颈不是更复杂的 world model，而是端侧能否稳定产生该 relative metric
measurement。

## sequence 分解

| sequence | opportunities | mean reduction | median reduction | better fraction | range reduction | bearing reduction |
|---|---:|---:|---:|---:|---:|---:|
| Clark Center | 1,100 | 47.71% | 60.32% | 83.00% | 80.00% | 44.84% |
| Gates Basement | 822 | 77.28% | 84.75% | 82.48% | 81.68% | 71.77% |
| Meyer Green | 161 | 54.96% | 81.58% | 80.12% | 67.55% | 61.01% |
| STLC 111 | 1,301 | 67.70% | 24.28% | 75.33% | 74.86% | 63.77% |

正结果不是单一 source、单一距离或 pooled aggregation 驱动。

## 因果与实现边界

- exact 3,384 opportunities / 53 identities
- history：current + previous 6 frames
- candidate：timestamp-aware OLS of `center_base_link_m` x/y/z
- target timestamp：exact `+15 frames`
- prediction 不读取 future
- 无速度裁剪、平滑、history/horizon、identity 或 sequence search

该 center history 是 source-native metric oracle，不是 detector-only runtime output。

## 与 D43.1 的断点

- D43.1：2D box state/slopes 无法跨 sequence 学出 metric residual，actual error
  恶化 `37.95%`
- D44：直接 relative metric history 使 actual geometry error 降低 `56.36%`

两者共同说明缺口是 metric measurement，不是 target 缺历史信息，也不是 Ridge
容量不足。因此不得回到 D43.1 加非线性；下一步应评价一个真实端侧 depth source。

## 下一授权

D44 只授权冻结一个 source-only shadow canary：

1. 选择一个 phone-causal metric-depth source；
2. 对 current person box/target 输出 relative depth measurement + quality +
   timestamp；
3. 与同一 target 的 7-frame metric history solver 对接；
4. 先评价 measurement coverage/error/latency，不读取 event/alert outcome；
5. source 未通过前不接入 production decision seam。

可选 source 的优先级由设备能力决定：硬件/AR depth 优先于未校准 relative monocular
depth；不能把 D44 oracle 直接当成 App 已实现。

## 复现

- D42/D43.1/D44 focused tests：5 PASS
- report 连续重建 SHA 稳定：
  `c96c37fca85f8a52fb37d372a8290a564982e241352e8d7a173e4b5a4ad03f09`
- report size：7,421 bytes

## claim ceiling

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

D44 不建立 runtime metric depth、event utility、Android runtime、主线、产品或安全
主张，也不覆盖 D35 device gate。
