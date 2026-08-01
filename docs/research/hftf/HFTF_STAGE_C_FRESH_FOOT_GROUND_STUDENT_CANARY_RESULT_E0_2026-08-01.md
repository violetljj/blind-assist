# HFTF Stage C fresh foot-ground student canary result E0

日期：2026-08-01

终态：`E0_FRESH_TEACHER_MECHANICS_NOT_EVALUABLE`

## 1. 结论

E0 在 student training 前按顺序门停止。六条 fresh source 的 exact media transport、
ground-plane recovery、anchor eligibility 和 role risk/no-risk opportunity 均足够；
失败精确定位在 `.8 s` candidate known-direction coverage：冻结门要求每 source/horizon
`>=.70`，六条中四条的 `.8 s` 为 `.6015–.6857`。

不得降低 E0 的 `.70` 门、删除失败 source、换 heldout 或在本 cohort 上训练后把结果
包装为原 E0 成功。`.8 s` formulation 在该 evidence version 关闭。

## 2. 报告绑定

- report：
  `artifacts.local/evidence/hftf/stage-c-e0-teacher-opportunity-20260801/teacher_opportunity.json`
- SHA-256：
  `770928a2e44776703f23185e2152326147e580256c25d2a76b92bdfbe3277e6b`
- protocol commit：`e61f2d4`
- opportunity runner commit：`adcef67`

双运行 payload byte-exact。报告没有持久化完整 teacher corpus，也没有训练或读取
student output。

## 3. Source mechanics

| role | source | anchors | plane known | speed eligible | known `.4 s` | known `.8 s` | result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | `2024_10_12...` | 105 | 1.000 | 1.000 | `.7505` | `.6857` | `.8 s` fail |
| train | `2024_12_27...` | 130 | `.9821` | 1.000 | `.7077` | `.6015` | `.8 s` fail |
| train | `2024_12_26...` | 139 | `.9976` | `.9928` | `.7855` | `.6812` | `.8 s` fail |
| train | `2025_01_03...` | 140 | `.9857` | 1.000 | `.8071` | `.7171` | pass |
| dev | `2024_11_15...` | 249 | 1.000 | 1.000 | `.7414` | `.6554` | `.8 s` fail |
| heldout | `2024_09_26...` | 120 | 1.000 | 1.000 | `.8167` | `.7517` | pass |

`.4 s` 在 6/6 source 均过原 `.70` 门；`.8 s` 仅 2/6 通过。这是 successor
hypothesis 的依据，不是对 E0 的事后修复。

## 4. Role opportunity

| role | risk cells | physical risk anchors | risk sources | directions | known no-risk |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 27 | 22 | 4 | 5 | 3,657 |
| dev | 8 | 4 | 1 | 4 | 1,731 |
| heldout | 36 | 19 | 1 | 5 | 905 |

三角色 opportunity 门全部通过，因此 blocker 不是缺少 risk proxy。所有 risk cells
仍是 geometry proxy，不是真实危险。

## 5. 唯一允许的 successor

可以另行冻结 `.4 s`-only E0.1：

- 原四条 train 作为已消费训练来源复用，不获得 fresh evidence credit；
- 原 dev/heldout 永久 burned，不得继续用于新 formulation 的选择或评价；
- dev/heldout 必须从未打开媒体的 healthy inventory 重新按 outcome-independent
  规则选择；
- model、阈值、success margin 必须在新 dev/heldout 媒体读取前冻结。

E0 当前不授权 teacher corpus、student training、完整 HFTF、研究主线、Android/App
或安全/产品 claim。
