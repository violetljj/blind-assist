# HFTF H1 forward-sector geometry teacher canary result R1

日期：2026-08-01

终态：`H1_GEOMETRY_TEACHER_NOT_EVALUABLE`

证据版本：`HFTF_H1_FORWARD_SECTOR_GEOMETRY_TEACHER_CANARY_R1`

## 1. 结论

R1 的前向扇区合同在全新 cohort 上不再受 R0 的 current observation coverage
blocker 阻断，但没有使完整 future teacher 合同可评价：

- 4/4 source authority、独立 parent sessions、exact frozen session set、usable
  anchors 与 single/multi consistency 均通过；
- camera-forward 6-bin field 的 current known coverage 为
  `22.07%–36.77%`，4/4 均超过冻结 `.15` 门；
- session `00c2a1cd…d4e3` 的 near/far coverage 只有 `3.34%/0%`，低于
  `.10/.10`，因此按第一顺序门终止；
- multi-height 与 future fractions 只作 failure localization，不能形成支持或否定。

由于 field contract 与 source cohort 同时改变，R1 不能把 current coverage 差异单独
归因于 sector；它只确认这个冻结 R1 evidence version 的 current support 可评价。

这不是 R0 threshold rescue：R1 在任何 teacher outcome 前冻结不同的
camera-forward support hypothesis，并使用四个全新 source sessions；R0/R1 的门、
UNKNOWN 与 denominator 没有放宽。

## 2. 冻结与执行身份

- protocol：
  `HFTF_H1_FORWARD_SECTOR_GEOMETRY_TEACHER_CANARY_PROTOCOL_R1_2026-08-01.json`
- protocol SHA-256：
  `c2e38c04d63d7c1473e11da4e5d938592ce90c67c2cf919f948f481ced87f39e`
- protocol commit：`e9226176a01cd91aed8acd9d28c156c9fd10ee77`
- runner commit：`133a515cb3f6ff955865a1fcf2c760dec3a9bb82`
- runner SHA-256：
  `324df2c5c850af48ac604959c0627553945be49f41b621f1c627062cbc0416e3`
- result：
  `artifacts.local/evidence/hftf/h1-forward-sector-geometry-teacher-canary-r1-20260801/teacher_canary.json`
- result SHA-256：
  `49b8a39119983b6c84187fc97b40365b4403e12c420d73a7f31bf73a194ab939`

正式 output path 在执行前不存在，runner 使用 exclusive create，未覆盖旧报告。执行后
四个 R1 sessions 永久 burned。

## 3. 冻结 denominator 结果

| Source session | U | required/horizon | current | near | far | height diagnostic | future diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `00c2a1cd…d4e3` | 18 | 1,944 | 0.220679 | 0.033436 | 0.000000 | 35/648 = 0.054012 | 15/1,944 = 0.007716 |
| `013e2db5…9d3a` | 21 | 2,268 | 0.277778 | 0.190917 | 0.111993 | 0/756 = 0.000000 | 0/2,268 = 0.000000 |
| `01c00b13…1a70` | 21 | 2,268 | 0.297178 | 0.196208 | 0.116402 | 25/756 = 0.033069 | 88/2,268 = 0.038801 |
| `026d78f9…e610` | 21 | 2,268 | 0.367725 | 0.241623 | 0.151235 | 44/756 = 0.058201 | 8/2,268 = 0.003527 |

所有 UNKNOWN/invalid cells 均保留在 required denominator。single-height 与三个
height layers 的 max consistency error 在四个 sessions 均为 `0`。

## 4. 顺序门判定

1. source authority：4/4 通过；
2. unique parent session 与 exact frozen set：通过；
3. usable anchors `>=12`：`18/21/21/21`，通过；
4. single/multi consistency `<=1e-12`：通过；
5. current coverage `>=.15`：4/4 通过，R0 blocker 在该新 evidence version 未复现；
6. near/far coverage `>=.10/.10`：`00c2a1cd` 失败；
7. 因第 6 项失败，multi-height/future 非冗余门不获得裁决资格。

所以机器终态必须为 `H1_GEOMETRY_TEACHER_NOT_EVALUABLE`，不能越级读取某些 session
的 height/future 通过值来宣称 mechanism supported。

## 5. Failure localization

burn 后的 pose 诊断显示，`00c2a1cd` 在 nominal `0.4/0.8 s` 的 camera translation
中位数约为 `3.60/7.14 m`；其他三 sessions 约为
`0.74–0.93/1.49–1.87 m`。固定 anchor-centric 0–8 m field 在这种快速前移下，future
observation 很可能已越过大量 anchor cells，使 probes 落到 future camera 后方或画面
外。

这是解释 future known collapse 的机制假设，不是已确认因果结论，也不授权在 R1 上
扩大 sector、缩短 horizon、改 coverage 门或选择性删除该 source。下一 evidence
version 若继续，必须在新 sessions 上预冻结 ego-motion-aware/temporal-fusion support
合同，并把一般 source authority 与 dynamic-opportunity eligibility 分开。

## 6. 权限与停止

- R1 source sessions 不再作为 fresh validation；
- H2 causal student 仍未授权；
- geometry proxy 仍不是人体、collision、event 或 safety truth；
- HFTF 仍是独立候选支线，研究主线、Android、提醒、默认 App、生产与安全权限均未改变。
