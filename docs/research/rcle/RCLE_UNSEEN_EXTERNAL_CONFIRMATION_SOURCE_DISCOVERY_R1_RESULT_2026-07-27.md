# RCLE unseen external confirmation Source Discovery R1 result

日期：2026-07-27

## 终态

`EXTERNAL_COHORT_NOT_EVALUABLE / VALID`

这不是 RGB algorithm failure，也不是 source-authority failure。两个 source family
均已通过 authority；失败点是 reviewed candidate lock 下的 OpenLORIS
geometry-only transport 不可满足。

## Candidate lock 与 review

- candidate lock：
  [RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R1_CANDIDATE_LOCK_2026-07-27.json](RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R1_CANDIDATE_LOCK_2026-07-27.json)
- lock SHA-256：
  `c1a0ea53dc698b1f12107db9951f7ecf88def7aa4aed2d2edd0e186093cb5a3c`
- 独立 review：
  [CANDIDATE_LOCK_REVIEW_PASS](RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R1_CANDIDATE_LOCK_REVIEW_RESULT_2026-07-27.md)
- review receipt SHA-256：
  `fbc727609a4c760c1f038e3941250c9f2319b8e6468add8eadeae25b4fcf8585`
- review errors：`[]`
- review 前 payload root bytes：`0`

lock 只含两个来源：

1. OpenLORIS corridor：`corridor1-1 / corridor1-2`；
2. MultiScan：`scene_00000_00 / scene_00000_01`。

原 R0 的 10 秒非重叠网格、cadence、coverage、positive/below 角色门与每来源
`1 positive + 1 below` 均未改变。

## Post-lock geometry-only transport preflight

candidate-lock review PASS 后，先复用 immutable 7z header 做 payload-before-read
transport preflight。只读 header bytes `495,819`；geometry payload、RGB member、
完整 archive 与算法 bytes 均为 `0`。

OpenLORIS archive 是 solid 7z，aligned-depth、raw depth、color 与 fisheye 在同一
solid pack stream 中交错。solid stream 不能从任意 member 独立开始解码；要到达一个
目标 depth member，解码器必须从所在 folder 开始顺序处理此前的 member bytes。

固定网格结果：

| capture | complete/cadence-eligible windows | geometry solid folders | RGB-free authorized windows |
| --- | ---: | ---: | ---: |
| `corridor1-1` | `28 / 28` | `4` | `0` |
| `corridor1-2` | `11 / 11` | `2` | `0` |

每个 cadence-eligible window 在到达最后一个所需 aligned-depth member 前至少经过
`299` 个 color members；因此没有一个窗符合 reviewed lock 的
“不访问任何 RGB/color member bytes”防火墙。不能把“解码后丢弃 RGB”偷换成
“没有 RGB access”，也不能退化为完整 archive download。

transport preflight：

- decision：
  `OPENLORIS_GEOMETRY_ONLY_TRANSPORT_NOT_EVALUABLE`
- status：`PASS`
- receipt SHA-256：
  `06c55a12ddccf4dd8beb25c9769baa88f96c18dc30e32280a35e1d60a6d559d1`

独立 offline validator 不 import preflight producer，重新检查 exact captures、
candidate-lock binding、solid/co-pack evidence、39 个固定窗、cadence、逐窗
preceding-color proof、range accounting 与 firewall：

- decision：
  `OPENLORIS_GEOMETRY_ONLY_TRANSPORT_NOT_EVALUABLE_VALID`
- errors：`[]`
- validation receipt SHA-256：
  `b03d5b354c2df6fc610d3419f113be800bc14059886d319126b9eb92efcddf98`

## Stop decision

candidate order 的第一来源已触发冻结
`transport_or_sync_failure -> EXTERNAL_COHORT_NOT_EVALUABLE`。因此：

- 不下载 OpenLORIS solid payload；
- 不继续取得 MultiScan depth payload；
- 不运行 geometry formula 或选窗；
- 不扩候选、不换来源、不改顺序、不降角色/coverage/cadence 门；
- 不读取或运行 RGB；
- 不启动 Android。

geometry role-complete independent source count 为 `0/2`，selected windows 为空。
外部 cohort 继续保持 `EXTERNAL_COHORT_NOT_EVALUABLE`。

## Evidence boundary

本结果证明的是 transport/firewall incompatibility，不证明 OpenLORIS 不含 positive
或 below motion，也不评价 MultiScan 的 motion roles。若未来允许 solid block 内部
经过但不保存/查看 RGB bytes，必须由用户另立新协议版本；当前 R1 不允许把这种扩权
作为“实现细节”静默完成。
