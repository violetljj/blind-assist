# HFTF H1 geometry teacher canary result R0

日期：2026-08-01

workflow：`DEVELOPMENT_STANDARD`

协议状态：`FROZEN_BEFORE_OUTCOME`

终态：`H1_GEOMETRY_TEACHER_NOT_EVALUABLE`

下一步：`STOP_OR_REFORMULATE_FAILED_H1_EVIDENCE_VERSION`

主线/App：`UNCHANGED / UNCHANGED`

## 结论

H1 R0 没有支持也没有否定 multi-height/future geometry mechanism；它在更早的
required-cell known coverage 门失败，因此两个 representation claim 都不可评价。

source authority、四个独立 session、canonical transform、local-ground proxy、
usable-anchor 数与 single/multi consistency 全部通过。失败来自 R0 精确定义的
`360° × 9 probes × anchor-centric future` 可观测性合同：current known coverage
只有 `5.62%–9.68%`，低于每 session `15%` 门；near 为
`0.54%–6.13%`，far 为 `0%–4.25%`，也低于 `10%` 门。UNKNOWN 全部保留在冻结分母，
没有被当作 safe 或从分母删除。

## 冻结执行

正式 runner 已在 commit `cf72847a509cfee818600cfb9c90fde0d1a7b5ed` 提交并推送后
运行一次。运行前已经冻结：

- 4 个完整 source session IDs 与 authority/manifest/spec/pose hashes；
- 24 theta × 6 distance × 3 horizon × 3 height field；
- `U=current+near+far all bound`；
- coverage `|U|×432`、height `|U|×144`、future `|U|×432` denominator；
- 9-probe known、semantic-0 UNKNOWN、anchor-centric future、risk 与终点顺序。

正式运行后未修改 bin、threshold、probe、denominator、anchor 或 terminal。

## 四 session 结果

| session | usable anchors | required cells/horizon | current known | near known | far known | height fraction | future fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e1ae36e0…de856` | 18 | 7776 | `.066872` | `.005401` | `.000000` | `.007330` | `.003472` |
| `001217c6…910a` | 20 | 8640 | `.086806` | `.061343` | `.042477` | `.010764` | `.006019` |
| `0099b54c…864c` | 18 | 7776 | `.056199` | `.006173` | `.005401` | `.009259` | `.000900` |
| `00bdf8ce…5896` | 18 | 7776 | `.096836` | `.007330` | `.000000` | `.023148` | `.000129` |

全部 session：

- authority validation：pass；
- usable anchors：`18/20/18/18`，均 `>=12`；
- single-height vs `max(foot,body,head)` error：`0`；
- source sessions：4/4 独立且与 frozen set 精确相同。

height/future fractions 仅保留作 failure localization。因为 known coverage 的第一顺序
门已失败，它们不能被解释为 representation 不支持，也不能用其中最好 session 救援。

## 证据

正式报告：

`artifacts.local/evidence/hftf/h1-geometry-teacher-canary-r0-20260801/teacher_canary.json`

SHA-256：

`53261fd930c9a1ffc1de03468d974a1e16624383fb12e241da8b26df0cf7809e`

正式报告记录 protocol、runner、authority dependency、四个 authority reports 和
manifest hashes；runner 还逐帧复算实际消费的 depth/mask bytes。

## 失败定位

R0 同时要求：

1. action-agnostic 360° field；
2. 每个 cell 9 probes 至少 5 个在单一 observation 中可见且 depth 到达；
3. future 仍在 anchor field 中评价，不随 future camera 重新居中；
4. UNKNOWN 留在全 required-cell denominator。

这些规则诚实地暴露了单个胸前相机对后方、强转头/平移后的 anchor cells 与边界角点的
可见性上限。R0 失败首先是 field support/observation contract 问题，不是 metric
transform、source mapping 或 local-ground proxy 问题。

## 后续规则

不得在这四个已消费 sessions 上降低 `.15/.10/.10` coverage 门、把 5/9 probes 改成
center-only、删除 UNKNOWN、只报告前方最好片段或重新解释 height/future fractions。

允许另立 H1 R1 的条件：

- 提出不同且有任务含义的 field support，例如预冻结前方 locomotion sector，或严格
  因果的 multi-view observed support；
- 使用新的、未参与 R0 outcome 的 source sessions；
- 在读取 R1 outcome 前重新冻结完整 session hashes、field、known、denominator 与门；
- R0 报告永久保留为 burned negative/diagnostic evidence。

## 未获得

- `GEOMETRY_PROXY_MECHANISM_SUPPORTED`；
- multi-height 或 future 非冗余的有效否定/肯定；
- H2 causal RGB student 授权；
- human collision/event/safety truth；
- 研究主线、Android、提醒、默认 App、生产或安全权限。
