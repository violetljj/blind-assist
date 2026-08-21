# P0-D3 targeted set-valued one-shot closure protocol V1

状态：`PROTOCOL_V1_FROZEN / ONE_FIXED_BATCH / SIX_PARENTS / NO_REPLACEMENT / NO_SECOND_BATCH / NO_MODEL_FIT / NO_SKY / NO_SCIENTIFIC_VERDICT`

日期：2026-08-21

## 目的与止损线

P0-D3 只给 P0-D2 数据前门最后一次固定机会。它不是继续建设 calibration dataset，也不保证所选 venue
最终可解析。一次性选择并物化 6 个 source-metadata 先验上较可能具有多个公共入口的全新 venue parents；完成一次
score-blind full-frame review 后立即终止数据采集。

只有两个合法终态：

1. 合并 consumed Development 后满足 `SET_VALUED>=4` 且 resolvable venue parents `>=12`：冻结 cohort，才允许
   按 P0-D2 V1 物化 runtime features 并运行 Logistic/Conformal；
2. 固定 6-parent batch 后仍不满足：记录
   `CURRENT_PUBLIC_DATA_SOURCE_INSUFFICIENT_FOR_CALIBRATION_DESIGN`，永久关闭这套 calibration frontdoor；不得补采
   第 7 个 parent，不得替换失败、无覆盖或 review 后为 `AMBIGUOUS` 的 parent。现有 consumed Development 仍可用于
   低 claim 的 rule-based、feature-usefulness、ranking、ambiguity detection 或 cheap discovery。

## 冻结 geographic/source scope

- Overture release：`2026-08-19.0`；
- 固定 source slices：Antwerp、Bruges、Brussels、Ghent、Leuven、Mechelen；
- Mapillary Graph API：以 venue POI 经纬度做 radius query，`radius=50 m / limit=100`；
- 只使用已存在的 Overture Places/Buildings source metadata 做 parent selection；Mapillary 图像只在 roster freeze 后获取；
- 不因城市、parent 或 frame 的实际 yield、像素内容或 truth 追加城市、扩大 bbox、替换 parent 或启动第二批。

## Outcome-blind parent selection

允许的 selection evidence 仅为：

- Overture `basic_category` 与 `taxonomy.hierarchy/primary`（旧 `categories.primary` 只作兼容 fallback）；
- Overture place confidence、名称与 POI 坐标；
- place point 到单一 Overture building 的既有 source crosswalk candidate；
- building footprint 的确定性 metric area、perimeter 与最长边；
- 已消费 roster/cohort 的 building/place/name 排除集合。

目标 venue families 只包括 mall、department store、supermarket、hospital、college/university、train/metro/subway station、
hotel、civic/government building、cinema/theatre、event/convention venue、stadium/arena。最小 footprint 为 `350 m²`，
最小最长边为 `20 m`；每城市最多 2 个 parent。固定排序为 venue family priority、footprint area、longest edge、
place confidence、city/name/place/building ID。每 building 最多一个 target venue。

明确禁止读取或使用：

- OSM `entrance=*`、入口 anchor 数或类型；
- Mapillary RGB、图像 yield、camera/sequence metadata（roster freeze 前）；
- Grounding DINO/YOLO/任何 entrance detector、proposal、confidence 或 candidate count；
- Terra/V3/Brain 输出、review resolution、acceptable regions、valid target 或 frontdoor score。

## Fixed Mapillary materialization

对 6 个冻结 parent 各执行一次 radius query。frame selection 只能使用 camera metadata、相机到 venue 的距离/朝向、
sequence ID、capture time 与 frame ID；每 parent 最多保留 4 帧，并优先 sequence diversity 与至少 3 m camera spacing。
没有合格 frame 的 parent保留为 acquisition failure，不可替换。保存原始响应摘要、request/selection policy、frame IDs、
source hashes 与图像 hashes；token 不写入 artifact。

本轮不运行 detector、Terra、V3、Logistic、Conformal、Sky 或其他拟合。图像只进入一次 score-blind full-frame truth
review，按 goal `进入该 venue` 标注：

- `UNIQUE`：一个合法公共入口；
- `SET_VALUED`：同一 venue 至少两个对目标等价合法的公共入口；
- `AMBIGUOUS`：像素与 source evidence 不能确定。

正门加员工门/紧急出口、不同商家门、门加窗均不得构成 `SET_VALUED`。以完整 6-parent batch 为审计单位，禁止看到
单个 `SET_VALUED` 后提前停止。

Claim ceiling：`CONSUMED_DEVELOPMENT_ONE_SHOT_DATA_CLOSURE_ONLY_NO_MODEL_OR_CALIBRATION_PERFORMANCE_CLAIM`。
