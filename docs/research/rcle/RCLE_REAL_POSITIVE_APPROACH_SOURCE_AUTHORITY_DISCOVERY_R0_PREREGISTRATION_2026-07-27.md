# RCLE real positive-approach source-authority Discovery R0 preregistration

日期：2026-07-27

状态：

`FROZEN_BEFORE_EXTERNAL_SOURCE_REVIEW / METADATA_ONLY`

## 唯一问题

只比较用户指定的三个不同数据族：

1. CID-SIMS，用户提出的 `Floor3`；
2. CoRBS，用户提出的 `E3`；
3. KITTI-360，一个待绑定的车载 sequence，仅允许压力候选角色。

本 Discovery 只审计精确 sequence/run/archive identity、适用许可、公开大小、
稳定官方入口、官方 checksum 是否存在，以及相机前向轴和
pose/depth/static-geometry 坐标链。它不读取 archive body、RGB、depth、pose
样本或任何 geometry/algorithm outcome。

ETH3D `sofa_3` R1 保持
`HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`；不修补、不重跑，
整个 ETH3D sofa scene family 不进入本版候选。

## 结果前冻结的选择规则

### 硬门

CID-SIMS `Floor3` 只有同时满足下列条件才成为本版唯一下一候选：

1. 官方或作者权威来源给出可唯一解析的 dataset、sequence/run 与 archive
   identity，不能只靠二手名称；
2. 有明确适用于该数据或下载物的许可/条款，且不禁止本项目的隔离内部研究；
3. 有稳定、无需绕过登录/付费/访问控制的官方入口，并能在不请求 archive
   body 的前提下绑定其精确下载对象；
4. metadata/论文/官方工具文档足以闭合相机 forward-axis、pose frame、
   depth frame、static-geometry frame 及其变换方向；未知项不得靠惯例猜测；
5. 仓库级 access audit 未发现该 exact sequence/capture 已读取
   geometry、RGB、claim-relevant outcome 或影响过当前选择；
6. 它不是 synthetic-only，也不属于 R1 已烧掉或排除的数据族。

官方 checksum 若发布，必须记录算法和值；若未发布，只能写
`OFFICIAL_CHECKSUM_NOT_PUBLISHED_OR_NOT_VERIFIED`。缺少官方 checksum 不得用
landing-page/source-descriptor hash 冒充，但可以在后继 R2 通过
claim-before-GET、single response artifact SHA-256 和冻结 container contract
补齐内容身份；这不豁免上述六个硬门。

### 决策顺序

- 若 CID-SIMS `Floor3` 六个硬门全部闭合：
  `SELECT_CID_SIMS_FLOOR3_AS_SOLE_NEXT_R2_CANDIDATE`。
- 若任一硬门不闭合：
  `HOLD_CID_SIMS_SOURCE_AUTHORITY_INCOMPLETE / NO_R2_CANDIDATE_SELECTED`。
  本 Discovery 不得改选 CoRBS；只能在新的协议中优先考虑 CoRBS `E3`。
- KITTI-360 无论 metadata 多完整，都只能是
  `VEHICLE_DOMAIN_STRESS_CANDIDATE`，不能在本版替代室内候选。

CoRBS 与 KITTI-360 的审计结论只形成后续来源地图，不构成自动 successor
authority。任何后继 R2 只能绑定一个预先选定的数据族、一个 exact
sequence/run 和一个 canonical archive；失败后不得 retry、mirror、repack、
replacement 或临时换源。

## 证据等级与缺失处理

证据优先级为：

`official dataset/download/license/calibration page or official repository`
`>` `dataset paper or author-hosted paper`
`>` `publisher metadata`
`>` `third-party implementation or catalog`。

二手来源只能帮助定位，不能单独闭合许可、archive identity、checksum 或
坐标链。重定向、页面生成的临时 URL 和文件名猜测不等于稳定官方入口。

每个未知字段必须显式记录为 `NOT_PUBLISHED_OR_NOT_VERIFIED`；不能把文件可下载、
论文提及、常见 OpenCV/KITTI 约定或相邻 sequence 的信息外推到目标 archive。

## 权限边界

本版最大权限是：

`METADATA_ONLY_SOURCE_AUTHORITY_DISCOVERY / UNIQUE_NEXT_CANDIDATE_LOCK`

它不授权 payload access、geometry、RGB algorithm、performance qualification、
confirmation、Kill Gate B、Replay、Android、human、safety 或 production。
