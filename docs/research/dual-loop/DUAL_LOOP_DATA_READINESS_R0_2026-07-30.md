# DUAL_LOOP_DATA_READINESS_R0

状态：`F-1A_DATA_AUDIT_ONLY / HOLD_DATA / DATA_PROTOCOL_VALID`

协议：`BLINDASSIST_DUAL_LOOP_PHASE_MINUS1_ADMISSION_R0 / F-1A`

审计时间：2026-07-30（Asia/Hong_Kong）

## 1. 访问前冻结

本轮只盘点既有连续 RGB、capture/session 身份、独立事件真值、负场景、时间/帧序和
`LEFT / CENTER / RIGHT` 区域标签。以下边界在读取候选数据内容前冻结：

- 不读取或运行 YOLO、Sparse LK、RCLE、风险分数、提醒或双环候选输出；
- 不连接 Android，不采集或下载新数据，不补标，不恢复 RCLE，不实现融合器；
- 只接受源文件/压缩包/视频/连续图像、来源与许可元数据、capture/session 清单、
  独立于上述候选输出的事件/窗口真值和既有自动化复核记录；
- 不以“YOLO 未检测到”、光流/几何响应或既有算法结果定义正例、负例、区域或角色；
- 连续 RGB 必须能由视频时间戳、源原生时间戳或无缺口的确定性帧序重建；抽样图片、
  拼接截图或无法证明顺序的帧集合不算连续 RGB；
- capture session 的独立性按源原生采集边界判定；同一 capture 的多个 clip、window、
  camera、pair、frame 或 cell 不得重复计为独立 session；
- 正事件必须有 `event_id`、正负属性、事件类型、`onset_interval`、
  `alertable_start_interval`、`end_or_clear_interval`、区域和独立真值来源；
- 负窗口必须明确 `should_alert=false`，并能归入转头/近原地旋转、正常步行抖动、
  横向经过/远离、静止、低纹理/模糊/遮挡之一；一个窗口只计入一个主类别；
- 身份相同或时间区间重叠的事件/窗口去重后再计数；来源或标签矛盾时不择优回填，
  相关项隔离并记为缺口；
- outcome access 必须按历史 ledger 如实记录；历史上用于同一 Sparse LK / 区域 / TTL /
  融合命题选择规则的 session 不得充当 decision。本轮自身只新增
  `CONTENT_INSPECTED_FOR_F1A`，不读取候选输出，也不把任何 session 新增标为
  `OUTPUT_INSPECTED` 或 `TUNED_ON`。

最小门槛固定为：至少 3 个独立 capture session、6 个正事件（至少分布在 2 个
session）、12 个负窗口、至少 4 类负场景且每类至少 2 个；全部纳入项均有连续 RGB、
可靠顺序、事件/窗口区间、区域和真值来源；冻结 1 个 development session 与 2 个
含正事件、未用于规则选择的 decision session。

角色分配只允许基于来源身份、连续性和独立真值覆盖，不得查看候选输出。若合格 session
超过最低数量，先按 `source_id / session_id / content identity` 稳定排序；优先把首个
同时含正负真值的 session 冻结为 development，其后前两个含正事件的独立 session
冻结为 decision。其余只列为 reserve，不进入 R0 最小分母。

合同 5.3 的“正向方向必须分别在两个 decision session 中成立”不能在 F-1A 合法判断，
因为合同 5.4 同时禁止本轮读取候选输出。本 R0 将其解释为 F-1B 的后续科学门；F-1A
只核验并冻结两个各含正事件且未用于规则选择的独立 decision session。若无法做到，
仍在本轮以 `HOLD_DATA` 或 `INVALID` 收口，不查看输出回救。

终点固定为：

- `READY`：上述全部门槛同时满足，并写出 hash/stable-identity 绑定的数据与角色表；
- `HOLD_DATA`：现有数据或标签不足，但已知身份未被破坏；只给一个有界补充方案；
- `INVALID`：身份、时间顺序、角色隔离或真值来源不可追溯，当前 evidence version
  不可签署。

## 2. 审计结果

### 2.1 证据绑定

本轮以文件内容身份而非目录名计数。主要审计输入如下：

| 输入 | SHA-256 | 角色 |
| --- | --- | --- |
| 阶段−1准入合同 R0 | `5814cb64...75ce23d8` | 冻结门槛 |
| SANPO event-labeled manifest | `3d7168ac...ed19d217` | 既有事件/负段真值候选 |
| SANPO stairs manifest | `ae6845c2...1ddb88a1` | 既有负序列与冲突复核 |
| SANPO public manifest | `cf38bffd...ef3d82f0` | session/派生身份交叉检查 |
| legacy LILOCBench truth windows | `383c1a8a...ce1445b5` | 只读缺口对照，不进入角色 |
| CrowdBot `11-51-18` frames ledger | `5b99e32b...051cc748` | outcome-blind 连续 RGB |
| CrowdBot `11-55-00` frames ledger | `88115772...3a39073` | outcome-blind 连续 RGB |

独立复核只检查数据粒度、身份、连续性、真值字段、去重、历史访问状态和本合同门槛；
未读取任何 YOLO、Sparse LK、RCLE、风险、提醒或双环候选结果。

### 2.2 连续 RGB 与 session 身份

存在可验证的连续 RGB，但尚不能与合规事件真值闭合：

| source / session | 连续 RGB 与身份 | outcome access | F-1A 角色 |
| --- | --- | --- | --- |
| CrowdBot `defaced_2021-03-27-11-51-18_filtered_lidar_odom` | `2239` 张 PNG，`000000..002238`，约 `177.041 s`；源时间戳严格递增，0 缺文件、0 重复 ID；原 bag SHA `c1538625...2d1760` | `sealed_holdout_input_not_truth_not_candidate_score`；`candidate_outputs_executed=false` | 未冻结；缺独立事件/负窗 truth join |
| CrowdBot `defaced_2021-03-27-11-55-00_filtered_lidar_odom` | `2183` 张 PNG，`000000..002182`，约 `169.289 s`；源时间戳严格递增，0 缺文件、0 重复 ID；原 bag SHA `55371f10...9778f5` | 同上 | 未冻结；缺独立事件/负窗 truth join |
| SANPO `-5O...`、`GxMb...`、`i2j...`、`bW85...` | 四个不同 source-native session；各有 `30` 帧 `10 fps` 连续派生序列、逐帧身份和源帧/源时间映射 | 历史内容/标签已查看；本轮未访问候选输出 | 仅作真值缺口审计；不冻结角色 |

另有多个本地完整公开视频和 `804` 帧连续 UB VisioGeoLoc 序列，但其既有标签是
training-only silver、稀疏 episode 或缺少 session/event 合同字段，不能用资产数量替代
可评价 session。ADVIO、TUM 和全部 RCLE lineage 均按用户边界排除；同源视频、
clip/reencode 和同一 capture 的多份派生不重复计数。

### 2.3 事件与负窗口

严格按自然事件/连续负段去重后，现有可追溯候选只有：

| 类型 | 候选数量 | session | 严格合规数量 |
| --- | ---: | --- | ---: |
| 正事件 | `2` | `GxMb...`、`i2j...` 各 1 个 | `0` |
| 连续 `should_alert=false` 区间 | `4` | 两个整段负序列 + 两个正事件后的 false 尾段 | `0` |
| 独立整段负序列 | `2` | `-5O...`、`bW85...` | `0` |

两个正事件候选均为 `CENTER`：

- `GxMb... / center_obstacle`：`f0–19=true`，`f20–29=false`；
- `i2j... / front_stairs`：当前 event ledger 为 `f0–15=true`，`f16–29=false`。

两者都从窗口首帧即为 true，因而 onset 与 alertable-start 左删失，不能回填所需区间。
两个 false 尾段复用正事件 ID，不是独立 `negative_window_id`。宽松解释时只能把这两个
尾段归为“经过/远离”一类；`parallel_curb` 两个整段负序列没有预先映射到合同五类，
不得事后强塞。转头/近原地旋转、正常步行抖动、静止、低纹理/模糊/遮挡均没有两个
合规窗口。

### 2.4 完整性、矛盾与隔离

以下问题阻止签署 READY：

1. `i2j...` 同一源帧身份在 accepted manifests 中存在 `30/30 true` 与
   `16 true + 14 false` 两个版本，尚无唯一权威 lineage；
2. `GxMb...` 的 `f12–19` 同时为 `should_alert=true` 与
   `expected_approach_state=RECEDING`；
3. event-ledger 的 dataset spec 与逐帧 source 对 official `test/train` 记录不一致；
4. 既有 AI review 只见融合后的 `independent_review_count=2/3`，未找到可分别重算的
   两个 reviewer 输入/输出和必要仲裁 ledger；
5. legacy LILOCBench ledger 虽有 `15` 正窗和 `15` 负窗，但只覆盖 `2` 个 session，
   本地连续 RGB payload 已清理，仅余 identity/frames manifests；同时缺区域、显式
   `should_alert=false` 和负类字段，且属于已关闭 USTRF 的历史用途，不能补入当前角色；
6. public-video silver R7 的 `11` 个 package / `21` 个 episode 记录中，
   `0` 个具有 onset interval，`0` 个具有 region，`0` 个具有独立 model directions，
   且多数是稀疏采样帧，不进入本轮分母。

这些冲突均可按内容身份局部隔离；剩余证据仍可追溯，因此不把整个 R0 判为
`INVALID`。隔离后数量进一步不足，合法终点为 `HOLD_DATA`。

### 2.5 门槛核验

| READY 门槛 | 要求 | 本轮严格可计 | 判定 |
| --- | ---: | ---: | --- |
| 完整且独立的 capture session | `>=3` | `0` 个同时闭合 RGB + truth + 区间 + 区域 | FAIL |
| 正事件 | `>=6`，至少 2 session | `0` 完整；仅 `2` 个不完整候选 | FAIL |
| 负窗口 | `>=12` | `0` 完整；仅 `4` 个不完整候选 | FAIL |
| 负场景类别 | `>=4` 类且每类 `>=2` | `0` 个预冻结合规类别；最多宽松 1 类/2 段 | FAIL |
| development role | `1` session | `0` | FAIL |
| decision role | `2` 个含正事件、规则选择隔离的 session | `0` | FAIL |

因此没有冻结 development、decision 或 reserve 的正式 F-1B 角色。把两个 CrowdBot
session 叫作“decision 候选”只描述其 outcome-blind RGB 状态，不授予 decision role。

### 2.6 数据质量结论

- 严重度：`CRITICAL_FOR_F-1B_ENTRY`；
- 置信度：`HIGH`；
- 主要短板：事件/负窗真值的完整性、版本一致性和 RGB join，而不是 RGB 数量；
- 受影响用途：任何 F-1B0 时间补测申请、F-1B 双环增量比较和角色冻结；
- 未评价：YOLO、Sparse LK、RCLE、Android、融合、实时性、效果、能效和安全。

### 2.7 唯一有界补充方案（未授权、未执行）

若用户未来决定补数据标签，只允许另立一次
`F-1A_EXISTING_RGB_LABEL_REPAIR_ONLY`：

1. 输入宇宙固定为上述两个 outcome-blind CrowdBot session，加一个已绑定 SHA 的既有
   development session；不新增来源、不采集、不查看候选输出；
2. 只对既有 RGB 建立 hash-bound 的 event/window ledger，使用隔离的两次自动复核和
   必要的第三模型仲裁；冻结 parent capture、时间基、自然 event/window 边界、
   `LEFT/CENTER/RIGHT` 与五类负场景；
3. 必须在固定宇宙内得到 `>=6` 正事件、`>=12` 负窗口、`4` 类各 `>=2`，且两个
   decision session 都有正事件和可评价负窗口；否则一次性停止并保留 `HOLD_DATA`；
4. 不允许换源、切碎负窗、把帧数当样本量、读取算法输出来选片或转入 F-1B0。

该方案只是 HOLD 后的最小选择，不属于本轮执行权限。

## 3. 唯一终点

```text
DATA_STATUS: HOLD_DATA
DATA_PROTOCOL_STATUS: VALID
TIMING_STATUS: NOT_RUN
SCIENCE_STATUS: NOT_RUN
RUNTIME_STATUS: NOT_RUN
EXECUTION_AUTHORITY: NONE
CLAIM_CEILING: THESIS_ROUTE_PROPOSAL_ONLY

DUAL_LOOP_DATA_READINESS_R0 = HOLD_DATA
F-1B0_ELIGIBILITY = NOT_ELIGIBLE
NEXT_AUTOMATIC_ACTION = NONE
```

本终点只表示现有数据尚不能合法评价双环；它不是 YOLO、Sparse LK、RCLE 或双环效果
失败。除非用户另行授权 2.7 的有界 existing-RGB label repair，否则本路线停在 F-1A。
