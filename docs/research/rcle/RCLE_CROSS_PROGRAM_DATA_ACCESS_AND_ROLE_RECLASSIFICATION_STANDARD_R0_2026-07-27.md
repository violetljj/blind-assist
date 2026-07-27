# RCLE 跨项目数据访问与角色重分类标准 R0

日期：2026-07-27

状态：`ADOPTED_FORWARD_ONLY / CURRENT_FROZEN_PROTOCOLS_UNCHANGED`

机器记录：
[R0 role overlay](RCLE_CROSS_PROGRAM_DATA_ROLE_RECLASSIFICATION_R0_2026-07-27.json)

## 结论

RCLE 不再把“在 USTRF 或其他项目中访问过”直接等同于“整个数据集永久不可用”。
历史访问只关闭它实际污染的证据角色和最小身份单元。数据可以在披露访问历史后转为
discovery、source characterization、regression、counterexample、stress case 或
cross-program canary；只有满足新的未见性与独立性条件，才可承担 pristine
admission 或 confirmation。

本标准不修改任何历史 terminal、claim、receipt、burned manifest 或冻结合同。
正在执行的
`RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R1`
继续只允许冻结的 ETH3D `sofa_3`，不得加入 OpenLORIS、`sofa_4` 或其他替代来源。

## RULE_CHALLENGE

被替代的过宽解释是：

> 只要数据曾在其他项目中出现、下载到本地或被某次宽泛检索触及，就把整个数据族从
> RCLE 的所有未来角色中永久排除。

该解释的问题是：

1. 它把 metadata、文件存在、geometry、人工 RGB 查看和算法 outcome 混成同一种
   “访问”；
2. 它把 window/sequence 的局部污染无证据地传播到 source family；
3. 它把“不能作为 unseen confirmation”错误扩写成“不能用于 discovery/canary”；
4. 它会反复制造数据短缺，却没有相应降低算法 outcome 泄漏风险；
5. 它与项目既有的最小失败传播和失败资产复用原则冲突。

替代规则保留以下 invariant：

- 不回写旧结果；
- 不在结果后改变当前协议的候选、窗口、门槛或终态；
- 不把已见 outcome 的单元包装成 pristine/unseen；
- confirmation 必须有结果前冻结的 identity、ancestry、independence group 和防泄漏
  检查；
- 许可、完整性、隐私或 source authority 无法控制时仍可 `DO_NOT_REUSE`。

## Access ledger：记录实际信息，不记录笼统的“碰过”

每个可复用单元至少记录：

| 字段 | 含义 | 单独是否烧毁 |
| --- | --- | --- |
| `metadata_identity` | 官网、许可、目录、大小、哈希、sequence/capture ID | 否 |
| `payload_presence` | payload 曾下载或缓存，但不证明 member 被读取 | 否 |
| `geometry_access` | pose、depth、LiDAR、scene/object geometry 或其统计被读取 | 只影响依赖该 geometry 未见性的角色 |
| `rgb_visual_access` | 人或模型看过 RGB、thumbnail、contact sheet 或视频 | 污染视觉盲选；不自动等于 RCLE algorithm outcome |
| `other_algorithm_outcome_access` | 读取过其他 target/算法的 score、alert 或结果 | 披露；是否相关需另判 |
| `claim_relevant_outcome_access` | 读取过当前 claim 或实质同机制的 outcome/诊断 | 关闭对应 canary/confirmation |
| `selection_or_tuning_influence` | 上述信息是否影响候选、窗口、门槛、实现或停止决策 | 若为 true，关闭依赖其盲性的角色 |
| `access_state` | `NO / YES / UNKNOWN` | `UNKNOWN` 只局部 fail closed |

访问单元必须尽量落到
`member/modality → frame/pair → window → sequence/capture → independence group`。
不得仅凭 dataset 名称、目录存在或跨项目引用把 `UNKNOWN` 扩大为全族 `YES`。

## 角色资格

### `DISCOVERY`

允许任何已披露访问的数据，前提是许可、完整性和隐私可控。Discovery 可以重复，
不得升级为 confirmation。

### `SOURCE_CHARACTERIZATION / REGRESSION / COUNTEREXAMPLE`

允许复用已见 geometry 或 algorithm outcome 的数据。必须引用旧结果，保留稳定
content identity，并声明新 claim 不依赖旧 outcome 的正式接受权威。

### `DISCLOSED_CROSS_PROGRAM_APPROACH_CANARY`

允许在其他项目看过 RGB、geometry 或不相关算法 outcome 的真实数据承担 RCLE
复用型 approach canary，条件为：

1. 新协议在读取 RCLE approach outcome 前冻结；
2. 明确列出既有访问；
3. sequence 和窗口由不读取既有 outcome 的确定性规则选择；
4. 不把它描述为 pristine/unseen；
5. 同一 capture/sequence 永久退出该 claim 的 confirmation；
6. 若既有工作已经读取实质同一 geometry/looming outcome，只能降为
   characterization/regression，不得承担这一角色。

### `RCLE_RGB_ALGORITHM_CANARY`

其他算法运行过不自动禁止 RCLE algorithm canary。必须证明：

- RCLE algorithm outcome 尚未读取；
- 其他算法的 target/机制/诊断没有被用于选择 RCLE 参数或实现；
- 数据角色先由允许的非 RGB-algorithm geometry truth 冻结；
- 该数据不进入 confirmation。

### `PRISTINE_ROLE_ADMISSION / CONFIRMATION`

必须在 claim-relevant 信息访问前冻结 exact identity 和 partition。metadata identity
访问允许；会影响选源/选窗的 geometry、RGB review、同机制 outcome 或 adaptive
selection 不允许。使用过的 sequence/capture 不再进入同一 claim 的 confirmation，
但同数据集内真正未访问、结果前已隔离且 ancestry 独立的 sequence 可以另行审计。

## 首批前向重分类

### OpenLORIS

USTRF route/event replay 已对 7 条 office 做稀疏视觉预筛，对 2 条 cafe 做完整连续
RGB review；`cafe1-2` 还产生过 VO、route-known 和 alert trace。这些访问污染了
pristine visual/source selection，但没有形成 RCLE positive-approach geometry outcome
或 RCLE RGB algorithm outcome。

前向角色：

- `DISCOVERY`：允许；
- `DISCLOSED_CROSS_PROGRAM_APPROACH_CANARY`：允许，必须使用新冻结的确定性
  sequence/window 规则；
- RCLE implementation/performance engineering canary：条件允许；
- pristine/unseen admission：不允许；
- confirmation：不允许。

USTRF 的“没有完整 route lifecycle”不能推出“没有 positive approach”；二者是不同
claim。

### ETH3D `cables_1`

USTRF 已读取固定 120 帧的 RGB-D/pose transport、12 对重投影统计和 RGB contact
sheet。允许 regression、transport/implementation canary 和 source characterization；
不得再作为 pristine approach admission 或 confirmation。该结论不自动传播到未访问
且 capture 独立的其他 ETH3D sequence；是否扩大到 sofa scene family 必须依据实际
capture ancestry，而不是 publisher 名称。

### Aria Digital Twin

已读取 groundtruth 并完成 geometry cell prescreen 的冻结 16 条 sequence 退出新的
blind geometry admission，但仍可 regression/characterization。其余未访问 sequence
不因共享 ADT 名称而自动烧毁；在 exact sequence/capture manifest、component
clustering、许可和 deterministic selection 闭合后，可进入新的 RCLE source audit。

### AV2、Waymo、UT CODa、HoloAssist

USTRF 的 pure-rotation cell、route lifecycle 或 source-specific authority failure
只关闭依赖这些条件的 USTRF claim：

- AV2/Waymo 可重新审视真实前进 approach 的车载-domain canary，但不能冒充手持/
  头戴 confirmation；
- CODa 可重新审视 pose + RGB/LiDAR approach，但 full archive checksum/version
  binding 仍是独立的 source-authority 阻断；
- HoloAssist 既有访问限于 split/annotation metadata，不因 USTRF 记录而烧毁；其
  pose/depth/连续几何能力仍需新审计。

### TUM 与 Bonn

已读取 RCLE geometry outcome 的具体 sequence/window 继续退出 blind admission 和
confirmation；允许 regression、counterexample、source characterization 和 disclosed
canary。未访问 sequence 不自动死亡，但由于同 family 已多次参与规则形成，若用于
高权威 confirmation，必须证明结果前 partition、capture independence 和无 adaptive
selection；优先使用不同数据族。

### EVIMO2 `Flea3/sanity_ll`

13 条 sequence 已读取完整 RCLE geometry outcome，继续只允许
source characterization、counterexample 和 regression。不能恢复为 unseen positive
approach admission 或 confirmation。

### Synthetic families

ICL-NUIM、TartanAir 和 Phase A synthetic 可以继续用于物理标定、实现测试和
regression，但不能填补 `REAL_POSITIVE_APPROACH`。

## “本地已有 payload”规则的纠偏

`ALL_PREEXISTING_ARTIFACTS_LOCAL_PAYLOADS` 只作为当前 R1 的一次性保守 containment
保留，不得自动继承到新协议。未来必须区分：

- 文件存在但 member 未打开；
- 读取了 identity/CRC；
- 读取了 geometry；
- 查看了 RGB；
- 读取了其他算法 outcome；
- 读取了 claim-relevant outcome；
- 是否实际影响 selection/tuning。

只有最后两类或无法局部控制的 integrity/license/privacy 风险，才可能关闭较高证据
角色。其余情况按最小单元降级，不判数据集全局死刑。

## 新协议采用要求

新 RCLE 数据协议在 preaccess contract 中必须：

1. 引用机器 reclassification overlay；
2. 为候选写完整 access vector，而不是单个 `burned=true/false`；
3. 分别声明允许的 discovery、canary、algorithm-canary 和 confirmation 角色；
4. 写明污染传播的最小 identity scope；
5. 写明哪些旧 outcome 可见、哪些必须由 firewall 隔离；
6. 声明是否为 `PRISTINE` 或 `DISCLOSED_CROSS_PROGRAM`；
7. 禁止用多个并行候选“谁通过选谁”；并行 source audit 必须预先分配不同角色，
   或只输出 discovery characterization。
