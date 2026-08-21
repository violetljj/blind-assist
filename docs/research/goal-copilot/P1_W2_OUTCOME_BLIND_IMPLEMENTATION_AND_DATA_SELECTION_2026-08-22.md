# P1-W2 outcome-blind implementation and data selection

状态：`IMPLEMENTATION_SELECTED / DEVELOPMENT_COHORT_IDENTIFIED / FRESH_PARENT_ROSTER_FROZEN / FRESH_PAYLOAD_NOT_MATERIALIZED / TARGET_CANDIDATE_ROSTER_NOT_FROZEN / NO_MODEL_DOWNLOAD / NO_EXECUTION`

Claim ceiling：`OUTCOME_BLIND_SELECTION_ONLY / NO_EMPIRICAL_CAPABILITY / NO_BUILDING_ENTRANCE_OR_P0_HANDOFF_CONFIRMATION / NO_PRODUCT_OR_SAFETY_AUTHORITY`

机器冻结文件：[`p1_w2_anchor_interface_selection_v1.json`](p1_w2_anchor_interface_selection_v1.json)

## 1. 本次完成和停止的位置

本 successor 只把 [`P1-W2 protocol`](P1_W2_REFERENT_ANCHOR_INTERFACE_FEASIBILITY_PROTOCOL_V1.md) 中尚未选择的接口
具体化，并用 metadata-only 规则冻结 fresh parent sequence。没有下载新 checkpoint、RGB 或 groundtruth，没有运行
matcher、identity encoder、17 个 Development cases 或任何 fresh case，也没有观察 P1-W2 outcome。

由于 fresh target、source frame、probe 与 same-scene confuser 必须读取 ADT private truth 后才能确定，本次不能诚实地
宣称完整 case roster 已冻结。当前合法终点是：**8 个父序列已 outcome-blind 冻结，但 target/candidate roster 尚不存在**。
因此下一步不是 execution，而是另行授权的
`P1_W2_FRESH_SOURCE_MATERIALIZATION_AND_PRIVATE_ROSTER_FREEZE`。

## 2. 最小 matched interface

### 2.1 Geometry path

唯一 correspondence provider 选择官方 EfficientLoFTR 的 Hugging Face checkpoint
`zju-community/efficientloftr@face1a79050ffa3e9da28720d1cf93aaf2e8f421`。选择时固定的
`model.safetensors` SHA-256 为
`00a5edc343fa222eba763643553548ad37a05aa3d00d266553c9f6cf67bb0e64`，许可为 Apache-2.0。
本地 `transformers 4.57.1 / torch 2.11.0+cu128` 只做了类与 CUDA 可用性检查；权重未下载，推理未执行。

Provider 的权限仍只有：

```text
EfficientLoFTR output
  -> confidence >= 0.20 correspondence candidates
  -> source AND probe endpoints inside referent core
  -> OpenCV 4.10.0 USAC_MAGSAC homography
  -> >=8 inliers, inlier ratio >=0.50
  -> source/probe inliers each occupy >=3/4 core quadrants
  -> GEOMETRY_SUPPORTED
```

输入是以 core 为中心、至少在四边各留半个 core 尺寸的 `4:3` bounded crop，越界处常量补零，再固定为
`640×480` grayscale。Context 可以影响 matcher 的局部表征，但 context endpoint 不进入几何模型；没有 core-core
support 时必须 fail closed。高 match count 本身没有 geometry authority。

### 2.2 Identity path

唯一 identity provider 选择已在 P1-A2 冻结并已本地校验的
`facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`。它只读取精确 core crop，直接 resize
`224×224`，形成 `16×16` L2-normalized patch tokens 和 mutual-nearest cosine correspondence；不读取 context。

四个 AND gate 原样采用 consumed P1-A2 Development prior：

```text
anchor_match_fraction >= 0.1640625
median match_confidence >= 0.755523741
spatial_consistency >= 0.423076923
anchor_coverage >= 0.875
```

这些数值明确带有 `outcome-aware historical Development` lineage；它们只是在任何 fresh P1-W2 outcome 可见前冻结的
先验，不被重新包装成独立或通用阈值。正因为已有一个可审计先验，本次不再引入 DINOv3 或 embedding zoo。

对一个 opaque candidate set，恰好一个 candidate 通过四门才是 `IDENTITY_SEPARATED`；零个为
`NOT_OBSERVABLE`，多个为 `AMBIGUOUS`。不按本轮相似度分布另造 margin。

### 2.3 Joint eligibility

Geometry-pass set 与 identity-pass set 的交集：

```text
size == 1  -> ELIGIBLE（随后由 private evaluator 判 correct / FALSE_BIND）
size == 0  -> NOT_ELIGIBLE
size > 1   -> AMBIGUOUS
```

Provider 只看到打乱的 opaque candidate IDs、RGB 与 region；`object_uid` 和 true/confuser mapping 只在输出封存后由
evaluator 读取。任何 unique wrong candidate 都是 `FALSE_BIND`，不能被 coverage 抵消。

## 3. 数据角色

### 3.1 Consumed Development

P1-W1 的 17 cases 全部保持 `CONSUMED_DEVELOPMENT_DIAGNOSTIC_ONLY`。它们没有在本次运行，未来即使另行授权调通
接口，也不能承担 confirmation、Stage A v2 verdict 或 fresh denominator。

### 3.2 Fresh proxy parent roster

从 ADT 的 236-sequence live inventory 中，只用 sequence metadata/name 和 artifact hash，在四个预定 strata 内做
`SHA256(salt | stratum | sequence_id)` 排序，各取前 2 个；排除了 seq100 sample、seq134/136 及此前已打开
groundtruth 的 16 个 USTRF parents。冻结的 8 个 parents 是：

| stratum | sequence |
|---|---|
| Lite recognition | `BirdHouseToy_seq032`, `BlackCeramicBowl_seq032` |
| Apartment single-user | `work_seq136`, `meal_seq133` |
| Apartment multiuser | `cook_seq116`, `clean_seq116` |
| Apartment multiskeleton | `party_seq121`, `party_seq117` |

完整 sequence ID、RGB/groundtruth filename、SHA-1、byte size、selection rank 与 inventory identity 均在机器冻结文件中。
selection identity SHA-256 为
`ce49c7634ce446accbc613e706b43951d2d88e30a442fbfc4d8db7c914d68189`。

这批数据有真实 `object_uid`、2D bbox、visibility 与 camera pose，适合形成物理 instance/confuser 的 private evaluator；
但它是 **ADT indoor-object proxy**，不是建筑入口数据。source core 未来也只能作为 evaluator-supplied、
candidate-conditioned 的 P0-handoff upper-bound surrogate。因此它不评价 proposal recall，不证明 named entrance anchor
coverage，也不能建立 BlindAssist 产品能力。

## 4. 下一 roster 的冻结规则

如果且仅当另行授权物化上述 8 个 parents 的 preview RGB 与 main groundtruth，private selector 才可按已冻结规则：

- 只纳入 `object + rigid + static` instance；source 要求 `visibility>=0.75`、bbox min-side `>=24 px`，probe 要求
  `visibility>=0.50`、bbox min-side `>=16 px`；
- confuser 优先不同 `object_uid` 的同 `prototype_name`，其次同 `category_uid`；最多 true + 3 confusers；
- 每 parent 只选一个 source referent，并要求同场 confuser 加至少一个 viewpoint stratum；
- probes 只由 truth pose/visibility 分为 rotation、small translation、large translation、reappearance 和
  same-scene confuser，按 frozen hash 选择，不读取 RGB feature 或模型输出；
- 不足 6/8 parents、任一预定 probe stratum 为零，或 same-scene-confuser probes 少于 6，终态为
  `NOT_EVALUABLE_DATA_SUPPORT`，不得换 parent 或降低门槛补救。

上述 selector 尚未实现、未运行；本文件只冻结它的输入权限和确定性规则。

## 5. 未来一次性判定与预算

若后续完整 roster 合法冻结并再次获得 execution 授权，分析单位是 parent-macro，同 parent probes 不能扩张成独立
样本。预冻结 feasibility gate 为：true-candidate geometry overall `>=0.70` 且每 viewpoint stratum `>=0.50`；
unique true identity overall `>=0.70` 且 false bind 为 0；correct joint eligibility overall `>=0.60` 且 false bind
为 0。它们只定义 proxy feasibility signal，不是统计推广或产品门槛。

一次执行预算上限为 8 parents、每 parent 5 probes、每 probe 4 candidates、每 provider 160 pair evaluations；禁止
online update、tracking、reacquisition、memory write、SAM2 propagation、SLAM、Stage A v2 或 Android/default-App。
任何模型/接口不可用、payload hash 不符、private mapping 泄漏或 denominator 不足都必须
`P1_W2_NOT_EVALUABLE_DATA_OR_INTERFACE`，不能记作模型负例。

## 6. 当前终点

```text
implementation/checkpoint/gates: frozen
consumed Development role: frozen, not run
fresh parent sequences: frozen metadata-only
fresh payload and private target/candidate roster: absent
P1-W2 execution: not authorized
```

唯一合法 successor：`P1_W2_FRESH_SOURCE_MATERIALIZATION_AND_PRIVATE_ROSTER_FREEZE`。它完成后仍须再次授权，才可
进行一次 P1-W2 execution；本文件不自动续跑。
