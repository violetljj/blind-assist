# P1-D0 Temporal Cohort Protocol V1

状态：`P1_TEMPORAL_DEVELOPMENT_COHORT_READY / DEVELOPMENT_STANDARD / ADT_GT_ONLY_SELECTION / NO_MODEL_OR_TRACKER_SELECTION / NO_ENTRANCE_CLAIM`

## 问题与边界

本任务只为 P1 Target Persistence 物化小型 Development temporal truth。它回答数据是否能给出：同一
physical instance 的连续身份、暂时不可观察、较长 loss、同一实例重现，以及同类 physical distractor
机会。它不验证 P0 grounding、`entrance_of`、任何 tracker、导航闭环、用户安全或默认 App。

P1 的权限边界保持：Persistence 可以维持、削弱、丢失或重新确认一个已经建立的 physical referent，
不能在 `NO_REFERENT` handoff 后自行建立 referent，也不能用 temporal consistency 反向证明 P0 semantic
correctness。

本批数据固定为 `CONSUMED_DEVELOPMENT_ONLY`。允许使用 ADT source GT 做 cohort selection 和 evaluator
truth；任何后续 P1 estimator 只能读取 RGB 与 P0 handoff，不能读取本 materializer 或 episode truth。

## 固定预算与来源

- Primary source：Aria Digital Twin；不启用 EgoTracks fallback。
- source budget：最多 6 条，本批实际复用 2 条已下载 sequence。
- episode budget：12–18，本批固定为 15。
- 每个 episode 只绑定一个 ADT `object_uid`；同一 physical target 在本批最多出现一次。
- 达到预算后停止；不增加第三数据源，不下载新 RGB/GT，不做人工 identity 标注。

实际来源为：

1. `ADT_Apartment_release_golden_skeleton_seq100_10s_sample_M1292`；
2. `ADT_Apartment_release_clean_seq136_M1292`。

RGB/GT 对齐不使用 detector 或 tracker output。ADT preview MP4 的 `description` metadata 保存逐帧 absolute
timestamp；materializer 将其与 `aria_trajectory.csv` 在 20 ms 内一一匹配。2D bbox row 再按相同时间门
映射到该 trajectory。无法对齐超过 1 个 GT frame 时 fail closed。

## Episode truth contract

每个 episode JSON 至少包含：

```text
episode_id
source_sequence_id
physical_target_id = adt:<object_uid>
source_object_uid
temporal_mode_tags
candidate_distractor_instance_ids
frames[]:
  source_frame_index
  timestamp_ns
  target_visibility_ratio
  target_visible
  target_bbox_xyxy | null
  candidate_distractor_instance_ids
```

`physical_target_id` 直接绑定 ADT source identity。不得用 tracker output 定义 identity、用 detector/model
output 选 episode、人工猜测跨帧 identity，或对缺失 GT 插值制造 truth。

## 确定性选择规则

统一 `visibility_ratio >= 0.10` 为 visible，visible run 至少 12 个 aligned frames。优先按每类 3 个 episode
选择，再在仍未达到 15 时以剩余 GT-only proposals 回填；每条 source 的 episode 数有确定性上限，优先
不同 `object_uid`。

- `CONTINUOUS_VISIBLE`：同一 identity 连续 visible 至少 90 帧；episode 取确定性 90 帧窗。
- `TEMP_OCCLUSION`：两个 visible run 间隔 2–44 帧，gap 中超过 20% 帧仍有低于 visibility 门的 target
  bbox row。
- `OUT_OF_VIEW_RETURN`：两个 visible run 间隔 2–44 帧，gap 中最多 20% 帧仍有 target bbox row。
- `LONG_LOSS`：两个 visible run 间隔 45–240 帧。
- `REACQUISITION`：上述 gap 后相同 ADT `object_uid` 再次 visible。
- `DISTRACTOR_PRESENT`：同 prototype 或 category 的另一个 ADT instance 同时 visible 至少 10 帧。

`TEMP_OCCLUSION` 与 `OUT_OF_VIEW_RETURN` 是 bbox-row/visibility 的机械 mode tag，不是 causal occluder 或
field-of-view 真值；不能将其扩写为来源没有直接提供的物理因果标注。

## P0 → P1 safety guard

manifest 为每条实际入选 source 引用一个机械 safety case：

```text
handoff = NO_REFERENT
required_persistent_referent_id = null
required_illegal_bind_rate = 0
```

后续 evaluator 必须把这一项作为硬门。画面出现高置信目标不能授权 P1 建立 referent。

## 稳定入口与输出

入口：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ba-adt-real-evidence `
  materialize_p1_temporal_cohort.py `
  --source <SEQUENCE_ID> <GROUNDTRUTH_ZIP> <PREVIEW_RGB_MP4> `
  --source <SEQUENCE_ID> <GROUNDTRUTH_ZIP> <PREVIEW_RGB_MP4> `
  --output-dir artifacts.local/evidence/p1_d0_temporal_cohort_v1 `
  --episode-budget 15
```

机器输出位于 ignored `artifacts.local/evidence/p1_d0_temporal_cohort_v1/`：

```text
p1_d0_manifest.json
summary.json
episodes/<episode_id>.json
```

## 本批结果与停止条件

本批得到 2 条 source sequence、15 episodes、15 个 physical targets、1,724 个 episode frames。mode count：

```text
CONTINUOUS_VISIBLE = 6
TEMP_OCCLUSION = 3
OUT_OF_VIEW_RETURN = 3
DISTRACTOR_PRESENT = 3
LONG_LOSS = 3
REACQUISITION = 9
```

缺失 mode 为 0；15/15 episodes 均有 source UID、per-frame visibility 与 bbox-or-null。运行计数固定为
`model=0 / detector=0 / tracker=0`，终态 `P1_TEMPORAL_DEVELOPMENT_COHORT_READY`。

P1-D0 到此关闭。不得为本 Development cohort 追加 sequence 或 episode；EgoTracks fallback 不触发。下一步
只允许冻结的 P1 representation/evaluator 与 baseline 消费这批数据。任何算法结果仍只属于 ADT indoor
physical-object Development，不能声称入口 persistence 已验证，也不能购买 fresh/domain confirmation，除非
先出现明确 persistence headroom 并另立后续协议。
