# Instance correspondence candidate annotation R0 result

状态：`COMPLETE / VALID_INPUT_AND_OUTPUT / DEVELOPMENT_CANDIDATE_ANNOTATION_ONLY`

## 结论

候选关联管线已完成并在固定 320-frame Development/consumed expansion 上成功批量
输出。当前输入没有产生唯一 `MATCH`；这是一项 honest negative/abstention result，
不是把“没有 detection”改写成未覆盖：

| 层级 | MATCH | NO_MATCH | ABSTAIN |
|---|---:|---:|---:|
| pair evidence | 0 | 4,798 | 892 |
| component annotation | 0 | 2,936 | 3,778 |
| detection annotation | 0 | 0 | 254 |

输入计数为 320 frames、6,714 components、254 YOLO detections、5,690 个有观测的
component/detection candidate pairs。每个 component 最多一个 detection，每个
detection 最多一个 component；独立复核未发现重复 selected identity。

`ABSTAIN` 的主要来源是没有 detection candidate、类别未知或几何证据不足；pair 中
`NO_MATCH` 只由冻结规则中的明确类别冲突或强几何分离产生。输出中的
`component_track_id`/`detection_track_id` 已生成并保留，但因为当前没有 selected
match，`temporal_continuity` 没有形成可用的跨-track correspondence evidence。

本批未提供 depth cluster 或 optical-flow sidecar，因此所有对应字段显式为
`UNKNOWN/null`，没有用 identity flow 或假 depth 填充：

- class compatibility：`3,676 COMPATIBLE / 1,523 INCOMPATIBLE / 491 UNKNOWN`；
- depth cluster rows assigned：`0`；
- flow pair rows assigned：`0`；
- temporal pair values assigned：`0`。

这说明当前批次把旧的 frame-level `ATTRIBUTION_UNCERTAIN` 拆成了结构化候选关系，
但没有获得足以升级 residual labelability 的 instance truth。既有 Atlas 仍保持
`RESIDUAL_WEAKLY_LABELABLE`；本结果不修改默认 YOLO、Android、risk、feedback、TTS、
振动或任何产品/安全结论。

## 可复现输入与输出

命令入口为
`scripts.research.dual_loop_segmentation_instance_correspondence.batch`，实现和配置
分别由输出 `provenance.json` 固定 SHA256。最终输出目录：

`artifacts.local/evidence/dual-loop-segmentation-instance-correspondence-r0/expansion-320/`

其中包含 `pair_evidence.jsonl`、`component_annotations.jsonl`、
`detection_annotations.jsonl`、`summary.json` 与 `provenance.json`。输入角色继续是
Development/consumed；不能把本批当作 fresh validation，也不能据此声称 detector
有效覆盖或 residual truth 已被恢复。

下一步若要使 depth/flow 进入真实批量输出，应先提供同一 source/frame/image identity
绑定的 sidecar，并重新运行相同合同；sidecar 缺失不是调阈值的理由。
