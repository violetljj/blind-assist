# P0-S0-V0 Visual Candidate Generator Admission

状态：`COMPLETE / P0_S0_VISUAL_CANDIDATE_GENERATOR_NOT_ADMITTED / NO_MODEL_RUN`

冻结记录：[`p0_s0_visual_candidate_generator_admission_v0.json`](p0_s0_visual_candidate_generator_admission_v0.json)

## Verdict

审计对象是固定上游
`project-terraforma/mapillary-entrances@3d3b85244b1a1ec2ba05a997d56d000936cc554a` 所引用的
`erantala1/yolov8s-entrance-detector/yolo_weights_750_image_set.pt`。终态为：

> `P0_S0_VISUAL_CANDIDATE_GENERATOR_NOT_ADMITTED`

本轮没有下载 checkpoint、Mapillary 图像或运行 inference。Hugging Face 元数据把 checkpoint identity 固定为
repository revision `524c7b7d1ea56f7c4b6f03389ffc6a73b75fcdda`、LFS SHA-256
`f0cceccf483ad87a5c0756044014d420a34df49a0b76afa877540ba3b7763b0a`、22,712,810 bytes。
checkpoint repository 只用 31-byte README front matter 声明 `Apache-2.0`，没有 model description、训练图像、
标注来源、数据许可或数据发布相容性。这个声明被保留，但不能补足训练 lineage。

## Fail-closed blockers

固定上游的 `hf_hub_download` 只传 `repo_id + filename`，没有绑定 revision 或 expected digest。README 示例给出
`conf=0.60 / iou=0.50 / device=cpu`，但没有冻结 `imgsz/max_det/agnostic_nms/augment/half/seed`、确定性算法或
canonical output ordering。requirements 固定 `ultralytics==8.3.224`，该发行版声明 `AGPL-3.0`，未来分发路径仍需
单独审查。

源码还有两个直接阻塞 proposal provenance 的问题：入口过滤表达式末尾含 `or True`，因此所有类别都会进入；
folder adapter 随后只返回整数 bbox tuple，丢弃 confidence、class 和 per-candidate model provenance。最终 geometry
cluster 也不是可替代的 image-space candidate receipt。

所以当前对象同时缺少训练数据 provenance、完整 replay envelope 和冻结 schema 所需的候选 lineage；不能通过
“仓库代码是 Apache-2.0”或 detector confidence 把它升格。

## Frozen authority boundary

任何后继 candidate generator 最多拥有 `VISUAL_PROPOSAL_ONLY`。它只能输出 `image_id + pixel bbox + confidence +
class + exact model/runtime/config provenance`。它不能建立 entrance、`entrance_of`、target-building、map、geometry、
multiview 或 evaluator truth，不能自行升格 `SILVER_A_PRIMARY`。map anchor、ray-wall geometry、multiview 和 conflict
check 必须独立成立；若被评系统复用相同 checkpoint/runtime lineage，必须披露 overlap，不能据此主张 evaluator
独立或 provider-coverage superiority。

## Minimal replacement audit shortlist

以下只是下一次可审计对象，均未获准运行：

1. `microsoft/Florence-2-base-ft@f6c1a25888ffc1d945ee8a1a77ac833c7303d46e`（model repo 声明 MIT）：需冻结
   prompt/task/output parser，并补齐 checkpoint 训练数据与标注 lineage。
2. `IDEA-Research/grounding-dino-base@12bdfa3120f3e7ec7b434d90674b3396eccf88eb`（model repo 声明
   Apache-2.0）：需冻结 text prompt、NMS/output semantics，并补齐 checkpoint 训练数据与标注 lineage。

## Next action and claim ceiling

V0 审计完成后，Windows 用户级 `MAPILLARY_ACCESS_TOKEN` 已配置，并以只返回 image ID 的极小只读 Graph API
请求通过鉴权；没有下载图像，Client secret 没有落盘。Mapillary access gate 因此已经满足。

仍不得运行 `P0-S0-R1`。唯一剩余前置条件是：修复并重新审计该 YOLO generator，或从上面 shortlist 选择一个做
同等 admission。candidate generator 获准后才可新建 S0-R1。

本轮只证明 exact checkpoint metadata 可定位，并识别 admission blockers；不证明模型性能、入口覆盖率、Silver
可产量、数据发布许可、S0 可通过、导航/安全/产品能力，也不改变冻结 P0/Silver evaluator semantics。
