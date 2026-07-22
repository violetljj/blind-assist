# SANPO P3 经同意前向手机序列接入（2026-07-13）

## 何时使用

全量 official-train 候选发现后，lateral 场景的仅有 4 个胸前候选均未通过精确 50-frame 几何门。因此，不能继续靠放宽 SANPO 稀疏阈值或重复同一 native session 来填补 P3 的 lateral session 缺口。经同意的前向手机序列是 A 层、可计入 P3 session 覆盖的补充来源。

这不是“上传视频即可训练”的入口。没有已验证的同意、PII 审核和人工像素标注时，planner 会拒绝该 session。

## 回执模板与硬门

每个 native session 使用一份独立的、无直接个人身份信息的 JSON 回执；以 [模板](../configs/sanpo_p3_consented_capture_receipt_template_20260713.json) 为起点。用于 P3 planner 前必须满足：

| 字段 | 接受值 |
|---|---|
| `format` | `blindassist_p3_consented_capture_receipt_v1` |
| `source_id` | `consented_forward_phone_v1` |
| `consent_status` | `granted` |
| `consent_record_ref` | 非空、假名化的同意记录引用 |
| `capture_mode` | `phone_chest_forward` 或 `phone_handheld_forward` |
| `residual_pii_review_status` | `passed` |
| `pixel_annotation_status` | `human_verified` |
| `annotation_quality` | `human` |
| `scene_review_status` | `approved` |
| `mask_taxonomy` | `blindassist_4class_mask_v1` |

原始视频、同意原件和原图/mask 只保存于 ignored `artifacts.local/`。Git 只可保存 schema、不可逆哈希和不含个人身份信息的 receipt reference；不得写入姓名、联系方式、人脸图或未脱敏画面。

## manifest / recipe 绑定

每个 manifest row 的 `source` 必须包含：

```json
{
  "source_id": "consented_forward_phone_v1",
  "session_id": "native-session-id",
  "consent_receipt_sha256": "sha256-of-the-receipt-file",
  "annotation_quality": "human",
  "residual_pii_review_status": "passed",
  "camera": "phone_chest_forward",
  "lens": "not_applicable",
  "source_width": 1920,
  "source_height": 1080
}
```

每个 manifest row 还必须有以下 `label_provenance`。经同意手机来源使用项目冻结的四类灰度 mask（`0=walkable`、`1=boundary_step_curb`、`2=obstacle`、`3=unknown_nonwalkable`）；它**不是** SANPO 的 31 类 panoptic ID。P3 planner 会按来源回执选择 taxonomy，并拒绝未声明、错配或出现非 `0–3` 值的模型共识 mask，避免把项目类别静默按 SANPO 类别误解。

```json
{
  "annotation_kind": "human_pixel_mask",
  "mask_taxonomy": "blindassist_4class_mask_v1"
}
```

每帧还必须给出相对 `package_root` 的 `image_path` 和对应 `image_sha256`。planner 重新计算 image/mask 哈希，并要求真实图片尺寸与 `source_width × source_height` 一致；同一 native session 内分辨率、camera 和 lens 不得变化。这样不会把脱离 RGB、尺寸被伪造或横竖屏切换的 mask 静默计入 P3 coverage。

recipe sequence 必须声明：

```json
{
  "source_id": "consented_forward_phone_v1",
  "native_session_id": "native-session-id",
  "official_split": "not_applicable",
  "consent_receipt_path": "relative/path/to/receipt.json"
}
```

`scripts/plan_sanpo_p3_session_split.py` 会在打开 candidate manifest 前验证 receipt；之后逐 row 验证 receipt SHA、人工 annotation quality 和 PII clearance。session 的 split 主键仍是 `source_id:native_session_id`，同一 session 的不同视角不得跨 train/dev。

## 标注与报告边界

- `machine` annotation 可以保留作预标注、复核工作队列或单独报告，但不能满足 P3 canonical/训练门。
- `model_consensus` annotation 仍需要通过 P3 planner 的 class-share、boundary 集中度和独立 session 门；模型共识不是自动训练许可。
- 每个候选保留上游原生 `HUMAN_ANNOTATED/MACHINE_ANNOTATED` provenance（若来源提供）以及本项目 `model_consensus` 准入状态，最终分层报告，不能把来源元数据与项目复核 authority 混为一项。
- 新采集必须覆盖真实的 target camera/view；车载、Mapillary 或合成素材仍不得代替 P3 dev/blind。
