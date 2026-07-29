# F-1A existing RGB 标签修复 R0 结果

状态：`COMPLETE / HOLD_DATA / PROTOCOL_VALID`

执行者：`viojjet`

## 结论

用户授权的 `F-1A_EXISTING_RGB_LABEL_REPAIR_ONLY` 已按固定输入宇宙和候选输出盲法完成。
最终账本包含 `17` 个正事件与 `19` 个明确负窗；两个 decision session 都同时含有
可评价正事件和负窗。数据门仍为 `HOLD_DATA`，唯一未通过项是负场景类别覆盖：

| 负场景 | 接纳窗口 |
| --- | ---: |
| `TURN_OR_NEAR_IN_PLACE_ROTATION` | 11 |
| `NORMAL_WALKING_SHAKE` | 5 |
| `LATERAL_PASS_OR_RECEDING` | 2 |
| `STATIC_SCENE` | 1 |
| `LOW_TEXTURE_BLUR_OR_OCCLUSION` | 0 |

冻结门要求至少四类各有两个窗口；当前只有三类满足。因此本轮不能进入 F-1B0。
这不是 YOLO、Sparse LK 或双环效果失败，而是数据类别覆盖不足。

## 固定输入与隔离

- 两个 decision session：CrowdBot `11-51-18` 与 `11-55-00`；
- 一个 development session：Wikimedia Commons Matoaka 连续步行视频；
- 两次隔离 RGB 复核均未见另一份复核，也未见 YOLO、Sparse LK、RCLE、风险、提醒或
  双环候选输出；
- 两次复核形成 `10` 条直接一致和 `55` 条分歧；独立第三裁决接纳其中 `26` 条，
  保守隔离 `29` 条；
- 不回收隔离项，不降低置信度、自然窗长度、类别数或 decision-session 条件。

## 可复算凭据

```text
review_bundle_subject_sha256:
b7fde445beb8e75a357ae09537a945a159521da77b250f4211c0a74e31259a6c

review_a_sha256:
9cd5017f22107c1fed99685eb693e1c9fc5c49af455282c819617e38b2445731

review_b_sha256:
9a3bce32dfcbaa18ed3b3b68c780a7b6daf1a6038d2daec46001954eb8b9d130

comparison_sha256:
31103e91c225e97d1fb8fafd23a4abbe56832862da6e5c9896208a9c09f704fd

adjudication_sha256:
97d9b9daa9ba704122b4d4b71dfe09d358323cc2e061af7359e82feed5cb0816

event_window_ledger_sha256:
ab9f7771edfab5015f6c3fca43947209482fa8f6c23ca072e31ea9fa0cf7cf70

validation_sha256:
bd9dd02880a5cbff5cc7693bb7f1c9b6d3d2090d8639053cd980829159a8f12e
```

正式本地凭据：

- `artifacts.local/evidence/dual-loop/f1a-existing-rgb-label-repair-r0/review-bundle/`
- `artifacts.local/evidence/dual-loop/f1a-existing-rgb-label-repair-r0/reviews/`
- `artifacts.local/evidence/dual-loop/f1a-existing-rgb-label-repair-r0/review-comparison.json`
- `artifacts.local/evidence/dual-loop/f1a-existing-rgb-label-repair-r0/adjudication.json`
- `artifacts.local/evidence/dual-loop/f1a-existing-rgb-label-repair-r0/event_window_ledger.jsonl`
- `artifacts.local/evidence/dual-loop/f1a-existing-rgb-label-repair-r0/validation.json`

## 后继边界

R0 的 `HOLD_DATA` 与全部隔离项保持不可变。根据用户“缺数据就找数据”的连续授权，
另立 `F-1A_NEGATIVE_CATEGORY_SUPPLEMENT_R1`，只允许在新的 development-only 既有 RGB
来源中补 `STATIC_SCENE` / `LOW_TEXTURE_BLUR_OR_OCCLUSION`。R1 不得重审 R0、替换
decision session、读取候选输出或更改数据门；只有合并后的冻结账本满足原合同，才可进入
F-1B0 时间凭据审计。
