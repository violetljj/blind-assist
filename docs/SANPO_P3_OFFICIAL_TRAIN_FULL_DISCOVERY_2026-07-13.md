# SANPO P3 official-train 全量候选发现（2026-07-13）

## 结论

已完成 SANPO-Real v0 official-train 索引 `0–559` 的全量、只读候选发现。结果**解除的是“公开 official-train 候选池未穷尽 / 长扫描不可恢复”的工程阻塞**，不是 P3 canonical 或训练门。

聚合出 `146` 条 candidate row：胸前视角 `86`（来自 `63` 个独立 native session），头戴视角 `60`（来自 `42` 个独立 native session）。所有候选仍是 source-mask 稀疏证据，必须继续经过 16-frame 局部预筛（如适用）、50-frame 几何门、RGB 下载与 PII 审核、人工场景/风险语义复核，才可以进入 P3 split planner。head 候选还必须先通过独立的 cross-view 门，不能静默混入 chest canonical。

因此：

- 可继续进行 candidate 的分层审核与 session-level 选取；
- 不可重建 P3 canonical、启动训练、改变 benchmark/blind 或替换 App 默认模型；
- 不可把 `146` 条 candidate row 当作 `146` 个独立、已标注、可训练 session。

## 覆盖与完整性

| 项目 | 结果 |
|---|---:|
| bound official-train session | 560 (`0–559`) |
| batch | 28 × 20 session |
| completed batch | 28 |
| unresolved network/data failure | 0 |
| aggregate candidate row | 146 |
| chest candidate row / unique session | 86 / 63 |
| head candidate row / unique session | 60 / 42 |
| aggregate SHA256 | `a5031bea47fae0c66bd59aa12b036a2ac420d3b0681d82ee7d25867078dd9889` |
| finalized checkpoint SHA256 | `9b0430be23724822b2ff3c3994cd226800b8296d287b796fd0c33cc80229d7ee` |
| bound session-order SHA256 | `a0f599cb2250232cf441c47f5eb412f2fd8615880e77f71a4f72c28b563f7a69` |

每一批都强制 `attempted_session_count == selected_session_count` 以及 `network_or_data_failure_count == 0`。GCS 的超时/SSL EOF 只允许在有限重试后继续；若仍未解决，runner 不会生成 aggregate。

## 候选构成

| view × profile | candidate row |
|---|---:|
| `camera_chest` × `step_curb` | 59 |
| `camera_head` × `step_curb` | 41 |
| `camera_chest` × `center_obstacle` | 23 |
| `camera_head` × `center_obstacle` | 19 |
| `camera_chest` × `lateral_pedestrian_or_ebike` | 4 |

这不是类别像素或场景配额。特别是 lateral 候选仍很少；而 `step_curb` 只是边界几何信号，不代表已确认的导航风险。后半段的大量零候选窗口是应保留的负证据，不能通过降低阈值或重复切分前段 session 来填补。

全量池中仅有的 4 条 lateral 候选已经逐条执行 exact 50-frame remote-mask gate，结果 `0/4` 通过：两条存在中心 hazard/target 污染，一条 lateral target 仅 15 帧，另一条也存在中心 hazard。尽管其中三条的 lateral target 连续性较高，它们不能作为 clean-lateral negative，因此 official-train 候选池无法单独解决该 P3 scene 的最小 session 配额。

相反，8 条此前未做 exact gate 的 chest center-obstacle 候选中有 3 条通过（`3ok1zz…`、`cBVS…`、`JtMY…`），5 条拒绝。前两条已完成 50 RGB+mask 隔离 draft、本地几何重放和哈希绑定审核请求；selection evidence SHA256 分别为 `93a9a34ebe8cb3b7363c520200f9639c05d82d3af409c695026e093ce764e659` 与 `6b1128d27ba201178bc314043f695971e2580a621e0db0820eb2d7aa2d834ff8`。它们仍是 `pending_review`，不能作为 center 的训练/指标样本；第三条 RGB 下载在时间上限后无 manifest，已停止且保持排除。

## 可恢复执行合同

新增 [run_sanpo_p3_discovery_batches.py](../scripts/run_sanpo_p3_discovery_batches.py) 将完整范围绑定到：官方 session 顺序、扫描参数、view policy、profile、标签、窗口阈值和重试次数。每完成一个 20-session batch 就保存 report SHA256；`--resume` 会在执行新请求前拒绝任何契约或官方 session 顺序变化。只有全部 batch 验证通过时才写 aggregate 并把 checkpoint 标记为 complete。

本次证据在忽略目录：

`artifacts.local/evidence/sanpo-p3-discovery-auto-20260713/`

读取范围仅为 SANPO 官方公开 train segmentation mask 清单与必要的稀疏 mask 像素；未读取 blind 标签、未训练模型、未改 App。

## 下一道可执行门

1. 按 `source_id:native_session_id` 去重，在 chest/head 两个池内分别选出独立 session；同一 native session 不得跨 split。
2. 执行 exact 50-frame geometry、RGB/PII 和人工 scene review；将 HUMAN/MACHINE annotation quality 写入每个 receipt。
3. 仅对完成 review 的 A 层 session 运行 P3 planner，检查每场景 `4–6 train + 2–3 dev`、像素 share、boundary 集中度和 camera×scene 覆盖。
4. 若 head 要与 chest 合并，先以预注册的分层 dev 和 cross-view gate 证明合并不会掩盖 domain shift；否则保留为独立评估/训练实验。

经同意的前向手机 A 层接入、receipt schema 和 planner hard gate 见 [consented capture intake](SANPO_P3_CONSENTED_CAPTURE_INTAKE_2026-07-13.md)。
