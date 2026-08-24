# SAGE-LM V1-E0 Privileged Geometry Teacher Ceiling

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / ARKIT_3DOD_MESH_TEACHER_LOW_CEILING / FOUR_BOUNDARY_10_OF_24 / TRUE_PAIR_9_OF_24 / GEOMETRY_9_OF_24 / R3_MISSING_RESCUE_3_OF_9 / STOP_BEFORE_STUDENT / R3_RETAINED / R6_NOT_RUN / B2_NOT_RUN`

## 问题与独立性

E0 只问 source-native privileged geometry 能否在冻结 V1-B-R2 24 episodes 上覆盖 R3 遗漏的 aperture boundary；不训练
RGB student，不融合 R3，也不运行 R6/B2。最终有效执行固定使用
`artifacts.local/evidence/sage-lm-v1b/correct-pose-cohort-r2/cohort.json`，其 SHA-256 为
`AB085770CB8EBD539B35DCD8C7F0BE8E4288BFAC96D2ABED26815484FF165CAB`。

teacher 对 11 个 source sequence 的官方 ARKitScenes `*_3dod_mesh.ply` 用官方 camera pose/intrinsics 做 256×192 raycast，
得到 metric depth 与 camera-space surface normal。相隔 2 px 的有效 ray 形成 signed depth-jump：近表面→更深内部为 LEFT，
反向为 RIGHT；竖直 coherence 形成 heatmap，mesh hole、无 hit、无 normal 和越界 depth 全部进入 invalid/ignore，而不是负标签。
首帧 top-8 boundary 再由 aperture-interior 深度提升到 3D，并投影到冻结 active view；该步骤只使用 mesh、pose、intrinsics，
不读取 RGB 内容。

每条 episode 保存两视图的：

```text
metric depth + surface normal
+ boundary heatmap
+ signed depth-jump field
+ label-valid mask
```

teacher generator 未调用 DeepLSD、Canny、RGB gradient、V1-C strong-line proxy、R3 output 或 evaluator truth。RGB 只在最终
overlay 中作为底图；绿色/洋红 evaluator line 只负责 outcome 可视化和 9 px 计分，不回流生成器。top-8、9 px、pairing 与
R2 triangulation 保持不变。

## Faro source support 边界

固定 24 条中，官方 3DOD mesh 对 `24/24` frame pair 可按相同 pose raycast；官方 Faro-projected high-resolution depth 只覆盖
部分 sequence，且 filtered 10 FPS 时间戳通常不与冻结 frame pair 重合。本地核对中仅 `qr_entrance-01` 与
`qr_entrance-03` 两条的两端都在 `0.10 s` 内有 Faro depth。E0 因此选择全分母一致的 3DOD mesh teacher，不把稀疏、
异时刻 Faro depth 混入 24 条主指标。当前结果关闭的是这套 **ARKit 3DOD mesh 表达 + aperture 映射**；它不是完整 Faro
depth ceiling，也不能外推为所有 privileged geometry 失败。

## 结果

| 指标 | R3 DeepLSD | V1-E0 mesh teacher | 继续门 / 裁决 |
|---|---:|---:|---|
| four-boundary Recall@8 | n/a | **10/24** | 低于所需完整覆盖 |
| true boundary pair | **15/24** | **9/24** | 未严格超过 15/24 |
| geometry output | **13/24** | **9/24** | 未严格超过 13/24 |
| R3 missing rescued | n/a | **3/9** | 未达到 6/9 |
| R3 pair retained | 15 | **6/15** | 丢失 9 个已有 pair |

九个输出 geometry 的 median center error=`0.0202 m`、median range error=`0.0844 m`。命中后的 metric geometry 质量良好，
失败层仍是 boundary coverage，而不是 triangulation 数值崩溃。三个新 rescue 为
`exact_shelf_target-01 / qr_entrance-01 / room_sign-08`，但 teacher 同时丢失九个 R3 pair；它不能作为 R3 的独立增量通道。

## 裁决

三个预定继续门全部失败：`3/9 < 6/9` rescue、`9/24 < 15/24` pair、`9/24 < 13/24` geometry。因此 V1-E 在 E0
停止，不训练 V1-C1 student、不训练新 backbone、不加入 signed-field student second arm，也不融合 R3。R3 继续是 champion。

该负结果定位为当前 ARKit mesh surface reconstruction / depth-jump-to-aperture 定义不足；其中哪一项主导尚未分离。只有未来
先建立与冻结 frame pair 真正同时间、同相机对齐且覆盖足够分母的 Faro depth，或另立独立 aperture truth/mesh mapping，才可
重新授权新的 teacher protocol；不得在当前已打开 24 条上扫 jump、coherence、top-k、pair score 或 localization gate。

本机证据：

- `artifacts.local/evidence/sage-lm-v1e/privileged-geometry-teacher-ceiling-e0-r1-r2-cohort/report.json`
  (SHA-256 `BB018B5ABAC250848ABA0060477FE19AAEE1D474CBD261951B55A560D11175B3`)；
- `teacher-ceiling-24-case-overlay.png`
  (SHA-256 `284690976E253BFC3DB4F3A1C17B915D0D2FB9DA4E088C018022E5316E6E4292`)；
- 每条 episode 的 compressed teacher fields 与独立 SHA-256 记录在 report rows 中。
