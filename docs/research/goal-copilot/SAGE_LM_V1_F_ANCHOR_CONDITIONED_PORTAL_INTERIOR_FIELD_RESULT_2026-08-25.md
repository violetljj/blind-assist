# SAGE-LM V1-F Anchor-Conditioned Portal Interior Field

日期：2026-08-25（Asia/Hong_Kong）

状态：`REVERSIBLE_EXPLORATION / FRESH_SOURCE_FRAME_DEVELOPMENT / TUNED_ON_DEVELOPMENT / R3_SAME_COHORT_TRUE_PAIR_18_OF_24_GEOMETRY_7_OF_24 / V1_F_TRUE_PAIR_0_OF_24_GEOMETRY_10_OF_24 / RETENTION_0_OF_18 / RESCUE_0_OF_6 / STOP_BEFORE_STUDENT / R3_RETAINED / FARO_NOT_RUN / R3_FUSION_NOT_RUN / R6_NOT_RUN / B2_NOT_RUN`

## 问题与实现

V1-F 检验 E0 的失败是否来自“mesh depth jump 直接映射 aperture boundary”这一表示：teacher 不读 RGB，先在 semantic
anchor 周围的官方 ARKitScenes mesh raycast depth 上拟合局部支撑平面；每条 ray 显式分为：

- 命中支撑平面；
- 穿过平面并在更深处命中有效 mesh，因而中间存在可观测后方空间；
- mesh 无命中或平面交点无效，保持 `UNKNOWN`，不当作自由空间。

后方有效 ray 在支撑平面坐标中形成连通场；teacher 选择最靠近 anchor、且两个 source-pose view 的平面区间一致的分量，输出
`portal interior soft mask / center bearing / range / width / target-front waypoint`。左右 image line 只在冻结旧 evaluator 接口处
从选中平面区间派生，不是 teacher 的内部目标。teacher 没有调用 DeepLSD、Canny、RGB gradient、R3 output 或 evaluator truth。

## fresh Development ancestry

materializer 以冻结 V1-B-R2 cohort 为显式排除清单，从 source `first_rgb + materialized window length` 复算旧 raw frame
identity；候选的整个 raw window 只要与旧窗口任一 frame 重叠就排除，并只接受已有官方 `*_3dod_mesh.ply` 的 sequence。
最终 24 cases 来自 12 个 ARKitScenes Training video，包含 307 个互异 raw frame；与旧 cohort 的 362 个互异 raw frame
交集为 `0`。每个 sequence 最多 4 cases，实际 selection score 最低为 `0.772`。

这只建立 **source-frame ancestry fresh**。D1 输出打开后，连通分量聚合在同一 24-case cohort 上修正并形成 D2；另一个 D3
边界汇总诊断退化后被拒绝，代码恢复到 D2。因此最终 operator 对该 cohort 是 `TUNED_ON_DEVELOPMENT`，不是
`SEALED_UNSEEN`、Confirmation 或独立泛化证据。

## 同批 R3 与 V1-F 结果

| 指标 | 冻结 R3（同 24 cases） | V1-F D2 | 晋级要求 | 裁决 |
|---|---:|---:|---:|---|
| true boundary pair | **18/24** | **0/24** | V1-F 至少比 R3 多 3 | FAIL |
| geometry output | **7/24** | **10/24** | V1-F 至少比 R3 多 3 | PASS |
| R3 success retention | n/a | **0/18 = 0%** | 至少 80%（15/18） | FAIL |
| R3 missing rescue | n/a | **0/6** | 至少 3 | FAIL |

V1-F 在 primary representation 上形成 10 条 geometry，较 R3 的 7 条多 3 条；但从 portal field 派生到冻结 evaluator
边界后为 0 条 true pair，且没有保留或救回任何 R3 pair。10 条 geometry 的 median center error=`0.2657 m`，median
range error=`0.4027 m`。这说明当前 connected-behind-plane 场能产出部分非退化几何，却没有与本 cohort 的旧
RGB/depth-supported vertical-opening proxy 对齐；它不能取代 R3，也不能作为增量 teacher。

D1 / D2 / rejected D3 的 `(true pair, geometry)` 分别为 `(0,2) / (0,10) / (0,9)`。三者 true-pair 结论一致，故停止不是
由挑选某个较差汇总版本造成。

## 裁决与边界

四项晋级门仅 geometry uplift 一项通过，最终为 `STOP_BEFORE_STUDENT`。不训练 E0/E1 student，不恢复 R3 fusion，
不补 Faro，不运行 R6/B2，也不接 Android/P1/default App。R3 保持当前 RGB boundary champion。

该结果不否定真实 portal-interior 表示本身。当前自动 cohort truth 是由 RGB vertical line + source depth discontinuity
构造的 opening proxy，并不保证对应 mesh 中一个完整、可穿越、与 anchor 同支撑面的物理 portal；另一方面，ARKit mesh
fragmentation、局部 plane 选择和跨视图连通分量截断也尚未分别消融。现有证据只足以拒绝这套冻结 D2 实现作为下一 student
teacher。

## 复现

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.materialize_rgb_cohort `
  --source-root artifacts.local/datasets/spatial-calibration-head-r1-arkitscenes-20260804/raw `
  --output-dir artifacts.local/evidence/sage-lm-v1f/fresh-portal-cohort-d1 `
  --seed 250825 `
  --exclude-cohort artifacts.local/evidence/sage-lm-v1b/correct-pose-cohort-r2/cohort.json `
  --required-mesh-root artifacts.local/evidence/sage-lm-v1e/privileged-source/raw/Training `
  --cohort-tag V1F --source-start-stride-frames 1 --minimum-selection-score 0.40

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.dense_boundary_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1f/fresh-portal-cohort-d1/cohort.json `
  --deeplsd-root artifacts.local/vendor/DeepLSD `
  --runtime-root artifacts.local/vendor/deeplsd-runtime `
  --checkpoint artifacts.local/vendor/DeepLSD/weights/deeplsd_md.tar `
  --arm b1 --output-dir artifacts.local/evidence/sage-lm-v1f/r3-same-cohort-d1

uv run --python 3.11 --with open3d --with opencv-python-headless --with numpy --with pillow python -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.portal_interior_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1f/fresh-portal-cohort-d1/cohort.json `
  --r3-report artifacts.local/evidence/sage-lm-v1f/r3-same-cohort-d1/report.json `
  --mesh-root artifacts.local/evidence/sage-lm-v1e/privileged-source/raw/Training `
  --output-dir artifacts.local/evidence/sage-lm-v1f/portal-interior-ceiling-d2

E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.test_portal_interior_teacher `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.test_privileged_geometry_teacher
```

本机 artifact：

- `fresh-portal-cohort-d1/cohort.json` SHA-256
  `E8DF42096BB789AD066041635EBFE45C603C75D602EA14D6911390B7F04AB3CD`；
- `r3-same-cohort-d1/report.json` SHA-256
  `455D35391EFE18D94353EAA821C6F8652879378E5D47A239E2F3969F67CD982C`；
- `portal-interior-ceiling-d2/report.json` SHA-256
  `FD3CC11EE0C0D9347D4FBDFD9BB265914F0CD23D4FA26EC51BC61670EC47459D`；
- 每条 case 的 soft mask、三类 ray mask、selected component mask 与 SHA-256 记录在 D2 report rows 中。
