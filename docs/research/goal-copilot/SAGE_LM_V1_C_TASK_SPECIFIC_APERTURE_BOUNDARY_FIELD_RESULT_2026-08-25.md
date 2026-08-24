# SAGE-LM V1-C Task-Specific Aperture Boundary Field

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / V1_C0_TASK_BOUNDARY_FIELD_BELOW_R3 / V1_C1_ANCHOR_CONDITIONED_FIELD_BELOW_R3 / R6_NOT_RUN / B2_NOT_RUN`

## 问题与固定面

V1-C 结束 DeepLSD/handcrafted selector 延续，直接从 RGB 预测 `P(left boundary)` 与
`P(right boundary)` 两张热图。C0/C1 使用相同轻量 RGB encoder/decoder；C0 不接 anchor，C1 只在 decoder
接入 anchor bbox heatmap。评估前两臂都遮掉 composited exact-anchor 的 RGB 像素，避免读取二维码/文字纹理；每个
frame/role 固定保留 top-8 peak，再进入 R2 source pose、9 px oracle localization 与原 triangulation。confidence、arrival、
R6、B2 均不裁决，也不修改现有 24-episode ARKitScenes Development cohort。

## 训练分母审计

排除评估 cohort 的 11 个 source sequence 后，余下 9 个本地 ARKitScenes Training sequence 在原双视图 motion、
visibility 与 opening-proxy gate 下为 `0` 个合格 episode；因此不能声称可直接生成几千个同合同双视图样本。

首个 r1 使用既有 TartanAir source-native door semantic mask：`1,100` 帧，train/val=`689/411`，包含
750 positive、200 hard-negative、150 background-negative。最佳 synthetic validation 双边界 Recall@8 为
C0=`244/290`（84.14%）、C1=`234/290`（80.69%），但迁移到 ARKitScenes 后分别仅 `2/24` 与 `0/24`
四边界直接覆盖，且 true pair/geometry 均为 `0/24`。该结果定位出明显的 semantic-door-mask 到 real
open-aperture 域/标签错配。

随后只移除训练不需要的 active-pair 门，对剩余 9 sequence 的全部 `1,350` 帧执行同一 RGB vertical-line +
source-native depth-discontinuity opening proxy。自动得到 `336` 帧；按 source sequence 隔离为 7 sequence / 260 train
与 2 sequence / 76 validation。它不是 ARKitScenes 官方 door annotation，也不是真实弱边界人工真值；selection 本身偏向
强可见竖线，provenance 保持为 `AUTOMATIC_OPENING_PROXY`。

## 同域 r2 结果

| 指标 | R3 DeepLSD | V1-C0 RGB field | V1-C1 + anchor | 目标 |
|---|---:|---:|---:|---:|
| held-out train-domain 双边界 Recall@8 | n/a | 57/76 (75.0%) | 55/76 (72.37%) | diagnostic |
| 24-episode 四边界 Recall@8 | n/a | **1/24** | **4/24** | >=20/24 |
| true boundary pair available | **15/24** | **1/24** | **3/24** | >=20/24 |
| geometry output | **13/24** | **1/24** | **3/24** | >=20/24 |
| missing | 9/24 | **23/24** | **21/24** | <=4/24 |

C1 相对 C0 有小幅恢复，但仍大幅低于 R3，也没有建立 task-specific representation uplift。三个 C1 geometry 的
median center error=`0.0388 m`、median range error=`0.1757 m`，说明少数命中仍可进入冻结 triangulation；主失败仍是
真实四边界 coverage，而不是 confidence 或 arrival。

## 裁决边界

本轮拒绝的是 `lightweight CNN + automatic strong-line/depth-discontinuity opening-proxy supervision`，以及当前
synthetic bbox-heatmap anchor conditioning；它没有证明 task-specific RGB boundary supervision 家族无效。训练分母虽然
source-disjoint，但只有 336 个自动正例，且标签生成机制系统性缺少目标中的弱/无显著线 aperture boundary，不能用更多 epoch、
head、loss、top-k 或在已打开 24 条上调参补救。

DeepLSD/handcrafted proposal family 保持终止；V1-C0/C1 当前实现同样终止。新的 task-specific successor 必须先提供
独立的、覆盖弱 aperture boundary 的 source-native/人工边界标签分母，不能把现有 opening proxy 重采样或改阈值冒充新信息。
R6/B2 保持不运行，Android/P1/default App 不变。

本机证据：

- TartanAir r1 `artifacts.local/evidence/sage-lm-v1c/task-specific-aperture-boundary-field-r1/report.json`
  (SHA-256 `561CBA97D6964FC741231135AC8091FDA8B46164D9771E3B489559925C988FF2`)；
- ARKitScenes source-disjoint materialization
  `artifacts.local/evidence/sage-lm-v1c/arkitscenes-source-disjoint-boundary-train-v1/receipt.json`
  (SHA-256 `37052397A9692BD15F78947168B8D4E7F176D7E19294C49303317CEB2B7BF382`)；
- ARKitScenes r2 `artifacts.local/evidence/sage-lm-v1c/task-specific-aperture-boundary-field-r2-arkitscenes/report.json`
  (SHA-256 `DA390E444C14F4CCAA50F82BC9BD5A961AAA714C621ED467DFFF0789A9F904D9`).
