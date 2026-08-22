# P1-PA1 target proposal rescue result

日期：2026-08-22（Asia/Hong_Kong）

终态：`P1_PA1_FIXED_TILED_SCALE_RESCUE_NOT_SUPPORTED_ON_FAILURE_COHORT`

证据角色：`POST_OUTCOME_SELECTED_CONSUMED_DEVELOPMENT_MECHANISM_DIAGNOSTIC_ONLY`

## 最窄答案

P1-PA1 在与 sealed PA0 完全相同的 7 个 target-visible first-poison frames、YOLOE-26n-seg checkpoint、visual
exemplar、score floor 和 K=10 contract 上，只把 full-frame 640 输入替换为固定 `2x2 / 20% overlap / 每 tile 640`
搜索。它没有建立预注册的主要信号：

> `IoU >= 0.30 Recall@10` 仍为 `0/7`；global-dedup 后的完整 postprocessed rank pool 也为 `0/7`。

因此，K=10 截断不是主门失败的原因。固定 tiled zoom 没有证明“让同一 provider 看清目标”即可恢复 proposal。

## PA0 对照

| 指标 | sealed PA0 full frame | PA1 fixed tiled |
|---|---:|---:|
| IoU >= 0.10 Recall@1 | 0/7 | 0/7 |
| IoU >= 0.10 Recall@3 | 0/7 | 1/7 |
| IoU >= 0.10 Recall@5 | 0/7 | 1/7 |
| IoU >= 0.10 Recall@10 | 2/7 | 1/7 |
| IoU >= 0.30 Recall@10 | 0/7 | 0/7 |
| IoU >= 0.50 Recall@10 | 0/7 | 0/7 |

PA0 的 5 个 IoU >= 0.10 absent cases 被救回 `0/5`。PA1 唯一 bounded hit 仍是 wall clock：从 PA0 rank 9、
best IoU `0.2179` 移到 rank 3、best IoU `0.2297`。wine rack 在 PA1 的完整 postprocessed pool 中仍有弱
IoU >= 0.10 candidate，但只位于 full rank 31（best IoU `0.1466`），被 K=10 cap 排除；其 bounded best IoU
仅 `0.0981`。因此 tiled search 改善了一个既有弱 candidate 的 rank，同时让另一个既有弱 candidate 被新增 proposal
竞争挤出，没有建立新的 target availability。

## 完整 rank 的 failure attribution

PA1 保留每个 tile 的 provider-postprocessed candidates、映射回全图的 bbox、global dedup decisions、完整 rank 和 K=10
cap decisions。per-tile pre-NMS proposals 仍不由 provider 暴露。

- IoU >= 0.10：full-rank recall `2/7`；其中一个在 rank 31，属于 bounded ranking/cap loss；
- IoU >= 0.30：full-rank recall `0/7`，七帧都没有足够质量的 postprocessed target candidate；
- IoU >= 0.50：full-rank recall `0/7`；
- pre-NMS/NMS 内部归因：`NOT_EVALUABLE_PROVIDER_INTERFACE`。

其余五帧的 full-rank best IoU 为 door A `0.0238`、wall artwork `0.0202`、door C `0.0756`、pot `0.0`、
smart-home display `0.0`。这包括 shortest side `46/57 px` 且 visibility `0.992/1.0` 的中等目标，所以本结果
进一步削弱“失败主要只是目标像素太小”的解释；它不证明所有多尺度搜索无效。

## Proposal 与 compute 增长

- 7 帧共 28 个 tile inference，严格为 PA0 的 4 倍输入图像数；
- per-tile postprocessed candidates：总计 437，mean `62.43/frame`；
- global dedup 后 full-rank candidates：总计 139，mean `19.86/frame`；
- bounded pool：总计 62，mean `8.86/frame`，6/7 帧达到 K=10；
- latency median `130.4 ms`、P95 `1639.2 ms`，P95 含 cold start；
- peak CUDA allocated/reserved：约 `412.5/834.0 MiB`。

PA1 增加了 proposal 数量和 compute，但没有增加 IoU >= 0.30 availability。

## 决策边界

- 关闭这一固定 tiled-scale rescue，不在已打开结果上搜索 tile 数、overlap、resolution、score floor、NMS 或 K；
- AMRM、reacquisition、verifier、VLM、VIO/SLAM、geometry 与 App 继续冻结；
- contrastive verifier 仍为 `NOT_EVALUATED`，proposal availability 尚不足以把它提升为主问题；
- 结果将主分叉继续推向 target representation / target-conditioned grounding mismatch，但不自动授权 parent-first、
  part/region grounding 或新模型 arm；
- parent-first 若后续单独建立，parent relation 必须来自公开冻结 target specification，不能读取 ADT private category/
  instance truth。

Claim ceiling：`FAILURE_COHORT_SCALE_RESCUE_MECHANISM_ONLY_NO_MODEL_SELECTION_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM`。

## Evidence

- `artifacts.local/evidence/p1_pa1_tiled_rescue_v1/manifest.json`
  SHA-256 `0006c97d6eb9c9694edf97299db4d01425ba9f5d011cd9eb82609cd004153ad0`
- `prediction.json` SHA-256 `1f9f06a817ccaeadafe5df6c96897de441a5af3d84d0106386710afe2bbbea47`
- `evaluation.json` SHA-256 `1eae8b2ea9b2957775b7510fe6477830a4f41e0db27c909b3b0584019d1031c4`

默认 App：不变。
