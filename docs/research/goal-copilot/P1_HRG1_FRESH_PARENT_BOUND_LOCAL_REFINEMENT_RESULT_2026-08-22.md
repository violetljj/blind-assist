# P1-HRG1 fresh parent-bound local-refinement result

日期：2026-08-22（Asia/Hong_Kong）

状态：`HRG0_PARTIAL_2_OF_7_AT_K10 / HRG1_NOT_OBSERVED_0_OF_7 / IDENTITY_VERIFIER_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

## 合法输入与 fresh cohort

本轮没有从 evaluator truth 反推 prompt。7 个真实产品任务先冻结为
`NAMED_BUILDING_ENTRANCE -> building entrance`，再解析 OSM 中属于目标建筑 parent way 的 entrance node，最后才访问
Mapillary metadata、pixel 和私有 truth。provider 只看到 public Goal Contract 与 current frame；bbox、visibility 和合法目标框
仅存在于 private evaluator input。

冻结 roster 中 4/7 个目标存在 parent-bound entrance 并取得 11 帧；pre-provider adjudication 得到 7 个 visible frame，覆盖
4/7 个 goal episode，另外 4 帧为 `NOT_VISIBLE`。没有 replacement、resampling、truth 后补图或 provider 后改框。

## 冻结的配对算法

HRG0 保持 commit `837424c5dac8826362bfba74e517ba1c0278e33e` 已冻结的接口：YOLOE semantic support、Grounding
DINO functional proposal、semantic-supported-first 排序、`1.5x` context region、`K=10`。

HRG1 在查看本批 pixel 前以 commit `355cc860e317a2afe83a4bd6ce6c0facd4b48ef5` 冻结：只取 HRG0 Top-5 coarse
region，在每个 crop 内以同一 Grounding DINO、同一 functional prompt 和同一阈值产生 Top-2 local boxes，映射回 full frame，
按 parent rank 再按 local provider rank 排序，最终 `K=10`。没有 identity、named-instance matching、private truth access、
prompt/threshold/model/parent-pool/local-pool sweep 或 retry。

## 一次性配对结果

Primary 是 7 个 visible frame 上 `IoU >= 0.30` 的 bounded candidate Recall@K：

| arm | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| HRG0 coarse functional-context | 0/7 | 0/7 | 0/7 | 2/7 |
| HRG1 Top-5 coarse-to-local refinement | 0/7 | 0/7 | 0/7 | 0/7 |

HRG0 的两个命中分别位于 Anne Frank House frame 02 rank 7（best IoU `0.4014`）和 Kunstmuseum Den Haag
frame 01 rank 10（best IoU `0.3738`）。HRG1 的最高未达标 case 是 Museum Speelklok frame 01，best IoU
`0.2969`；冻结阈值为 `0.30`，因此必须计为 miss，不能降 gate。

终态为：

- HRG0：`P1_HRG0_PARTIAL_HIERARCHICAL_TARGET_AVAILABILITY_ON_FRESH_COHORT`
- HRG1：`P1_HRG1_LOCAL_REFINEMENT_TARGET_AVAILABILITY_NOT_OBSERVED`
- paired verdict：`HRG1_DID_NOT_IMPROVE_HRG0_AND_DID_NOT_ESTABLISH_TARGET_AVAILABILITY`

## 失败层归因与下一边界

结果写完后的只读诊断显示，7 个 visible target center 在 HRG0 coarse pool 中首次被包含的 rank 为
`[none, 7, 9, 6, 8, 1, 2]`。因此 4 个 target 首次进入 coarse parent 时已经晚于 HRG1 的 Top-5 parent gate，1 个在
bounded coarse pool 中完全没有 parent；HRG1 对这些 case 在接口上就不可恢复。剩下进入 Top-5 的两个 case 中，local box
也没有越过 `IoU 0.30`。

这只支持一个窄结论：固定的 Top-5 parent-rank local refinement 没有在本 fresh cohort 建立 proposal availability；失败同时包含
coarse parent selection coverage 和 local localization precision，不能归因为 instance identity。Contrastive Identity Verifier、
AMRM 恢复和 App 集成仍不授权。

如果启动后继实验，必须使用另一批 fresh cohort，并先冻结一种不依赖 private entrance truth 的 public spatial contract 或
global local-candidate reranking contract；本 cohort 不允许扩大 parent pool、改排序、改阈值或重跑。

## Evidence identity

- formal run manifest SHA-256 `d0b4c70faa6e5f9275703b8a77c87d0994095922a99245a102350c7e4ac56620`
- HRG0 prediction SHA-256 `e421aacdc88156b00af08aefc53f7948c7e30054ec054bf0d8cb51474b86a484`
- HRG0 evaluation SHA-256 `ea83bbb1afabcc24e667d9181f74d0f194444922a6b50c9cf871d81aab6ef502`
- HRG1 prediction SHA-256 `70707bd6f8159ad4e10bce71c8a5bf7f148322a7906f43e2610f9c221b8b1821`
- HRG1 evaluation SHA-256 `d19f0e1d4d87ddc1bd61621bc5f8793ae90e90cdb131ddbd743f25f04ca7113b`

Claim ceiling：`FRESH_PAIRED_PROPOSAL_AVAILABILITY_ONLY_NO_IDENTITY_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM`。
