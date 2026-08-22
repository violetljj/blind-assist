# P1-PA3 goal-semantic 与 functional-region proposal 结果

日期：2026-08-22（Asia/Hong_Kong）

状态：`PA3_SEMANTIC_TARGET_AVAILABILITY_NOT_OBSERVED / FRG1_PARTIAL_FUNCTIONAL_REGION_SIGNAL_ON_CONSUMED_DEVELOPMENT_COHORT / HRG0_FRESH_SINGLE_VISIBLE_CASE_RANK1 / DEFAULT_APP_UNCHANGED`

## 先建立合法输入

本轮没有从旧 7-case private category 回填 prompt。先冻结两个 product-authored Goal Contract cohort，统一通过
`NAMED_BUILDING_ENTRANCE -> "building entrance"` 全局映射生成 canonical prompt：

1. station-centroid prospective acquisition：7 个 goal 先于本项目首次 pixel access 与 truth；几何规则物化 4 帧，
   private pre-provider adjudication 为 `0 VISIBLE / 4 NOT_VISIBLE`。该 cohort 不运行模型、不替换样本，证明 place-centroid
   heading 不是足够的 target-visible observation acquisition interface。
2. fresh museum entrance-anchor development：另冻结 7 个新 goal，再用 OSM `entrance=*` 节点冻结采集 anchor；物化
   `6/7` 帧，private pre-provider adjudication为 `2 VISIBLE / 3 NOT_VISIBLE / 1 UNADJUDICABLE`。公开 Mapillary 原图的
   物理拍摄早于 goal，因此 receipt 只主张
   `GOAL_BEFORE_FIRST_PROJECT_PIXEL_ACCESS_AND_TRUTH`，明确不主张 goal 先于 physical capture。

所有零候选、不可见和不可裁决样本均保留；没有 outcome 后换地点、补图或重新标成 negative。

## PA3：YOLOE goal-semantic proposal

合法 public/private input 后只执行一次 `YOLOE-26n-seg / Ultralytics 8.4.52 / imgsz=640 / conf=0.001 /
max_det=100 / K=10` semantic proposal。Primary 仍为 visible `SET_VALUED` case 上 IoU >= 0.30 的
Recall@1/3/5/10：

| endpoint | 结果 |
|---|---:|
| evaluable | 2 |
| Recall@1 | 0/2 |
| Recall@3 | 0/2 |
| Recall@5 | 0/2 |
| Recall@10 | 0/2 |
| terminal | `P1_PA3_SEMANTIC_TARGET_AVAILABILITY_NOT_OBSERVED_ON_COHORT` |

Van Abbemuseum 与 Frans Hals Museum 的 best IoU 分别为 `0.2769`、`0.1005`；两者在 IoU >= 0.10 都有 proposal，
但不能改阈值升级 PA3 primary。post-outcome functional-anchor diagnostic 只报告 target anchor 是否落在候选 region：
`@1=1/2 / @3=@5=@10=2/2`。这说明 region/extent interface 有信号，不把它重写成 PA3 成功。

### provider dependency caveat

首次 text prompt 调用时 Ultralytics 自动取得 `mobileclip2_b.ts`，实际 SHA-256 为
`35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f`。原 run manifest 未在调用前冻结该
text-encoder 哈希，因此 PA3 只保留为 consumed development result，不能升级为严格 one-shot confirmation。
当前 runner 已 fail-closed 要求显式 `--text-encoder`、固定文件名和上述哈希，并从指定本地目录加载；本 cohort 不重跑。

## FRG1：pre-existing frozen functional-region prompt

PA3 terminal 后，复用在其 outcome 前已经冻结的 Grounding DINO Tiny：revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`，prompt
`door . doorway . entrance . building entrance . storefront entrance . gate .`，box/text threshold `0.15/0.10`，
NMS `0.50`，K=10。它没有使用 private truth，也没有 prompt/threshold/model sweep。

| endpoint | 结果 |
|---|---:|
| evaluable | 2 |
| Recall@1/3/5 | 0/2 |
| Recall@10 | 1/2 |
| Frans Hals first correct rank | 6 |
| Van Abbe best IoU | 0.2894（仍为 fail） |
| terminal | `P1_FRG1_FUNCTIONAL_REGION_AVAILABILITY_OBSERVED_ON_CONSUMED_COHORT` |

因此唯一窄结论是：functional-region proposal 在 consumed development cohort 上建立了部分 bounded availability，
但没有建立稳定 coverage。它授权下一步构造 fresh hierarchical grounding contract；不授权 Contrastive Identity
Verifier、AMRM、按当前 outcome 拼框、调阈值/prompt/model、App 集成或产品/安全主张。

## Evidence identity

- PA3 prediction SHA-256 `a712aa2337aef9d1b1fdf3900671ef79e294ac0d03c7013367da231e453556bd`
- PA3 evaluation SHA-256 `c9f693d43cb00a4066fb3ab95afbd3b00ad64f2149bbd49a39c7a7b4290e8889`
- anchor diagnostic SHA-256 `c64f7fa8ee90427ea4690a206e50cfb2a234d5094a145bc4d1eed6323d55c47a`
- FRG1 prediction SHA-256 `e8955c32cfba2176aeed59707a4fdfe448477df8358617cd996dcaaa92249c54`
- FRG1 evaluation SHA-256 `831be73ffe74ab9ce06f3607d06aa12aecd996d953eb485343e8fa7c8c7e48f8`

Claim ceiling：`CONSUMED_DEVELOPMENT_GOAL_SEMANTIC_AND_FUNCTIONAL_REGION_PROPOSAL_MECHANICS_ONLY_NO_FRESH_CONFIRMATION_IDENTITY_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM`。

Fresh successor 已执行，结果见
[`P1-HRG0 fresh hierarchical functional-context result`](P1_HRG0_FRESH_HIERARCHICAL_FUNCTIONAL_CONTEXT_RESULT_2026-08-22.md)。
它在唯一 visible fresh case 上 rank 1 命中，但 `1/4` captured visibility yield 不足以形成泛化确认或授权 identity。
