# GRAIL M1 V2b Development Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / EVALUABLE / NO_CLEAR_UPLIFT_OVER_STRONGEST_BASELINE / FORMAL_TEST_UNOPENED / STOP_BEFORE_M2 / DEFAULT_APP_UNCHANGED`

V2b 在模型或 test outcome 前移除 V1 target-centering leak：每个 query position 只从目标方向附近 `{-30,0,30}` 度按 sample hash 选择一个 yaw，采用首个目标可见位置。train 418 positives、257 wrong-target cases；dev 78 positives、43 wrong-target cases。train 目标的水平绝对偏心中位数为画面宽度 16.7%，仅 66/418 在中心 ±5%，故 B2 不再能依赖固定正前方目标。

冻结 DINOv2-S pooled/local-token features、Depth-Anything-V2-S、B0/B1/B2、K=3 GRAIL heads 与 dev guardrail threshold 后：

| 方法 | Interaction Pose Success | Wrong-target | Absence false commit | Permutation |
|---|---:|---:|---:|---:|
| B0 pooled cosine + fixed 1 m | 14/78 | 18/43 | 3/78 | n/a |
| B1 pooled cosine + relative depth | **23/78** | 18/43 | 3/78 | n/a |
| B2 direct single pose | 17/78 | NOT_EVALUABLE | 29/78 | n/a |
| GRAIL local-token factorized K-set | 22/78 | **16/43** | **3/78** | 78/78 |

B2 不输出 candidate identity，且本 collection 没有每个 distractor 的 pose truth，因此旧 evaluator 的 `0/43` 只是未分配 selected candidate，不能解释为零 wrong-target；结果表改记 `NOT_EVALUABLE`。

只读层级归因：oracle target candidate 下 GRAIL pose head 成功 `64/78`；referent top-1 选中目标 `44/78`，二者同时成立 `34/78`，加入为 wrong-target/absence 预注册 guardrail 选择的拒绝阈值后最终为 `22/78`。主要瓶颈仍在 reference-conditioned referent selection/abstention，不在 set-valued pose head。

## 裁决

GRAIL 的 wrong-target 略优于 B1、absence 相同、permutation 全过，但主 Interaction Pose Success 为 `22/78`，没有超过最强 baseline B1 的 `23/78`，更不满足 V2 预注册的 +10 个百分点门。因此没有清晰 M1 信号：V2 test roster 保持未采集，禁止 formal test 重放、threshold/loss/K/head sweep、B2/GRAIL fusion、长期记忆、主动搜索、M2 temporal belief 与 Android 三环境测试。

M0 native teacher upper bound 仍成立；本结果否定的是当前 `synthetic RGB + oracle masks + reference-only frozen DINO evidence` 已足以建立 GRAIL student 优势。若未来另开 successor，必须改变独立 referent 信息源（例如可信语义/关系目标权威），不能继续消费同一 reference-only arm。

证据 identity：train collection SHA-256 `2fcfbbd8f022dbefb31011a457dd75ebf400b5745cefa5fbeaca5d23097ec03f`；dev collection `5a7478f4ccd871f684f318f278ae56e34ce1163f58b81ddc245072b1a13f0037`；checkpoint `d838e8c1f648a771a41a32df7cbc0146b6bcebe98715fcd7f7c6c24ed7988b18`；development result `9657dd5b9306b1400a2dc1ef3e5fa7e0db26b2bcce7f81b82aa5b539e9a83fcb`。Claim ceiling 仅为 synthetic ProcTHOR Development；无正式 test、自然场景、proposal、text-goal、Android、用户、产品或安全结论。
