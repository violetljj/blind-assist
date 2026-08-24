# Near-Identity Hard-Negative Unary V0 Result

状态：`DEVELOPMENT / CONSUMED_SINGLE_RUN / NEAR_IDENTITY_HARD_NEGATIVE_UNARY_MIXED_WITH_COLLATERAL_DEVELOPMENT / RELIABLE_VERIFIER_NOT_ESTABLISHED / NO_P1 / DEFAULT_APP_UNCHANGED`

## 判决

预注册的 DINOv2-S mean-pool + 小型 near-identity projection head 不晋级。它在 135 个 sealed target-present pair
上相对冻结 baseline 产生 `4 rescue / 17 collateral / net -13`，且 baseline-correct controls 只保留
`1/18=5.6%`。Challenger 的 target-absent false accept 为 0，但只接受 `5/135=3.7%` 的 present pairs；
因此这不是可靠拒绝，而是 coverage gate 明确拦截的近乎全 abstain。

终态为：

```text
NEAR_IDENTITY_HARD_NEGATIVE_UNARY_MIXED_WITH_COLLATERAL_DEVELOPMENT
RELIABLE_VERIFIER_NOT_ESTABLISHED
NO_P1
DEFAULT_APP_UNCHANGED
```

这只拒绝本次冻结 arm：相同 DINOv2-S backbone、384→256→128 projection head、固定 NearID-style loss 和
CORe50 三重 source-disjoint protocol。它不等同于运行或否定官方 NearID SigLIP2/MAP checkpoint，也不否定所有
learned identity objective、layout evidence 或单张 RGB 可解性。

## 运行身份与密封顺序

pre-outcome protocol 已先作为提交 `57a063b0` 推送，之后才下载数据。官方 128×128 CORe50 archive 为
`5,892,103,007` bytes，SHA-256 `f3d65017490dfc4392bc5e41b66930600f1d00898cae47fc4bd1a01cf92e1c6a`；
ZIP central directory 含 165,428 entries。roster 在不解码像素时冻结，计数为：

```text
selected samples       465
train tuples           180
calibration pairs       60 present + 60 absent
sealed test pairs      135 present + 135 absent
```

train、calibration、test 的 physical object、category、session 三重不相交。train/calibration feature、最终 step-1200
head、两臂阈值和 quality cutoffs 写入 `pretest-lock.json` 后，程序才 materialize `s8--s11` test pixels。

| 收据 | body SHA-256 |
| --- | --- |
| protocol freeze | `1d9eb30a30f6e8264be41409d8f817346e3bd5ec3e0b13a198d42ca62c3fd56b` |
| frozen roster | `4fe6b044d89ef6e6a060d863829f082bf15438cd51016abeda56bcada4b78a5f` |
| pre-test lock | `7431e078dc44d52cc7248448ed5446dbe6c78f03087dd6a85e6bfc7b46f4c0fa` |
| raw test scores | `4f056d463625ec1b60876b6a1d1aa8ad797e4c0ab8a26c5d091dbe1d92f5187a` |
| final report | `afdd8af798ba73045b262f276fc0c4fa46d9a1a1d96e858fa77bd4e77ae6a74a` |

最终 head SHA-256 为 `d2da094587ee288a1f07c460649f4b5d0335ea1e3f9fdcabf2460493e772e893`。
正式 artifacts 位于
`artifacts.local/evidence/public-identifiable-referent-near-identity-hard-negative-unary-v0/run-20260824T073300Z/`；
`final-report.json` 已存在，入口拒绝覆盖或重跑。

## 冻结训练与 calibration

训练 loss 从 step 1 的 `1.40317` 降至 step 1200 的 `0.000518`，最终 step 按协议直接采用，没有 early stopping、
checkpoint selection 或 outcome sweep。低 train loss 与 sealed generalization 失败并存，支持的是 seen tuples 上拟合充分，
不支持跨新 physical objects/categories/sessions 的 identity rule 已建立。

两臂各自在 calibration 的 60 个 absence pair 上允许最多 `3/60=5%` false accept：

| arm | 冻结阈值 | calibration absence false accept |
| --- | ---: | ---: |
| DINOv2-S symmetric local baseline | `0.6472855210` | `3/60` |
| near-identity projection challenger | `0.7848621607` | `3/60` |

quality Laplacian-variance cutoffs 冻结为 `815.13181 / 1239.43953`。CORe50 未提供本协议可用的 occlusion
truth，因此该分层保持 `NOT_EVALUABLE_SOURCE_LABEL`。

## Sealed test 指标

| metric | Baseline | Challenger |
| --- | ---: | ---: |
| same-instance recall | `18/135=13.3%` | `5/135=3.7%` |
| same-class false commit | `9/135=6.7%` | `0/135=0%` |
| accepted coverage | `27/135=20.0%` | `5/135=3.7%` |
| selective risk | `9/27=33.3%` | `0/5=0%` |
| target-absent false accept | `19/135=14.1%` | `0/135=0%` |
| ordinary-negative false accept | `5/135=3.7%` | `0/135=0%` |
| candidate permutation invariance | `100%` | `100%` |

Challenger 的三组 test candidate-session recall 为 `2.2% / 2.2% / 6.7%`，最低组低于冻结的 40% 门；
组差仅 4.4 个百分点虽然通过 gap 门，但表示的是跨组一致地低，而不是稳定身份能力。category recall 也只有
`2.2% / 2.2% / 6.7%`。quality 三组均约 2.5%--4.3%，没有一个分层提供可晋级的覆盖。

## 预注册 gates

| Gate | 结果 |
| --- | --- |
| `rescue > collateral` | **FAIL**：`4 <= 17` |
| control retention `>=80%` | **FAIL**：`1/18=5.6%` |
| permutation invariance `=100%` | PASS |
| test target-absent false accept `<=5%` | PASS：`0/135` |
| ordinary-negative false accept `<=5%` | PASS：`0/135` |
| accepted coverage `>=50%` | **FAIL**：`5/135=3.7%` |
| matched-context construction | PASS |
| worst session recall `>=40%` | **FAIL**：`2.2%` |
| session recall gap `<=35pp` | PASS：`4.4pp` |

absence 与 ordinary-negative gates 的通过完全由极低 accepted coverage 限制；coverage gate 正确阻止把它解释为
calibrated `NONE` 成功。Baseline 也从 calibration 的 5% 漂移到 sealed test 的 14.1%，说明即使冻结阈值，
跨 category/session 的绝对分数仍明显不稳定。

## 科学边界与下一路由

本 arm 证明：仅在冻结通用 DINO 表示上加一个小型 hard-negative projection objective，可以把训练 loss 压低，
但没有迁移成 source-disjoint identity margin；其净效应比 baseline 更差。不得在已打开结果上扫 head 宽度、step、
temperature、rank weight、sampling、threshold 或 DINO+head fusion。

这次没有运行官方 NearID checkpoint；因此官方 NearID architecture 仍可在完全新协议中另作新 arm，但不能把本结果
写成“NearID 普遍失败”。同时，审计中的 `LOCAL_LAYOUT_INFORMATION_LOST=3` 仍只保留为独立假设，不能用本次失败
事后混入 layout fusion。当前主状态仍是：exact-instance representation 未建立、open-set physical authority 未建立、
P1 与 App 不启动。
