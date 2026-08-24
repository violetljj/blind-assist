# Near-Identity Hard-Negative Unary V0 Protocol

状态：`FROZEN_BEFORE_DATA_DOWNLOAD_OR_MODEL_EXECUTION / DEVELOPMENT / EXECUTION_AUTHORIZED / NO_OUTCOME`

## 唯一问题与 claim ceiling

`NEAR_IDENTITY_HARD_NEGATIVE_UNARY_V0` 只回答：在相同冻结 DINOv2-S backbone 和新鲜、source-disjoint
数据上，显式 `same identity > near identity > ordinary negative` 训练目标是否比通用 appearance 表示更可靠地
区分同类不同物理对象。它不是 NearID 论文复现，也不回答定位、建筑入口、proposal、native `NONE` head、belief、
tracker、P1、控制、安全或 App。即使通过，最高结论仍是 Development identity signal，不是 physical-instance authority。

旧 C2、two-reference 与 T-LESS/PDM cohort 只用于提出本假设；不得进入训练、校准、threshold、checkpoint、
结构选择或本次测试。PDM、fusion、layout verifier、REBASE、Deep Sets、multiple references、threshold sweep、
Active Search、belief/tracker/P1 和默认 App 接线全部禁止。

## 冻结数据

唯一新数据为官方 CORe50 `core50_128x128.zip`，固定来源为
`http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip`，预期字节数 `5,892,103,007`。
下载后先记录 archive SHA-256，再只读取 ZIP central directory，以固定规则生成 roster；此时不得解码像素。

| split | category / physical object | source session | 用途 |
| --- | --- | --- | --- |
| train | categories 1--4 / `o1--o20` | `s1--s4` | 训练唯一 projection head |
| calibration | categories 5--7 / `o21--o35` | `s5--s7` | 两臂绝对阈值与质量分层 |
| sealed test | categories 8--10 / `o36--o50` | `s8--s11` | 单次最终判决 |

对象、类别和 session 必须三重不相交。每个 category 固定含连续五个对象。帧按 ZIP member 名排序，并以
`floor(q*(n-1))` 选定。train 使用 `q={.25,.50,.75}`，anchor 为 `s1`，positive 为 `s2--s4`；
calibration reference 为 `s5/q=.50`，candidate 为 `s6--s7/q={.33,.67}`；test reference 为
`s8/q=.50`，candidate 为 `s9--s11/q={.25,.50,.75}`。同一 pair 的 target、hard negative 与 ordinary
negative 必须共享 candidate session 和 quantile；candidate slot 由 `SHA256(pair_id)` 最低位确定。

对每个 anchor，hard negative 是同 category 的下一个对象（五对象循环）；ordinary negative 是下一个 split
category 中相同序号的对象（category 循环）。预期 train tuples=`180`、calibration present/absence pairs=`60/60`、
sealed test present/absence pairs=`135/135`。任何缺对象、缺 session、重复 member、计数漂移或跨 split 泄漏均为
`NOT_EVALUABLE`，不得换图补齐。

## 冻结两臂

Baseline 是既有 `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`：224×224
ImageNet normalization、最后层 16×16×384 L2-normalized patch tokens；候选分数是既有双向 mean-nearest
patch cosine。Challenger 使用完全相同 patch tokens，先对 256 tokens 求均值并 L2 normalize，再通过唯一 head：

```text
Linear(384,256) -> GELU -> Linear(256,128) -> L2 normalize
```

head 以 seed `20260824` 初始化，AdamW `lr=1e-3 / weight_decay=1e-4`，batch `32`，固定 `1200` steps，
不用 early stopping 或 checkpoint selection。每个 tuple 为 `(anchor, positive, hard, ordinary)`；冻结损失为：

```text
L_disc = -log softmax([sim(anchor,positive), sim(anchor,hard), sim(anchor,ordinary)] / 0.07)[0]
L_rank = softplus((sim(anchor,ordinary) - sim(anchor,hard)) / 0.07)
L      = mean(L_disc + 0.5 * L_rank)
```

只使用最终 step-1200 head。两臂均逐候选独立打分；candidate 集合、slot 或另一个候选不得进入 encoder/head。

## 校准、密封与判决门

每臂只在 60 个 calibration target-absent pair 上选一次绝对阈值。令允许 false accept 数为
`floor(0.05*N)`；阈值固定为降序 absence-max 中第 `allowed+1` 个值的向 `+inf` 下一浮点数，从而在 calibration
上不超过 5%。不得观察 test 后改阈值。calibration target 图像 Laplacian variance 的 33%/67% 分位数冻结
low/mid/high quality strata；源数据无遮挡标签，故 occlusion 始终 `NOT_EVALUABLE_SOURCE_LABEL`。

阈值、head SHA、archive/roster/protocol/model hashes 与 quality cutoffs 全部写入 `pretest-lock.json` 后，才允许读取
`s8--s11` 像素。两臂在相同 135 present 与 135 absent pairs 上一次运行，逐 pair 保存 raw unary scores/margin。

Challenger 晋级必须同时满足：

- `rescue > collateral`，且 baseline-correct control retention `>=80%`；
- candidate permutation invariance `=100%`；
- sealed target-absent false accept `<=5%`，ordinary-negative false accept `<=5%`；
- accepted coverage `>=50%`；
- matched-context construction 全部成立；按 test candidate session 的 same-instance recall 最低组 `>=40%`，
  且最高与最低组差 `<=35` 个百分点。

报告还必须给出 same-instance recall、same-class false commit、coverage、selective risk/risk--coverage、按 category、
session、quality 的最坏组，以及每个 pair 的 raw scores。任何一门失败均不晋级；有 rescue 但总门失败记为 mixed
negative，不能 outcome-driven 调参或 fusion。

## 单次、恢复与终态

正式执行目录必须绑定本协议文件 SHA。train/calibration 阶段中断时可以删除该未锁定 run 并以相同 roster、seed、
代码与模型从头确定性重跑；不得复用部分 step 选择 checkpoint。写入 pre-test lock 后，只允许在同一 hashes 下恢复
确定性 test inference/evaluation。`raw-test-scores.json` 和 `final-report.json` 原子写入；final report 存在后程序必须
拒绝覆盖、重跑或另开同 roster 的第二次判决。无外部模型调用，不存在可追加 provider attempts。

合法科学终态只有：

```text
NEAR_IDENTITY_HARD_NEGATIVE_UNARY_SIGNAL_SUPPORTED_DEVELOPMENT
NEAR_IDENTITY_HARD_NEGATIVE_UNARY_MIXED_WITH_COLLATERAL_DEVELOPMENT
NEAR_IDENTITY_HARD_NEGATIVE_UNARY_NOT_SUPPORTED_DEVELOPMENT
NEAR_IDENTITY_HARD_NEGATIVE_UNARY_NOT_EVALUABLE
```

无论终态如何，都保持 `RELIABLE_VERIFIER_NOT_ESTABLISHED / NO_P1 / DEFAULT_APP_UNCHANGED`，除非未来另有授权与证据。
