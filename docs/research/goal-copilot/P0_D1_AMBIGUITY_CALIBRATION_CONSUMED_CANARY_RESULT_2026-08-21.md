# P0-D1 Ambiguity Calibration consumed-development canary

状态：`CONSUMED_DEVELOPMENT_MECHANISM_CANARY_COMPLETE / V1_OVER_REFUSAL / V2_OVER_REFUSAL / NOT_ADMITTED / PARENT_DISJOINT_CONFIRMATION_NOT_STARTED / NO_SCIENTIFIC_VERDICT`

日期：2026-08-21

## 结论先行

当前失败应称为 `unsupported grounding / evidence-authority overreach`，而不是把 23 个选择写成已有人类
gold 证明的错误入口。原始 Brain 在 31 个 `AMBIGUOUS` episodes 中 23 次选择具体门、0 次主动输出
`AMBIGUOUS`；但这些 episodes 集中于 8 个 venue parents，venue-parent macro unsupported-commit rate 为
`0.6667`，不能把 31 帧当作 31 个完全独立场景。

两个只改变 Brain 决策原则的 consumed-development canary 都显著减少 unsupported commitment，但都破坏
resolvable retention，因此均不准入：

| policy | SELECT / AMBIGUOUS / ABSTAIN | unsupported commit | venue-parent macro | baseline correct grounding retained | UNIQUE refusal | resolvable refusal |
|---|---:|---:|---:|---:|---:|---:|
| 原始 baseline | 39 / 0 / 8 | 23/31 | 0.6667 | 13/13 | 0/12 | 0/16 |
| V1 双证据硬门 | 19 / 0 / 28 | 12/31 | 0.2969 | 4/13 | 5/12 | 9/16 |
| V2 两级 support | 8 / 17 / 22 | 6/31 | 0.2031 | 2/13 | 10/12 | 14/16 |

这不是 calibration improvement。V1/V2 都把“少乱猜”换成了“多数情况不回答”；V2 虽首次让同一个
Terra Brain 主动表达 17 次 `AMBIGUOUS`，但保护项明显失败。

## 固定面与唯一变量

两次 canary 都复用同一个 47-goal / 43-frame consumed Silver-B cohort、Grounding DINO Tiny proposals、
threshold、NMS、score-neutral images、opaque case IDs、冻结 evaluator、`Codex CLI 0.149.0 +
gpt-5.6-terra + medium`。未更换模型、Provider、proposal、truth 或 evaluator。

- V1：SELECT 前要求 place identity 与 exact entrance relation 两类证据都足够；branding、signage、facade
  proximity 不能单独建立具体门归属。
- V2：显式输出 `place_support` 与 `entrance_relation_support`，取值为 `STRONG / WEAK / ABSENT`；说明证据
  是替代且累积的，一条直接证据可单独成为 STRONG，多条中等证据也可合成 STRONG，并把不可排除的竞争门
  计入 entrance relation support。

V2 的 support/action 分布为：`ABSENT/ABSENT/ABSTAIN=22`、`STRONG/STRONG/SELECT=8`、
`STRONG/WEAK/AMBIGUOUS=5`、`WEAK/STRONG/AMBIGUOUS=2`、`WEAK/WEAK/AMBIGUOUS=10`。它证明结构化
两级 belief 可以诱导 ambiguity 表达，但没有建立可用的非对称证据预算；尤其 4 个 `SET_VALUED` 全部被
拒绝或判 ambiguous，不能声称集合语义改善。

## 统计与 evaluator 口径修正

`baseline correct grounding retained` 只计算原始 baseline 的 13 个 `CORRECT_GROUNDING` 在 candidate 中
仍为 `CORRECT_GROUNDING`。不能使用 evaluator 的 `top1_correct_given_available` 作为 retention：即使 Brain
abstain，适配器仍会保留 Provider rank，因而该字段可能为 true，却没有发生 grounding。

原始 unsupported commits 的 venue-parent 分布为：Maki Maki `8/8`、NTGent Café `8/8`、SuPe `3/3`、
Theaterzaal `2/6`，其余 4 个 parents 各有 1–2 episodes。D1 因此同时报告 frame micro 与 venue-parent
macro，不把重复视图伪装成独立建筑证据。

## D1 当前终态

这两个结果只是在已经观察过的 cohort 上进行机制开发，不是 parent-disjoint confirmation，更不是 fresh、
held-out 或 formal baseline。当前环境没有 `MAPILLARY_ACCESS_TOKEN` 或 `MAPILLARY_TOKEN`，所以要求的
25–40 个新 venue-parent-disjoint episodes 尚未采集；旧帧不得重命名为新 slice。

停止继续 prompt fishing。下一步先取得新 Mapillary acquisition authority，构造 signage-positive / relation-
uncertain、facade conflict、multiple-plausible 与少量明确 UNIQUE 的新 parent-disjoint Development slice。只有
当同一失败在新 parents 上复现，才继续设计可学习或可校准的 evidence-budget surface；当前不启动 Sky，
不修改 detector/evaluator，也不把两级 belief 表述升格为已准入算法。

## Evidence

- cohort SHA-256：`aa724c31f1fb9f906c28bcc417dfa7dc83175fcaad07bf1919ad4d27415236a7`
- V1 Brain report SHA-256：`cd2d3ef7098f204c55fb69d22d36930173219e55d921d4ca507ed70e0edf980a`
- V1 calibration audit SHA-256：`3f258301b10b912929f9ce8db2ce9ca9009c4d0f37c48436522ed6ab24a62b00`
- V1 model audit SHA-256：`7454d58b085fce36ef8cc8d6db59c058d375fd90f8a6363df6f0c134edb2d7f1`
- V2 Brain report SHA-256：`7f23eb6533609ac69e293becbb048f0ad7d79920c0c819c086b7af78415fe2c4`
- V2 calibration audit SHA-256：`76626a549c5892cc99e5f9c30c4d1d6f4df4ec14b50431a9fb6272a20f3f771b`
- V2 model audit SHA-256：`e580a5569275abd758b8000b11cd8ab2027e59bb90257f9b217aea6f8bd24f85`
- 两次 model audit 均为 `NO_TOOL_OR_EXTERNAL_CALL_EVENTS`：各 12 个 batch、12 个 agent message，0 个
  tool/shell/web 事件。

## Claim ceiling

`CONSUMED_SILVER_B_DEVELOPMENT_MECHANICS_ONLY_NO_GENERALIZATION_OR_SCIENTIFIC_VERDICT`
