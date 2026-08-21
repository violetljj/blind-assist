# P0-D1 venue-parent-disjoint Development confirmation

状态：`PARENT_DISJOINT_DEVELOPMENT_CONFIRMATION_COMPLETE / BASELINE_CALIBRATION_FAILURE_REPRODUCED / NO_V3_PROMPT / NO_SCIENTIFIC_VERDICT`

日期：2026-08-21

## 结论先行

原始 Brain 的 ambiguity-calibration failure 在新的 Brussels venue parents 上再次出现。新的 24-goal / 24-frame
Development slice 中，原始 baseline 对 20 个 `AMBIGUOUS` episodes 仍有 13 次具体 `SELECT`，unsupported-commit
micro rate 为 `0.6500`；按 9 个 ambiguous venue parents 等权计算的 macro rate 为 `0.7444`。与此同时，4 个
`UNIQUE` episodes 全部 `SELECT` 且全部得到 `CORRECT_GROUNDING`，resolvable refusal 为 `0/4`。

因此，当前证据支持把问题从“旧 cohort 中少数 storefront 的偶然诱导”提升为一个跨 venue 重复出现的
Development failure mechanism：同一个 Brain 在可解析入口上能选对，却仍会把地点、招牌或附近 facade 的证据
过度升级成具体入口。它不支持继续做 prompt fishing；下一步研究对象应是 commit / ambiguous / abstain 的小型
calibration boundary，而不是把 V1/V2 的硬 gate 再放宽一句。

这仍不是正式 baseline、held-out、统计泛化或科学性能结论。新 slice 是在已经知道 failure hypothesis 后构造的
Development confirmation，且只有 4 个 resolvable episodes；它只确认机制值得继续研究。

## 新数据与独立性

数据来自两个新的 Brussels acquisition batches。Mapillary token 从用户级环境变量注入进程，未写入仓库、日志或
结果文档。请求方式先按 [Mapillary Image API 官方说明](https://www.mapillary.com/developer/api-documentation#image)
核对 `bbox=west,south,east,north` 与字段查询；密集城区整 bbox 返回 `Please reduce the amount of data` 后，没有继续
调大 `limit` 或重试同一请求，而是采用现有 runner 的 60 m 单 anchor 查询。Mapillary 官方人员也说明密集区域受
S2 cell 密度影响，应拆分 bbox，较大覆盖发现应改用 coverage vector tiles，且空间参数不提供传统分页；见
[官方论坛说明](https://forum.mapillary.com/t/webapp-ai-fetch-issue-looking-for-help/9973)。

两个 acquisition batches 原始导出 `18 + 14 = 32` 个 Silver-B goal episodes。去除跨 batch 的同帧重复以及同一
frame 被多个附近 POI 重复物化的条目后，保留 24 个独立 goal / 24 个独立 RGB frames，没有为了达到预估的
25-goal 下界复制帧或放大分母。score-blind full-frame review 在 baseline 输出前完成，得到：

- `UNIQUE=4 / AMBIGUOUS=20 / SET_VALUED=0`；
- 10 个 venue parents，其中 9 个出现在 ambiguous denominator；
- 与旧 47-goal cohort 的 target-name overlap `0`；
- 与旧 cohort 的 frame-ID overlap `0`；
- episode-ID duplicate `0`。

4 个 `UNIQUE` 只在直接招牌与一个可见物理门形成明确关系时成立：Pylones 1 帧、Moneytrans 1 帧、Hotel Floris
Arlequin Grand Place 2 帧。Jimmy Fairly 的直接 storefront branding 没有自动升级为 UNIQUE：当视图只建立店面、
但没有隔离一个具体 doorway 时仍标为 `AMBIGUOUS`。这正是本轮要审计的 place identity / entrance identity 边界。

## 固定运行面

只运行原始 `baseline` decision policy，没有运行 V1、V2 或任何 V3。固定面为：

- `Codex CLI 0.149.0`，ChatGPT login preflight 通过；
- executable SHA-256 `14b7e6b2356e82d1d9275579eaa588757b4e0a501b65dcc19fccdf77bd83dc00`；
- `gpt-5.6-terra / medium`；
- 同一 Grounding DINO Tiny proposal path、冻结 evaluator 与 opaque model case IDs；
- 6 batches，每 batch 4 episodes，全部一次成功；
- model-run audit：6 个 `agent_message`，0 个 tool / shell / web item，终态
  `NO_TOOL_OR_EXTERNAL_CALL_EVENTS`。

## 结果

| cohort | raw SELECT / AMBIGUOUS / ABSTAIN | unsupported commit | venue-parent macro | correct grounding | resolvable refusal |
|---|---:|---:|---:|---:|---:|
| 旧 consumed 47-goal | 39 / 0 / 8 | 23/31 = 0.7419 | 0.6667（8 parents） | 13/13 baseline-correct | 0/16 |
| 新 parent-disjoint 24-goal | 17 / 0 / 7 | 13/20 = 0.6500 | 0.7444（9 parents） | 4/4 | 0/4 |

新 cohort 的 ambiguous parent 分解：

| venue parent | unsupported SELECT / ambiguous |
|---|---:|
| Au Coin Gourmand | 2/2 |
| Bruxelles Accueil Porte Ouverte | 1/2 |
| Chez Massimo | 2/2 |
| Cliff Brussels | 0/2 |
| Global Enterprises Tours - Brussels Beer Tasting Tour | 1/5 |
| Jimmy Fairly | 4/4 |
| La Chaloupe d'Or | 1/1 |
| Moneytrans | 1/1 |
| Pylones | 1/1 |

原始 Brain 仍然从未主动输出 `AMBIGUOUS`。7 个安全结果全部是 `ABSTAIN`；这说明原始策略不是没有拒绝能力，
而是缺少把“地点已建立、具体门仍未建立”表达成 ambiguity 的稳定决策边界。

## Evidence

- cohort content SHA-256：`5788b5f790f3628821d9473c25ad5ecc484650c4175f367dda68cd4074cb63bd`
- cohort file SHA-256：`8ba11387250b6f2f308d634e9786ac240f07fa1372be26fc63e336854d3c6964`
- Brain report content SHA-256：`cdd1e3a290b27b96cb031b47f1a62df29d8a3a2704f37dd7327b1aa026c63ff8`
- Brain report file SHA-256：`de4206866c8e69ef875574b15d2e3f06975dd4d7c205adb7fc634b7885f899da`
- model-run audit SHA-256：`76d73da4af0fd4248494ccb2393741160657227c1c6fcb76a49dfea3e1a47639`
- cohort：`artifacts.local/evidence/p0-s0/2026-08-21-p0-d1-brussels-parent-disjoint-cohort-v1/brain-cohort.json`
- Brain run：`artifacts.local/evidence/p0-s0/2026-08-21-p0-d1-brussels-parent-disjoint-baseline-v1/brain-baseline-report.json`

## 决策与下一步

`P0-D1_BASELINE_CALIBRATION_FAILURE_REPRODUCED_ON_PARENT_DISJOINT_DEVELOPMENT`。

停止追加 prompt 例外，也不回头调 detector、proposal threshold 或 evaluator。下一步可以正式设计一个小型
evidence-budget / calibration surface，输入 place support、exact entrance relation evidence、candidate competition、
direct signage relation、map relation 与 cross-view agreement，输出 `COMMIT / SET / AMBIGUOUS / ABSTAIN`。
第一轮应继续使用 consumed Development 数据学习或定边界，并保留一个更大、更多 resolvable parents 的新 slice
做后续 adjudication；当前 24-goal confirmation 本身不得再充当 fresh test。

## Claim ceiling

`PARENT_DISJOINT_SILVER_B_DEVELOPMENT_MECHANISM_CONFIRMATION_ONLY_NO_GENERALIZATION_OR_SCIENTIFIC_VERDICT`
