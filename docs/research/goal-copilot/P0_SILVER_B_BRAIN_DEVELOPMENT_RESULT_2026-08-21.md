# P0 Silver-B Development cohort and single-Brain mechanics result

状态：`DEVELOPMENT_COMPLETE / BLINDED_CASE_IDS / 47_GOAL_EPISODES / 43_UNIQUE_FRAMES / UNIQUE_12 / SET_VALUED_4 / AMBIGUOUS_31 / GPT_5_6_TERRA_MEDIUM / NO_SCIENTIFIC_VERDICT`

日期：2026-08-21

## 结论先行

Silver-B Development 已从最初 `4/4 AMBIGUOUS` 扩展到：

| 计数单位 | 数量 |
|---|---:|
| 唯一真实 Mapillary frames | 43 |
| goal episodes | 47 |
| `UNIQUE` | 12 |
| `SET_VALUED` | 4 |
| `AMBIGUOUS` | 31 |

同一帧上的 generic multi-entrance goal 与 `rightmost/right-hand` referring expression 分别计为不同 goal
episode，但只计为一个 unique frame；因此没有用语言改写虚增物理观测数。Grounding DINO Tiny、prompt、
threshold、NMS、P0-S0 materializer 与冻结 evaluator 均未修改。

单一 Brain 按用户指定固定为 `Codex CLI 0.149.0 + gpt-5.6-terra + medium`。有效 run 使用 opaque
`case-001...case-047`，输入只有 natural-language goal、score-neutral 原图和全部编号 Grounding DINO
proposals；manual referent regions、resolution、真实 episode ID 与 review notes 不进入 prompt。alias→episode
映射只在全部模型调用结束后写入。12 个 batch 全部一次完成，47/47 输出通过冻结 output contract；JSONL
审计只有 12 个 `agent_message`，没有 tool、shell、web 或其它外部调用事件。

本结果只允许称为：

> `conditioned candidate selection / weak grounding mechanics on Silver-B Development data`

不得称为入口定位准确率、Grounding DINO recall/precision、Brain 正式性能、端到端导航效果或 BLV 用户效果。

## Cohort 构造与 truth 边界

数据来自 Ghent、Antwerp、Leuven、Bruges、Mechelen 的 bounded source slices，以及对多入口建筑
`Theaterzaal`、`hofbladelin` 的定向 acquisition。原始 Silver-B row 可因同一 frame 对应多个 OSM anchor
而重复；Brain cohort 按 Mapillary frame ID 去重后为 43 个 unique frames。

review 流程使用隐藏 proposal score 的全帧候选图。`UNIQUE / SET_VALUED` 的 acceptable regions 由逐帧
review 独立手工绘制；proposal candidate ID 仅作为 audit reference，不能生成 truth。未能在 Silver-B
权限下建立合法物理 referent 集的 31 帧保持 `AMBIGUOUS`，不被压成 bbox truth。

`SET_VALUED=4` 已足够首次观察 mechanics，但仍是明显薄层，不能支持普遍结论。它们来自 4 个多入口
视图；对应的 4 个 `rightmost/right-hand` goal 是同帧派生 referring-expression episodes，已由
`unique_source_frame_count` 单独约束。

## 单 Brain 观察

### 原始动作

| Brain action | 数量 |
|---|---:|
| `SELECT` | 39 |
| `AMBIGUOUS` | 0 |
| `ABSTAIN` | 8 |

### 冻结 evaluator mechanics

| 观察项 | 结果 | 解释边界 |
|---|---:|---|
| contract-valid outputs | 47/47 | 只证明接口与适配器 mechanics 可运行 |
| correct proposal available on resolvable goals | 14/16 | frozen IoU 下的 Provider availability，不是 detector recall |
| Brain top-1 correct given available | 13/14 | Silver-B conditioned selection mechanics，不是正式 accuracy |
| `AMBIGUOUS` 上 ambiguous/abstain | 8/31 | 全部为 abstain；Brain 从未主动返回 ambiguous |
| frozen end-to-end success accounting | 21/47 | 仅作 evaluator audit，不得对外称 end-to-end accuracy |

### 按 referent semantics

- `SET_VALUED`：4/4 都返回多个 candidate，而不是强行唯一化；返回集合大小依次为 `2/3/3/2`，且首项
  均命中 valid set。当前没有观察到预设的 “VLM 总会把 SET 强行变 UNIQUE” 瓶颈，但 `n=4` 太小，不能
  据此关闭该风险或发明新算法。
- `UNIQUE`：12/12 都选择 candidate。10 个在 frozen IoU 下存在正确 proposal，其中 9 个 top-1 命中；
  唯一 selection miss 是一个 LA Look 视图选择 candidate 10，而正确 proposal rank 为 4。
- 另外 2 个 `rightmost` UNIQUE episode 在语义上选择了预期的小门 candidate，但该 proposal 与独立
  manual region 的 IoU 未达到 0.5，因此 evaluator 归为 Provider-candidate unavailable / spatial error。
  这是 proposal localization 与 region-boundary sensitivity，不应伪装成 Brain selection failure。
- `AMBIGUOUS`：只有 8/31 abstain，0/31 返回 `AMBIGUOUS`，其余 23/31 直接 grounding。23 次选择覆盖
  Maki Maki 8/8、NTGent Café 8/8、SuPe 3/3、El Sombrero 1/1、30CC Minnepoort 1/1，以及 Theaterzaal
  2/6；8 次 abstain 是 Theaterzaal 4、Usa Nails 2、VDD Project Development 2。Silver-B review 未建立
  exact physical referent，而 Brain 常把它认为可见的 branding、storefront 或邻近门洞直接升级为入口归属。
  这是当前最清楚的 over-grounding mechanics 信号，但弱 truth 不能证明这些现实入口事实上错误。

## 真实 failure 与下一步

少量 SET 样本上，现成 Brain 已主动返回集合；但 blind run 暴露了明确的 ambiguity-calibration 问题。
当前值得继续验证的瓶颈是：

1. Brain 几乎不表达 referential ambiguity：31 个 unresolved goals 中 `AMBIGUOUS=0`，只在缺少任何可靠
   identity clue 时 abstain；
2. branding/signage/facade association 被过快升级为 exact entrance referent；
3. correct entrance proposal 存在时仍可能选择相邻入口框，但本轮只观察到 1 个 such miss。

这些问题先复用现有方法而不是造新模型：ambiguity/association failure 沿用 `To Ask or Not to Ask`、DialFRED/ELBA 的
ask/guess/abstain calibration 设计；后者先借 BridgeNav 的 near-stage door refinement，并保持 Grounding
DINO proposal-only。只有在扩大 SET 与 signage-conflict Development slices 后仍稳定失败，才有理由定义
BlindAssist-specific 方法。

ABotN/POIBench 当前只吸收 episode/entrance-frame/arrival/SR/SPL/collision 定义；不把 ABot-N1 当可安装
模型，也不把 3DGS evaluator 混入当前 Mapillary Silver-B。Project Guideline 的 world-coordinate
persistence、STOP 与 logging 仍留到 P1/P2。

## Evidence

- cohort：`artifacts.local/evidence/p0-s0/2026-08-21-silver-b-dev-cohort-v1/brain-cohort.json`
- cohort report SHA-256：`aa724c31f1fb9f906c28bcc417dfa7dc83175fcaad07bf1919ad4d27415236a7`
- Brain report：`artifacts.local/evidence/p0-s0/2026-08-21-silver-b-brain-gpt-5.6-terra-blind-v2/brain-baseline-report.json`
- Brain report SHA-256：`1e9e7b3176d231fa3e67fa67f266da14a31672f3814bb836866c6446a0b3813c`
- model-execution audit：`NO_TOOL_OR_EXTERNAL_CALL_EVENTS`，audit SHA-256
  `33acda60c2b06758c200e0d55043719dc78b1c956ef8718bd74f98c55edb784f`；
- provider receipt 固定 CLI path/version/executable SHA、model 与 reasoning effort；12 个 batch 各有 dispatch、
  prompt、原始 response、stdout/stderr 与 response hash。
- 先前 `gpt-5.6-sol` partial run 已写 `ABORTED.json`，未产生 aggregate report，也不进入本结果；首次 Terra
  v1 因 model-visible episode ID 泄露 `unique/set/ambiguous` 后缀而标记 `NOT_EVALUABLE_INPUT_LEAKAGE`，
  其全部 aggregate 数值已撤回。

## Claim ceiling

`SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY`

本轮没有 Silver-A/human gold、未运行 POIBench、没有 closed-loop navigation、没有 P1 persistence、没有
Android/default-App 接线、没有 BLV 用户研究，也没有创新算法准入。
