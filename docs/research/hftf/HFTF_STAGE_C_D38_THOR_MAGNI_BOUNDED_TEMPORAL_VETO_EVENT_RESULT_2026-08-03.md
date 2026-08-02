# HFTF Stage C D38 THOR-MAGNI bounded temporal veto event result

日期：2026-08-03（Asia/Hong_Kong）

## 结论

终态：

`D38_THOR_MAGNI_BOUNDED_TEMPORAL_VETO_EVENT_NOT_SUPPORTED`

D38 证明 D37 定位到的 system seam 是真实的：把逐帧 scene-scale veto 延续为
与 production evidence TTL 相同的固定 250 ms feedback-only latch 后，
negative triggered windows 从 `251` 降到 `217`，绝对减少 34，5 folds 中 4 folds
出现减少。相比 D37 只减少 1，temporal persistence 明显把更多 frame-level
suppression 转化成了 window-level effect。

但这不是可接受的 Pareto 改善。candidate 同时损失：

- 16 个 positive anchors（114 -> 98）
- 6 个 positive events（79 -> 73）

negative relative reduction 也只有 `13.55%`，未达到冻结的 `20%`。因此固定、
无条件 250 ms persistence 实例不支持继续进入独立验证或 App。

保留的窄正信号：

`BOUNDED_TEMPORAL_VETO_CHANGES_EVENT_TERMINALS_DEVELOPMENT_ONLY`

明确拒绝：

`FIXED_250MS_UNCONDITIONAL_SCENE_VETO_PERSISTENCE`

## 数据角色与单变量边界

D38 是看到 D37 后冻结的：

`POST_D37_ADAPTIVE_OUTCOME_OPEN_DEVELOPMENT`

它不是独立验证。原样复用：

- 19 sessions / 530 anchors
- 157 positive anchors / 373 negative anchors / 107 positive events
- D36 truth-free detections 与 producer receipt
- production scene-scale producer 的默认 threshold、association、minimum matches
- production risk/event/planner path

唯一变量是新增且独立于 D37 的：

`DualLoopRuntimeMode.ACTIVE_CONTRADICT_TTL`

规则固定为：

- 当前 admitted contradiction 立即 veto feedback
- latch 保持到 `decisionAtNs + 250 ms`
- 新 contradiction 可以延长到新的 `+250 ms`
- abstain/evidence absent 不延长
- session start/reset 清空
- 只作用 feedback，不改变 risk 或 event state

250 ms 原样复用 production scene evidence TTL，未做 duration search。原
`ACTIVE_CONTRADICT_ONLY` mode 保持字节级 replay 结果不变。

## source-only production replay

- D37 replay SHA 仍为：
  `390fa479ce1bedec904d6b22ff70fa97b32288e89a3cc26d1d1695e37856622e`
- D38 admitted contradiction frames：`682`
- D38 direct + carried suppressions：`1,000`
- latch-only suppressions：`492`
- latch-only suppression anchors：`231`
- latch-only suppression sessions：`19`
- baseline triggered windows：`365`
- D37 candidate triggered windows：`364`
- D38 candidate triggered windows：`315`
- raw/stable risk mismatch：`0 / 0`
- non-scene source observations：`0`
- candidate-only frames/windows：`0 / 0`

因此 latch opportunity gate 充分，D38 可以解释 effect。

## pooled paired outcomes

| 指标 | baseline | D38 candidate | 差异 |
|---|---:|---:|---:|
| positive anchor alerts | 114/157 (72.611%) | 98/157 (62.420%) | -16 / -10.19 pp |
| positive event hits | 79/107 (73.832%) | 73/107 (68.224%) | -6 / -5.61 pp |
| negative anchor alerts | 251/373 (67.292%) | 217/373 (58.177%) | -34 / -13.55% relative |
| candidate-only windows | — | 0 | 0 |

fold outcomes：

| Fold | negative reduction | positive anchor losses | positive event losses |
|---:|---:|---:|---:|
| 0 | 9 | 4 | 1 |
| 1 | 9 | 2 | 2 |
| 2 | 6 | 5 | 0 |
| 3 | 0 | 2 | 1 |
| 4 | 10 | 3 | 2 |

## gates

全部 evaluability gates 通过，包括：

- contradiction opportunity：PASS
- latch-only opportunity：`231 anchors / 19 sessions`，PASS
- source/risk parity：PASS

support gates：

| Gate | 结果 |
|---|---|
| zero positive event losses | **FAIL：6** |
| positive anchor noninferiority | **FAIL：16 / -10.19 pp** |
| negative absolute reduction >=10 | PASS：34 |
| negative relative reduction >=20% | **FAIL：13.55%** |
| negative reduction in >=3/5 folds | PASS：4/5 |
| no candidate-only window | PASS |

## 科学解释

D37 的失败并非因为 causal contradiction 无法作用于系统；D38 已证明延长 veto
作用域会显著改变 event terminal。真正的问题是无条件时间保持缺少“何时安全继续
抑制”的判别：同一个 latch 同时覆盖 negative 和 positive event opportunities。

因此下一步不应搜索 100/200/300/500 ms。固定时长搜索会在已消费 outcome 上优化
trade-off，违反本协议，也不能解决语义缺失。若继续该路线，新科学变量必须回答：

> persistence 是否能绑定到可解释的 event/target continuity 与解除条件，而不是
> 单纯依赖 wall-clock duration？

该变量可以先做 truth-free 机制设计，但任何 event utility 支持必须使用新鲜、
独立 outcome evidence。当前支线仍不超过主线。

## 工程与控制面

- 新 D38 mode 没有改写 D37 mode；重复 D37 replay 的 TSV SHA 完全不变。
- 合并验证命令曾从 `scripts/research/hftf` cwd 使用 repo-relative evaluator/
  artifact path，因路径重复而失败；没有改写 D38 report，改回 repo root 后按同一
  evaluator 重建成功。这是可修复命令/path error，不产生科学终态。
- D36+D37+D38 evaluator 共 10 tests 通过；`core:assist` 全量测试通过。

## 可复现 artifact

目录：

`artifacts.local/evidence/hftf/stage-c-d38-thor-magni-bounded-temporal-veto-event-v0/`

- `kernel_replay.tsv`
  - bytes：`63,212`
  - SHA-256：
    `8cf20b345f30fa757307c430e5eeeb63a2859450d238c06a50ad5fbd22394930`
- `report.json`
  - bytes：`33,825`
  - SHA-256：
    `af97a203f06208f6256a1e1bee45191908c46bda41a5dc45793216f4a4ef09d7`

input bindings：

- D12 samples：
  `9a099a52d29da60f889d40cacc1a2e267e506c23dc4aafa7fba1764eb1d64a54`
- D36 detections：
  `5083db4c86ff20c01d12a47aa9b419d3cd1727b8f8ed1383020b33601ad6f731`
- D36 producer receipt：
  `26f2991f4f836c2611224bc9ea1c9066d00b6bf03b600155736fafcbf8ab5ade`

## claim ceiling

D38 不改变：

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

它不覆盖 D35 device gate，不建立 independent generalization、产品或安全主张。
