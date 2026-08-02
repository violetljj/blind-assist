# HFTF Stage C D39 THOR-MAGNI confirm-release veto event result

日期：2026-08-03（Asia/Hong_Kong）

## 结论

终态：

`D39_THOR_MAGNI_CONFIRM_RELEASE_VETO_EVENT_NOT_SUPPORTED`

对称 scene confirm 成功执行了预期机制：

- admitted confirm：1,247 frames
- confirm opportunities：406 anchors / 19 sessions
- 实际解除 live latch：331 frames，272 anchors / 19 sessions
- latch-only suppressions 从 D38 的 492 降到 73
- D38 的 6 个 positive event losses 全部恢复为 0

但解除过于积极，negative event utility 也基本回到 D37：

- negative windows 仅从 `251` 降到 `250`
- relative reduction `0.398%`
- 只有 1/5 folds 出现 negative reduction
- 仍损失 2 个 positive anchors

因此显式 confirm-release mechanism 是可评价、可运行的，但当前对称单帧
release 规则不支持 event utility。按冻结 stop rule：

`HFTF_SCENE_SCALE_PERSISTENCE_FAMILY_STOP`

不得在同一 outcome 上继续搜索 asymmetric thresholds、confirm count 或更多 hold
durations。

## 单变量实现

新增 source identity：

`CAUSAL_SCENE_SCALE_BIDIRECTIONAL_R1`

它保持 production scene producer 的 association、median rate、minimum matches、
quality、target binding 与 maximum gap 不变，只增加严格对称的 tri-state：

- `rate <= -0.05/s`：`CONTRADICT_APPROACH`
- `rate >= +0.05/s`：`CONFIRM_APPROACH`
- 中间 deadband：`ABSTAIN`

新增独立 mode：

`ACTIVE_CONTRADICT_TTL_CONFIRM_RELEASE`

- contradiction：当前帧 veto 并激活 250 ms hard-cap latch
- confirm：立即解除 live latch，但不创建 alert
- abstain/evidence absent：不延长也不解除

D37/D38 modes 未改变；重复 replay 的 artifact SHA 分别仍为：

- D37：
  `390fa479ce1bedec904d6b22ff70fa97b32288e89a3cc26d1d1695e37856622e`
- D38：
  `8cf20b345f30fa757307c430e5eeeb63a2859450d238c06a50ad5fbd22394930`

## source-only replay

- cohort：530 anchors / 19 sessions
- admitted contradiction：682 frames / 351 anchors / 19 sessions
- admitted confirm：1,247 frames / 406 anchors / 19 sessions
- live-latch releases：331 / 272 anchors / 19 sessions
- latch-only suppressions：73 / 53 anchors / 16 sessions
- total feedback suppressions：581
- baseline triggered windows：365
- candidate triggered windows：362
- raw/stable risk mismatch：`0 / 0`
- non-bidirectional-source observations：`0`
- candidate-only frames/windows：`0 / 0`

全部 evaluability gates 通过。

## pooled paired outcomes

| 指标 | baseline | D39 candidate | 差异 |
|---|---:|---:|---:|
| positive anchor alerts | 114/157 (72.611%) | 112/157 (71.338%) | -2 / -1.274 pp |
| positive event hits | 79/107 (73.832%) | 79/107 (73.832%) | 0 |
| negative anchor alerts | 251/373 (67.292%) | 250/373 (67.024%) | -1 / -0.398% relative |
| candidate-only windows | — | 0 | 0 |

fold outcomes：

| Fold | negative reduction | positive anchor losses | positive event losses |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 3 | 0 | 1 | 0 |
| 4 | 0 | 1 | 0 |

## gates

evaluability：

- complete/source/risk parity：PASS
- contradiction opportunity：PASS，351 / 19
- confirm opportunity：PASS，406 / 19
- confirm releases live latch：PASS，272 / 19
- latch-only suppression opportunity：PASS，53 / 16

support：

| Gate | 结果 |
|---|---|
| zero positive event losses | PASS |
| positive anchor noninferiority | **FAIL：2 / -1.274 pp** |
| negative absolute reduction >=10 | **FAIL：1** |
| negative relative reduction >=20% | **FAIL：0.398%** |
| negative reduction in >=3/5 folds | **FAIL：1/5** |
| no candidate-only window | PASS |

## 科学解释

D38 与 D39 给出了有界但清楚的两端：

- 无条件维持 250 ms：能显著降低 negative windows，但误伤 positive events
- 单帧对称 confirm 立即解除：恢复 positive events，但同时释放掉几乎全部 negative
  effect

这说明当前 scene-scale rate 只足以产生方向性 frame evidence，不足以单独定义安全的
event persistence/release state。继续调 deadband 或 duration 只会在已消费 outcome
上寻找折中，不会补上缺失的 event semantics。

下一步必须离开当前 family：

1. 使用新的独立 evidence 建立 target/event continuity 或可验证的解除状态；或
2. 获得新鲜 event outcome cohort，对预先冻结的新状态变量进行评价。

不能把 D39 包装成新主线，也不能撤销 D33/D34 的 mechanism/parity 正结果。

## 工程边界

首次尝试一次性修改 producer/kernel/tests 时因 patch context 不匹配而整体未应用；
随后拆成小 patch 完成。该故障发生在任何 D39 replay/truth join 前，没有 artifact
或 outcome 被读取，不烧毁 cohort。

重复 D37-D39 replay 与 `core:assist` 全量测试通过；D36-D39 evaluator 共 12 tests
通过。

## 可复现 artifact

目录：

`artifacts.local/evidence/hftf/stage-c-d39-thor-magni-confirm-release-veto-event-v0/`

- `kernel_replay.tsv`
  - bytes：`65,147`
  - SHA-256：
    `3b3a3d7a587a95baa5942b3b343ad9bd31a3cf788f5ef3c6929f4d25216ea832`
- `report.json`
  - bytes：`34,733`
  - SHA-256：
    `bfad01a931d169178e5060e13e2fcb4f40aefccf612e57c8bf03158cd5e7abb7`

input bindings：

- D12 samples：
  `9a099a52d29da60f889d40cacc1a2e267e506c23dc4aafa7fba1764eb1d64a54`
- D36 detections：
  `5083db4c86ff20c01d12a47aa9b419d3cd1727b8f8ed1383020b33601ad6f731`
- D36 producer receipt：
  `26f2991f4f836c2611224bc9ea1c9066d00b6bf03b600155736fafcbf8ab5ade`

## claim ceiling

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

D39 不覆盖 D35 device gate，不建立 independent generalization、产品或安全主张。
