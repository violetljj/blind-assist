# HFTF Stage C D37 THOR-MAGNI production scene-scale veto event result

日期：2026-08-03（Asia/Hong_Kong）

## 结论

终态：

`D37_THOR_MAGNI_PRODUCTION_SCENE_SCALE_VETO_EVENT_NOT_SUPPORTED`

D37 已经跨过 D36 的 evidence opportunity 瓶颈：production scene-scale producer
在 `351/530 anchors`、全部 19 个 sessions 上产生 admitted contradiction，共
682 frames，并实际造成 508 次逐帧 feedback suppression。因此它是可评价的，
不是 coverage 不足。

但直接 event utility 没有成立。positive event/anchor 均零损失，说明 veto
方向保守；然而 508 次逐帧 suppression 只让 negative triggered windows 从
`251` 降到 `250`，绝对减少 1，relative reduction `0.398%`，远低于冻结的
`>=10` 且 `>=20%`。只有 1/5 folds 出现任何 negative reduction。

这同时保留两个不同层级的结论：

- 正机制：
  `PRODUCTION_SCENE_SCALE_CONTRADICTION_HAS_REAL_EVENT_OPPORTUNITY`
- 直接系统终态：
  `FRAME_LOCAL_SCENE_SCALE_VETO_EVENT_UTILITY_NOT_SUPPORTED`

不能把后者倒推为 scene-scale evidence 不存在；真正断裂发生在 evidence 已经
触发 suppression 之后、window/event terminal 之前。

## frozen paired replay

- cohort：19 sessions / 530 anchors
  - positive onset anchors：157
  - negative anchors：373
  - positive events：107
- source：原样复用 D36 truth-free detector TSV
  - 3,710 unique source frames
  - 14,364 person detections
- baseline：production `AssistDecisionKernel` + `OFF`
- candidate：production `AssistDecisionKernel` +
  `ACTIVE_CONTRADICT_ONLY`
  - 不注入外部 evidence
  - 直接调用 kernel 内 production
    `CausalSceneScaleTristateGeometryProducer`
- raw/stable risk mismatch：`0 / 0`
- non-scene source observations：`0`
- candidate-only triggered frames/windows：`0 / 0`

production 默认参数未变：

- scene scale rate threshold：`-0.05/s`
- maximum gap：`500 ms`
- minimum matches：`2`

未搜索 threshold、gap、association、minimum matches 或 history/window length。

## pooled paired outcomes

| 指标 | baseline | candidate | 差异 |
|---|---:|---:|---:|
| positive anchor alerts | 114/157 (72.611%) | 114/157 (72.611%) | 0 |
| positive event hits | 79/107 (73.832%) | 79/107 (73.832%) | 0 |
| negative anchor alerts | 251/373 (67.292%) | 250/373 (67.024%) | -1 |
| candidate-only windows | — | 0 | 0 |
| positive event losses | — | 0 | 0 |

evidence/action diagnostics：

- anchors with admitted contradiction：`351`
- sessions with admitted contradiction：`19`
- admitted contradiction frames：`682`
- anchors with at least one suppression：`268`
- suppressed frames：`508`
- scene abstain observations：`2,175`
- evidence-absent frames：`853`

## gates

全部 evaluability gates 通过：

| Gate | 结果 |
|---|---|
| complete cohort | PASS |
| D31 anchor detector parity | PASS |
| raw/stable risk path parity | PASS |
| admitted source identity | PASS |
| baseline positive/negative opportunities | PASS |
| contradiction opportunity >=10 anchors / >=5 sessions | PASS：351 / 19 |

support gates：

| Gate | 结果 |
|---|---|
| zero positive event losses | PASS |
| positive anchor noninferiority | PASS |
| negative absolute reduction >=10 | **FAIL：1** |
| negative relative reduction >=20% | **FAIL：0.398%** |
| negative reduction in >=3/5 folds | **FAIL：1/5** |
| no candidate-only window | PASS |

## 科学解释

scene-scale contradiction 已经大量到达 production feedback seam，且没有创建
新 alert；问题不是 producer 覆盖，也不是 admission。当前 active path 只抑制
当帧 feedback，而一个 7-frame sample/window 内常有其他 baseline-triggered
frames。于是大量真实 suppression 被同一 window 内的延迟或替代 trigger 覆盖，
几乎不改变 event terminal。

因此下一变量应位于 system seam，而不是再换模型或调 scene threshold：

> 检验 causal contradiction 是否需要 bounded temporal/event-scoped persistence，
> 才能把 frame-level safety veto 转化为 event-level utility。

D37 outcome-open Development evidence 可以用于定位和设计这个变量，但不能把同一
cohort 上的新设计称为 independent validation。新的主线替换主张仍需要独立
outcome evidence。

## 控制面错误与修复

1. 首次 Python unittest 从 repo root 调用，module path 不成立；改到脚本目录后
   generic Python 又缺少 `cv2`。两次均发生在 kernel replay/truth join 前，未读取
   outcome，改用项目既有 research venv 后 8 tests 通过。
2. 首次 evaluator 的科学 `status` 正确写为 `NOT_SUPPORTED`，但
   `status.endswith("_SUPPORTED")` 把布尔字段误序列化为 `supported=true`。
   这是纯报告语义 bug，不涉及输入、gates 或阈值；改为 exact status equality，
   加回归测试并原样重跑后得到 `supported=false`。

这些错误不烧毁 cohort，也没有被记录为算法负结果。

## 可复现 artifact

目录：

`artifacts.local/evidence/hftf/stage-c-d37-thor-magni-production-scene-scale-veto-event-v0/`

- `kernel_replay.tsv`
  - bytes：`63,201`
  - SHA-256：
    `390fa479ce1bedec904d6b22ff70fa97b32288e89a3cc26d1d1695e37856622e`
- `report.json`
  - bytes：`32,243`
  - SHA-256：
    `875d2b092cd110d9dae60bdf94490c8dd61a150e8a48604709d37730d23309bb`

input bindings：

- D12 samples SHA-256：
  `9a099a52d29da60f889d40cacc1a2e267e506c23dc4aafa7fba1764eb1d64a54`
- D36 detections TSV SHA-256：
  `5083db4c86ff20c01d12a47aa9b419d3cd1727b8f8ed1383020b33601ad6f731`
- D36 producer receipt SHA-256：
  `26f2991f4f836c2611224bc9ea1c9066d00b6bf03b600155736fafcbf8ab5ade`

重复 Kotlin replay 的 counts 与 kernel SHA 完全一致；随后 evaluator 重建得到上述
report。`core:assist` 全量测试与 D36+D37 evaluator 8 tests 均通过。

## claim ceiling

D37 不支持当前 frame-local scene-scale veto 的真实事件效用，不改变：

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

D33/D34 的 track future mechanism/parity 正结果、D35 的
`READY_FOR_DEVICE_EXECUTION / NOT_EVALUATED` 也均保持原边界。
