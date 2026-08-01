# HFTF Stage C causal future-label mechanics result D1

日期：2026-08-01

终态：`D1_CAUSAL_FUTURE_LABEL_MECHANICS_SUPPORTED`

## 1. 结论

冻结的 history-origin-causal future-label mechanics 在两个 consumed EgoWalk
calibration sources 上通过全部 causality、odometry mapping、anchor eligibility、
label support、future opportunity 与 determinism 顺序门。

这支持 future depth observation 能在不让真实 future path 选择 origin 或输出方向的
前提下，增加 `.4/.8 s` geometry-proxy teacher label support。它不支持 hazard/safe
truth、自然 prevalence、student agreement/effect、助盲事件效果或主线晋升。

## 2. 报告绑定

- report：
  `artifacts.local/evidence/hftf/stage-c-d1-causal-future-label-20260801/future_label_mechanics.json`
- SHA-256：
  `e0c86898539602d6323958edc0ac01935f3fbc74375c85575db187e3948fc8c3`
- protocol commit：`bee4232`
- runner commit：`eb458b2`

完整 payload 第二次运行 byte-exact。future pose 只用于把 future depth observation
变换到 world；报告明确记录其未用于 causal origin 或 output direction。

## 3. Source/horizon metrics

| source | horizon | eligible anchors | candidate known | known fraction | future-added known | known lost | risk proxies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| outdoor `2024_08_15...` | `.4 s` | 128 | 593/640 | `.9266` | 186 | 0 | 13 |
| outdoor `2024_08_15...` | `.8 s` | 128 | 561/640 | `.8766` | 280 | 0 | 10 |
| indoor `2024_07_11...` | `.4 s` | 131 | 521/655 | `.7954` | 303 | 0 | 1 |
| indoor `2024_07_11...` | `.8 s` | 131 | 497/655 | `.7588` | 490 | 0 | 0 |

两条 source 的 history-speed eligible fraction 均为 1.0。motion-yaw mapping 结果为：

| source | horizon | circular resultant | median absolute error |
| --- | ---: | ---: | ---: |
| outdoor | `.4 s` | `.8986` | `6.41°` |
| outdoor | `.8 s` | `.8404` | `6.95°` |
| indoor | `.4 s` | `.9691` | `4.54°` |
| indoor | `.8 s` | `.9619` | `4.60°` |

所有 source/horizon 的 candidate known fraction 均高于 `.70`，future-added known
均不少于 100，known loss 与 UNKNOWN→SAFE violation 均为 0。cohort 共 24 个
risk-proxy cells、20 个 source/horizon/frame instances、5 个方向
`[-30,-15,0,+15,+30]°`，越过冻结的 `2/2/2` opportunity 门。它们没有 reference
truth，不能解释为 24 个真实危险。

七个 structural canaries 全过：future-pose perturbation 不改变 causal origin/grid
orientation、已知 future observation 可填充 baseline UNKNOWN、unmatched observation
保持 UNKNOWN、candidate 不删除 baseline known、缺失 future depth 不变成 safe，以及
identical-input determinism。

## 4. 下一边界

唯一新权限：

`FREEZE_FRESH_SESSION_DISJOINT_TEACHER_CORPUS_AND_STUDENT_CANARY_PROTOCOL_ONLY`

下一阶段必须在读取新媒体 outcome 前冻结 trajectory/session-disjoint
train/dev/held-out cohort、选择规则、source hashes、UNKNOWN denominator、学生输入和
比较臂。学生假设仍是 history RGB only；必须同时比较 single-frame RGB、
history-RGB current-only labels 与 history-RGB current+future layered labels。

当前不授权 fresh acquisition、teacher-corpus generation、student training/effect、
研究主线、Android/App 或安全/产品 claim。
