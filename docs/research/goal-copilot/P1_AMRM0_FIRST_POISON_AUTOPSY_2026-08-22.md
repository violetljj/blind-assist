# P1-AMRM0 first-poison autopsy

日期：2026-08-22（Asia/Hong_Kong）

角色：`POST_OUTCOME_READ_ONLY_FAILURE_AUTOPSY`

冻结终态保持：`P1_AMRM0_MEMORY_POISONING_FAIL`

## 最窄回答

17 次 poisoned admission 分布在 9 个 episode，因此真正需要向上游追溯的 first-poison event 是 **9** 个。

```text
first-poison events:                     9
correct candidate absent:               9
correct candidate present but lost:     0
multiple plausible / ambiguous:         NOT_EVALUABLE_SINGLE_CANDIDATE_STREAM
explicit authority contradiction bug:   0
absolute-authority false confirmation:   9
```

其中 7/9 在真目标仍可见时发生，2/9 在目标不可见时发生。9/9 first-poison candidate 都是 background，
没有一个是其他 ADT instance。

## 为什么这是 proposal bottleneck

P1-A4 frozen interface 每帧只暴露一个 candidate。9 个 first-poison frame 中，这唯一 candidate 都不是 referent，
所以正确 candidate 在冻结候选池中全部 absent；“正确与错误 candidate 同时存在但 verifier 选错”和 local
discriminative margin 在本 cohort 中无法评估。

尤其是 7 个目标可见事件，background candidate 与 target GT 的 IoU 全部为 `0.0`。这不是两个相邻合理候选之间的
细微排序错误，而是 proposal 已经离开目标。因此第一分叉明确落在：

> `PROPOSAL BOTTLENECK`，而不是 clean multi-candidate identity-verifier bottleneck。

另外 2 个事件发生时目标不可见，正确行为只能是 `NONE / STALE`。

## 它为什么仍拿到 VERIFIED

9/9 first-poison event 的非排他 authority path 完全相同：

- original target dense match：9；
- original masked-context dense match：9；
- tracker spatial prediction：9，且实现中固定写为 `SUPPORTED`；
- bearing：0；
- local alternative contrast：0；
- negative veto：0；
- fallback/precedence：0。

这三项是 conjunctive 放行路径，当前 outcome 没有单变量消融，因此不能把 9 次错误排他地分给 target、context 或
tracker spatial 中某一个。但可以确定：它们共同把 background 当成获得 referent identity authority 的独立证据。

7 个可见事件的 identity score 范围为 `0.747–0.910`，context score 范围为 `0.851–0.977`，尽管 candidate-target
IoU 全为 0。这直接证明本 canary 中的高 absolute similarity 不具有 identity 诊断性。

## 防止“全拒绝”假修复

冻结 canary 中，newly accumulated verified KF 对 true reacquisition 的 observed contribution 为 0。因此当前没有
证据表明禁止这些 poisoned admissions 会损失由新 KF 支持的正确重捕获；但这只是 observed contribution，未执行
counterfactual snapshot-only rerun，不能写成反事实性能结论。

## 决策边界

- AMRM1/2/3、VLM、VIO、SLAM、geometry 继续冻结；
- 本 cohort 不具备检验 contrastive identity verification 的 multi-candidate interface；
- 若开 successor，应先以独立 proposal-availability 问题验证：目标可见时，正确 target candidate 是否能进入候选池；
- 在 correct candidate 与 plausible alternatives 同时可用之前，不宣称 contrastive verifier 是已支持的修复。

## Evidence

- autopsy：`artifacts.local/evidence/p1_amrm0_matched_canary_v1/first_poison_autopsy.json`
- autopsy SHA-256：`351a6baab3ecaa22bd7d262c58c6563b68bed24a5e2654d8bd15c1cda38faada`
- sealed canary result：`artifacts.local/evidence/p1_amrm0_matched_canary_v1/result.json`
