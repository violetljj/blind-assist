# P0-A1 Ambiguity Gate Discovery result

状态：`ONE_FROZEN_FEATURE_SWEEP_COMPLETE / CLEAR_SIGNAL_COMPACT_POLICY_NEXT / CONSUMED_DEVELOPMENT_ONLY / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`

日期：2026-08-21

## 结论先行

现有、已实际观测的 runtime representation 含有明显的 commit/abstain discrimination signal。P0-A1 在协议
提交 `f634e0b6` 后只运行了一次冻结 feature sweep；没有新增 parent、没有新模型调用，也没有运行
Logistic、Conformal、ranking、Sky 或 teacher。冻结选中规则为：

```text
retain the existing Terra SELECT only when
    brain_confidence >= 0.85
and candidate_center_dispersion <= 0.2423407460503519
otherwise ABSTAIN
```

它只给既有 Terra `SELECT` 加 gate，不改变 selected candidate，也不能把原来的 abstain 反向变成 commit。

| policy | AMBIGUOUS false commit | ambiguous parent macro | RESOLVABLE commit coverage | committed resolvable correctness |
|---|---:|---:|---:|---:|
| ungated Terra baseline | 36/51 = 70.59% | 70.78% over 17 parents | 20/20 = 100% | 17/20 = 85% |
| frozen selected conjunction | 11/51 = 21.57% | 19.61% over 17 parents | 20/20 = 100% | 17/20 = 85% |
| always abstain | 0/51 = 0% | 0% | 0/20 = 0% | undefined |

因此 ambiguous false commitment 下降 `49.02` 个百分点，venue-parent macro 下降 `51.18` 个百分点，
同时没有牺牲当前 20 个 resolvable episodes 的 commit coverage，也没有改变 committed correctness。它通过了
预冻结 A 类止损门，终态为：

```text
CLEAR_SIGNAL_COMPACT_POLICY_NEXT
```

这授权的下一步只有 `P0-A2 compact ambiguity policy discovery`；它不把当前 conjunction 准入 App、正式
baseline 或科学确认，也不直接授权 Sky。

## 一次性 sweep 范围

P0-A1 只纳入具有同一冻结 Grounding DINO candidate surface、同一 Terra baseline decision 与同一冻结 evaluator
结果的两组 consumed Development：

- 初始 Silver-B：47 episodes / 11 venue parents；
- Brussels parent-disjoint Development：24 episodes / 10 venue parents。

合计 71 episodes / 21 parents，其中 51 个 `AMBIGUOUS` episodes 来自 17 parents，20 个
`UNIQUE + SET_VALUED` resolvable episodes 来自 9 parents。parent 集合可跨 truth class 重叠，不能把
`17 + 9` 当成 26 个独立 parents。

当前全部 consumed public bank 的 92 episodes / 34 parents 没有被伪装成完整 runtime cohort：D2 enrichment
没有 Terra baseline，D3 没有 detector/Brain runtime evidence，`NOT_OBSERVED` 也没有视觉输入。因此这些行均按
协议列为 `MISSING_RUNTIME_EVIDENCE / NOT_OBSERVED`，不插值、不用人工或 evaluator 字段代替 runtime feature。

## 冻结搜索面

八个 feature 都能在决策时由现有 runtime surface 得到：Brain confidence、detector top-1 score、top1-top2
margin、candidate count、0.05 near-tie count、selected candidate rank、selected score margin 与 candidate center
dispersion。跨 prompt/provider agreement、跨 view stability、place identity 和 entrance relation 在两组输入中不
统一可用，因而没有补值。

阈值只按未看标签的 observed-value grid 生成；每个 feature 最多 11 个 cut points。固定方向的单 feature rule
与全部两 feature conjunction 共 3,159 条。94 条通过 clear gate，644 条通过较弱 gate（包含 clear），去重后
safety-coverage Pareto frontier 有 15 个点。冻结 selector 先最大化 resolvable coverage，再最小化 ambiguous
false commit，因此选中上面的 100%-coverage conjunction。

## 解释边界

这是一个真实而有用的 Development signal，但还不是可部署算法：

- 阈值选择和指标都来自同一 consumed cohort；没有 fresh、held-out、置信区间或 p-value；
- 3,159 个规则属于预冻结但多重的有限搜索，selected threshold 很可能包含 cohort-specific optimism；
- Brain confidence 与 proposal geometry 相关性只说明当前 representation 有可分信号，不证明两个 feature 是
  因果机制，也不证明跨城市、设备、天气或视角稳定；
- 当前规则保留了全部 3 个 resolvable localization errors，所以它解决的是“是否承诺”的开发信号，不是 ranking；
- D2/D3 缺失 runtime evidence 仍是明确 coverage gap，不能把 71-episode结果外推成全部 92 episodes 的表现。

因此不再重跑 P0-A1、不加 feature、不改 threshold。P0-A2 若继续，只能在同一 consumed evidence 上寻找更紧凑、
可解释且稳定的 ambiguity policy；达到预先定义的 winner 后，才值得购买一次有限科学确认。

## Evidence

- frozen protocol：[`P0_A1_AMBIGUITY_GATE_DISCOVERY_PROTOCOL_V1.json`](P0_A1_AMBIGUITY_GATE_DISCOVERY_PROTOCOL_V1.json)
- frozen protocol content SHA-256：`810c50835bc667193a0add5cb776c91dc656e0faec5ee822d556ac959468c484`
- result content SHA-256：`31350c374ea4cb737252468ea5ef5ef75e84f777a1e4d814c21aa8c1f0352984`
- result file SHA-256：`91a35f039f2bc7b7439cbcbd218b64549109f87f2674d750cbba04021fcea0b6`
- protocol freeze commit：`f634e0b6`
- result artifact：`artifacts.local/evidence/p0-s0/2026-08-21-p0-a1-ambiguity-gate-discovery-v1/result.json`

Claim ceiling：`CONSUMED_DEVELOPMENT_AMBIGUITY_FEATURE_DISCOVERY_ONLY_NO_GENERALIZATION_OR_SCIENTIFIC_VERDICT`。
