# HFTF Stage C D4 opportunity ecology and recruitability R0

## 结论

D3-Q0.1 已诚实关闭为六来源 cohort 不足，不能靠第 41 个 slot、降门或追认 near-miss
继续。D4 改变研究问题与数据角色：

> 在一个全新、前瞻定义的来源总体中，all-four opportunity 的 source-level rate
> 是多少；pre-truth metadata 能否在有限成本内定义一个可招募、并与未来 effect
> 样本完全互斥的来源池？

当前状态为
`FROZEN_AFTER_D3_Q0_1_ATLAS_BEFORE_D4_METADATA_CENSUS_OR_FRESH_SOURCE_CONTENT`。
此设计只授权冻结 M0 metadata-census execution contract，不授权现在扫描、打开新
pose/media/support/truth、运行 ecology/effect 或 student。

## 为什么这是新问题

Q0.1 问的是：按冻结顺序和 40-slot 预算，能否组出 6 个 all-four qualified parents。
答案是不能。D4 不改变 `.10/5/20`、252 denominator、body/head × `.4/.8 s` 或
UNKNOWN 规则，而把“机会生态是否可招募”与“有机会时算法 effect 是否成立”拆成两个
互斥研究阶段。

Q0.1 的 37 个合法 selectors 只作 consumed pilot。整体 qualified 为 5/37；post-hoc
按 fps 看，5 Hz 为 3/15、20 Hz 为 2/22。这只生成“5 Hz metadata stratum 可能更易
招募”的假设，不证明 enrichment，也不能进入 fresh confidence interval。

## M0：只做完整 metadata census

official SANPO-Synthetic train split 固定 1560 sessions。原完整 HFTF exclusion 为
84 个，Q0 roster 40 个与其零重叠；D4 必须排除全部 124 个，不论 Q0 slot 是 carry、
failure、not-qualified 或 qualified。实现前复核发现，这个 124-parent global union
包含 6 个 official-test parents，并不属于 train；因此 train 内投影是 118 个排除项，
而不是 124 个。M0 必须保留全局 124 权威，同时对完整 1560-row train ledger 中的
118 个 in-split exclusions 零请求跳过，并对剩余 1442 IDs 各做且只做一次 metadata
census：

- 允许验证 description、camera schema、5/20 Hz、pose object receipt，以及 exact
  normalized 13-frame mask/depth object receipts；
- 禁止读取 pose CSV 内容、RGB/mask/depth bytes；
- 禁止计算 support、truth、clearance、effect 或 sealed payload；
- 输出完整 scan ledger、5 Hz/20 Hz eligible IDs、ineligible reason、authority
  hashes、fsynced 5 Hz pool manifest、one-shot seed receipt、单一 randomized rank
  及机械实例化的 ecology/effect/unassigned IDs；
- interruption/failure 不靠重跑补齐。

完整 5 Hz eligible-pool manifest 必须先 fsync。随后 allocator 先排他写入并 fsync
绑定 manifest hash 的 attempt，再仅调用一次 OS CSPRNG 生成 32-byte seed，写入并
fsync seed receipt；此时所有 fresh content 仍未打开。任何 orphan attempt/seed、
中断、重抽或手工覆盖都直接 M0 invalid，不得恢复抽签。

每个 parent 的 rank 固定为
`SHA256("HFTF_D4_R0_ALLOC|" || seed_bytes || "|" || lowercase_session_id)`，
按唯一 digest 升序；任意 digest collision 直接 M0 invalid，不使用 tie-break 继续。
完整 pool + 一次性随机 seed 形成无放回 random permutation；它不能读取 Q0 成败、
margin、scene、motion 或 failure signature。

## M0 后仍必须再冻结一次

M0 只是 inventory，不自动授权新 source content，但它也不再留下人工挑配额的空间。
R0 的 target subpopulation 固定为 fresh 5 Hz metadata-eligible natural parents；
20 Hz 只完成 census，不允许在本轮打开 content，也不能在 5 Hz 数量不足时补位。

外生 source-content cost cap 固定为 128。令 M0 得到的 fresh 5 Hz parent 数为 `N`：

- `N < 64` 直接
  `D4_M0_FRESH_5HZ_METADATA_RECRUITABILITY_POOL_INSUFFICIENT_STOP`；
- 否则 `C=min(N,128)`；
- ecology 数 `n=floor(3C/8)`，effect reserve 数 `B=C-n`；
- 单一 frozen hash rank 的前 `n` 个进入 ecology，接下来 `B` 个进入 effect
  reserve，其余永不在 R0 打开。

因此最大 ecology/effect source 数为 48/80，每个 source 最多 1 pose + 11 mask +
11 depth = 23 objects，总上限 2944。M0 只能把 `N` 代入这个函数，不能根据 5/20
composition 人工重选 target、sample、budget 或 fallback。

推断也固定为有限总体无放回 exact hypergeometric，而不是 IID binomial。Ecology 必须
跑满 exact `n` parents，不设 sequential early stop。success、not-qualified 或
acquisition/execution/orphan failure 都必须有 durable receipt；任何 failure 的
operational qualification indicator 固定为 0，且不 replacement。因此 estimand 是
“一个 fresh 5 Hz metadata-eligible parent 完成冻结 acquisition/qualification workflow
并通过 all-four gates 的概率”，不是纯 scene opportunity prevalence。

记 exact `n` parents 中 operationally recruitable 数为 `x`。对全体 `N` 中未知成功数
`K`，以 `alpha=.05` 做 one-sided 95% inclusive exact tail inversion：

- `K_L` 是使 `P[X >= x] >= .05` 的最小可行整数 `K`，`x=0` 时固定为 0；
- `K_U` 是使 `P[X <= x] >= .05` 的最大可行整数 `K`，`x=n` 时固定为 `N`；
- `R_L=max(0,K_L-x)`，`R_U=min(N-n,K_U-x)`。

再定义：

`R_min = min R : P(Hypergeom(N-n, R, B) >= 6) >= .90`

及仅用于报告的 `p_min=R_min/(N-n)`。只有 `R_L >= R_min` 才 GO；
`R_U < R_min`、`B<6`、`R_min` 不存在或物理上不可能留出 6 个成功源则 STOP；其余只能
`NOT_EVALUABLE_NO_EXPANSION`。灰区不得扩 ecology、改 allocation、转用 20 Hz、
复用 ecology 或从 effect reserve 补位。

Ecology 与 effect reserve 在 M0 时按 natural parent/session group 一次性不可逆分配。
camera/fps/file 或 derived artifact variant 不能被当成新 parent。Ecology 一旦打开
support/truth 就永久 consumed，永远不能转 effect；effect reserve 也不能补 ecology
failure。即使 recruitability supported，也只授权为已锁定 `B` 个 5 Hz parents 冻结
新的 sealed-effect contract，不自动授权 effect。若 `N>C`，所有未分配 parents 在
R0 永久禁用，不能在 STOP/NOT_EVALUABLE 后补入。

## Opportunity 定义保持不变

每个 source 仍是 7 anchors × 6 theta × 6 distance = 252 cells/stratum，四个 strata
全部必须满足：

- common-known coverage `>=.10`；
- future truth risk count `>=5`；
- future truth safe count `>=20`；
- UNKNOWN→SAFE `=0`。

不得删 head × `.8 s`、池化 strata、降 risk 门、改分母、按 near-miss 追认或在 fresh
outcome 后扩样本。

## Claim ceiling

M0 只产生 metadata-pool authority；后续 ecology 最多产生 fresh 5 Hz
metadata-eligible SANPO-Synthetic official-train subpopulation 的 prospective
operational recruitability 证据。它不估计 20 Hz 或 overall official-train
prevalence。未来 effect 若获准，也只能属于 5 Hz enriched qualified conditional
cohort。它不是目标人群自然风险 prevalence、人类事件或 safety evidence。

Q0/Q0.1、fresh effect、RGB student、研究主线、默认 App/Android、生产与 safety
权限全部保持关闭。

冻结前独立科学与工程终审均为 `CLEAR`、0 blocker。实现前工程复核随后识别并修正
了一个计数错误：124 个全局排除 parent 中只有 118 个在 official train，另 6 个是
已消费/保留的 official test；所以候选是 1442，不是 1436。该修正没有改变全局
124-parent exclusion authority、hash、target、cost cap 或任何 outcome gate。工程终审复算四个 parent
bindings、124-parent exclusion union（8060 bytes，
`156bf17c54ecfba41f181a12df209aecc56b3c9a6a85f27b2db2f340737252f2`）、
seed/allocation/CI/failure 闭集和全部链接；机器合同 SHA-256 为
`d7d26ac2267fe43c2a80d36cfe164a5544e34034c3b80509544be1591e3f0a68`，
并由 M0 execution contract 重新绑定。
