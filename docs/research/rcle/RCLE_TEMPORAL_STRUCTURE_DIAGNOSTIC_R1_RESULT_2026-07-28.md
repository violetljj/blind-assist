# RCLE 时间结构诊断 R1 结果

日期：2026-07-28

终态：`HOLD_MIXED_OR_INSUFFICIENT_TEMPORAL_EVIDENCE / VALID`

## 结论

当前四个 ADVIO Development session **不支持把高响应主要归因于 feature collapse /
blur / forward-backward 测量失效**；但也**没有证明高响应与 pose-derived 周期运动同步**，
因此还不能把开发优先级正式转向 motion decomposition 或 temporal modeling。

最关键的组合证据是：

- 四个 session 都有很强的 `0.7–3.0 Hz` pose 频带能量
  （band-energy fraction `0.729–0.924`）；
- flow direction 可评估覆盖为 `75.4%–99.2%`，相邻 pair 方向余弦中位数为
  `0.976–0.993`，说明大部分时段并非普遍跟踪崩溃；
- 但 flow 在各 session 的 pose 主频上只有 `R²=0.020–0.035`，没有达到冻结的
  `0.20` 周期同步门；
- 高响应与 measurement-failure 的重叠为 `17.6%–47.1%`，四个 session 都低于
  `50%`；没有 session 满足完整 measurement-failure routing；
- 因而 motion routing 为 `0/4`，quality routing 也为 `0/4`，合法结论是
  `HOLD`，不是二选一强判。

这意味着“高响应发生时 flow support 往往尚可”是可支持的描述，但“这些 flow 与
步态周期同步”没有得到支持。它挑战了把低 feature count 当统一解释的方向，也没有
把 pose 周期代理升级为因果原因。

## 冻结设计

[R1 合同](RCLE_TEMPORAL_STRUCTURE_DIAGNOSTIC_R1_CONTRACT_2026-07-28.json)
在正式 flow-direction 提取前冻结，SHA-256 为
`69c4f26acd44ae33660d95d5f4687c72dd5b8bbcf863d6b755553365d24bec6e`。

- 独立描述单位是 capture session；pair 是重复纵向测量；
- sequence13、14、15、17 各固定 `601` pair，sequence16 保持
  `SEALED_UNSEEN`；
- Stage 1 只读取已开放的 RGB、timestamp 和 source pose，禁止读取 response、
  trigger、风险/障碍/人工 gait 标签；
- Stage 2 才连接冻结 direction ledger、R0 proxy ledger 和 unchanged R3 ledger；
- R3、strict `>0.01/s`、三连续 pair、窗口和 PairState 均未改变；
- “高响应”仍是 session 内 R3-evaluable absolute compensated response 的最高
  20%，不是风险错误标签。

正式输出前的合成测试发现 absolute response 可能在一个周期的两个半周期形成稳定
双峰，因此普通相位锁定会互相抵消。协议在任何新 flow-direction 输出产生前改用
轴向相位锁定 `|mean(exp(2iφ))|`。同一预执行检查还加入 radial flow，避免全局
median `(dx,dy)` 把空间上天然相消的径向扩张场误判为方向失效。正式输出后没有改变
频带、阈值、窗口或终态。

## 逐 session 结果

| session | pose component / 主频 | pose band energy | 周期数 | 高响应周期比例 / 最长连续周期 | axial PLV | direction coverage / adjacent cosine | flow-at-pose-frequency R² | high-response direction coverage | failure overlap / RR | routing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 13 | translation-y / 1.597 Hz | 0.908 | 15 | 46.7% / 7 | 0.151 | 96.2% / 0.979 | 0.035 | 100.0% | 17.6% / 0.885 | neither |
| 14 | translation-x / 0.899 Hz | 0.898 | 11 | 63.6% / 7 | 0.536 | 80.0% / 0.976 | 0.034 | 68.9% | 47.1% / 2.040 | neither |
| 15 | angular-z / 1.100 Hz | 0.729 | 11 | 63.6% / 7 | 0.254 | 99.2% / 0.993 | 0.022 | 100.0% | 33.3% / 2.017 | neither |
| 17 | translation-z / 1.597 Hz | 0.924 | 12 | 91.7% / 7 | 0.477 | 75.4% / 0.985 | 0.020 | 85.0% | 45.0% / 1.269 | neither |

sequence17 最接近 motion routing：除 flow-at-pose-frequency `R²` 外，其余八项冻结
条件均通过；但 `0.020` 远低于 `0.20`，不能用“几乎通过”替代终态。sequence14 的
高响应 direction coverage 为 `68.9%`，也低于 `70%`。

## Measurement failure 时间结构

| session | feature-collapse pairs | round-trip-failure pairs | union failure pairs | failure events / 最长 event | 高响应与 failure 重叠 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 13 | 6 | 1 | 122 | 12 / 57 pairs | 17.6% |
| 14 | 93 | 2 | 188 | 15 / 88 pairs | 47.1% |
| 15 | 1 | 1 | 121 | 10 / 54 pairs | 33.3% |
| 17 | 126 | 1 | 238 | 16 / 49 pairs | 45.0% |

union failure 还包含 session 内 sharpness 最低 20%，因此即使在最宽的冻结定义下，
高响应 overlap 仍没有一个 session 达到 50%。sequence14、15 的 failure-vs-good
高响应 prevalence ratio 分别为 `2.040`、`2.017`，说明局部富集确实存在；但 overlap
不足，不能把它概括为主要机制，也不能据此重新设计统一 quality gate。

## 周期复现与同步

高响应在 sequence14、15、17 的多数 pose-derived 周期中反复出现，最长连续
`7/7/7` 个周期；sequence13 为 `46.7%`。这支持“存在重复时间结构”作为后续解释
线索。

但是：

- 这些周期来自 source pose 的频带和零交叉，不是人工 gait phase；
- flow 的 pose-frequency `R²` 在四个 session 都很低；
- axial phase locking 只有 sequence14、17 达到 `0.40`；
- 因而“反复出现”不等于“与步态同步”，更不等于步态振荡是因果原因。

## Learning record

- observation：pose 高频能量与逐 pair flow 方向支持广泛存在，measurement failure
  不是多数高响应的共同伴随条件；但 flow 没有在 pose 主频上形成冻结要求的周期拟合。
- supported inference：停止把当前 feature-count/FB gate 当作统一解释；同时不把
  pose 高频代理直接升级为 motion-model failure。
- alternatives：十秒窗可能不足以稳定估计局部 flow mode；全局/径向三维 PCA 仍可能
  混合多个空间区域；pose-derived 周期可能不是 gait；真实接近运动也可能贡献响应。
- challenged constraint：单 pair support 良好不能替代跨时间同步证据，周期 recurrence
  也不能替代 phase/frequency alignment。
- reuse：四份 direction ledger、径向场合成反例、周期/axial-PLV 测试和独立 validator
  保留为 Development negative evidence 与回归 fixture，不升级为 unseen evaluation。

## 判读边界

这批数据没有障碍、风险、接近或人工 gait 真值，所以不能回答：

- 高响应或 trigger 是否是假警；
- pose-derived 周期是否是正常步态；
- 步态/头动是否造成高响应；
- quality gate、motion decomposition 或 temporal model 是否改善性能；
- 其他 session、来源、Android 或真实助行条件是否成立。

## 复现与验证

- Python `3.11.9`
- NumPy `2.1.3`
- OpenCV `4.10.0`，threads `1`，RNG seed `20260728`
- 9 项 focused synthetic/real-LK/invalid-cycle-gap/contract tests：`PASS`
- analysis：
  `artifacts.local/evidence/rcle_temporal_structure_diagnostic_r1/session_analysis_r1.json`
  SHA-256
  `e2bf758d4d7afd86e67a7604f6fd67dd732bd5f5cea369ea8cd1b3cbc3ec82e1`
- 独立 validator 未导入 production summarizer：`VALID / failures=[]`
- validation receipt SHA-256：
  `7b302a0881540bcab9268c90813fde8e8b1ddc5c3252e392b57ac714557cc5fb`

四份 direction ledger SHA-256：

- sequence13：
  `d03b608858a4b32b0cc73e762272602db1c481b42a482ce5afaec7617f808a1a`
- sequence14：
  `62b02250de735fedccbde7e01bf732d85f27bf67332c86f8472447664bbac220`
- sequence15：
  `30c3f7d14ead72009913d5283053d85177e151565c18e2db85938dea6a22bc33`
- sequence17：
  `3d29133da70d13426fcd8cae9825f35c2e4117813a579839a8ec7c09b3a5dae3`
