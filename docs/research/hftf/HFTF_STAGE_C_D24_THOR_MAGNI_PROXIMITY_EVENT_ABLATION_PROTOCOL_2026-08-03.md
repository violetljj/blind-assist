# HFTF Stage C D24：THOR-MAGNI proximity event ablation protocol

日期：2026-08-03

证据角色：Development / real-recorded-trajectory event proxy

研究主线：不变

默认 App：不变

## 科学问题

D23 已建立 history arm 相对独立训练 current arm 的 proximity-onset
representation increment，但没有证明模型在事件层实际使用了历史动态。D24 不训练新模型，
只复用 D23 的 5 folds × 3 seeds history checkpoints，对每个 held-out source 执行同权重
输入消融：

1. `history`：真实五帧历史与 D22 dense flow；
2. `zero_dynamics`：同一 checkpoint、重复当前帧与全零 flow。

它只问：

> 在真实记录的 THOR-MAGNI 轨迹上，移除历史动态是否会降低 proximity 进入事件的
> 排序、低 false-active 工作点召回和提前量？

同权重消融检验的是 checkpoint 对历史信息的因果依赖，不替代 D23 的独立训练
current comparator，也不建立部署效用。

## 冻结 cohort 与事件

- 样本固定为 D12 的 1,078 个 THOR-MAGNI anchors、19 个 source sessions；
- 只评价 `proximity_eligible=true`：
  - `proximity_onset=true` 为正 anchor；
  - `proximity_onset=false` 为负 anchor；
- 在同一 source 内，anchor frame 间隔不超过 45 frames 的相邻正 anchors 合并为一个
  positive event；45 只容纳原始 30-frame anchor stride 的连续采样，不跨越一个缺失
  anchor；
- positive event score 是组内最大 proximity score；
- negative observations 保持为负 anchors，避免事后发明无真值的负事件边界；
- source 必须同时有 positive events 与 negative anchors 才进入 source-macro 指标。

首次进入 `1.25 m` 的时间用 D8 已绑定的原始 scenario CSV、QTM anchor 和同一
`0.10 s / 2.0 s` source-native future scan 重建。它是几何代理，不是人工风险事件真值。

## 冻结指标

每个 checkpoint 在自己的 held-out fold 上产生一个 paired unit。每个 source、每种输入
独立取“负 anchor 中尽可能宽松且观测 false-active rate 不超过 10%”的阈值。阈值仅由
该 source 的负分数决定，不看正分数，用于评价固定 false-active 包络，而不是宣称已得到
可部署阈值。

主要指标：

- source-macro event AUROC；
- source-macro event recall at `false-active <= 0.10`；
- source-macro positive-anchor recall at 同一工作点；
- source-macro lead-time credit：对每个正 anchor，命中则计首次 `1.25 m` 进入时间，
  漏检计 0 秒，再在 source 内取均值；
- first subsequent eligible-negative clearance 作为诊断，不进入 gate。

所有差值均为 `history - zero_dynamics`。

## 冻结 gate

D24 仅在以下条件全部满足时支持事件层动态依赖：

1. source-macro event AUROC mean delta 至少 `+0.010`；
2. source-macro event recall mean delta 至少 `+0.020`；
3. event recall delta 的 3/3 seed means 为正；
4. event recall delta 的 fold seed-mean 至少 3/5 为正；
5. event recall delta 至少 10/15 paired units 为正；
6. positive-anchor recall mean delta 不低于 `-0.010`；
7. lead-time credit mean delta 大于 `0`。

通过终态：

`D24_THOR_MAGNI_PROXIMITY_EVENT_DYNAMICS_SUPPORTED`

失败终态：

`D24_THOR_MAGNI_PROXIMITY_EVENT_DYNAMICS_NOT_SUPPORTED`

失败不撤销 D23 的 representation 正结果；它只说明当前 checkpoint 的历史动态依赖尚未
转化为冻结的事件代理增量。

## 工程故障与主张边界

checkpoint 加载、路径、CSV 扫描、显存、缓存、序列化、落盘或中断错误属于可修复工程
故障。同一冻结协议可从头重跑，不烧毁 source，也不记为科学负结果。只有 15 个 paired
units 全部产生后才判 gate。

即使通过，D24 也不建立：

- human-event truth、用户提醒价值或安全收益；
- 可部署阈值、线上 false-alert 或 clearance；
- corridor / broad future-risk transfer；
- 研究主线替换、默认 App、生产或安全权限。

