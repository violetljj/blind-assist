# HFTF Stage C D24：THOR-MAGNI proximity event ablation result

日期：2026-08-03

证据角色：Development / real-recorded-trajectory event proxy

研究主线：不变

默认 App：不变

## 结论

D24 完整产生 5 folds × 3 seeds = 15/15 paired units，但冻结 gate 仅 2/7
通过，终态为：

`D24_THOR_MAGNI_PROXIMITY_EVENT_DYNAMICS_NOT_SUPPORTED`

同一 D23 history checkpoint 使用真实历史与 flow，相对重复当前帧和零 flow：

| source-macro metric | mean delta | 正单元 | 正折 | 正 seed |
|---|---:|---:|---:|---:|
| event AUROC | -0.00641 | 5/15 | 1/5 | 2/3 |
| event AP | -0.00873 | 7/15 | 2/5 | 2/3 |
| event recall @ false-active ≤ 0.10 | -0.00132 | 6/15 | 1/5 | 2/3 |
| positive-anchor recall | +0.00964 | 7/15 | 2/5 | 2/3 |
| lead-time credit | +0.02175 s | 9/15 | 1/5 | 1/3 |
| clearance diagnostic | -0.00602 | 4/15 | 3/5 | 1/3 |

因此，D23 的 history-arm representation increment 没有在同权重输入消融下形成稳定的
事件排序或低 false-active 召回增量。

## 这项负结果否定什么

D24 否定的是：

> 当前 D23 history checkpoint 在 THOR-MAGNI proximity event proxy 上，稳定依靠
> 实际历史动态获得事件级增量。

它不否定 D23 已建立的事实：独立训练的 history arm 相对独立训练 current arm，在
proximity source-macro AUROC/AP 上有跨 seed、跨 fold 的 Development
representation increment。两个结论可以同时成立：历史输入改变了训练所得表示，但当前
模型没有把该信息稳定地用于事件决策；增量可能来自训练期正则化、source-specific
interaction 或未校准的动态残差，而不是可直接部署的时间推理。

lead-time credit 平均 `+0.02175 s` 且 9/15 单元为正，是受限的机制线索；但只有 1/3
seed mean 和 1/5 fold seed-mean 为正，不能单独升级为 event utility。

## Cohort 与执行完整性

- 1,078 samples，19 source sessions；
- 530 个 proximity-eligible anchors：
  - 157 positive onset anchors；
  - 373 negative anchors；
- 连续正 anchors 固定合并为 107 positive events；
- 157/157 个正 anchor 都从原始 scenario CSV 重建了首次 `1.25 m` 进入时间；
- 15 个已绑定 checkpoint 各做 history/zero-dynamics 两次推理，共 30 passes；
- 没有新训练、模型选择、seed 搜索或 held-out 正标签阈值搜索；
- 每个 source/arm 的 10% 工作点仅由该 source 的负分数确定，是诊断包络，不是部署
  阈值。

本次没有工程 invalid；若曾发生 checkpoint、CSV、CUDA、路径或落盘异常，按协议只会
修复后从头重跑，不会烧毁 source 或记为科学负结果。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d24-thor-magni-proximity-event-ablation-v0/
    report.json
    report.json.sha256
    scores.npz
    scores.npz.sha256
```

- report SHA-256：
  `60f461be0560f7518eefc8b474d8a349e31513315e6d6e8e5f0460b3cbfa6838`；
- scores SHA-256：
  `632fca1e2bd753198cdaf465eeb0bf989f3ce1f0b856a1b2e53479d9f4b09e9d`；
- report 约 112 KiB，score matrix 约 42 KiB；数值数组独立落盘，没有再次制造大型单
  JSON 控制面问题。

## 下一科学变量

不再增加 seed、改阈值或把 D24 的局部正 fold 包装成系统成功。下一候选应直接处理
“表示增量没有变成决策增量”：

> 在 train sources 上只学习一个有界的动态残差增益，把同一 checkpoint 的
> `history_logit - zero_dynamics_logit` 显式加入静态 logit，并在 held-out sources
> 上检验事件排序与召回。

这会把下一实验限制为一个可解释的标量决策桥，而不是再训练一个主模型。若 train-only
事件目标无法稳定学到非零、跨 source 可迁移的残差增益，则停止当前 D23→event
conversion 路线，同时保留 D23 表示层正结果。
