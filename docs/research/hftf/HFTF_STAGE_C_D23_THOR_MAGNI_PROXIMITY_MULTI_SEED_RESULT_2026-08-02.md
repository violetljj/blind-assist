# HFTF Stage C D23：THOR-MAGNI proximity multi-seed result

日期：2026-08-02

证据角色：Development / post-hypothesis target-specific robustness

研究主线：不变

默认 App：不变

## 结论

D23 的冻结 proximity gate 10/10 全部通过：

`D23_THOR_MAGNI_PROXIMITY_MULTI_SEED_ROBUSTNESS_SUPPORTED`

合并 D22 已观察的 seed17 与 D23 新执行的 seeds23/41，共 5 folds × 3 seeds：

| metric | overall mean | 正单元 | 正折 | 正 seed |
|---|---:|---:|---:|---:|
| proximity source-macro AUROC | +0.04098 | 12/15 | 4/5 | 3/3 |
| proximity source-macro AP | +0.03242 | 11/15 | 5/5 | 3/3 |
| proximity pooled AUROC | +0.01391 | 8/15 | 4/5 | 2/3 |
| proximity pooled AP | +0.01877 | 10/15 | 3/5 | 2/3 |

source-macro AUROC 的各 seed 五折 mean：

- seed17：`+.03669`；
- seed23：`+.04996`；
- seed41：`+.03628`。

source-macro AP：

- seed17：`+.03660`；
- seed23：`+.04407`；
- seed41：`+.01658`。

因此 D22 的 proximity 信号不是 seed17 单次偶然；它在三个 seed、五个
source-session-held-out folds 与 pooled 指标上建立了稳定的 Development
representation increment。

## 主张边界

D23 是在看到 D22 seed17 proximity 正信号后冻结的 post-hypothesis 检查，不是 fresh
confirmation。其准确结论是：

> D20-style aligned dense-flow dynamics 对 THOR-MAGNI current-negative
> proximity onset 具有跨 source-session、multi-seed Development robustness。

它不建立：

- corridor 或完整未来风险场的 broad transfer；
- human-event reminder utility；
- first-warning lead time 或 false-alert 改善；
- 研究主线替换；
- App、生产或安全主张。

这些是后一层尚未检验的主张边界，不是否定当前正结果。

## corridor 负结果保持

同一 15 units 的 corridor source-macro：

- AUROC mean `-.00763`，仅 1/3 seed mean 为正；
- AP mean `+.00224`，仅 2/3 seed mean、1/5 fold seed-mean 为正。

因此 D22 的 broad-transfer 失败仍然有效。D23 不是事后删除 corridor 来宣称 D22
通过，而是对一个预先存在 target 上的新假设做独立标注的 Development successor。
当前机制更像接近速度/looming signal，而不是完整 ego-path corridor reasoning。

## 可复现证据

新增 seeds23/41：

```text
artifacts.local/evidence/hftf/
  stage-c-d23-thor-magni-proximity-multiseed-v0/
    additional_seeds_report.json
```

- additional report SHA-256：
  `8ee936d14ccf23904313f7f6ef454e32ebfa5f7e9b52172c1909d41def7e04cb`。

三 seed 聚合：

```text
artifacts.local/evidence/hftf/
  stage-c-d23-thor-magni-proximity-multiseed-v0/report.json
```

- aggregate report SHA-256：
  `9b0db1615323dee41634b4ab54d8ba66b955a1245a0186570df441cd0e43b7b3`。

新增执行 20 个 training runs；连同 seed17 共计 30 个 current/history training
runs。所有 units 使用相同 1,078 samples、19 source sessions、五折、模型、flow、
loss、30 epochs 与 fixed-final-epoch evaluation。

## 下一科学变量

D23 只授权冻结一个真实事件层的 proximity-onset decision test。下一实验应优先复用
已有真实连续序列与 source-native timing，回答：

> D23 proximity score 能否相对 current comparator 提前发现真实近距进入，同时不
> 明显增加 false-active exposure？

门控应围绕 event recall、first-warning lead time、false-active rate 与 clearance，
而不是继续增加 seed、调模型或把 synthetic cell AUROC 当作系统效用。若本地资产无法
形成有正负 event 的可评价 cohort，终态只能是 event-layer `NOT_EVALUABLE`，但不撤销
D23 representation 正结果。
