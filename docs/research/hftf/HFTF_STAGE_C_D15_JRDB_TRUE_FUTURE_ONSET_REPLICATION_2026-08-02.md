# HFTF Stage C D15：JRDB true-future-onset 独立复现

日期：2026-08-02

证据角色：Development / independent-dataset onset replication

研究主线：不变

默认 App：不变

## 结论

D13 在 THOR-MAGNI true future onset 上得到的弱 history signal 没有在 JRDB
独立来源复现。JRDB onset cohort 的两个固定 source-pair folds 都包含 corridor
正负例，因此这是可评价后的科学负结果，不是 opportunity 不足或控制面终止。

终态：

- `D15_JRDB_CORRIDOR_FUTURE_ONSET_TWO_FOLD_READY`
- `D15_JRDB_FUTURE_ONSET_HISTORY_REPLICATION_NOT_SUPPORTED`

primary corridor history-minus-current：

| fold | AUROC | AP |
|---:|---:|---:|
| 0：clark + gates | -0.00595 | -0.00641 |
| 1：meyer + stlc | -0.00641 | -0.05556 |
| 两折 mean | -0.00618 | -0.03098 |

AUROC/AP 都是 0/2 folds 为正。proximity 负对照一正一负，aggregate AUROC/AP
`-.00222/+.00054`，不触发 target 切换。

D12 修正后的 true-onset estimand 继续保留；D13 也保留为 THOR source-local weak
representation signal。但当前 frozen MobileNet history representation 不再进入更多
head、seed、threshold 或同数据集搜索。下一步需要更大的独立 onset-rich source，
或实质不同的预训练任务；不能把 THOR 的弱正结果升级为跨来源效应。

## JRDB transition census

D9 的 JRDB future window 是 `anchor+1 ... anchor+30`，不包含 `t=0`，因此没有
THOR 原标签的同一实现缺陷。D15 仍显式计算 anchor-frame 3D-person state，只在当前
安全样本中定义 onset：

```text
eligible = current state is safe
onset = eligible and any future frame is risky
clearance = current state is risky and every future frame is safe
```

总机会：

| target | eligible | onset positive | onset negative | clearance |
|---|---:|---:|---:|---:|
| proximity | 102 | 14 | 88 | 0 |
| corridor | 71 | 10 | 61 | 1 |

按固定 source-pair fold：

| fold | proximity 正/负 | corridor 正/负 |
|---:|---:|---:|
| 0 | 10 / 40 | 8 / 35 |
| 1 | 4 / 48 | 2 / 26 |

fold 1 corridor 只有两个 positives，AP 置信度很弱；但不是单类，且运行前已明确只在
两折都为正时支持 replication。因此该不确定性限制证据强度，不把负结果改成
`NOT_EVALUABLE`。

## 等容量 replication

- frozen JRDB MobileNet `5×576×4×7` spatial maps；
- current arm 重复 current map；
- history arm 使用真实五帧；
- 两臂共享同一个 13,586-parameter temporal-spatial head；
- seeds `17/23/41`、120 epochs；
- target-masked、source-balanced BCE；
- fixed final epoch，无 held-out 选模。

沿用 D9 的 corridor primary gate：

- corridor AUROC/AP seed-mean delta 在两个 source-pair folds 都为正；
- 两项各至少 4/6 fold×seed units 为正；
- proximity 仅作负对照。

六个 units 中 fold 0 三个 seeds 的 corridor AUROC/AP 全部为负；fold 1 只有两个
精确零或微正 units，另一个 seed 明显为负。两折 aggregate 因此不支持 replication。

## 可复现证据

```text
artifacts.local/evidence/hftf/
  stage-c-d15-jrdb-future-onset-v0/
    samples.jsonl
    report.json
  stage-c-d15-jrdb-future-onset-replication-v0/
    report.json
    report.json.sha256
```

- onset samples SHA-256：
  `b60848c48097a6b18830b7bd7285e20f1d51ac220a02f28761310a9ae9d43b06`
- census report SHA-256：
  `ee61c8fd6d4bfd8e91eaa57da4ed864f37554f6e87304733774ce305d0befc64`
- frozen spatial features SHA-256：
  `530dedf0005f709eb0009f4ddbb79cbbe27d6b3b44047d916980da12d466a8c9`
- replication report SHA-256：
  `044e02340a2fd3e0184d181d7b35f649ae5946d906eb04e6d1cd73c29e9325d4`

## 主张边界

JRDB 是 robot-centric RGB360 + source-native person 3D geometry，不是盲人佩戴相机
或 human reminder truth。D15 只裁决 D13 weak history signal 能否跨到第二个几何
来源；它不裁决所有 HFTF 表示、人体三高度碰撞场或未来取得更大真实步行数据后的模型。
