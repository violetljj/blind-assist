# HFTF Stage C D31：THOR-MAGNI full-resolution measurement result

日期：2026-08-03

证据角色：Development / full-resolution current measurement replication

研究主线：不变

默认 App：不变

## 结论

D31 完整校验 19 个原视频 hash、顺序解码 530 个 current anchors，并复用 D30 的
冻结 measurement contract。gate 通过 6/8，整体终态为：

`D31_THOR_MAGNI_FULL_RESOLUTION_MEASUREMENT_RELATION_NOT_SUPPORTED`

全分辨率明确修复 detection、bearing 与大部分 nearest-person opportunity，但没有
修复 source-general distance rank。因此不能进入完整 bearing-distance state filter。

## D30 → D31 单变量比较

| metric | D30 128×224 | D31 source resolution | change |
|---|---:|---:|---:|
| detector anchor coverage | 74.15% | 87.36% | +13.21 pp |
| detector + visible-person anchors | 289 | 322 | +33 |
| accepted / assigned pairs | 61.88% | 67.74% | +5.86 pp |
| nearest-person accepted coverage | 46.51% | 58.43% | +11.92 pp |
| source-macro bearing Pearson | 0.7089 | 0.7847 | +0.0758 |
| source-macro bearing MAE | 14.12° | 11.23° | -2.89° |
| source-macro distance Spearman | 0.2867 | 0.2485 | -0.0382 |
| pooled distance Spearman | 0.4246 | 0.3410 | -0.0836 |
| evaluable sources | 17/19 | 18/19 | +1 |

D31 的 322 anchor opportunity、67.74% accepted fraction、bearing Pearson/MAE、
5/5 positive distance folds 与 18 evaluable sources 均通过。失败仅有：

- nearest-person coverage `58.43% < 60%`；
- source-macro distance Spearman `0.2485 < 0.30`。

nearest coverage 接近门槛不能四舍五入为通过；distance rank 的下降更说明继续只靠
box height 调 detector/assignment 不是有证据的下一步。

## 保留的正结果

D30 的 bearing signal 在更高分辨率上得到强化复现：

- pooled box-x / predicted-x Pearson `0.7806`；
- source-macro Pearson `0.7847`；
- pooled/source-macro bearing MAE `11.91°/11.23°`；
- accepted pairs 从 310 增到 399；
- visible-body assignment fraction 从 `72.82%` 增到 `85.61%`。

因此 current person bearing 是跨 source、跨输入分辨率可测的 Development 信号。
这不是完整 world measurement，也不被 distance gate 失败抹掉。

## distance 断点

全分辨率 height 没有提高 distance relation：

- pooled Spearman `0.3410`，仍为正；
- 5/5 folds 仍为正；
- 但 source-macro 从 `0.2867` 降到 `0.2485`。

这更符合 source-specific camera/person-scale calibration 异质性，而不是像素不足。
继续在 THOR 上搜索 FOV、height formula、confidence、NMS、assignment cost 或
source normalization 会消费同一 current geometry outcomes，不再授权。

## 非 person 边界

D31 仍只评价 `Helmet_*` Visitor/Carrier person。D26/D27 还包含
`DARKO_Robot` 与 `LO1` carried object；完整 collision field 未来必须有非 person
measurement channel。person bearing 正结果不能升级为全对象 collision coverage。

## 工程与复现

- 19/19 source videos SHA-256 verified；
- full-resolution boxes SHA-256：
  `ecc30d0106372245c26cae6e5bece1b051036a2037ddb8e5908a4d75ff27701f`；
- report SHA-256：
  `bb8f68214cb617729ca289fc4762ab700b4e452e04fa386403e442dc4c0bb860`；
- 463/530 anchors 有 person，1,990 raw detections，1,885 selected slots；
- 42 anchors 超过 8 slots，selection 按冻结规则截断；
- 无 future outcome read、无训练、无阈值/FOV/assignment 搜索；
- stderr 为空，无 hash、decode、path、cache、serialization 或 fsync invalid。

```text
artifacts.local/evidence/hftf/
  stage-c-d31-thor-magni-full-resolution-measurement-v0/
    full_resolution_boxes.npz
    full_resolution_boxes.npz.sha256
    report.json
    report.json.sha256
    stdout.log
    stderr.log
```

## 下一科学变量

按冻结 stopping rule，停止 THOR current-box distance fitting。下一候选转入已有
source-authority 证据的原生 2D/3D identity-bound person trajectory source：

1. 以 native identity 同步 2D box、3D person position 与短时 trajectory；
2. 分解 bearing、distance、velocity 三个 measurement errors；
3. 冻结一个不依赖 THOR source-specific scale 的 state estimator；
4. 在独立 sequences 做 sequence-heldout replication；
5. 只有 state measurement 通过，才回到 action-conditioned collision field；
6. THOR 仅作为后续 external transfer，不再作为 measurement 拟合 source。
