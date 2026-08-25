# GRAIL-R1C-L Task-Trained Pairwise Owner Coordinate Protocol

状态：`AUTHORIZED / REVERSIBLE_EXPLORATION / FRESH_HOUSE_DISJOINT / RGB_MASK_ONLY_AT_INFERENCE / ONE_ARCHITECTURE_TWO_SEEDS_MAX / FINAL_TEST_UNOPENED / STOP_BEFORE_DEPTH_GEOMETRY_AND_M2`

## 问题与边界

R1C-O 已建立 privileged owner-local coordinate ceiling；R1C-V deterministic rule 与 R1C-P fixed zero-shot OA-V2 都没有从 RGB 获得该 coordinate。R1C-L 是当前 observation contract 下最后一次 RGB-only learned successor，只回答：大量 ProcTHOR 原生监督下，任务化 pairwise 模型能否从 reference/query owner-group RGB 与 mask 学出下游 slot 所需的相对规范坐标。

部署输入仅为两视图 owner-group RGB crop、sibling union mask、sibling centroid mask 及其直接图像几何。native owner yaw、camera pose、metric depth、AI2-THOR coordinates、OA-V2 prediction 与 target-slot truth 禁止进入模型输入；native identity/pose 只生成训练标签和 evaluator truth。M1 selector、candidate set、pose head、threshold 与 absence logic 冻结。

## 冻结实现与数据

唯一执行 manifest 是 `scripts/research/grail/grail_r1c_l_manifest_v3.json`，SHA-256 `736ef95ad7b71aea007492b34a16fe1c4f27a2e49c20ef666e3f7198dcf35c81`。v1/v2 仅用于 5-house mechanics pilot；pilot 只暴露 group/view/pair denominator 与 simulator timeout，没有训练、OA-V2 或 learned-model outcome。v3 在任何模型 outcome 前冻结相同 house roster，并将机械修正固定为 native-owner 内 2–4 target-local sibling neighborhood、至少 2 个共同 physical siblings 与 ordered `reference→query` pairs：

- ProcTHOR train 160 houses 用于 20k–50k pairs；另 20 houses 用于 2k–5k validation pairs；
- ProcTHOR test 12 个独立、排除既有 GRAIL test roster 的 houses 只承担一次 final adjudication；
- physical owner 与 house 不跨 role；同一 owner group 最多 8 个 quadrant/near-far balanced views；
- pair 两端各有 2–4 个可见 siblings，且至少共享 2 个 physical siblings；大于 4 的 native owner group 用去重 4-nearest local neighborhoods；ordered `reference→query` 两个方向都保留；Drawer/Doorway 在 loader 平衡；
- final test 在 validation 相对固定 OA-V2 slot uplift 未达到 `+8` 时不得渲染或访问。

唯一模型为 shared DINOv2-S RGB encoder（仅最后 2 blocks 可训练）+ 2-channel mask adapter + 2-layer bidirectional cross-attention + 36×10° yaw head。训练最多 seeds `1701/2701`，不扫 backbone、crop、bin、loss weight、aggregation 或 ensemble。

## Slot-marginalized objective

每个 yaw bin 只按其对 sibling horizontal direction 导出的 permutation 参与监督。所有能产生 native-correct slot permutation 的 bins 共同构成正集合：

```text
L_slot = -log sum_{b: pi(b)=pi*} p(b | reference, query)
```

90°/270° 两个与水平 slot direction 正交的 bins 不进入正集合；模型仍输出完整 36-bin distribution。交换一致性权重固定为 `0.05`，比较 `p(Q,R)` 与 `inverse(p(R,Q))`。

## 停走与最终门

validation 固定 OA-V2 baseline 物化一次。两个 seeds 中按 validation slot accuracy 选一个；相对 OA-V2 uplift `< +8` 时终态为 `STOP_R1C_L_WITHOUT_FINAL_TEST`，不得用同 cohort 调参，也不打开 final roster。

若 validation 通过，final test 同 cohort 同时运行冻结 OA-V2、R1C-L 与 evaluator-native R1C-O。78-positive 门为：slot 比 OA-V2 `+12` 且 `>=55`；referent `+12` 且 `>=55`；complete `+8` 且 `>=40`；wrong-target `<=2/43`；absence false commit `<=1/78`；candidate permutation `156/156`；Drawer 与 Doorway 都有正 uplift。成功只开放后续另行冻结的 M2/formal；失败关闭 single-RGB owner-orientation，并且本协议不自动授权 R1C-G depth/geometry。

主张上限始终是 synthetic ProcTHOR Development mechanism evidence；不建立 natural-scene、live device、Android/default-App、product 或 safety authority。
