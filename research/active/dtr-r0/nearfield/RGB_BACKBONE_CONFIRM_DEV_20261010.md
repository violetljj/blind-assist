# RGB 主干六个新 capture 冻结工作点比较（Development，2026-10-10）

UniDepth V2 raw 在六个新 capture 的近带见证均高于 DAV2 raw，近带从 3/292 提升至 103/292，中远合计从 464/1047 提升至 775/1047。但原 pooled304 工作点的误支持代价未迁移：近带 FREE 从 0/515 增至 12/515，超过预封存容许增量 5；中远 FREE 从 7/107 增至 17/107，超过容许增量 2。因此本次主替换信号 **未出现**，保留 **DAV2 raw R0 的 RGB ≥0.8m 职责，近带交给 ToF**。

这项失败发生在 FREE 成本约束，不能据此否定 UniDepth 的近带信息。近带逐 query 救 100、损 0，中远救 357、损 46；应保留这些收益与局部损失。本次没有通过调阈值、重拟合或选 capture 补救。远带 FREE 只有 6 个，两个模型均为 0/6，不能据此声称远带误支持已充分验证。

## 范围与冻结协议

本次为 EXPLORE / fresh-capture Development，仅比较 DAV2 Metric Indoor Large raw R0 与 UniDepth V2 ViT-L raw R0。沿用原 query/ray 交集及逐 ray `min(z-entry, exit-z)` 的原第 16 顺序统计量；正见证只对冻结 native first-return POS 参考域应用相同统计量。无训练、微调、affine 重拟合、读出变体或已缓存预测重跑。

PLAN 于 2026-10-10 13:36:32 UTC 封存，SHA256 `ebe6605d4c051d417731d0011cc673a570c158cd8d9a0e67fb4b46e688aa0000`。主工作点逐字节复制上一 readout run 已封存 pooled304：DAV cutoff `0.24403834342956543 m`，Uni cutoff `0.09616100788116455 m`；原 cal 两者均为 29/588 sampledFREE 支持。完整原切点文件复制 SHA256 `c65b3d5c76c76a779a03ed919a2094e6ef0e8dc04db7373b8167cc78f2cd6ba8`，没有重新估计 pooled cut。

辅助三折只用旧六 capture 的 FREE：旧顺序 `arkit16, arkit_40777060, arkit_40777065, new_arkit_41069021, new_arkit_41069042, new_arkit_41069048`，序号 `%3` 定 fold，每折四个旧 capture 校准、两个旧 capture 留出。strict sampledFREE 5% 保留整组 ties；841 个旧 FREE query、86 个含 FREE 帧参与校准。全部六个辅助 cut 于 13:39:23 UTC 前封存，随后才读 fresh 模型分数；辅助结果不替代主工作点。

Fresh 数据按官方 ARKitScenes raw Validation CSV 原顺序，排除已消费 visit，取首六个满足冻结近带 POS≥16 query 的 visit，每个取 16 个时间排序的均匀对齐帧，共 96 帧、2592 query。候选 41125696 的近 POS 为 9，按预声明 coverage gate 跳过，保留选择收据；不是按模型输出筛选。六个 accepted 的近 POS 合计 292。近 POS 富集意味着结果不代表普通场景分布。

Fresh public valid 为预封存的完整注册 256×192 grid 与有限几何域；不新跑 DepthPro，不取 DAV/Uni finite-mask 交集。每模型 invalid depth 失去支持，不能删除 FREE query 或改分母。旧辅助数据保留原冻结 public domain。UniDepth 使用公开 K，DAV 按官方 metric indoor 输入方式；RGB、坐标映射、native sampling 与原冻结输入链一致。

主判定仅使用六个 fresh capture 与原 pooled304 cut：近带至少 4/6 capture 的 Uni W 严格大于 DAV，aggregate 净 FREE 增量不超过 `floor(.01×nearFREE)`；中远合计见证净损满足 `100×(DAVW−UniW)≤3×DAVW`，且 aggregate 净 FREE 增量不超过 `floor(.02×midfarFREE)`。不加逐 capture FREE gate，不用 gross FREEadded 替代净增，不把旧数据并入主分母。W/POS 为正见证，POS支持另列；FREE 与 UNKNOWN 为支持格数，不是体积 clearance 或安全证明。

## 完整结果

## Fresh6主判定整数条件

| 条件 | 测量 | 要求 | 通过 |
| --- | --- | --- | --- |
| near改善capture | 6 | ≥4/6 | True |
| near aggregate净FREE增量 | 12 | 5 = floor(.01×515) | False |
| midfar见证净损 | -311 | 13 = floor(.03×464)；exact100×loss≤3×DAVW | True |
| midfar aggregate净FREE增量 | 10 | 2 = floor(.02×107) | False |
| 新6capture完整 | True | True | True |
| 主替换信号 | False | near与midfar全部通过 | False |

改善near capture：confirm_arkit_41125718, confirm_arkit_41125756, confirm_arkit_41142278, confirm_arkit_41159503, confirm_arkit_41159519, confirm_arkit_41159529

## Fresh6逐capture近带

| capture | DAV W/POS | Uni W/POS | DAV FREE/FREE | Uni FREE/FREE | Uni ΔW | Uni净ΔFREE | Uni救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confirm_arkit_41125718 | 0/26 | 6/26 | 0/111 | 2/111 | 6 | 2 | 6/0 | 2/0 |
| confirm_arkit_41125756 | 0/43 | 18/43 | 0/92 | 3/92 | 18 | 3 | 18/0 | 3/0 |
| confirm_arkit_41142278 | 0/92 | 36/92 | 0/34 | 1/34 | 36 | 1 | 36/0 | 1/0 |
| confirm_arkit_41159503 | 3/65 | 23/65 | 0/60 | 3/60 | 20 | 3 | 20/0 | 3/0 |
| confirm_arkit_41159519 | 0/46 | 14/46 | 0/95 | 3/95 | 14 | 3 | 14/0 | 3/0 |
| confirm_arkit_41159529 | 0/20 | 6/20 | 0/123 | 0/123 | 6 | 0 | 6/0 | 0/0 |

## Fresh6距离带与中远合计

| 范围 | 模型 | W/POS | POS支持/POS | FREE/FREE | UNKNOWN/UNKNOWN | Uni救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3-0.8m | dav_raw | 3/292 | 4/292 | 0/515 | 0/57 | - | - |
| 0.3-0.8m | uni_raw | 103/292 | 110/292 | 12/515 | 12/57 | 100/0 | 12/0 |
| 0.8-1.5m | dav_raw | 244/642 | 303/642 | 7/101 | 47/121 | - | - |
| 0.8-1.5m | uni_raw | 492/642 | 508/642 | 17/101 | 28/121 | 267/19 | 13/3 |
| 1.5-3m | dav_raw | 220/405 | 253/405 | 0/6 | 105/453 | - | - |
| 1.5-3m | uni_raw | 283/405 | 303/405 | 0/6 | 42/453 | 90/27 | 0/0 |
| midfar | dav_raw | 464/1047 | 556/1047 | 7/107 | 152/574 | - | - |
| midfar | uni_raw | 775/1047 | 811/1047 | 17/107 | 70/574 | 357/46 | 13/3 |

## 主原pooled304：fresh6每capture×距离带

| capture | fold | 模型 | 带 | W/POS | POS支持/POS | FREE/FREE | UNKNOWN/UNKNOWN | Uni救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confirm_arkit_41125718 | - | dav_raw | 0.3-0.8m | 0/26 | 0/26 | 0/111 | 0/7 | - | - |
| confirm_arkit_41125718 | - | dav_raw | 0.8-1.5m | 42/108 | 58/108 | 2/22 | 2/14 | - | - |
| confirm_arkit_41125718 | - | dav_raw | 1.5-3m | 38/82 | 42/82 | 0/0 | 1/62 | - | - |
| confirm_arkit_41125718 | - | uni_raw | 0.3-0.8m | 6/26 | 7/26 | 2/111 | 0/7 | 6/0 | 2/0 |
| confirm_arkit_41125718 | - | uni_raw | 0.8-1.5m | 91/108 | 91/108 | 4/22 | 1/14 | 53/4 | 3/1 |
| confirm_arkit_41125718 | - | uni_raw | 1.5-3m | 60/82 | 63/82 | 0/0 | 3/62 | 25/3 | 0/0 |
| confirm_arkit_41125756 | - | dav_raw | 0.3-0.8m | 0/43 | 0/43 | 0/92 | 0/9 | - | - |
| confirm_arkit_41125756 | - | dav_raw | 0.8-1.5m | 33/113 | 39/113 | 0/15 | 3/16 | - | - |
| confirm_arkit_41125756 | - | dav_raw | 1.5-3m | 46/78 | 56/78 | 0/2 | 22/64 | - | - |
| confirm_arkit_41125756 | - | uni_raw | 0.3-0.8m | 18/43 | 18/43 | 3/92 | 0/9 | 18/0 | 3/0 |
| confirm_arkit_41125756 | - | uni_raw | 0.8-1.5m | 88/113 | 91/113 | 0/15 | 1/16 | 61/6 | 0/0 |
| confirm_arkit_41125756 | - | uni_raw | 1.5-3m | 41/78 | 41/78 | 0/2 | 7/64 | 7/12 | 0/0 |
| confirm_arkit_41142278 | - | dav_raw | 0.3-0.8m | 0/92 | 1/92 | 0/34 | 0/18 | - | - |
| confirm_arkit_41142278 | - | dav_raw | 0.8-1.5m | 37/96 | 57/96 | 0/0 | 26/48 | - | - |
| confirm_arkit_41142278 | - | dav_raw | 1.5-3m | 7/11 | 8/11 | 0/0 | 29/133 | - | - |
| confirm_arkit_41142278 | - | uni_raw | 0.3-0.8m | 36/92 | 39/92 | 1/34 | 4/18 | 36/0 | 1/0 |
| confirm_arkit_41142278 | - | uni_raw | 0.8-1.5m | 89/96 | 93/96 | 0/0 | 9/48 | 54/2 | 0/0 |
| confirm_arkit_41142278 | - | uni_raw | 1.5-3m | 6/11 | 6/11 | 0/0 | 4/133 | 0/1 | 0/0 |
| confirm_arkit_41159503 | - | dav_raw | 0.3-0.8m | 3/65 | 3/65 | 0/60 | 0/19 | - | - |
| confirm_arkit_41159503 | - | dav_raw | 0.8-1.5m | 72/114 | 77/114 | 3/3 | 8/27 | - | - |
| confirm_arkit_41159503 | - | dav_raw | 1.5-3m | 11/37 | 13/37 | 0/0 | 8/107 | - | - |
| confirm_arkit_41159503 | - | uni_raw | 0.3-0.8m | 23/65 | 23/65 | 3/60 | 7/19 | 20/0 | 3/0 |
| confirm_arkit_41159503 | - | uni_raw | 0.8-1.5m | 92/114 | 96/114 | 1/3 | 9/27 | 25/5 | 0/2 |
| confirm_arkit_41159503 | - | uni_raw | 1.5-3m | 25/37 | 26/37 | 0/0 | 4/107 | 14/0 | 0/0 |
| confirm_arkit_41159519 | - | dav_raw | 0.3-0.8m | 0/46 | 0/46 | 0/95 | 0/3 | - | - |
| confirm_arkit_41159519 | - | dav_raw | 0.8-1.5m | 38/105 | 47/105 | 0/26 | 6/13 | - | - |
| confirm_arkit_41159519 | - | dav_raw | 1.5-3m | 55/90 | 65/90 | 0/3 | 21/51 | - | - |
| confirm_arkit_41159519 | - | uni_raw | 0.3-0.8m | 14/46 | 17/46 | 3/95 | 1/3 | 14/0 | 3/0 |
| confirm_arkit_41159519 | - | uni_raw | 0.8-1.5m | 78/105 | 81/105 | 0/26 | 7/13 | 41/1 | 0/0 |
| confirm_arkit_41159519 | - | uni_raw | 1.5-3m | 69/90 | 76/90 | 0/3 | 7/51 | 17/3 | 0/0 |
| confirm_arkit_41159529 | - | dav_raw | 0.3-0.8m | 0/20 | 0/20 | 0/123 | 0/1 | - | - |
| confirm_arkit_41159529 | - | dav_raw | 0.8-1.5m | 22/106 | 25/106 | 2/35 | 2/3 | - | - |
| confirm_arkit_41159529 | - | dav_raw | 1.5-3m | 63/107 | 69/107 | 0/1 | 24/36 | - | - |
| confirm_arkit_41159529 | - | uni_raw | 0.3-0.8m | 6/20 | 6/20 | 0/123 | 0/1 | 6/0 | 0/0 |
| confirm_arkit_41159529 | - | uni_raw | 0.8-1.5m | 54/106 | 56/106 | 12/35 | 1/3 | 33/1 | 10/0 |
| confirm_arkit_41159529 | - | uni_raw | 1.5-3m | 82/107 | 91/107 | 0/1 | 17/36 | 27/8 | 0/0 |

## 辅助三折：fresh6按acceptedindex%3每query一次

| capture | fold | 模型 | 带 | W/POS | POS支持/POS | FREE/FREE | UNKNOWN/UNKNOWN | Uni救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confirm_arkit_41125718 | 0 | dav_raw | 0.3-0.8m | 4/26 | 5/26 | 10/111 | 3/7 | - | - |
| confirm_arkit_41125718 | 0 | dav_raw | 0.8-1.5m | 77/108 | 85/108 | 4/22 | 8/14 | - | - |
| confirm_arkit_41125718 | 0 | dav_raw | 1.5-3m | 38/82 | 50/82 | 0/0 | 7/62 | - | - |
| confirm_arkit_41125718 | 0 | uni_raw | 0.3-0.8m | 11/26 | 11/26 | 4/111 | 3/7 | 8/1 | 4/10 |
| confirm_arkit_41125718 | 0 | uni_raw | 0.8-1.5m | 95/108 | 98/108 | 6/22 | 1/14 | 23/5 | 4/2 |
| confirm_arkit_41125718 | 0 | uni_raw | 1.5-3m | 71/82 | 75/82 | 0/0 | 12/62 | 34/1 | 0/0 |
| confirm_arkit_41125756 | 1 | dav_raw | 0.3-0.8m | 18/43 | 20/43 | 2/92 | 8/9 | - | - |
| confirm_arkit_41125756 | 1 | dav_raw | 0.8-1.5m | 71/113 | 79/113 | 0/15 | 6/16 | - | - |
| confirm_arkit_41125756 | 1 | dav_raw | 1.5-3m | 54/78 | 62/78 | 0/2 | 37/64 | - | - |
| confirm_arkit_41125756 | 1 | uni_raw | 0.3-0.8m | 30/43 | 30/43 | 7/92 | 3/9 | 12/0 | 6/1 |
| confirm_arkit_41125756 | 1 | uni_raw | 0.8-1.5m | 91/113 | 95/113 | 0/15 | 3/16 | 31/11 | 0/0 |
| confirm_arkit_41125756 | 1 | uni_raw | 1.5-3m | 46/78 | 47/78 | 0/2 | 10/64 | 4/12 | 0/0 |
| confirm_arkit_41142278 | 2 | dav_raw | 0.3-0.8m | 28/92 | 38/92 | 1/34 | 13/18 | - | - |
| confirm_arkit_41142278 | 2 | dav_raw | 0.8-1.5m | 85/96 | 92/96 | 0/0 | 45/48 | - | - |
| confirm_arkit_41142278 | 2 | dav_raw | 1.5-3m | 8/11 | 10/11 | 0/0 | 62/133 | - | - |
| confirm_arkit_41142278 | 2 | uni_raw | 0.3-0.8m | 83/92 | 85/92 | 5/34 | 14/18 | 59/4 | 4/0 |
| confirm_arkit_41142278 | 2 | uni_raw | 0.8-1.5m | 94/96 | 96/96 | 0/0 | 32/48 | 9/0 | 0/0 |
| confirm_arkit_41142278 | 2 | uni_raw | 1.5-3m | 9/11 | 9/11 | 0/0 | 19/133 | 1/0 | 0/0 |
| confirm_arkit_41159503 | 0 | dav_raw | 0.3-0.8m | 17/65 | 22/65 | 7/60 | 13/19 | - | - |
| confirm_arkit_41159503 | 0 | dav_raw | 0.8-1.5m | 97/114 | 106/114 | 3/3 | 18/27 | - | - |
| confirm_arkit_41159503 | 0 | dav_raw | 1.5-3m | 18/37 | 20/37 | 0/0 | 18/107 | - | - |
| confirm_arkit_41159503 | 0 | uni_raw | 0.3-0.8m | 41/65 | 45/65 | 9/60 | 17/19 | 24/0 | 4/2 |
| confirm_arkit_41159503 | 0 | uni_raw | 0.8-1.5m | 105/114 | 110/114 | 1/3 | 17/27 | 10/2 | 0/2 |
| confirm_arkit_41159503 | 0 | uni_raw | 1.5-3m | 30/37 | 31/37 | 0/0 | 10/107 | 12/0 | 0/0 |
| confirm_arkit_41159519 | 1 | dav_raw | 0.3-0.8m | 13/46 | 19/46 | 3/95 | 2/3 | - | - |
| confirm_arkit_41159519 | 1 | dav_raw | 0.8-1.5m | 71/105 | 75/105 | 0/26 | 11/13 | - | - |
| confirm_arkit_41159519 | 1 | dav_raw | 1.5-3m | 65/90 | 76/90 | 0/3 | 38/51 | - | - |
| confirm_arkit_41159519 | 1 | uni_raw | 0.3-0.8m | 24/46 | 30/46 | 5/95 | 1/3 | 13/2 | 2/0 |
| confirm_arkit_41159519 | 1 | uni_raw | 0.8-1.5m | 83/105 | 85/105 | 0/26 | 8/13 | 16/4 | 0/0 |
| confirm_arkit_41159519 | 1 | uni_raw | 1.5-3m | 71/90 | 77/90 | 0/3 | 10/51 | 15/9 | 0/0 |
| confirm_arkit_41159529 | 2 | dav_raw | 0.3-0.8m | 4/20 | 4/20 | 4/123 | 0/1 | - | - |
| confirm_arkit_41159529 | 2 | dav_raw | 0.8-1.5m | 43/106 | 50/106 | 12/35 | 2/3 | - | - |
| confirm_arkit_41159529 | 2 | dav_raw | 1.5-3m | 78/107 | 90/107 | 0/1 | 31/36 | - | - |
| confirm_arkit_41159529 | 2 | uni_raw | 0.3-0.8m | 17/20 | 17/20 | 8/123 | 0/1 | 13/0 | 5/1 |
| confirm_arkit_41159529 | 2 | uni_raw | 0.8-1.5m | 71/106 | 75/106 | 20/35 | 3/3 | 28/0 | 9/1 |
| confirm_arkit_41159529 | 2 | uni_raw | 1.5-3m | 92/107 | 104/107 | 0/1 | 21/36 | 23/9 | 0/0 |

## 辅助三折：held旧capture诊断（不进入主判定）

| capture | fold | 模型 | 带 | W/POS | POS支持/POS | FREE/FREE | UNKNOWN/UNKNOWN | Uni救/损 | FREE增/减 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arkit16 | 0 | dav_raw | 0.3-0.8m | 0/44 | 2/44 | 0/96 | 0/4 | - | - |
| arkit16 | 0 | dav_raw | 0.8-1.5m | 66/114 | 69/114 | 0/9 | 21/21 | - | - |
| arkit16 | 0 | dav_raw | 1.5-3m | 55/70 | 70/70 | 0/0 | 31/74 | - | - |
| arkit16 | 0 | uni_raw | 0.3-0.8m | 30/44 | 32/44 | 8/96 | 2/4 | 30/0 | 8/0 |
| arkit16 | 0 | uni_raw | 0.8-1.5m | 84/114 | 84/114 | 1/9 | 0/21 | 27/9 | 1/0 |
| arkit16 | 0 | uni_raw | 1.5-3m | 63/70 | 69/70 | 0/0 | 24/74 | 9/1 | 0/0 |
| arkit_40777060 | 1 | dav_raw | 0.3-0.8m | 6/11 | 6/11 | 1/133 | 0/0 | - | - |
| arkit_40777060 | 1 | dav_raw | 0.8-1.5m | 45/69 | 47/69 | 5/75 | 0/0 | - | - |
| arkit_40777060 | 1 | dav_raw | 1.5-3m | 107/122 | 108/122 | 1/12 | 2/10 | - | - |
| arkit_40777060 | 1 | uni_raw | 0.3-0.8m | 1/11 | 1/11 | 0/133 | 0/0 | 0/5 | 0/1 |
| arkit_40777060 | 1 | uni_raw | 0.8-1.5m | 35/69 | 35/69 | 3/75 | 0/0 | 3/13 | 1/3 |
| arkit_40777060 | 1 | uni_raw | 1.5-3m | 103/122 | 108/122 | 0/12 | 5/10 | 4/8 | 0/1 |
| arkit_40777065 | 2 | dav_raw | 0.3-0.8m | 0/0 | 0/0 | 0/144 | 0/0 | - | - |
| arkit_40777065 | 2 | dav_raw | 0.8-1.5m | 31/48 | 33/48 | 12/93 | 0/3 | - | - |
| arkit_40777065 | 2 | dav_raw | 1.5-3m | 128/131 | 130/131 | 8/10 | 3/3 | - | - |
| arkit_40777065 | 2 | uni_raw | 0.3-0.8m | 0/0 | 0/0 | 0/144 | 0/0 | 0/0 | 0/0 |
| arkit_40777065 | 2 | uni_raw | 0.8-1.5m | 38/48 | 39/48 | 26/93 | 2/3 | 10/3 | 17/3 |
| arkit_40777065 | 2 | uni_raw | 1.5-3m | 127/131 | 127/131 | 3/10 | 2/3 | 2/3 | 0/5 |
| new_arkit_41069021 | 0 | dav_raw | 0.3-0.8m | 0/44 | 0/44 | 0/89 | 2/11 | - | - |
| new_arkit_41069021 | 0 | dav_raw | 0.8-1.5m | 38/79 | 42/79 | 0/30 | 32/35 | - | - |
| new_arkit_41069021 | 0 | dav_raw | 1.5-3m | 59/83 | 65/83 | 0/3 | 18/58 | - | - |
| new_arkit_41069021 | 0 | uni_raw | 0.3-0.8m | 18/44 | 20/44 | 0/89 | 6/11 | 18/0 | 0/0 |
| new_arkit_41069021 | 0 | uni_raw | 0.8-1.5m | 61/79 | 63/79 | 0/30 | 24/35 | 23/0 | 0/0 |
| new_arkit_41069021 | 0 | uni_raw | 1.5-3m | 72/83 | 77/83 | 0/3 | 11/58 | 13/0 | 0/0 |
| new_arkit_41069042 | 1 | dav_raw | 0.3-0.8m | 44/75 | 45/75 | 14/62 | 2/7 | - | - |
| new_arkit_41069042 | 1 | dav_raw | 0.8-1.5m | 96/117 | 108/117 | 0/4 | 20/23 | - | - |
| new_arkit_41069042 | 1 | dav_raw | 1.5-3m | 26/39 | 34/39 | 0/0 | 27/105 | - | - |
| new_arkit_41069042 | 1 | uni_raw | 0.3-0.8m | 41/75 | 43/75 | 0/62 | 4/7 | 16/19 | 0/14 |
| new_arkit_41069042 | 1 | uni_raw | 0.8-1.5m | 103/117 | 113/117 | 0/4 | 11/23 | 14/7 | 0/0 |
| new_arkit_41069042 | 1 | uni_raw | 1.5-3m | 25/39 | 34/39 | 0/0 | 14/105 | 6/7 | 0/0 |
| new_arkit_41069048 | 2 | dav_raw | 0.3-0.8m | 11/51 | 14/51 | 10/80 | 9/13 | - | - |
| new_arkit_41069048 | 2 | dav_raw | 0.8-1.5m | 79/128 | 88/128 | 1/1 | 15/15 | - | - |
| new_arkit_41069048 | 2 | dav_raw | 1.5-3m | 17/33 | 26/33 | 0/0 | 91/111 | - | - |
| new_arkit_41069048 | 2 | uni_raw | 0.3-0.8m | 37/51 | 40/51 | 4/80 | 10/13 | 27/1 | 1/7 |
| new_arkit_41069048 | 2 | uni_raw | 0.8-1.5m | 120/128 | 123/128 | 1/1 | 9/15 | 43/2 | 0/0 |
| new_arkit_41069048 | 2 | uni_raw | 1.5-3m | 20/33 | 21/33 | 0/0 | 55/111 | 7/4 | 0/0 |

## 辅助fresh三折合计（不能替代主工作点）

| 带 | 模型 | W/POS | FREE/FREE | UNKNOWN/UNKNOWN |
| --- | --- | --- | --- | --- |
| 0.3-0.8m | dav_raw | 84/292 | 27/515 | 39/57 |
| 0.3-0.8m | uni_raw | 206/292 | 38/515 | 38/57 |
| 0.8-1.5m | dav_raw | 444/642 | 19/101 | 90/121 |
| 0.8-1.5m | uni_raw | 539/642 | 27/101 | 64/121 |
| 1.5-3m | dav_raw | 261/405 | 0/6 | 193/453 |
| 1.5-3m | uni_raw | 319/405 | 0/6 | 82/453 |

## 复制主切点与辅助三折

| 主模型 | cutoff m | 原cal FREE支持/分母 |
| --- | --- | --- |
| dav_raw | 0.24403834342956543 | 29/588 |
| uni_raw | 0.09616100788116455 | 29/588 |

| fold | held old | cal old | DAV cutoff m | Uni cutoff m | DAV calFREE | Uni calFREE |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | arkit16,new_arkit_41069021 | arkit_40777060,arkit_40777065,new_arkit_41069042,new_arkit_41069048 | 0.06464843750000004 | -0.0007804632186889648 | 30/614 | 30/614 |
| 1 | arkit_40777060,new_arkit_41069042 | arkit16,arkit_40777065,new_arkit_41069021,new_arkit_41069048 | -0.017459935690204542 | 0.028909921646118164 | 27/555 | 27/555 |
| 2 | arkit_40777065,new_arkit_41069048 | arkit16,arkit_40777060,new_arkit_41069021,new_arkit_41069042 | -0.054333400726318315 | -0.07706530094146724 | 24/513 | 25/513 |

原primary thresholds.json完整字节复制SHA：`c65b3d5c76c76a779a03ed919a2094e6ef0e8dc04db7373b8167cc78f2cd6ba8`。所有辅助cut在fresh评分之前封存。

## 官方顺序选择流（没有模型筛选）

| capture | visit | metadata index | nearPOS | 状态 |
| --- | --- | --- | --- | --- |
| 41125696 | 382321 | 222 | 9 | SKIP_NEAR_REFERENCE_COVERAGE |
| 41125718 | 382320 | 225 | 26 | ACCEPTED |
| 41125756 | 382319 | 228 | 43 | ACCEPTED |
| 41142278 | 384651 | 389 | 92 | ACCEPTED |
| 41159503 | 381893 | 417 | 65 | ACCEPTED |
| 41159519 | 381879 | 420 | 46 | ACCEPTED |
| 41159529 | 381875 | 424 | 20 | ACCEPTED |


## 资源、许可证与可复算产物

本次只下载新数据与官方小型元数据，累计传输 **88,595,794 B / 1,000,000,000 B**，包括选择失败与重试；没有新模型权重下载。两臂 GPU 所属推理进程 wall 合计 **98.6519364 s**，预算收据保守计 **103/600 s**。CPU 计 command-wall 而非 CPU 内核时间：根代理保守计 200/250 s（含 GPU 命令等待及整合交付）、acquisition 82/350 s、评价与报告 35/200 s、独立审计 34/100 s，合计 **351/900 s**；这是保守计账，实测与未单独计时的启动开销逐项保存在 budget/各 resource receipts。校准实测 3.8017766 s、final 17.2496668 s。模型单帧部署统计沿用本次各 inference terminal 的逐帧延迟；本次资源边界不作为效果门槛。

模型沿用上一 backbone run 已固定 revision / SHA 权重与许可证收据：DAV2 Large metric weights 为 CC-BY-NC-4.0（仓库代码 Apache-2.0），UniDepth V2 为 CC-BY-NC-4.0。ARKitScenes 使用官方 Apple research license，本次已重新获取官方 LICENSE 且与旧字节一致。仅本地非商业研究，不再分发权重。URL、revision、权重 SHA、许可证全文及收据关联见 `license_receipts.json` 指向的上一 run provenance，原缓存均保留。

所有产物位于 canonical junction `artifacts.local/work/rgb-backbone-confirm-dev-20261010/`，物理根为 `F:\ba-data\blindassist-artifacts-20260805\work\rgb-backbone-confirm-dev-20261010`：

- `PLAN.json`、`plan_seal.json`、`calibration/primary_original_thresholds.json`、`calibration/thresholds.json` 与 cal inputs/source snapshot，证明冻结顺序与旧 FREE-only 校准。
- `official_receipts.json`、`consumed_inventory.json`、`acquisition_summary.json`、`new_public_roster.json`、`new-eval-sensor/<cohort>/dataset_manifest.json`，保留 official source/Range/CRC/SHA、选取、跳过及 native reference。
- `dav2/predictions.json`、`unidepth/predictions.json`、`new_prediction_manifests.json` 及 native npz，保留新 96 帧预测、公开 K 和输入 SHA；原预测与权重只读沿用。
- `final/primary/`、`final/supplemental-fresh/`、`final/supplemental-held-old/` 的 `results.json`、`summary.csv`、`query_pairs.csv`，加 `final/<cohort>_scores.json` 和 `final/summary.json`，完整保留所有逐 query 判断与配对。Final summary SHA256 `17f25c842ac2fa4cbf4d65a7e9e57f18b3ef03a875064f43e696f6b3661a3e9b`。
- `report-summary/` 保留首次汇总；`report-summary-v2/` 是最终文字汇总，仅把 accepted selection 状态的显示从缺省 None 补为 ACCEPTED，无评分或切点变化。一次尝试写已存在 summary 目录被 helper 的保留保护拒绝，随后改用新目录，旧产物没有覆盖。
- `license_receipts.json`、各 inference terminal、`gpu_cleanup.json` 与最终 `budget_receipt.json`，记录复用许可、实测/保守计账及释放资源。

评价实现为 [rgb_backbone_confirm_evaluate.py](rgb_backbone_confirm_evaluate.py)。Cal/final 运行 source SHA256 相同：`60ee6532d0bdc23626371c98c88595d49db4a6f6f905ce722fb77b1b1659b87d`。最终只做相关 AST/产物结构核查与独立复算，不重复神经推理。

## 独立复算核验

独立审计不导入生产评分函数，GPU、下载、训练均为 0。第一次已核对原切点、旧 FREE 三折、15552 query decisions、7776 paired queries/CSV rows 及主判定字段，随后 source 收据 schema 的 `local_path` 读取产生 KeyError；这次失败收据保留。修正收据路径字段后，只续审受影响的 source/native 核验，不重复已通过数学部分。最终 **PASS**（independent_audit_attempt2.json，SHA256 `568a6fa01ab9da868464bd6a63403cc26a1c45d19536efbba3fc1d61de926478`）：5 份官方文件、378 次网络请求、320 个 ZIP 成员 CRC/SHA、6 条有序新 visit / 96 帧、2592 个参考标签面、首条被跳过 capture 的 144 个 near 标签面及 10368 个 native 评分分量全部吻合。审计已记录 command-wall 组件 30.4533 s，保守计 34 s；异步启动未单独计时，收据明确记录该限制。GPU/下载/训练均为零，所有任务 GPU PID 与 CPU worker session 均已释放；主评分和切点未因审计改变。
