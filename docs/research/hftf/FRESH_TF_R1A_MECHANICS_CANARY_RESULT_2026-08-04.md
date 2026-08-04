# FRESH-TF R1-A C1 mechanics canary 结果

日期：2026-08-04
终态：`FRESH_TF_R1A_C1_FROZEN_MECHANICS_NOT_SUPPORTED_CANARY_ONLY`
范围终态：`LOCAL_GEOMETRIC_VALIDITY_EFFECT_NOT_EVALUATED`

## 结论

冻结 C1 的状态机和投影机制成功执行，也确实产生了大量局部遮挡、新暴露和出界机会；
但它只保留了 macro `28.91%`、worst-session `19.12%` 的 cell support。这个覆盖水平不足以
支持继续把该实现作为实用 local-validity 候选。

因此关闭的是这一具体组合：source-native pose reprojection、full-resolution Farneback、
12×8 cell、32 点、60% support、1.5 px forward/backward、3 px geometry-flow residual
和当前硬状态优先级。不能在已经打开的三条序列上调整这些参数救援。

这不等于 local geometric validity 总问题失败。正式 effect gate 没有运行：当前没有冻结的
direction/traversability truth，而且每种机制只有一个 session，低于协议要求的两个。

## 固定结果

| sequence | frames | supported cells | cell coverage | 主要拒绝来源 |
| --- | ---: | ---: | ---: | --- |
| `freiburg1_rpy` | 241 | 4,423 | 19.12% | low-flow 7,091；out-of-frame 4,249 |
| `freiburg1_desk` | 199 | 4,511 | 23.61% | low-flow 5,098；newly-exposed 2,905 |
| `freiburg3_sitting_static` | 236 | 9,971 | 44.01% | occluded 7,236；newly-exposed 3,431 |

总计 676 帧、64,896 cells。状态机识别到 12,890 个 `OCCLUDED`、8,956 个
`NEWLY_EXPOSED` 和 6,329 个 `OUT_OF_FRAME` cell opportunity，并按构造使这些状态
不能继承旧 support。这里的“误继承为 0”只是 fail-closed 状态机不变量，不是遮挡检测
准确率、false-clear 或安全效果证据。

当前结果也不能把失败单独归因给光流、pose/depth registration、cell aggregation 或遮挡
门；这些机制在本 canary 中耦合。若提出继任版本，必须先形成机制驱动、非阈值搜索的
修改理由，再在新的 parent/session-disjoint 数据上冻结执行，同时补齐方向/通行真值和
每机制至少两个 session。R1-B 与 NPU 主动调度仍不得启动。

## 证据与验证

- protocol SHA-256：`2379D50E497ED417C6EF8BF6D9CFDD793AF64709B22AD494061E861687D345F9`
- implementation SHA-256：`69492C75913E3065D879BB9B7BDDB810B73ACF1FDB002B1A0B687010D082C913`
- ignored result SHA-256：`3702EBB7712FB0DDA1A6A9DA79EB973A96DED36977FE57918535FA70A8035971`
- ignored trace SHA-256：`04AB6C2A0DAF504ADDFC68C2A1515A4670473C304B70FCA0A158B7B87D7961C2`
- focused tests：9/9 通过；独立汇总、cell 数量、coverage 和遮挡优先级 invariants 通过。

本结果没有 App、手机 pose、NPU、方向提醒、生产、导航或安全权限。
