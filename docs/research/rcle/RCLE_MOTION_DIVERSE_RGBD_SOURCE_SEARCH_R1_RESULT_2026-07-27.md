# RCLE motion-diverse RGB-D source search R1 result

状态：`DEVELOPMENT_SIGNAL_DIRECTION_SUPPORTED / VALID`

## 结论

本轮已经找到可用的 motion-diverse RGB-D **开发 cohort**，无需继续下载或筛选
Floor3/ETH3D 候选：

| 角色 | 冻结窗 | 来源性质 | geometry |
|---|---|---|---:|
| positive | `desk_changing_1@4065.364250422` | ETH3D real development | coverage 0.8111；positive 0.8037；最长 6.5616 s |
| positive | `japanesealley/Hard/P002@000260` | TartanAir synthetic development anchor | coverage 1.0；positive 0.8283；最长 7.7 s |
| below-reference | `TUM_RGBD_FR2_RPY@2` | burned real development anchor | coverage 1.0；below 0.9967；最长 9.5046 s |
| below-reference | `TUM_RGBD_FR2_RPY@7` | burned real development anchor | coverage 1.0；below 0.9130；最长 6.0327 s |

四窗身份在读取新 RGB 前冻结；cohort receipt SHA-256 为
`f623f5b2609b0541a28839d3f00f7f2aa8281a312215cd21fbf527e10b9c5b56`。

## 高效漏斗实际结果

1. 先复用已冻结的一个 ETH3D real positive 和两个 TUM real below anchors，只搜索
   缺失的第二个 positive。
2. 利用本地已有的 TartanAir JapaneseAlley 包做 pose-only 排序，共 214 个候选，
   固定非重叠队列后逐窗 geometry。
3. 第 1 窗因最长 positive 连续段仅 4.2 s 淘汰；第 2 窗因 fixed-denominator
   positive 仅 0.7778 淘汰；第 3 窗通过即停。
4. 只取最终四窗 RGB：ETH3D 271 张，range transport 实际传输 98,748,962 bytes；
   TartanAir 从本地包提取 100 张；TUM 复用本地两窗各 300 张。

没有下载完整 1.4 GB ETH3D mono ZIP，没有视觉查看候选，没有后补窗或按结果重排，
没有改算法或阈值。

## RGB development canary

冻结的既有 RCLE RGB pair algorithm 对 967 个 pair 执行，四窗 coverage 均达到
0.8：

| 角色聚合 | compensated expansion median | trigger coverage fixed denominator |
|---|---:|---:|
| positive | 0.2796300 /s | 0.7427609 |
| below-reference | -0.0009581 /s | 0.3478261 |

两个冻结方向比较都满足 `positive > below-reference`，因此 terminal 为
`DEVELOPMENT_SIGNAL_DIRECTION_SUPPORTED / VALID`。正式 result SHA-256：
`74222a5e47d30f02b6240e744ccb1d3874c73be0992dd8a187710b0cfa78de42`。

前两个 RGB 启动分别暴露了 ETH3D per-pair pose abstention 和 TUM duplicate-pose
timestamp 的来源适配错误；两次均无 result/window/aggregate metric 写出。修订只改
adapter failure handling，保留原窗、0.8 coverage、0.01 trigger、固定分母、
8-worker、算法与 aggregation。

## 独立验证

独立 validator 不导入 producer，重新检查：

- 671 个 RGB 文件的 bytes/SHA-256；
- 967 行 pair ledger 的窗口、顺序、角色、trigger rule 与 hash；
- 每窗 coverage、abstention、median、trigger coverage 和最长连续 trigger；
- 两类 role aggregate、terminal、implementation lock 和 authority。

验证为 `PASS`，receipt SHA-256：
`9203091caa820060a5b4c2897a5fbb8f530412ada1a59f8b818e4a47c09db18f`。

## 权限边界

这证明“当前四窗开发 cohort 可用于后续算法开发比较”，但不证明：

- 全 real 的 cross-source holdout；
- 独立 confirmation；
- 性能、Android、真人效果、产品或安全资格。

下一步若要扩大验证，应另立真正的 all-real cross-source holdout，不能把本轮
TartanAir synthetic positive 或 burned TUM anchors 包装为 unseen confirmation。
