# RCLE Phase B PB-H1 role proxy R0 结果

状态：`SUPPORT / VALID`

日期：2026-07-26

终态：`PB_H1_SUPPORTS_TUM_FR2_RPY_METADATA_GEOMETRY_AUDIT`

## 结论

旧的 raw translation-speed rotation gate **因果错位**。在受控 fixture 中，小幅
横移与同速前向接近的 raw speed 都是 `0.300 m/s`，旧代理完全相同；但投影几何
明确不同。公式通过纯旋转零响应和前向平面解析值校准，值得进入下一次单来源
TUM `fr2/rpy` metadata/pose/depth geometry audit。

这个支持结论有一个重要限定：**median absolute radial expansion 单独不是
approach 判据**。本 fixture 中横移的 absolute median 甚至高于前向接近。可区分
二者的是 signed radial coherence（signed median + positive fraction）与
time-normalized parallax 的联合结构。不得把本轮结果简化成另一个 absolute
threshold。

## 冻结公式

对前一相机帧中的深度点 `X`，使用 source-native 相对位姿：

```text
X_r = R X
X_f = R X + t
radial expansion = log(rho(project(X_f)) / rho(project(X_r))) / dt  [s^-1]
parallax = angle(unit(X_r), unit(X_f)) / dt                          [rad/s]
raw speed = ||t|| / dt                                               [m/s]
```

`rho` 以标定主点为中心。半径 `<8 px`、非正深度、任一投影出界的点退出；full
projection 的同一取整目标像素只保留最近深度。Bonn 使用 `8 px` raster；pair
内汇总 signed/absolute radial median、positive fraction 与 parallax Q90，
window 再对 pair summary 取 median。

## 受控 fixture

固定 `3.0 m` 正视平面、`dt=0.1 s`、`910` 点。纯旋转为 `8 deg/s` yaw；横移和
前向接近均为 `0.300 m/s`。

| 运动 | raw speed m/s | signed radial median s^-1 | absolute radial median s^-1 | positive fraction | parallax Q90 rad/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 纯旋转 | 0.000 | 0.000000 | 0.000000 | 0.000 | 0.000000 |
| 横向平移 | 0.300 | -0.008131 | 0.198717 | 0.486 | 0.098267 |
| 前向接近 | 0.300 | 0.100503 | 0.100503 | 1.000 | 0.041063 |

前向解析值为
`log(3.0 / (3.0 - 0.3*0.1)) / 0.1 = 0.1005033585 s^-1`，实现误差在
`1e-12` 内。六项检查全部通过：

- 纯旋转 translation geometry 为零；
- 横移与接近的 raw speed 完全相同；
- 前向解析 radial 标定通过且全点正扩张；
- 横移 signed radial 近零、正负约各半；
- 横移 parallax Q90 高于前向接近。

因此公式的坐标、符号、`dt` 和单位在本 fixture 上正确；旧速度门无法表达运动
方向与投影后果。

## 烧掉的 Bonn diagnostic window

窗口不是按新指标选择，而是固定 B1A denominator 的第一个窗口：
`rgbd_bonn_crowd2:0`，`1548339892.26121–1548339902.26121`。输入为旧 B1A
ledger SHA `5d734979…270` 与本地 archive SHA `e751ca1b…840`；没有下载新数据。

`294/294` candidate pair 可评价，pair coverage `1.000`，median valid fraction
`0.99924`：

| raw speed median m/s | signed radial median s^-1 | absolute radial median s^-1 | positive fraction | parallax Q90 rad/s |
| ---: | ---: | ---: | ---: | ---: |
| 0.082983 | 0.0000167 | 0.047438 | 0.5001 | 0.061515 |

旧 `0.02 m/s` 门会拒绝该窗；直接几何却显示接近零的 signed median、约
`50/50` 正负扩张和显著 parallax，更像横向/混合 translation contamination，
而不是 coherent forward approach。这里支持的是“旧 gate 会把非接近平移与接近
混为一谈”，不是给 Bonn 补发 rotation role，也不回救 INVALID B1A。

## 证据强度与下一边界

- 强：受控 fixture 的解析物理校准与相同 speed 反事实。
- 中低：一个 burned Bonn window 的 source characterization；ledger 属于旧
  INVALID execution，且当前公式不估计独立物体运动。
- 未建立：真实纯旋转来源上的分布、跨序列稳定性、RCLE algorithm 效果、
  confirmation、人体/安全/产品能力。

下一步值得审计 TUM `fr2/rpy`，但严格只做一个 source 的 metadata/pose/depth
geometry audit：先核 native pose/depth/time authority，再复用本文公式报告连续
分布；不预设 `0.02 m/s` 新硬门，不下载其他候选，不读取 RCLE RGB algorithm
outcome。若 `fr2/rpy` 不能提供可评价窗口或纯旋转仍出现无法解释的 direct
translation geometry，则局部停止该来源。

机器证据：
`artifacts.local/evidence/rcle_pb_h1_role_proxy_r0/discovery_r0/`；
result SHA `50bc54d0…3de7`。receipt 同时绑定 runner、geometry、experiment、
B1A ledger 与 Bonn archive 的 SHA-256；`--validate-existing` 独立返回 `VALID`。
