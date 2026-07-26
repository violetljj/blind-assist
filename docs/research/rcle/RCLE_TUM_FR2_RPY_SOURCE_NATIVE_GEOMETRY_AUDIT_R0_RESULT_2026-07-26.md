# TUM fr2/rpy source-native geometry audit R0 结果

状态：`PASS / VALID`

日期：2026-07-26

终态：`PASS_SOURCE_NATIVE_GEOMETRY_CANARY_DESIGN_MAY_FOLLOW`

## 结论

TUM `fr2/rpy` **确实提供可评价的、低平移且以旋转为主的真实数据窗口**。
10 个预先固定的非重叠 10 s 窗中 9 个可评价；其中冻结诊断带判定
`0, 3, 6` 为 rotation-active 且 translation-induced geometry
rotation-like。来源权威、RGB/注册深度、mocap color-camera pose、时间戳和
坐标变换均足以对齐。

但本 source **没有验证旧 `0.02 m/s` 门仍会窗口级错杀**。三个
rotation-dominant 窗的 median raw speed 都低于 `0.02 m/s`。超过旧门的
window `8, 9` 同时出现更大的 absolute radial expansion、parallax ratio，
以及不再平衡的 signed/positive structure；它们不能被事后包装成错杀。

这不推翻 PB-H1 的一般结论：raw speed 仍是因果错位的代理；只表示在本次
固定 `fr2/rpy` 窗口上，`OLD_0P02_GATE_FALSE_KILL` 没有出现。

## 冻结与来源

在下载目标 TGZ、查看 pose speed 或运行几何前，先冻结
[R0 contract](RCLE_TUM_FR2_RPY_SOURCE_NATIVE_GEOMETRY_AUDIT_R0_CONTRACT_2026-07-26.json)：

- shared coverage 内，整数 Unix 秒锚定的全部非重叠 10 s 窗；
- 每窗保留 `dt in [0.020,0.050] s` 的相邻 RGB pair；
- 按 TUM `associate.py`：offset `0`、唯一最近邻、最大差 `0.020 s`；
- pose 在 RGB timestamp 线性/SLERP 插值，不外推，bracket `<=0.050 s`；
- 原样导入 PB-H1 `geometry.py`，固定 8 px raster、可见性与 pair 汇总；
- 窗口 coverage `>=0.80`、median valid depth fraction `>=0.50`；
- 不按 geometry 换窗，不换 TUM sequence，不读取 RCLE RGB outcome。

metadata audit 在任何 geometry 之前发现 ground-truth 有三组重复 timestamp。
[Implementation amendment](RCLE_TUM_FR2_RPY_SOURCE_NATIVE_GEOMETRY_AUDIT_R0_IMPLEMENTATION_AMENDMENT_2026-07-26.json)
仅冻结 TUM Python 文本解析一致的 `last text row wins`，并设重复组
`1 mm / 0.5 deg` fail-closed；未改变窗口、公式、阈值或停止规则。

TUM 官方材料：

- [benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
- [download index](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
- [file formats and calibration](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
- [association tools](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools)
- [fr2/rpy source](https://cvg.cit.tum.de/rgbd/dataset/#freiburg2_rpy)

官方将本序列描述为原地缓慢绕 R/P/Y 三轴旋转；整体统计为
`109.97 s`、`1.506 m`、平均 `0.014 m/s` 与 `5.774 deg/s`。下载的唯一目标
TGZ 为 `2,045,614,831 bytes`，与 HEAD 完全一致；SHA-256
`3a35b799…2b51f`。TGZ 内有 `3290` RGB PNG、`3287` 注册 depth PNG、
`rgb.txt`、`depth.txt`、`groundtruth.txt` 和 `accelerometer.txt`。
内部 ground-truth 与预先取得的官方独立文本逐字节一致。

## 对齐、覆盖与弃权

- RGB：`3290/3290` 为 `640x480` RGB PNG；
- depth：`3287/3287` 为 `640x480` uint16 PNG，按官方 `raw/5000 m`；
- RGB-depth：`3221` 个唯一匹配，stream coverage `0.9790`；
- pose：`32992` raw rows，`32989` unique timestamps；
- 重复 pose 最大离散：`0.0001 m / 0.04287 deg`，通过冻结上限；
- 固定窗口：`9/10` 可评价；window 4 因 source-depth raster
  中位覆盖 `0.34125 < 0.50` 弃权；
- pair：`2852/2990 = 0.95385` 可评价；
- 唯一弃权：`138` 个 `RGB_DEPTH_UNMATCHED_OR_REUSED`；
- source-depth nonzero fraction：median `0.72625`，Q10 `0.47292`，
  min `0.22667`；
- PB-H1 对已非零深度点的 projection visibility：median `0.99917`，
  最小 pair `0.99453`。

空间语义来自 TUM source-native 定义：depth 已由 OpenNI 重投影到 RGB color
camera；ground-truth 是 RGB color-camera optical center 的 world pose。使用
TUM 推荐的 ROS default `fx=fy=525, cx=319.5, cy=239.5`，不对已注册深度做
第二次去畸变。对 `T_world_camera`：

```text
R_current_from_previous = R_world_current^T R_world_previous
t_current_from_previous = R_world_current^T (c_world_previous-c_world_current)
```

## 连续 pair 分布

| 指标 | min | Q10 | Q25 | median | Q75 | Q90 | Q95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw speed m/s | 0.00000 | 0.00709 | 0.01043 | 0.01535 | 0.02205 | 0.02967 | 0.03577 | 0.08907 |
| signed radial s^-1 | -0.06982 | -0.01754 | -0.00714 | -0.00039 | 0.00422 | 0.00937 | 0.01484 | 0.07496 |
| absolute radial s^-1 | 0.00000 | 0.00525 | 0.00813 | 0.01335 | 0.02322 | 0.03513 | 0.04405 | 0.08708 |
| positive fraction | 0.000 | 0.188 | 0.352 | 0.493 | 0.579 | 0.736 | 0.834 | 1.000 |
| Q90 parallax rad/s | 0.00000 | 0.00451 | 0.00698 | 0.01087 | 0.01658 | 0.02304 | 0.02777 | 0.06479 |

整体 median 是低平移且 signed expansion 接近零、positive fraction 接近
`50/50` 的结构；但 pair tails 明确非零，所以不能把序列级均值或名称冒充每窗
纯旋转。

## 固定窗口

| 窗 | coverage | speed m/s | angular deg/s | signed s^-1 | absolute s^-1 | positive | parallax rad/s | parallax/angular | rotation-like | old gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 1.000 | 0.01461 | 6.514 | 0.00103 | 0.01024 | 0.519 | 0.01035 | 0.091 | yes | pass |
| 1 | 0.873 | 0.01613 | 6.949 | -0.00059 | 0.01552 | 0.494 | 0.01345 | 0.111 | no | pass |
| 2 | 1.000 | 0.01416 | 4.861 | -0.00094 | 0.00925 | 0.481 | 0.00922 | 0.109 | no | pass |
| 3 | 1.000 | 0.01293 | 6.055 | 0.00068 | 0.01103 | 0.511 | 0.00939 | 0.089 | yes | pass |
| 4 | 0.980 | 0.01218 | 4.936 | 0.00208 | 0.00718 | 0.559 | 0.00708 | 0.082 | depth abstain | pass |
| 5 | 0.913 | 0.01070 | 4.562 | 0.00371 | 0.00972 | 0.615 | 0.00698 | 0.088 | no | pass |
| 6 | 1.000 | 0.01407 | 6.385 | -0.00272 | 0.01343 | 0.457 | 0.00998 | 0.090 | yes | pass |
| 7 | 1.000 | 0.01923 | 8.225 | -0.01184 | 0.02501 | 0.394 | 0.01590 | 0.111 | no | pass |
| 8 | 0.839 | 0.02524 | 8.462 | -0.00537 | 0.02666 | 0.433 | 0.01893 | 0.128 | no | reject |
| 9 | 0.933 | 0.02402 | 9.573 | -0.01086 | 0.02875 | 0.296 | 0.01688 | 0.101 | no | reject |

`rotation-like` 同时要求 angular `>=5 deg/s`、absolute radial `<=0.02/s`、
`abs(signed)<0.01/s`、positive fraction `[0.40,0.60]` 和
parallax/angular `<=0.10`。这些是冻结 discovery diagnostics，不是新
confirmation gate。

## 证据边界与下一步

强证据：

- TUM 官方 source-native RGB/注册 depth/color-camera mocap 定义；
- 目标 archive 完整清单和全量 PNG header；
- 固定窗口内的 pose interpolation、depth sampling 与 PB-H1 continuous
  geometry；
- 两次独立执行得到相同 result SHA，7 项 PB-H1/TUM 专项测试通过。

未建立：

- RCLE RGB algorithm 效果；
- approach role 或跨来源稳定性；
- confirmation、人体、安全或产品能力；
- 旧 `0.02 m/s` 在 fr2/rpy 上的窗口级错杀。

本 source gate 已通过，因此**可以**在单独预注册后设计一个小型
real-data geometry canary；不能直接把本 audit 当作 canary 或算法证据。

机器证据：
`artifacts.local/evidence/rcle_tum_fr2_rpy_geometry_audit_r0/`；
result SHA `ae388f8e…6c578b1`。
