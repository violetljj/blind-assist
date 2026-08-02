# HFTF Stage C D42 JRDB ego-object metric teacher result

日期：2026-08-03（Asia/Hong_Kong）

## 结论

终态：

`D42_JRDB_EGO_OBJECT_METRIC_TEACHER_SUPPORTED_DEVELOPMENT_ONLY`

冻结 7/7 support gates 全部通过。相对 current-relative static baseline，完整
ego+person metric teacher：

- mean horizontal error：`0.80935 -> 0.34757 m`，降低 `57.06%`
- median horizontal error：`0.74938 -> 0.14080 m`，降低 `81.21%`
- horizontal error better fraction：`79.994%`
- mean absolute range error：`0.56035 -> 0.10200 m`，降低 `81.80%`
- mean absolute bearing error：`5.30386° -> 2.45105°`，降低 `53.79%`
- 四个 sequences 的 mean horizontal error 全部改善

D42 建立的是 source-native metric-geometry information ceiling，不是 RGB/IMU
student、事件效用或 App 结果。

## 三臂贡献分解

| Arm | mean horizontal error | 相对前一层 |
|---|---:|---:|
| current-relative static | 0.80935 m | — |
| ego-kinematic / object-static | 0.73127 m | 相对 baseline -9.65% |
| ego+object kinematic | 0.34757 m | 相对 ego-only -52.47% |

因此 D41 的几何缺口不能主要归因于缺少 ego compensation。ego motion 有正贡献，
但更大的增量来自 same-identity person world motion。当前 2D box-history recipe
失败的核心是没有稳定恢复 metric person motion，而不是 target 本身没有历史信息。

## sequence 分解

| sequence | opportunities | mean horizontal reduction | median reduction | better fraction | range reduction | bearing reduction |
|---|---:|---:|---:|---:|---:|---:|
| Clark Center | 1,100 | 48.89% | 61.96% | 83.73% | 84.57% | 44.31% |
| Gates Basement | 822 | 77.28% | 84.74% | 82.48% | 81.68% | 71.76% |
| Meyer Green | 161 | 54.53% | 81.91% | 79.50% | 70.05% | 59.73% |
| STLC 111 | 1,301 | 67.69% | 24.28% | 75.33% | 74.86% | 63.77% |

所有 sequences 均满足无 material harm；正结果不是单一 source 驱动。

## cohort 与 parity

- producer source frames：480/480
- detector track occurrences：5,366
- paired opportunities：3,384
- distinct native identities：53
- evaluable sequences：4/4
- `odom <- base_link` transform parity maximum error：
  `1.1368683772161603e-13 m`

全部 evaluability gates 通过。

## 冻结预测

三臂全部只读 current 及前 6 frames：

1. current relative center 不变；
2. 7-frame ego odom translation + unwrapped yaw OLS，object world center 固定；
3. 在 arm 2 基础上再对 same-identity `center_odom_m` x/y/z 做 OLS。

future packet 只作为 `+15 frames center_base_link_m` truth。没有速度裁剪、平滑、
history/horizon、coordinate subset、regression order 或 sequence search。

## 与 D27/D33/D41 的层级关系

- D27：THOR-MAGNI action-conditioned field 中存在强 world-motion information
  ceiling；
- D33：真实 detector tracks 的 image-scale trend 能高精度判断 future range
  direction；
- D41：2D translation 有局部 IoU 增量，但 log-scale extrapolation 不稳定；
- D42：JRDB same-identity metric world motion 能强、跨 source 地恢复一秒后相对
  geometry。

这些证据共同把下一瓶颈定位为：

> 如何从 phone-causal detector track、RGB history 与可用 IMU 中恢复 D42
> teacher 的 person-relative metric motion。

它不是继续搜索 2D box-state 子集，也不是直接接入 alert。

## D43 授权边界

D42 只授权冻结一个轻量 student contract：

- target：D42 full teacher 相对 current-static 的 future displacement residual；
- inputs：phone-causal 2D track history + 同步 IMU，禁止 native identity、pose、
  3D center 或 future truth 进入 inference；
- split：sequence/session isolation；
- first gate：teacher residual regression 与 metric ranking；
- 不读取 event/alert outcome，不进入 production decision seam。

D43 尚未因 D42 自动获得支持。

## 复现

- D32/D33/D41/D42 evaluator tests：13 PASS
- report 连续重建 SHA 稳定：
  `1b8a8b9458edb2dd7b5f34eca95b5c0bdd9b0715efa8881cbbf8a43d5e1f5dfb`
- report size：10,060 bytes
- tracks SHA：
  `efa249fdfe8114dfeb1da419ffdb359189e3d4e6b1f406fabad04a31a39a0fa1`
- producer receipt SHA：
  `fa91162274222b9fe2254ae675ccb95af3fcdd6dca50ab267d476d74764be318`

artifact：

`artifacts.local/evidence/hftf/stage-c-d42-jrdb-ego-object-metric-teacher-v0/report.json`

## claim ceiling

`RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

D42 不建立 RGB/IMU student learnability、event utility、Android runtime、独立泛化、
产品或安全主张，也不覆盖 D35 device gate。
