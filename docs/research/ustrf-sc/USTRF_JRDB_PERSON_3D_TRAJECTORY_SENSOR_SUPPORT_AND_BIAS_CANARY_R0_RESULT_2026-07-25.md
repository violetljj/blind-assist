# JRDB person 3D trajectory sensor support and bias canary R0 结果（2026-07-25）

状态：`SENSOR_SUPPORT_AND_BIAS_PROFILE_AVAILABLE_WITH_ABSTENTION / VALID`

权限：`SEEN_DEVELOPMENT_DIAGNOSTIC_ONLY`

## 结论

父 R1 的 annotation-derived 3D 轨迹不是全无真实传感器支持，但也不能把“1,350/1,350 可计算”改写成“1,350/1,350 被 LiDAR 证实”：

- `1,105/1,350 = 81.8519%` 个可计算 3D object-frame 达到冻结的 `>=3` 个 box 内真实 LiDAR return；
- `89/1,350 = 6.5926%` 在双 PCD 完整解码后为 0 点，只能保留 `annotation-only`；
- `156/1,350 = 11.5556%` 有 1–2 个点，质心支持不足，局部 `abstained`；
- `0/1,350` invalid。没有因局部缺支持枪毙整段。

相邻运动中，`1,044/1,336 = 78.1437%` pair 的两端都得到 sensor support；其余为 `67 annotation-only + 225 abstained + 0 invalid`。三帧加速度单元为 `1,001/1,322` sensor-supported。

因此唯一合法终态是：

`SENSOR_SUPPORT_AND_BIAS_PROFILE_AVAILABLE_WITH_ABSTENTION / VALID`

它回答了“得到多少 LiDAR 支持、偏差多大”，但不证明 annotation center 是人体真实中心，也不验证人体真实速度或 USTRF 效果。

## 点云支持与质心偏差

240 份 `binary_compressed` PCD 均从 raw bytes 重新解码；每帧分别保留 upper/lower 点数、质心、timestamp，再作不跨传感器去重的描述性融合。1,350 个 3D object-frame 的支持模式为：

| 模式 | 总数 | 达到 `>=3` |
| --- | ---: | ---: |
| upper + lower | 1,028 | 1,020 |
| upper-only | 209 | 81 |
| lower-only | 24 | 4 |
| neither | 89 | 0 |

对 1,105 个 sensor-supported object-frame，融合点云质心相对 annotation center 的 3D 残差为：

- median `0.1949m`；
- P90 `0.3923m`；
- P95 `0.4807m`；
- P99 `0.6705m`；
- max `0.9360m`。

水平残差 median/P95 为 `0.0922m / 0.2995m`。有符号均值为 `x +0.0075m / y +0.0239m / z -0.0944m`；负 z 偏差与“LiDAR 质心落在可见表面而非人体几何中心”的机制一致，但本单序列不能把它外推为通用标注偏差。

## 系统性偏差检查

距离是最强的支持退化轴：

| range | 分母 | sensor-supported | 支持率 | residual median / P95 |
| --- | ---: | ---: | ---: | ---: |
| `<10m` | 504 | 504 | 100.00% | `0.1930 / 0.3834m` |
| `10–20m` | 164 | 161 | 98.17% | `0.1617 / 0.3261m` |
| `20–40m` | 538 | 418 | 77.70% | `0.2052 / 0.5404m` |
| `>=40m` | 144 | 22 | 15.28% | `0.4965 / 0.9022m` |

cross-modal missingness 也不是随机的：

- `3D+2D`：`1,104/1,321 = 83.57%` supported；
- `3D-only`：仅 `1/29 = 3.45%` supported，另有 `16` annotation-only、`12` abstained；
- `2D-only`：24 个因没有 3D box 全部只在 union ledger 局部 abstain，不伪造轨迹。

遮挡分层支持率为 Fully visible `92.21%`、Mostly visible `78.50%`、Severely occluded `75.27%`、Fully occluded `22.22%`。其中 3D-only 的 occlusion 为 unknown，未拿 2D 遮挡字段替代 LiDAR visibility truth。以上是单序列描述性 cluster profile，不是因果效应。

## 相邻运动、跳变、加速度与 pose 敏感性

在 1,044 个双端 sensor-supported pair 上：

- sensor-centroid motion 与 annotation motion 的 3D 差值 median/P95 为 `0.0495m / 0.2535m`；
- annotation speed median/P95 为 `1.489 / 5.135m/s`，冻结 `>4.5m/s` flag 为 `146/1,336`；
- sensor-centroid speed median/P95 为 `1.362 / 5.255m/s`，flag 为 `98/1,044`；
- `>0.5m/pair` jump 为 annotation `4/1,336`、sensor centroid `10/1,044`；
- annotation acceleration `>12m/s²` 为 `402/1,322`；sensor centroid 为 `595/1,001`，表明二阶量对点云质心抖动更敏感，不能当人体真实加速度。

pose 敏感性按冻结定义比较 dynamic odom displacement 与 left-pose-frozen displacement。annotation 与 sensor centroid 的 median/P95 分别为 `0.0657/0.2717m` 与 `0.0656/0.2716m`；这说明相邻运动不能忽略机器人 pose。上下 PCD timestamp skew median `2.299ms`、max `4.230ms`；R0 未伪造 deskew，分别保留两路 timestamp，并把未 deskew 融合作为限制。

## 证据与复算

- config SHA-256：`004a97b6af6dbac8d8b554f71bf5ccfcc42cbaf83a47bef69aecc1b701e9dda3`
- ledger SHA-256：`850fff5ae9df09fc9453bba4706784b12ac87bd436ac9b1af77aa64bf47692a8`
- receipt SHA-256：`6d3bd96b5501b87eb2d42f8a8db7d2af6901d4b866cb23fa575343abc6ee9978`
- validation SHA-256：`34adc7e18247fe44e2c43829b9ed3320392cc691519b67e60c40a9ce018680d7`
- focused tests：`5/5 OK`
- 独立 validator：`16/16 checks true / VALID`
- producer 与 validator 均逐份重验 PCD SHA、完整 LZF 解码、field-major XYZ、声明点数、upper/lower transform、oriented-box query、四类守恒和父 `1,350/1,336/1,322` 分母。

## 权限和下一边界

点落入 annotation box 是独立的 sensor evidence，但 ROI 本身由 annotation 定义；它最多支持“该标注体积内有 LiDAR return”以及可见点质心残差，不能独立证明 person identity、人体真实中心或 annotation accuracy。

本轮不产生 candidate selection、route risk、event lifecycle、提醒、Android、人体/独立行走或生产权限。若继续，下一独立边界必须先解决以下之一：独立 person trajectory truth、跨新 sequence 的偏差复现，或 upper/lower time-aware deskew/robust centroid 的预注册验证；不得直接进入 route/event/safety。
