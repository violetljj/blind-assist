# 双环可归因区域级接近证据源 Discovery R0

状态：`SOURCE_FOUND_FOR_DEVELOPMENT / RUNTIME_SOURCE_NOT_EVALUATED`

阶段：`DISCOVERY`

日期：2026-07-30（Asia/Hong_Kong）

## 结论

新双环路线已经找到一个能输出“可归因区域级接近证据”的开发期几何真值源：
**REveL Dynamic 的 RGB 人框、green/yellow helmet 目标身份与 Vicon
person/sensor 轨迹组合**。

本结论只完成来源发现与既有本地证据的只读连接复算，不实现运行时算法、不运行新的
候选输出、不打开确认集，也不改变旧 F-1B 的
`NO_INCREMENT / VALID / decision SEALED` 终点。

```text
GEOMETRY_TRUTH_SOURCE: FOUND_FOR_DEVELOPMENT
RUNTIME_GEOMETRY_SOURCE: DESIGN_CANDIDATE_ONLY / NOT_EVALUATED
CONFIRMATION: NOT_AUTHORIZED
PRODUCT_OR_SAFETY: NOT_AUTHORIZED
```

## 问题和判据

来源至少应能在同一时间绑定中提供：

```text
target identity
+ image region: LEFT | CENTER | RIGHT
+ radial state: approaching | quasi_static | receding
+ provenance and failure reason
```

这里的“可归因”要求几何量能落到一个明确目标，而不是只有全帧或中心走廊残差；
“区域级”使用框中心的固定水平三等分；“接近”要求有带符号的径向变化，而不是仅有
光流幅值、残差或图像尺寸。

## REveL 本地证据

官方 [REveL 页面](https://uts-ri.github.io/revel/) 说明 Dynamic 采集包含 RGB、
event、LiDAR、IMU、Vicon sensor pose，以及由 green/yellow helmet 区分的两名
人员轨迹与 2D 标注。本地既有审计已经形成以下独立证据链：

- `dynamic.bag`：`7,287,305,421 bytes`，SHA-256
  `6b10752b0d4cb401751e57f3ac55ebe45fcbb785f89d8a43fe1cbfd30dc0b08a`；
- 8,580 个 RGB—label 完整配对，13,018 个 green/yellow 标注框；
- person/sensor Vicon 米制轨迹和 range/range-rate；
- RGB 标注与 Vicon marker 的同源重投影；
- 512 个抽样帧、770 个 GT 框的 source radial motion 分层。

既有来源、同步、重投影与运动定义详见
[USTRF-SC research metrics](../ustrf-sc/USTRF_SC_RESEARCH_METRICS_2026-07-20.md)。
径向状态冻结为：

```text
v_r <= -0.10 m/s  -> approaching
v_r >=  0.10 m/s  -> receding
otherwise         -> quasi_static
```

### 本轮只读连接复算

本轮没有生成新的候选算法输出。复算只连接两个既有逐框账本：

- GT 框：
  `artifacts.local/evidence/ustrf-sc/revel-yolo11n-guarded-bounded-512-20260720-r2/details.jsonl`
  ，SHA-256
  `47cfb30d7cf1862dd85628332f3b9526708c1de76deaa1e24691beeb4396f530`；
- Vicon 径向运动：
  `artifacts.local/evidence/ustrf-sc/revel-yolo11n-vicon-radial-stratification-20260720-r1/details.jsonl`
  ，SHA-256
  `155863e2725ccac5a237b98153fd275fb4f64faf764fe4ab6f828e219059d3ef`。

连接键为 `selected_index + exact normalized_area`；区域按 GT 框中心
`x < 1/3`、`1/3 <= x <= 2/3`、`x > 2/3` 固定为 LEFT、CENTER、RIGHT。
结果为 `770 joined / 0 unmatched`，其中 `488` 框有 source motion：

| region | approaching | quasi-static | receding |
| --- | ---: | ---: | ---: |
| LEFT | 79 | 29 | 56 |
| CENTER | 79 | 47 | 73 |
| RIGHT | 46 | 27 | 52 |

接近框的目标身份分布为：

| region | green helmet | yellow helmet | unique sampled frames |
| --- | ---: | ---: | ---: |
| LEFT | 41 | 38 | 77 |
| CENTER | 43 | 36 | 78 |
| RIGHT | 20 | 26 | 45 |

逐框账本中的 `same_class_frame_ambiguous` 在这 770 行内为 `0`。因此该固定子集确实
同时包含 `target identity + region + approach direction`，不是从全局统计量推断区域
或目标。`TTC-proxy < 3s` 仅 LEFT/CENTER/RIGHT `5/4/1` 框，样本过少，不能支持
近碰撞、安全或秒级性能结论。

## 来源角色和限制

REveL 在新双环中可以承担：

- 目标归因和区域划分的开发真值；
- causal runtime candidate 的机制调试、方向符号检查和 abstention 检查；
- 与旧 Sparse LK 缺失语义的直接对照。

它不能承担：

- 手机运行时输出：当前 range-rate 使用严格包围图像时刻的前后 Vicon pose，
  `offline_noncausal=true`；
- physical assistive TTC：helmet marker 不是人体包络，source sensor marker 不是
  手机或眼镜 body frame；
- 独立确认：本来源的旧检测和几何输出已经被检查；
- 数据再分发或商业授权：REveL 页面中的 CC BY-SA 文案明确指向网站源码，不能自动
  推定数据载荷许可，保留 `HOLD_LICENSE_FOR_REDISTRIBUTION`。

## 运行时几何源候选

最小运行时路线是一个独立组件：
`target/track-conditioned causal radial geometry`。

它接收已有检测/跟踪给出的目标身份与 ROI，只使用当前和过去帧，计算框面积对数增长
和/或 ROI 内稀疏径向光流，并对全局相机运动、低支持和过期结果 abstain。最小输出
合同应冻结为：

```text
source_frame_id
captured_at_ns
target_id
region: LEFT | CENTER | RIGHT
approach_state and/or signed approach_rate
optional_ttc_proxy
quality
valid_until_ns
abstention_reason
```

若 `target_id/ROI` 来自 YOLO，这条路线是“同目标几何确认侧环”，不是完全独立的第二
感知源，也不能在 YOLO 尚未发现目标时提前告警。它是否仍能改善首次有效提醒、风险
连续性或误提醒，只能由后续事件级对照评价回答。

## 后备来源

1. **EVIMO2 `Flea3/imo`**：官方
   [下载页](https://better-flow.github.io/evimo/download_evimo_2.html) 和
   [真值格式](https://better-flow.github.io/evimo/docs/ground-truth-format.html)
   提供逐像素 object ID、毫米深度、相机/对象 pose 与 200Hz 轨迹；适合做对象级
   几何机制真值。RGB NPZ 约 33GB，当前本地未下载，且它不是助盲人员域。
2. **JRDB**：官方 [dataset page](https://jrdb.erc.monash.edu/dataset/) 提供
   egocentric 2D/3D 人体标注，适合生态复核；本地已检查部分的 3D 标注全部为
   interpolated，因此只能作 Development/diagnostic，不能取代直接物理真值。

## 证据传播和唯一下一步

- 旧 Sparse LK F-1B、sealed decision set 和 timing-only receipt 保持不变；
- REveL 旧输出改作新路线 Development，不升级为 unseen Confirmation；
- 后续 Confirmation 必须在输出访问前另行冻结独立 session/source partition、
  事件单位、阈值、统计量和停止规则；
- Discovery 结论不产生 Android、融合器、默认模型、提醒、产品或安全权限。

唯一建议的下一步是：**冻结一个 LITE Development round，定义上述 causal runtime
接口、REveL 开发子集、按目标/区域的评价单位、abstention 和 falsifier；完成设计
检查后，才另行激活最小实现与离线 replay。**
