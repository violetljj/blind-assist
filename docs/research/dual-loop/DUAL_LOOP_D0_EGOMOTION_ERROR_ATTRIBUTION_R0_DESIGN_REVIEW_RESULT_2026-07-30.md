# D0 ego-motion error attribution R0 design review

状态：`DESIGN_REVIEW_PASS / CONTRACT_FROZEN / NOT_RUN`

日期：2026-07-30（Asia/Hong_Kong）

冻结合同 SHA-256：
`f43add496d1dab53072bc9a27ddd28e716ec9480360a41cca6243ed4b326cf62`

## 审查结论

[D0 合同](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R0_PROTOCOL_2026-07-30.json)
已经把问题冻结为 burned LITE R2/REveL 的单次 Development 误差归因。它不修改或
重跑算法，不产生新的有效性或泛化结论，也不访问旧 F-1B decision 输出。

本审查的 `PASS` 只证明设计边界完整、输入与出口可检查；不代表 D0 已实现、已执行，
更不代表任一机制结论成立。

```text
SCIENTIFIC OUTCOME: NOT_RUN
PROTOCOL STATUS: DESIGN_REVIEW_PASS
EXECUTION AUTHORITY: NONE
OLD F-1B DECISION: SEALED / ZERO ACCESS PRESERVED
```

## 已核验

- 前序精确绑定到
  `DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R2`，而不是同名的 RCLE
  periodic synthetic R2；前序终点、一次性消费与 no-rerun 保持不变。
- 诊断只使用已烧毁的 `REVEL_DYNAMIC_V1`：13,014 个 ROI replay rows、26,028
  个 R2 arm rows、469 个 primary parent natural events，以及同一 capture 的
  Vicon person/sensor 6DoF 和冻结 calibration。
- 独立单位是 parent natural event；frame、pair、track 只作事件内重复测量。按
  target 与 anchor region 输出描述性分层，但不把同一事件按 region 切开。
- person 与 sensor 径向分量使用可与有限差分 range-rate 精确闭合的 chord range
  gradient；同一 person-pair 时间基和 frozen nearest-sensor indices 已明确。分量的
  符号、求和闭合和相机光心 6DoF 派生已分开定义。
  camera optical-center motion 只作诊断，不改写原 sensor-marker-to-person truth。
- 相机角速度、相机平移、ROI 尺寸/中心抖动、事件长度、flow MAD、flow sign flip、
  surviving tracks、features、quadrants、forward-backward error 与 coverage 均有
  event-level 字段。
- 统计只报告固定 capture 的事件计数、median/IQR、Cliff delta 与
  target/region/truth-state 方向一致性；无 frame-level 独立性、无泛化 p 值、
  无置信区间或 population claim。用于路由的指标必须在 approaching 与 receding
  内分别达到 `Cliff delta >=0.33`；任一可评价 region 出现 material opposite
  direction 都进入不可识别，不能用 2/3 多数或 pooled composition 救援。
- `TEMPORAL_NOISE_DOMINANT` 必须同时具备至少一个直接时间不稳定指标
  （flow MAD/sign flip）和一个独立 support/persistence 指标；相关的低支持指标不能
  机械叠加成 temporal dominance。
- 科学出口严格只有
  `EGO_MOTION_DOMINANT`、`TEMPORAL_NOISE_DOMINANT`、
  `MECHANISM_NOT_IDENTIFIABLE`。混合、矛盾、person-motion 竞争解释或分层支持不足
  一律进入第三出口，不做 subgroup rescue。

## 后继锁

- 只有 `EGO_MOTION_DOMINANT` 可进入最后一次 EVIMO2v2 受控机制 canary。
- canary 只允许一个变化：目标外背景 robust affine 补偿后计算目标内 residual
  radial expansion；同轮禁止 homography 备选、平滑、深度、复杂融合、阈值或窗口
  搜索。
- canary 再失败，图像尺度/径向光流路线作为完整负结果停止。
- `TEMPORAL_NOISE_DOMINANT` 只允许在新数据、新版本上前瞻冻结 causal multi-frame
  robust trend；禁止在 REveL 上优化后自评。
- `MECHANISM_NOT_IDENTIFIABLE` 直接停止该路线。
- EVIMO2v2 canary 通过后才可进入 JRDB 人员域 Development；Confirmation 仍需
  未用于选择或调参的独立 session/source。

## 官方来源核验

- [EVIMO2v2 官方下载页](https://better-flow.github.io/evimo/download_evimo_2.html)
  列明 motion-segmentation/object-recognition 数据含独立运动物体、pixelwise mask、
  depth 与 trajectory；Flea3 RGB NPZ 为 33 GB，完整 NPZ 约需 525 GB 磁盘。
- [JRDB 官方页](https://jrdb.erc.monash.edu/dataset/) 列明 54 个室内外序列、
  360° RGB + LiDAR、2D/3D 人体框，以及 stationary 与 moving sensor perspectives。
- [REveL 官方页](https://uts-ri.github.io/revel/) 列明总时长 14.1 分钟、四个
  ROS bag、手持移动传感器和两名 Vicon 跟踪人员；这些事实不把本地已烧毁的单
  capture 升级为独立 Confirmation。

## 未核验与下一权限

- 未实现 D0 extractor、event table、validator 或 runner。
- 未读取任何新的算法输出，未运行 D0，也未下载 EVIMO2v2/JRDB。
- 若要执行 D0，必须另立实现 identity、测试、只读输出根、独立实现审查与一次性
  activation；本合同及本审查不自动授予这些权限。
