# USTRF-SENSOR-REPLAY-R3 连续闭环结果（2026-07-22）

状态：`SOURCE_ADMISSION_FAILED / DO_NOT_SELECT_HARDWARE`

## 结论

R3 的可执行闭环已经建立，但本轮自动取得的三条 TUM 动态 RGB-D+mocap 序列没有通过“真正适合身体绑定前向路线与障碍生命周期”的来源门。两个隔离模型分别检查覆盖全部 `2714` 帧的 `29` 张连续帧 sheet，均在看不到 candidate alert 的条件下一致拒绝三源。因此没有合法的 onset、alertable、passed/cleared anchors，五项事件指标必须保持 `not_evaluable`，不能写成 0，也不能做宏平均。

最终 verdict 为 `DO_NOT_SELECT_HARDWARE`。`120 episode`、`U0`、重复 ARCore、硬件选择、Android runtime 和生产替换继续关闭。

## 本轮实现

- 自动下载并 SHA-256 冻结 TUM `fr3/walking_xyz`、`walking_halfsphere`、`walking_rpy`；TUM 数据许可为 CC BY 4.0。
- 新增 TUM RGB-D 一对一时间关联 Adapter，规范化完整连续序列，而不是只取 120 帧。
- 生成不含 `camera_to_world` 或 pose 时间戳的 sanitized estimator ledger；独立 `opencv_orb_rgbd_pnp_r3_v1` 进程只能读取 RGB、depth、intrinsics。
- evaluator 才读取 mocap GT，以首个公共帧做 SE(3) 对齐并量化 drift；没有 GT pose 自比自。
- route truth 由未来 mocap camera center 投影产生；causal route prediction 只使用当前及过去 estimated pose，逐帧显式 `known/unknown`。
- route truth 与 causal prediction 物理写入两套 JSONL ledger，并由独立 manifest 绑定各自 SHA-256；manifest hash 为 `76ecf68f…a728e0`。
- `ustrf_metric_depth_causal_route_r3_v1` 在审核前冻结逐帧 route-depth risk 和 alert trace；review bundle 不包含任何 candidate alert。
- 双模型完整序列 review 支持正向 anchors、哈希绑定、隔离字段和 fail-closed consensus。
- replay gate 新增逐来源 event recall、critical miss、false alerts/min、clearance rate、clearance p95 ms；总门必须每源 AND，按每个指标分别报告 worst source，不能用宏平均救失败源。
- R2 旧合同保持兼容。

## 三来源量化

| 来源 | 帧数 | transport | 独立 pose 覆盖 | pose RMSE / endpoint | rotation RMSE | route p50 / p95 | unknown rate | 双模型来源门 |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- |
| `tum_fr3_walking_halfsphere` | 1021 | fail；depth reprojection p95 `0.416m` | `0.993` | `0.979 / 1.346m` | `21.22°` | `72.98 / 72.98px`（仅 1 个公共 known 帧） | `0.999` | reject / reject |
| `tum_fr3_walking_rpy` | 866 | pass；`0.277m` | `0.962` | `1.867 / 2.053m` | `24.72°` | not evaluable | not evaluable | reject / reject |
| `tum_fr3_walking_xyz` | 827 | pass；`0.226m` | `0.931` | `0.913 / 1.012m` | `18.73°` | `236.74 / 375.19px` | `0.937` | reject / reject |

candidate 在审核前已冻结：halfsphere 产生 `1` 个 alert interval，rpy/xyz 为 `0`；由于来源与事件 truth 未准入，这些只是诊断 trace，不能参与五项效果计分。

## 为什么三源被拒绝

- `walking_halfsphere`：轨迹由半球绕行和强旋转主导，路线投影稀疏，不能绑定稳定前向身体走廊。
- `walking_rpy`：主要是原地 roll/pitch/yaw 扫动，不是前向路线。
- `walking_xyz`：姿态相对稳定，但只是办公室内局部 XYZ 平移；可见人物、椅子与桌面不能形成可审核的“路线交叉—alertable—passed/cleared”因果生命周期。

这说明数据集名称中的 `walking` 指动态人物，并不自动等于可用于助行路线闭环。自动取得成功不等于来源准入成功。

## 证据

- 预注册：`configs/ustrf_sensor_replay_r3_prereg_v1.json`
- 来源与 archive hashes：`configs/ustrf_sensor_replay_r3_sources_v1.json`
- 独立 pose：`artifacts.local/evidence/ustrf-sensor-replay-r3/pose-estimates-v1.json`
- 冻结 candidate：`artifacts.local/evidence/ustrf-sensor-replay-r3/candidate-evaluation-frozen-v1.json`
- 分离路线账本：`artifacts.local/evidence/ustrf-sensor-replay-r3/route-ledgers-v1/manifest.json`
- 隔离 review：`artifacts.local/evidence/ustrf-sensor-replay-r3/review-inputs-v1/`
- 共识：`artifacts.local/evidence/ustrf-sensor-replay-r3/review-consensus-v1.json`
- 最终报告：`artifacts.local/evidence/ustrf-sensor-replay-r3/replay-report-v1.json`

## 下一步

不要在这三源上调 route horizon、unknown 阈值、candidate 深度阈值或 review 口径。下一轮只替换来源：先用低成本预览筛选具有持续前向载体轨迹、RGB-D、独立 pose、路线内障碍接近与可观察 passed/cleared 的连续序列，再下载完整资产。OpenLORIS dynamic office/cafe、具明确协议的 egocentric RGB-D+head-pose 数据可进入候选筛选，但只有完整片段双模型准入后才能成为三源之一。
