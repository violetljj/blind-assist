# RCLE Synthetic Scene Mechanism Validation R0 preregistration

状态：`DEVELOPMENT_PASS / SEALED_REVISE / CLAIM_CONSUMED / NO_RETRY`

## 研究问题

在相同的确定性三维场景、纹理、内参和时间采样下，只改变相机运动时，RCLE
旋转补偿局部扩张及其 causal three-pair R1 是否能够：

1. 在真实接近与“旋转 + 接近”中保留正扩张和连续触发；
2. 在静止与纯旋转中保持低参考响应；
3. 对横移和后退 stress 保持可解释的非接近行为；
4. 在未见的 scene-family split 中维持相同方向，而不共享场景、纹理或派生帧。

## 证据设计

权威生成后端是 CPU NumPy/OpenCV analytic ray-caster，不使用扩散视频作为几何
真值。它对有限平面与 bounded panels 求每像素最近交点，逐帧输出 RGB、
camera-positive-Z metric depth、`T_world_camera`、`T_camera_world`、内参和精确
时间索引。所有几何计算使用 float64；`.npy` depth 是权威真值，PNG 只用于交换和
查看。

同一 scene family 的六种运动全部留在同一 split。development 与 sealed 使用不同
scene id、seed、panel layout 和程序纹理；禁止按帧拆分或把失败序列替换到另一 split。
冻结配置是
`scripts/research/egomotion_compensated_looming/rcle_synthetic_scene_r0/dataset_spec.json`。

每个序列固定 `101` 帧、`100` pair，frame index 为闭区间 `0..100`，timestamp
严格为 `frame_index / 10`；pair 归因到 current-frame timestamp。相机使用右手坐标，
世界和零 yaw 相机均为 `+X` 向右、`+Y` 向上、`+Z` 光轴向前；pose 是
`T_world_camera`。translation 在世界坐标中积分，yaw 是绕世界 `+Y` 的右手旋转。
由于像素 `+V` 向下，投影使用 signed
`K=[[fx,0,cx],[0,-fy,cy],[0,0,1]]`；配置中的 `fy` 是正幅值。

## 算法与真值防火墙

算法层只读取 RGB 的灰度派生、时间戳、内参以及由 pose 计算的相邻帧旋转。
depth、translation、motion role、surface id 和生成器元数据只进入独立 truth/QA
层。old 与 R1 必须从完全相同的 expansion pair ledger 派生；任何 abstention、窗边界
或 `<= 0.01/s` pair 都重置 R1 streak。

真值层用上一帧 metric depth 与相邻双向 pose 独立重投影，计算 rotation-compensated
dense correspondence，再按同一 3×3 空间单元拟合几何 expansion。它不得调用 RGB
光流或算法 estimator。

算法 adapter 接收一个独立 `algorithm_staging/` 下的严格 allowlist schema：灰度图、由
renderer ray-hit `surface_id>0` 在 staging 前派生的 valid mask、
相邻时间戳、内参和 `R_current_from_previous`。它不能打开完整 pose、depth、role、
surface id 或 generator metadata。depth、translation、role 和 surface-id poison
mutation 必须在保持同一 staged RGB/mask/rotation-only input 时证明不会改变 algorithm
ledger；manifest 多一个字段或路径逃出 staging root 都必须 fail closed。

所有科学门逐 sequence 取逻辑 AND，禁止跨 sequence pooled rescue。每个 sequence
固定分母为 `100` pair，abstention 保留在 trigger/coverage 分母。positive retention
定义为 `R1 coverage / old coverage`，old coverage 为零时 FAIL；首次触发额外延迟定义为
`first R1 current timestamp - first old current timestamp`，而不是从序列起点计算。
stress lateral/receding 在 R0 只完整报告，不进入 PASS。

truth 以 signed `K` 反投影上一帧点，使用
`R_world_current.T @ (P_world - t_world_current)` 进入当前相机，再投影并用
`inverse(K @ R_current_from_previous @ inverse(K))` 把 current pixel 去旋转到 previous
坐标。算法与 truth 的 sequence median absolute error、positive 绝对 expansion/trigger
coverage、pure-yaw `median(|raw|)-median(|compensated|)` 和同 scene 的
rotation+approach/approach 差值均有独立门；两种 positive 近零不能通过。

## 执行顺序

1. 实现 renderer、manifest、truth builder、algorithm adapter 和独立 validator。
2. 运行单帧/短序列 fixture、双运行字节哈希确定性与错误注入测试。
3. 只生成并运行 development split，保存所有 abstention 和失败。
4. 在 RCLE current 原子治理收口及 separate activation 以前，不生成或读取 sealed
   split 的算法结果。

sealed activation 必须绑定 spec、renderer、builder、truth、algorithm adapter、
validator 和依赖身份，先独占创建 claim，再确认 canonical output 为空；claim 一经消费
不得重试、替换或扩大。当前 sealed scene seeds 对实现者可见，因此它只是
outcome-unseen fixed holdout，不称为 cryptographically hidden holdout。

## 结论上限

最大允许结论是
`CONTROLLED_SYNTHETIC_CAUSAL_MECHANISM_EVIDENCE_ONLY`；即使未来 sealed PASS，也
只能说明四个固定 synthetic scene families 和冻结运动幅度上的方向复现。PASS 不构成真实数据外部
confirmation、跨真实来源泛化、Android/产品晋级、人体有效性或安全证据；既有
`NOT_EVALUABLE`、`INVALID`、consumed claim 和 real-data terminal 均不回写。
