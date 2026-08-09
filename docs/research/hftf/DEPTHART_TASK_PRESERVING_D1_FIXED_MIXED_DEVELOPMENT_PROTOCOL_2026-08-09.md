# DepthART task-preserving D1 fixed-mixed Development screen

状态：`CONTRACT_AND_METADATA_ROSTER_FROZEN / EXECUTION_NOT_ACTIVATED / NO_OUTCOME_ACCESSED`

机器合同：[`DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_PROTOCOL_2026-08-09.json`](DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_PROTOCOL_2026-08-09.json)

## 问题与边界

D0 的 FP16、W8A16、INT8 三臂已经在 outcome 前技术关闭，不能修改后重跑。D1 是一个新问题：
只重建一个产品纵横比 `fixed-mixed` 候选，在独立 Development 数据上验证它是否保持 B0 的
clearance、occupancy、false-clear、false-block、UNKNOWN 和 temporal 合同。既有 G4-C
`448x448` 图只是 source/control，不能直接成为 D1 或 R2 候选。

strict G4-D 的
`CURRENT_QAIRT_2_47_SM8650_HTP_STANDARD_FLOAT_PATH_STRICT_G4D_NOT_SUPPORTED`
保持不变。D1 PASS 也只允许锁定一个 R2 candidate；它不访问 R2 outcome，不授权测性能、替换
DA2、修改默认 App、生产或 safety 主张。

## 冻结产品输入

- CameraX 请求 `640x480 / 4:3`，决策视角固定为 display-upright portrait；
- 保留完整 FOV，禁止额外 crop 和 padding；
- portrait tensor 固定为 `1x3x608x448`；
- K 按 `source crop -> clockwise upright rotation -> final sx/sy resize` 动态传播；
- transform、K 或 gravity 缺失、过期、非有限时必须输出 `UNKNOWN`，不能填成 clear；
- landscape 只可作为 label-blind coverage 记录，不得救援 portrait 任务门。

## 冻结任务后处理

truth、canonical PyTorch reference 与 fixed-mixed candidate 使用同一 B0 reader/policy。预测 depth
只以 finite、`0.25–6.0m` 作为深度有效性；ARKit pose 提供的 display-upright gravity vector 对
三臂相同。每帧必须产生 `left/center/right × 1.0/1.5/2.0m` 九个 cell；null、无地面、无支撑
或无效输入均映射为 `UNKNOWN_GROUND`，`UNKNOWN` 永远不是 negative。

质量 screen 同时要求对独立 truth 的绝对门和相对 reference 的非劣门，且 pooled、parent-macro、
session-macro、worst-parent 与全部 band/horizon 都完整 finite。任一缺失或零分母 FAIL。只有全部
质量门通过，才允许把这个单一图锁定为 R2 candidate；性能仍需等 R2 质量 PASS 后另行进入。

## Development roster

[`D1 ARKit Development roster lock`](DEPTHART_TASK_PRESERVING_D1_ARKIT_DEVELOPMENT_ROSTER_LOCK_2026-08-09.json)
冻结 8 个 primary 和 8 个 reserve Training visit/session。规划器在冻结提交上同时扫描 HFTF 与
Assistive Geometry 两条路线，排除既有 R0、G4-D、R2 和 B0/B1 身份。当前只读官方 split CSV
的 Git blob；媒体、truth、模型输出均未打开，也没有下载授权。

reserve 只能按预冻结的 label-blind portrait/pose/RGB-D 连续性规则替换不合格 primary，禁止按
reference/candidate/truth outcome 替换。

## 当前唯一后继

先扩展 reviewed ARKitScenes use scope 到这 16 个锁定身份，再做 label-blind media integrity 与
portrait/pose/RGB-D continuity preflight。该门通过后才允许重建 `608x448` fixed-mixed 图并冻结
单一 candidate/reference/postprocess 身份；当前不得下载媒体、运行模型或读取任务 outcome。
