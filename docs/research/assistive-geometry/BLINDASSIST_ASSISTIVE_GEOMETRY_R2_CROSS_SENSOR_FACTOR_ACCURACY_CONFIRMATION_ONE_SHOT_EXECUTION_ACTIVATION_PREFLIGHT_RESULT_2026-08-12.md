# AG R2 独立跨传感器 factor-level Confirmation 激活前检结果

状态：`FAIL_CLOSED / EXECUTION_LOCK_NOT_ISSUED / ONE_SHOT_UNCONSUMED / SCIENTIFIC_NOT_RUN`

用户已授权创建并冻结唯一 successor 的 one-shot execution lock；本次授权不包含在同一步消费锁、
读取真实 archive member、运行模型或执行 Confirmation。前置核对未通过，因此没有伪造
`AUTHORIZED_UNCONSUMED`：

- ETH3D 官方文档说明该 camera-IMU calibration 来自 Kalibr；Kalibr 的官方格式以 YAML
  `T_cam_imu` 嵌套 `4×4` 矩阵表示 IMU→camera 变换。冻结 parser 却只接受同一文本行上的
  `<key> + 16 row-major floats`。不使用真实 archive 的 official-shaped synthetic control 稳定得到
  `ContractError F2_IMU_CALIBRATION_MATRIX`。
- 冻结 execution-lock schema 是 exact-key：它不能表达 calibration encoding、变换方向、IMU
  column/frame/axis、specific-force sign 或这些约定的官方/独立证据 binding；加入字段会命中
  `F2_EXECUTION_CALIBRATION_BINDING_SCHEMA`。
- `camera_imu_calib_radtan.zip` 的 exact member 尚未枚举或读取，不能从公开示例猜 member/key。
- 11 个 runtime role 中已有 10 个本地候选完成 bytes/SHA 核对，但必需的
  `DEPTHART_SOURCE_MANIFEST`（schema `blindassist.depthart.source_manifest.v1`）不存在，无法形成完整锁。

七个 archive direct-child 的文件名和总计 `721,072,411 bytes` 与 data identity 一致；本步没有重哈希
archive 内容。真实 archive bytes/member、RGB、depth、IMU、trajectory、calibration payload 均未读取；
checkpoint 未反序列化，model inference、source truth、factor scoring、Confirmation 和 exclusive evidence
root 均为 `0`。当前没有 execution lock，one-shot 未消费，科学状态仍为 `NOT_RUN`。

唯一 successor 是非执行的
`BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CONTROL_FORMAT_AND_RUNTIME_BINDING_REPAIR_IMPLEMENTATION_LOCK`：
先让 schema 能 hash-bind 官方控制证据、实现并合成验证 Kalibr YAML、生成并独立复核 DepthART source
manifest，同时实现一个未来需另行授权的 calibration-control-only preflight。该 successor 本身仍不得
枚举/读取真实 member、加载模型、创建 Confirmation root 或评分。

官方控制依据：

- ETH3D SLAM documentation: <https://www.eth3d.net/slam_documentation>
- Kalibr YAML formats: <https://github.com/ethz-asl/kalibr/wiki/yaml-formats>
- ROS REP 145 IMU sensor drivers: <https://ros.org/reps/rep-0145.html>
