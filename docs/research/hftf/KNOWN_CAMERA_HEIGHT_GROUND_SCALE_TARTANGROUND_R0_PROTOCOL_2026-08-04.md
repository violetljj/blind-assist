# Known camera height ground scale TartanGround R0 protocol

日期：2026-08-04

状态：`FROZEN_BEFORE_UNUSED_ENVIRONMENT_METADATA_QUALIFICATION_OR_PAYLOAD`

## 独立后继

ARKitScenes R0 已在 DA/effect 之前因 ground-height authority 不足终止为
`HOLD_SOURCE_AUTHORITY_NO_REPLACEMENT`。本协议不修改该 cohort，也不改变候选算子或
效果门；它把 source population 改为 TartanGround differential-drive `P1000`，因为该
source 的 metadata 显式提供 `robot_height`、`time_step` 和动态 `lcam_front` pose。

TartanGround 是 synthetic ground-robot proxy。通过最多证明 fixed-height geometry
mechanism，不证明眼镜佩戴、人体通行、真实事件、默认 App、生产或安全。

## 冻结算子与评价

候选实现逐常量继承
[`KNOWN_CAMERA_HEIGHT_GROUND_SCALE_R0`](KNOWN_CAMERA_HEIGHT_GROUND_SCALE_R0_PROTOCOL_2026-08-04.md)：

- lower ROI `0.55`、stride `4`、RANSAC `240`、seed `1729`；
- minimum candidates/inliers/fraction `100/80/0.08`；
- `abs(n_y) >= 0.55`、normalized plane residual `<=0.035`；
- `s = robot_height / h_rel`，scale `[0.25,4.0]`；无 offset、无分带尺度、无 smoothing；
- 同一三带 clearance 与 UNKNOWN；
- fresh gate 仍为 coverage `0.60`、MAE `0.25 m`、agreement `0.90`、false-clear
  `0.05`、temporal delta `0.15 m`、scale median/p90 relative error `0.10/0.25`，并要求
  至少 `3/4` parents 同时 MAE 优于 raw DA 且 false-clear 不恶化。

TartanGround local camera 为 NED pinhole：`640x640`、`fx=fy=320`、`cx=cy=320`，depth
表示 local forward；送入 optical-depth 算子前固定重排
`[forward,right,down] -> [x=right,y=down,z=forward]`。该坐标链继承既有 source
reprojection 证据，不搜索轴或符号。

## Metadata-only 资格与选择

权威 provider 固定为 `theairlabcmu/TartanGround` revision
`388faf9c800568cfc6828fa47e063f8369397eb3` 及既有 archive URL map SHA-256
`C3961C4C32F16AF040745681E0A8CED4B9DCA37BF96BFB11F1CB71A6FA2EE957`。

先排除所有在 base、expansion、outcome-unseen transfer 中已经打开 payload/outcome 的
21 个 environments。对剩余 P1000 parents，只允许下载 `metadata.zip` 并检查：

- finite `robot_height` 在 `[0.80,2.20] m`；
- `time_step == 0.1 s`；
- `pose_lcam_front.txt` 为至少 60 行、每行 7 个 finite 值；
- metadata `num_poses` 与 pose 行数一致；
- archive map 同时存在 `image_lcam_front.zip` 与 `depth_lcam_front.zip`。

资格检查不得读取 RGB、depth、seg、DA 或任何效果 outcome。eligible parents 按
`sha256("KNOWN_HEIGHT_TARTANGROUND_R0|environment")` 升序选前 4 个，environment 是
独立单元；选择后不得替换。

每个 parent 固定等间隔取 60 个 frame ids。media 只允许抽取这些 `lcam_front` RGB/depth
成员；不能按内容、地面覆盖或候选结果换帧。source metric depth 是 comparator/truth，
`robot_height` 是 synthetic fixed-height input；seg 不进入候选主臂。

## 终态

- metadata 不足 4 个 eligible environments：
  `TARTANGROUND_FIXED_HEIGHT_SOURCE_NOT_EVALUABLE`；
- payload/坐标/共同帧资格失败：
  `TARTANGROUND_FIXED_HEIGHT_COHORT_NOT_EVALUABLE_NO_REPLACEMENT`；
- ground opportunity coverage 不足：
  `TARTANGROUND_GROUND_SCALE_OPPORTUNITY_INADEQUATE_NOT_EVALUABLE`；
- opportunity 充分但效果门失败：
  `TARTANGROUND_KNOWN_HEIGHT_GROUND_SCALE_NOT_SUPPORTED_STOP`；
- 全门通过：
  `TARTANGROUND_FIXED_HEIGHT_SCALE_MECHANISM_SUPPORTED / REAL_WEARABLE_NOT_EVALUABLE`。

失败后禁止修改 source、height range、frame count/sampling、ROI、RANSAC、residual、scale、
clearance 或效果门救援。下一变量必须另立协议。
