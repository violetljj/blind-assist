# USTRF-SC 临时手持刚体标定执行单

状态：可选的准备采集；不是通过的 calibration manifest。
临时假设：当前已真机审计的 SM-S9280 以手持方式使用。本假设只建立 `handheld-device-body-v1` 的刚体坐标，不代表人体胸前或眼镜佩戴坐标，也不授权生产几何。

## 坐标与适用条件

| 字段 | 临时值 | 约束 |
| --- | --- | --- |
| `cameraFrame` | `arcore-camera-v1` | 后置相机；相机切换、分辨率/配置变化即失效 |
| `bodyFrame` | `handheld-device-body-v1` | 手机本体刚体，不是人体参考点 |
| `calibrationId` | `sm-s9280-handheld-r1` | 换机、换壳、镜头附件或安装方式改变后必须新建 |
| 使用范围 | isolated benchmark | 不写入默认 App、风险格或反馈链 |

在本执行单中，手机 body axes 以设备背部中心为原点：`+x` 为屏幕朝上时设备右侧，`+y` 为屏幕朝上时设备顶部，`+z` 垂直离开屏幕面。后置相机相对该 body 的完整变换必须由实测填入；不得假设 identity 或仅填 yaw。

## 最小现场物料

1. 打印 [A4 20 mm 棋盘格标定靶](../../../artifacts.local/calibration/ustrf-sc/ustrf_sc_checkerboard_a4_20mm_r1.pdf)，并参照同目录的 `ustrf_sc_checkerboard_a4_20mm_r1.manifest.json` 核对 SHA-256、方格规格和 100 mm 尺寸线。必须 A4、100% 打印、关闭适应页面；若实测线不是正好 100 mm、靶被裁切/弯折/反光，则拒绝使用。
2. 同一手机、相同保护壳和相同后置相机配置。
3. 可由第二人复核的原始 artifact 存储位置；不把原始画面写入仓库。
4. 安全、静止环境；本轮不测试导航、不让用户依照应用输出行动。

当前产品路线不以本执行单作为手持手机阶段的前置条件；手机可继续做 reference-free shadow。只有用户选择开启手机几何实验时才需要完成本单。未来眼镜设备也不能复用本单，见 [设备阶段策略](USTRF_SC_DEVICE_PHASE_POLICY.md)。

## 采集顺序

1. 记录设备、壳体、相机配置、参考物编号与 `calibrationId`。
2. 在至少 5 个距离/角度桶中采集 30 个以上有效样本；记录每个样本的 source-frame、depth/confidence freshness、标定板观测与失败原因。
3. 仅以 depth timestamp 和 confidence timestamp 同时等于 source frame timestamp 的样本计算 depth registration 误差。
4. 在两次拆装/复装后重复测量完整 camera-to-body translation + quaternion，计算平移和旋转 repeatability。
5. 对 raw artifact 与计算脚本生成 SHA-256 manifest，由与采集者不同的复核者核对。
6. 将汇总填入 `UstrfCalibrationTrialEvidence`；只有 verifier 返回 `Available`，才可将对应的内参/登记/完整外参作为后续 shadow 的实验输入。

## 禁止事项

- 不将 ARCore `TRACKING`、短窗稳定的 image intrinsics 或 raw depth candidate 填作外参/登记真值。
- 不将手机刚体标定推广成胸前或眼镜坐标；那两种安装形态需单独 `bodyFrame` 和新 manifest。
- 不以单人自评替代独立 review，不保存/共享原始影像到仓库，不在缺失 artifact digest 时填写 `independentReviewApproved=true`。

完成本执行单后，仍须经历受控几何与动态事件 shadow，不能直接接入默认助盲体验。
