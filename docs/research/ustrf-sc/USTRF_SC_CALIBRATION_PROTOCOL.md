# USTRF-SC 设备几何校准与证据协议

状态：实验协议；不构成生产、医疗、导航或用户安全授权。
适用范围：独立 `:core:ustrf` 与 `:ustrf-shadow-benchmark` 的下一阶段；不接入默认 App。

## 1. 目的与边界

本协议为 `UstrfIndependentCalibrationEvidenceVerifier` 和 `UstrfMetricGeometryReceiptPromoter` 准备可审计的外部证据。它解决的是“允许实验性像素到身体坐标投影”的前置条件，不解决场景语义、地面分割、动态目标、风险策略或反馈。

协议不得把以下任一项当作通过：ARCore `TRACKING`、raw depth 存在、同一短窗内内参未变、单人自评、未复核的截图、或缺失原始证据的汇总数字。

## 2. 先决条件

1. 固定本轮设备、相机模组、壳体/支架和身体安装位置；任一改变都必须产生新的 `calibrationId`。
2. 先定义坐标：`cameraFrame` 为实际成像相机，`bodyFrame` 为固定刚体参考。没有实体固定 body frame 时，不得填写外参通过。
3. 自动设备采集数据、原始影像、标定板信息与机器记录放在受控的外部实验存储；仓库和 trace 只保存匿名汇总、manifest 路径标识与 SHA-256。
4. 采集 Agent 与验证 Agent 必须使用相互隔离的运行上下文；验证 Agent 要能访问原始 artifact 而不是只看到摘要。

## 3. 四类证据

| 证据 | 最小采集 | 输出指标 | 未通过时的动作 |
| --- | --- | --- | --- |
| 相机内参 | 至少 30 帧、5 个不同姿态/距离桶的平面标定目标 | P95 reprojection error（px）、内参版本 | 禁止构造 independently verified intrinsics receipt |
| depth-to-camera 登记 | 同一目标在相机图像与 raw depth 中都有可测参考点 | P95 registration error（m）、depth coordinate frame、transform ID | raw depth 只能保持 candidate/reprojected metadata |
| camera-to-body 完整外参 | 固定支架下重复装夹并测量平移与三维旋转 | translation/rotation repeatability、SE(3) quaternion | 不得以 yaw-only receipt 代替完整外参 |
| 独立自动复核 | calibration-auditor 模型/验证器核对 manifest digest、样本数、姿态桶、算法输出与失败样本 | review approval、agent ID、model/config/prompt/input hashes | 所有几何授权保持 unavailable |

代码的实验起始阈值是：样本数 `>=30`、姿态桶 `>=5`、内参 P95 `<=1.5 px`、登记 P95 `<=0.03 m`、外参平移 repeatability `<=0.01 m`、旋转 repeatability `<=1°`。这些是复现实验的入场阈值，不是安全性能或量产规格。

## 4. 现场流程

1. 由设备脚本记录设备型号、系统版本、相机/支架装配影像编号（不写入仓库）、`cameraFrame`、`bodyFrame`、采集 Agent 和时间。
2. 以固定标定板/参考刚体，在近、中、远距离以及不同俯仰/偏航角中采集；每个姿态桶至少保留一个有效样本与一个可追溯的失败原因。
3. 记录 raw depth 与 confidence 时间戳。仅当两者都等于对应 camera source timestamp 时，才计算登记误差；重投影 depth 必须计入拒绝数量，不得混入有效样本。
4. 至少重复一次拆装/复装后测量 mount transform；以完整 quaternion + translation 计算 repeatability，不能只记录 yaw。
5. 离线生成不可变 source artifact（包含原始证据引用、测量脚本版本、指标和失败样本索引），计算 SHA-256；敏感原始数据留在受控存储。
6. 由独立 calibration-auditor 模型与机械 verifier 审查原始 artifact 与摘要。只有哈希绑定的自动复核通过后，才可写 `independentReviewApproved=true`；失败或不确定即 unavailable，不转人工。

## 5. 回填模板

提交给 verifier 的字段以 `UstrfCalibrationTrialEvidence` 为准。下面只展示结构，数值必须由实测填入：

```text
calibrationId = "mount-<device>-<date>-r<N>"
cameraFrame = "<actual-camera-frame>"
bodyFrame = "<fixed-rigid-body-frame>"
cameraCalibrationVersion = "<intrinsics-version>"
sourceArtifactSha256 = "<64 lowercase hex>"
collectorId = "<automated acquisition agent/run>"
reviewerId = "<isolated calibration auditor model/run>"
independentReviewApproved = <true only after review>
sampleCount = <measured>
poseCoverageBins = <measured>
intrinsicsP95ReprojectionPx = <measured>
depthRegistrationP95ErrorM = <measured fresh pairs only>
mountTranslationRepeatabilityM = <measured>
mountRotationRepeatabilityDeg = <measured>
collectedAtNs = <monotonic receipt time>
validUntilNs = <explicit experiment expiry>
```

`Available` 只表示该 manifest 满足实验性证据合同。要构造后续的内参、登记和完整外参 receipt，还必须使 frame/version/TTL 与当前 capture 完全一致；`UstrfMetricGeometryReceiptPromoter` 将再次校验这些绑定。

## 6. 停止条件与复验

立即停止几何投影尝试并使 receipt 失效的情形包括：装配松动/更换、相机配置或分辨率改变、参考刚体变化、任何 digest 不匹配、复核撤回、样本覆盖不足、误差超过阈值、时间有效期到期，或后续 shadow 中出现 freshness/coordinate-frame 不一致。

通过本协议后仍需独立验证：逐像素 depth 置信校准、地面/台阶/头部几何、动态 TTC、热/时延、连续自动多模型事件参考和默认反馈链的独立安全门。上述事项全部完成前，USTRF 继续是 production-isolated experiment。

当前 SM-S9280 手持刚体的预采集约束单见 [临时手持执行单](USTRF_SC_PROVISIONAL_HANDHELD_CALIBRATION_RUNBOOK.md)。它不是 manifest，也不代替最终胸前或眼镜形态的单独标定。

无现成标定物时可使用仓库生成的 [A4 20 mm 棋盘格靶](../../../artifacts.local/calibration/ustrf-sc/ustrf_sc_checkerboard_a4_20mm_r1.pdf)。它有独立 SHA-256 manifest 和 100 mm 打印核验线；打印核验通过只意味着参考物尺寸可追溯，仍不等于 calibration manifest 或 production authorization。
