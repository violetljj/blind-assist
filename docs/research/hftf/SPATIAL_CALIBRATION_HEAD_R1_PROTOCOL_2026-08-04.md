# Spatial Calibration Head R1 protocol

日期：2026-08-04

状态：`FROZEN_BEFORE_COHORT_ROSTER_OR_MEDIA_ACCESS`

## 决策

R1 是三条已消费 metric-depth 路线之后的新窄实验。它固定读取 DA V2 Metric Hypersim
ViT-S 第 11 层空间 patch tokens、raw DA 三带 clearance、区域深度统计和归一化内参，
用一个左右中共享小头输出每带 `scale / offset / confidence`：

```text
clearance'_b = exp(log_scale_b) * clearance_DA_b + offset_b
confidence_b < 0.5 -> UNKNOWN
```

DA V2 的 patch stride 是 `14`，所以冻结的是 transformed
`H/14 × W/14 × 384` patch map，不使用示意性的 `H/16`，也不再只读 CLS。

共享 MLP 为 `781 -> 12 -> 3`、SiLU，精确 `9,423` 个可训练参数。781 维输入由
384 维区域 token 均值、384 维区域 token 总体标准差和 13 个深度、梯度、有效率、
位置、内参标量组成。比例限制到 `[0.25, 4]`，offset 限制到 `[-3, 3] m`。

## 数据与真值防火墙

主数据只使用 ARKitScenes raw。先根据官方 CSV 做 metadata-only roster，再固定 24 个
互不重叠的 `visit_id`：16 train、4 validation、4 sealed；每个 visit 只取一个 video，
每个 video 固定等间隔 150 个 RGB/depth/confidence 匹配帧，共 3,600 帧。parent 是
`visit_id`，帧不是独立样本。

绑定 CSV 中 `visit_id=381879` 同时出现在官方 Training 和 Validation；它与 `NA` visit
全部排除。角色按 train→validation→sealed 顺序选择，后一个角色必须排除前面已选 parent，
所以官方 fold 名称本身不被当作 parent-disjoint 证明。

训练和 validation 真值是 ARKitScenes `lowres_depth` 的 confidence-2 sensor returns；
Metric3D 不充当教师、真值或运行时输入。sealed 的 depth、confidence、trajectory、
DA 输出和 outcome 在 activation receipt 之前关闭。唯一例外是 roster 与许可 receipt
之后，由隔离工具只读取 sealed `lowres_wide` RGB，完成 hash 与 pHash/crop/mirror
capture-identity 审计；不得读取语义、预测或米制真值。跨角色候选必须完成裁决；任何
`SAME_CAPTURE`、`UNKNOWN`、分歧或未审保持 `HOLD_COHORT_INDEPENDENCE`。

SANPO 只有在另立未消费 session 身份和 source-native sparse-depth 合同后才能做外部
助盲域验证；ARKitTrack、DIODE 等主 sealed 决策后再开；Hypersim 不进入 R1。

## 固定训练与比较

四个主臂固定为：

1. raw DA V2；
2. train-parent 常数 global affine；
3. 第 11 层 CLS、ridge `lambda=10`、770 参数 global head，在每个 R1 train split 重拟合；
4. 9,423 参数 spatial head + confidence/UNKNOWN。

常数和 global head 的训练标签使用固定 global affine：在 confidence-2 sensor depth 上
stride 4 采样、至少 500 对，执行一次 OLS→3×1.4826×MAD inlier→OLS；inlier fraction
至少 0.5、slope `[0.25,4]`、inlier median residual 不超过 0.25 m。

另有一个 9,410 参数无 confidence spatial 消融，只诊断拒答机制，不能授权晋级。

损失固定为 Huber clearance、1.0/1.5/2.0 m occupancy BCE、occupied 正例权重 3 的
false-clear 非对称惩罚、confidence correctness BCE 和 90% coverage hinge。AdamW、
`lr=1e-3`、`weight_decay=1e-4`、80 epochs、batch 64、seed `20260804`；无 scheduler、
augmentation 或 early stopping，只取 epoch 80。

## 评价与晋级

16 个 train parents 固定分成四折，每折 12 train / 4 held-out。spatial 必须至少 3/4
折同时满足 MAE 优于常数且 false-clear 不差于常数。随后用全部 16 train parents
拟合一次，在四个 validation parents 上必须通过：

```text
known coverage >= 90%
clearance MAE <= 0.25 m
envelope agreement >= 90%
false-clear <= 5%
temporal delta MAE <= 0.15 m
confidence ECE <= 0.10
```

晋级门按 `visit_id` 无权 parent-macro 裁决；pooled 帧指标只能诊断。validation 通过且
实现锁、预测 receipt、sealed exclusive activation receipt 全部 durable 后，才允许
一次性打开四个 sealed parents。

sealed 还必须通过五项门、ECE、MAE 优于常数、false-clear 不差于常数和冻结的相对
validation non-degradation 界限。通过才授权手机 shadow；有效失败则关闭本轮纯 RGB
尺度扩展，并另冻同摄像头、同 session 的 spatial-vs-VL53L5CX E 臂协议。

完整数值、选择算法、终态和禁止项以同名 JSON 合同为机器权威。

## 官方来源

- <https://github.com/apple/ARKitScenes>
- <https://github.com/apple/ARKitScenes/blob/main/DATA.md>
- <https://github.com/apple/ARKitScenes/blob/main/LICENSE>

协议绑定官方 main commit `7283761bf26c27570ec59a5dc0f8686fbff07726`；媒体下载前
仍须生成本地许可复核 receipt。
