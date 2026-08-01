# HFTF Stage C fresh foot-ground student canary E0

日期：2026-08-01

状态：`FROZEN_BEFORE_FRESH_RGB_DEPTH_OR_GEOMETRY_LABEL_OUTCOME`

## 1. 研究问题与上限

E0 检验：在 source/session 隔离、学生只能读取当前及过去 RGB 的条件下，
history-RGB + current/future teacher labels 是否比 single-frame future 与
history current-only 两个直接基线更能预测 fresh held-out 的短期 foot-ground
continuity geometry proxy。

这是原始 HFTF 的最小时间轴 canary，不是完整
`F(theta,rho,tau,z)`。它只预测 5 个方向、`[0,.4,.8] s` 的 foot-ground
known/risk-proxy 状态；body/head、距离分辨输出、人类危险真值和事件效用仍未评价。

## 2. Fresh cohort

在任何六条 fresh source 的 RGB/depth 或 geometry-label outcome 被读取前，从已绑定
SHA 的 95 条 metadata-healthy inventory 中：

1. 排除 D0/D1 已消费的两条轨迹；
2. 按 `pose+RGB+depth total bytes`、trajectory ID 升序；
3. 逐条保留尚未出现的 recording date；
4. 前六条按位置固定为 `4 train / 1 dev / 1 heldout`。

| role | trajectory | date | rows | total bytes |
| --- | --- | --- | ---: | ---: |
| train | `2024_10_12__16_04_11` | 2024-10-12 | 530 | 133,942,728 |
| train | `2024_12_27__15_41_58` | 2024-12-27 | 657 | 152,272,089 |
| train | `2024_12_26__13_31_58` | 2024-12-26 | 703 | 158,303,165 |
| train | `2025_01_03__16_51_04` | 2025-01-03 | 705 | 159,463,904 |
| dev | `2024_11_15__16_32_22` | 2024-11-15 | 1,251 | 176,089,927 |
| heldout | `2024_09_26__15_36_54` | 2024-09-26 | 609 | 176,111,646 |

机器合同绑定每个 pose/RGB/depth 的 repo path、size 与 SHA-256。六个 recording dates
互异；selection 不能因媒体内容、label opportunity 或 student outcome 换样。

## 3. 顺序门

必须依次通过：

1. **source lock**：从 inventory 独立复算相同六条与角色；
2. **media transport**：每个精确文件 size/hash、完整 decode、pose/RGB/depth
   frame count 和物理 timebase 通过；
3. **teacher mechanics**：每 source 至少 80 anchors，plane known `>=.95`、
   history-speed eligible `>=.95`、每 future horizon candidate known `>=.70`，
   known loss 与 UNKNOWN→SAFE 为 0；
4. **role opportunity**：train/dev/heldout 分别具有预冻结的 risk/no-risk 支持。

任何前门失败都停止，不训练模型。特别地，heldout 必须至少有 2 个 risk-proxy cells、
2 个 anchors；否则终态为 `E0_FOOT_GROUND_STUDENT_CANARY_NOT_EVALUABLE`，不得换一条
更有利 heldout。

## 4. Teacher 与输入防泄漏

每个 anchor 的学生输入固定为 `anchor-2, anchor-1, anchor` 三帧 RGB（`.4 s`
history）。teacher 可使用 anchor、`+2`、`+4` depth/pose，严格复用 D0/D1 mechanics；
future pose 只重投影 observation，不能进入学生或选择 origin/direction。

全部 `5 directions x 3 horizons` 都保留在 known-head 分母中；UNKNOWN 不可从分母
删除。risk loss/metric 只在 teacher-known cells 上计算。semantic class、annotation、
future RGB/depth/pose 均不进入 student。

## 5. 三个等预算 arms

三臂共用冻结的 torchvision MobileNetV3-Small ImageNet1K V1 encoder、三 slot
编码和同一 1728→256→两头网络；encoder 冻结，参数量必须完全相同。

- `SF_FUTURE`：三个 slot 都复制 current RGB，监督 current/.4/.8；
- `HIST_CURRENT`：输入三帧 history，只监督 current，评价 future 时复制 current
  概率；
- `HIST_FUTURE`：输入三帧 history，监督 current/.4/.8。

预训练权重 SHA-256 为
`047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`。
三个固定 seed、50 epochs、AdamW 与阈值网格全部写入机器合同；heldout 不参与训练、
阈值或架构选择。

## 6. 成功与停止

co-primary 是 heldout `.4/.8 s` risk macro-F1 的均值。`HIST_FUTURE` 三 seed 中位数
必须同时超过两个基线至少 `.03`，每个 seed 对两基线均为正增量；current risk 与
known macro-F1 相对最佳基线不得低于 `-.02`。

成功终态也只到
`E0_FOOT_GROUND_TEMPORAL_STUDENT_CANARY_SUPPORTED`：单一冻结 heldout trajectory
上的 geometry-proxy agreement canary。它没有 source-level precision/generalization
能力，不授权完整 HFTF、研究主线、Android/App 或安全 claim。

若失败，关闭这一 foot-ground temporal student formulation；不得在 heldout 上调参。
任何新 formulation 必须使用新的 fresh dev/heldout sources。

机器可读真源：
[HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.json](HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.json)
