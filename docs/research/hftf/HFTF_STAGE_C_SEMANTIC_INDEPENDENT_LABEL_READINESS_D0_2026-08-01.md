# HFTF Stage C semantic-independent label readiness D0

日期：2026-08-01

状态：`FROZEN_AFTER_CONSUMED_CALIBRATION_BEFORE_FORMAL_D0_REPORT`

## 1. 目的与诚实边界

C0.1 只证明两条 EgoWalk RGB/depth/pose transport 可用，不能直接训练 student。
D0 先回答：不用 semantic class 或 annotation，仅从 metric depth 能否生成稳定的
ground-plane、分段 support height、`KNOWN/UNKNOWN` 与少量自然 geometry-proxy
opportunity。

两条 source 已是 consumed Development calibration。冻结前打开了每条 32 个均匀 depth
frame、RGB/depth montage，以及两个室外 proxy frame，用于把 reader 做到基本合理。
因此 D0 不声称 fresh、Confirmation、prevalence 或效果；formal 报告只检验冻结后的
determinism/readiness。

## 2. 校准得到的关键约束

底半图受 camera-height 约束的 RANSAC 在室内走廊和室外道路预览上均能恢复地面平面。
但相机高度约 1.3 m，当前成像几何看不到约 1.2 m 内的脚下地面，所以该区域必须
`UNKNOWN`，不能用平面外推冒充观测。

可观测 section 固定为 `1.4/1.8/2.2/2.6/3.0 m`，方向固定
`-30/-15/0/+15/+30°`。每个 section 从高度直方图的 dominant mode 拟合局部平面，
且局部 normal 必须与 ground normal 在 30° 内；这会拒绝墙面等 vertical mode。

校准中，加入该 orientation gate 后室内四个 wall-derived 假台阶降为 0；室外剩余
两个 proxy 对应花坛/路缘方向。它们只能叫 foot-risk geometry proxy，不是 hazard truth。

## 3. 冻结 mechanics

depth 按官方 `gray16le mm -> m / zero -> UNKNOWN` 解码，使用 exact RGB-aligned
intrinsics 投影到 OpenCV camera frame。正式帧按 `0,5,10,...,last` 选择，即 1 Hz；
selection 不读 RGB/depth outcome。

ground plane 使用 deterministic 256-iteration RANSAC：

- bottom-half、水平 5%–95%、stride 4、depth `.25–20 m`；
- normal 与 camera `+Y` 至少 50° 内；
- plane distance 与 source camera height 相差不超过 `.35 m`；
- `.04 m` inlier、至少 200 点，胜出后 SVD refit。

每个方向至少 4/5 horizontal support sections 才是 known。相邻 section rise
`>.18 m` 或 drop `<-.15 m` 才输出 risk proxy。缺点、近场和无有效平面全部 UNKNOWN。

## 4. 顺序门

1. exact source/decoder/七个 structural canaries；
2. 每 source plane known `>=.95`、median inlier fraction `>=.40`、height error
   median/P90 `<=.25/.30 m`；
3. 每 source known direction fraction `>=.70`、known no-risk directions `>=50`；
4. cohort 至少 2 个 risk-proxy cells、2 frames、2 directions；
5. 第二次相同输入的规范化报告 payload byte-exact。

顺序终态从 mechanics、plane、profile、opportunity fail closed；全过才是
`D0_SEMANTIC_INDEPENDENT_LABEL_READINESS_SUPPORTED`。

## 5. 权限

D0 成功也只允许冻结 fresh-source label opportunity + student canary protocol。
它不授权 fresh acquisition execution、teacher dataset 生成、student training/effect、
主线、Android/App 或安全/产品 claim。

机器可读真源：
[HFTF_STAGE_C_SEMANTIC_INDEPENDENT_LABEL_READINESS_D0_2026-08-01.json](HFTF_STAGE_C_SEMANTIC_INDEPENDENT_LABEL_READINESS_D0_2026-08-01.json)
