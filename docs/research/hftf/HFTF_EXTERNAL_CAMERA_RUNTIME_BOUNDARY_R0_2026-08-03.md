# HFTF external-camera runtime boundary R0

日期：2026-08-03

决策：`RGB_ONLY_RUNTIME_CORE / DEPTH_TEACHER_ONLY / D45_NOT_A_DEPLOYMENT_CORE`

## 为什么现在改

预期最终输入来自普通外接 USB、Wi-Fi 或眼镜摄像头。此类视频流不能默认进入手机
ARCore session，也不能默认提供 raw depth、hardware depth、稳定 IMU 外参或手机本机
camera id。把 D44 的 metric-history ceiling 直接落实成 ARCore runtime dependency，
会让候选路线在目标硬件上先天不可部署。

D45 因此只保留为一次有价值的 source-feasibility 诊断：它证明当前手机 context 也无法
稳定提供 fresh raw depth，同时暴露了最终硬件形态与该 source 路线不一致。它不再是
HFTF 的下一工程主轴。

## 最小运行时合同

候选在线核心只允许必需输入：

- causal RGB frames；
- source-monotonic frame timestamp；
- 冻结的 camera profile：分辨率、方向、裁剪/缩放和可用时的 intrinsics；
- 模型自身 causal state。

以下均不得成为核心必需输入：

- ARCore、raw/automatic depth 或 hardware depth；
- YOLO detection、person track 或语义类别；
- phone CameraX camera id；
- IMU、SLAM、VIO 或外接相机到手机的固定外参。

IMU、检测、单目深度或校准几何可以作为独立 baseline/ablation；缺失时核心仍必须产生
同一 HFTF field 语义或明确 `UNKNOWN`，不能切换成另一套产品行为。

## Teacher 与 student 的新边界

训练/标签侧仍可使用 source-authoritative depth、pose、future frames、人体轨迹与几何
包络。它们只产生 `risk/known` teacher target，不进入 student inference graph。

下一科学问题恢复为候选章程的原问题：

> 在外接相机式 human-egocentric source 上，causal RGB-history field student 能否在
> 相同 backbone/预算下超过 current-frame RGB baseline，并保留跨 camera/source
> 最差组非劣？

这不是重跑已关闭的 THOR whole-frame dense-flow、YOLO box-flow slot 或 JRDB 2D
linear metric residual。新实验必须使用新的 human-egocentric source/data-role boundary，
直接预测 action-agnostic future layered field；不得把恢复 metric depth 设为成功前提。

## 最小评价与停止

在任何新模型设计前只冻结四类比较：

1. equal-budget current RGB；
2. equal-budget causal RGB history；
3. teacher ceiling（只作 information ceiling）；
4. camera/source-held-out worst group。

主效应仍是 future field/event 增量，不是 depth error、mAP 或 teacher agreement。history
不能超过 current，便关闭该新 source 上的 formulation；不得靠 ARCore、metric-depth
fallback 或更大模型回救。

这个边界不授权训练、数据下载、App 接入或主线晋级；它只撤销一个与目标硬件不一致的
运行时前提，并保持主线不变。
