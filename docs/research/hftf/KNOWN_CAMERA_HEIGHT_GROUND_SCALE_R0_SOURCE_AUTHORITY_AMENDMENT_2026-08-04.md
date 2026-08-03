# Known camera height ground scale R0 source-authority amendment

日期：2026-08-04

状态：`FROZEN_BEFORE_DA_OR_EFFECT_OUTCOME / AUTHORITY_DOWNGRADED`

## 修正

fresh roster 已锁定、媒体已下载，但尚未运行 DA、候选尺度或任何效果汇总。结构检查发现
ARKitScenes raw 提供 metric depth、confidence、intrinsics 和 camera trajectory，却不
提供逐像素 ground identity 或独立 floor-plane receipt。因此 roster 中的
`SOURCE_TRUTH_DERIVED_PER_FRAME_ORACLE` 过强，现降级为：

```text
SOURCE_DEPTH_CONFIDENCE2
+ OFFICIAL_TRAJECTORY_CAMERA_TO_WORLD
+ FROZEN_WORLD_VERTICAL_PLUS_Z
+ LOWER_ROI_HORIZONTAL_PLANE_PROXY
-> per-frame H_proxy
```

它只是重力约束的水平面 proxy；桌面、台阶平台或其他水平面仍可能被误认。它不得叫
真实地面高度，也不能替代卷尺测得的固定眼镜安装高度。

## 冻结 proxy reader

- trajectory 按官方 `TrajStringToMatrix` 语义解析：axis-angle 与 translation 先组成
  world-to-pose extrinsics，再求逆得到 camera-to-world；
- ARKitScenes 当前 source version 的 world vertical 固定为 `+Z`；不在四段中逐段搜索
  X/Y/Z 或符号；`up_camera = R_camera_to_world.T @ [0,0,1]`；
- pose 与 frame timestamp 的最大差 `0.05 s`，平局或超时拒绝；
- 只读 confidence `==2`、`0.25–6.0 m` sensor depth，stride `4`；
- lower ROI、最小 candidates/inliers/fraction 与 candidate operator 相同；
- 沿冻结 up 轴计算 offset，禁止 sign flip；只接受 `[0.80,2.20] m`；
- offset histogram bin width `0.04 m`，选最高计数 bin；支持带为 mode center `+/-0.08 m`；
- 最终 height 取 support median，normalized median residual 必须 `<=0.035`；
- 每个 parent 的 150 帧中至少 `90` 帧得到 proxy height，否则整个 cohort 终止
  `HOLD_SOURCE_AUTHORITY_NO_REPLACEMENT`；不得更换 vertical、ROI、mode 或 source。

source qualification 允许读取 sensor depth/confidence/intrinsics/trajectory 和结构 hash，
但禁止运行或读取 DA、候选 scale、candidate clearance、raw-DA comparator 或 effect 指标。

## 新 claim ceiling

本 cohort 即使通过全部原效果门，也只能终止为：

`GRAVITY_PLANE_PROXY_SCALE_MECHANISM_SUPPORTED / TRUE_GROUND_AND_WEARABLE_HEIGHT_NOT_EVALUABLE`

原协议的真实固定高度条件性支持终态仍要求新的固定安装、卷尺高度与内参 receipts；本
amendment 不降低该要求。协议常量、4-parent roster、无替换规则和效果门均不改变。
