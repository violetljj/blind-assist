# HFTF Stage C D2.1 signed-clearance transport 定义澄清

## 冻结结论

在 mechanics 实现前发现两处会改变结果的定义冲突，因此暂停实现并新增本澄清，而不修改
已经冻结的 D2 文档。冻结时只打开了 metadata（ID、fps、intrinsics、对象 receipts）；
RGB、mask、depth、pose 内容与任何 geometry/effect outcome 均未打开。

本澄清只解决两点：

1. exact G0 的 point population 是否允许预先过滤 field-domain 外点；
2. ground-aligned SE(2) 的 yaw 符号、投影轴与 predicted basis 公式。

其他 source、opportunity、metric、effect、terminal 与 no-retuning 规则全部不变。

## 冲突一：忠实 exact G0

D2 原文一处要求 exact hash-bound G0 field，另一处却写域外点从 predicted field
排除。两者不兼容：exact G0 会把全部有限、语义准入的 stride4/offset2 obstacle
points 送入每个 cell 的 closed-box proxy；不属于该 cell 的点为 `inside=false`，
产生正 SDF，并继续参与 second-smallest order statistic。

本澄清冻结为 exact G0 优先：

- 不按全局 theta/distance domain 预过滤 obstacle points；
- 每个 cell 单独判 longitudinal/lateral/height membership；
- nonmember 保留为正 absolute closed-box SDF，零点使用朝正无穷的 `nextafter`；
- 对全部 admitted points 的 per-cell signed proxy 取第二小，再 clip；
- 不创建新的 D2 field definition。

## 冲突二：几何 yaw 与 predicted basis

SANPO pose 固定为 `xyzw` quaternion，OpenCV camera point 使用
`p_world = R_xyzw @ p_camera + position`，camera forward 为 `[0,0,1]`。

对每个 anchor：

1. 只用 current mask/depth，按 hash-bound pose/ground authority 的 stride16、
   ground classes `{1,3,5,6,17,30}` 与 source-frame seed 拟合 current local plane；
2. `u` 为朝相机的 ground normal，`o` 为 current camera ground projection；
3. history/current camera forward 都投影到 **current** `u` 的切平面并归一化；
4. `delta = wrap[-pi,pi)(atan2(u·cross(f_history,f_current),
   clip(f_history·f_current,-1,1)))`；
5. `omega = delta / 0.4`，在 horizon `h` 用 Rodrigues 绕 `u` 将 current forward
   旋转 `omega*h`；
6. predicted right 始终为 `normalize(cross(predicted_forward,u))`；
7. predicted origin 为 current ground projection 加上 history→current camera
   translation velocity 在 current tangent plane 的投影乘 `h`。

任一 forward projection norm `<=1e-8`、plane 缺失或非有限量，走既有
`D2_NOT_EVALUABLE_TIMEBASE_OR_POSE_AUTHORITY_INADEQUATE_NO_SOURCE_REPLACEMENT`。
future pose 不定义 candidate origin、forward、right 或 up。

## Per-anchor leakage

每个 anchor prediction 只能读自己的 history/current pose rows 与 current mask/depth，
并在处理后续 anchor 前 durable 写入。后续 anchor 输入不得改变较早记录 bytes；
扰动该 anchor 的 future depth/mask/pose 也不得改变 prediction bytes。全部
`6×7×2=84` anchor-horizon records 封存后，才允许 truth join。

## 权限

本澄清仍不授权媒体采集、preprocessor、truth/effect 或 RGB student。只有独立定义复审
`CLEAR` 后，才可冻结新的 hash-bound one-shot media/mechanics implementation contract。
研究主线、App、Android、生产与安全权限均不改变。
