# DepthART D2R1 target-support window recovery protocol

状态：`PRE_OUTCOME / FROZEN / AWAITING_EXPLICIT_SOURCE_SCOPE`

D2 Phase-B 只得到 2/8 个 support-qualified identity，门限保持不变。主要失败集中在
首个固定 300-frame portrait window 缺少 clear/known denominator，而不是资产不可用；其中一个
session 只有 `clear=179`，另一个只有 `known=1734`，均按冻结门被拒绝。

D2R1 不降低阈值、不增加身份、也不读取模型。它只针对同一 16 个 session，重新请求完整
intrinsics/depth/confidence，在 pose-derived portrait continuous runs 中枚举所有 300-frame
window。每帧 sensor truth 最多解码一次，窗口计数用 prefix sum；每身份只取按时间排序的
第一个全门通过窗口。若少于 8 个身份通过，仍直接停止且不分配任何 TRAIN/DEVELOPMENT 角色。

冻结 HEAD 显示 depth+confidence body 共 `2,826,084,750` bytes，连同 intrinsics 的总请求上限
约 2.90 GB。该访问超出上一 receipt 的“exact first window”范围，因此必须获得新的显式授权；
授权前不得执行 HEAD、GET 或 window scan。R2、RGB、模型输出、性能、默认 App、production
与 safety 均不在范围内。
