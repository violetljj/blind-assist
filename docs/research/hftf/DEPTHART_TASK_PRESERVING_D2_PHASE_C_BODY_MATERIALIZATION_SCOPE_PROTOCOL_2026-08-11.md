# DepthART D2 Phase-C exact-eight body materialization scope

状态：`PRE_OUTCOME / FROZEN / AWAITING_EXPLICIT_SOURCE_SCOPE`

Phase-C 只对 D2R1 已锁的 4 TRAIN + 4 sealed DEVELOPMENT identity 及其各 300 个 exact
frame stems 物化源文件。每身份请求 `lowres_wide.zip`、`lowres_wide_intrinsics.zip`、
`lowres_depth.zip`、`confidence.zip`；32 个 ZIP body 精确总量为 `5,281,655,713` bytes，
激活上限冻结为 `5,282,000,000` bytes。trajectory 复用本地 Phase-A 锁，不再 GET。

物化只允许安全枚举 ZIP、提取 exact members 并记录 CRC/bytes/SHA。不得解码图像、重新选择
窗口、派生 truth、运行模型或训练；sealed DEVELOPMENT 源文件保持不可解码，直到训练完成并锁定
唯一 head hash。该范围不能从 RGB HEAD 授权自动继承。
