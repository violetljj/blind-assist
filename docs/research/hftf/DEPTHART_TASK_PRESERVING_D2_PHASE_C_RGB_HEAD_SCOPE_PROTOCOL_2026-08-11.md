# DepthART D2 Phase-C RGB HEAD scope

状态：`PRE_OUTCOME / FROZEN / AWAITING_EXPLICIT_SOURCE_SCOPE`

D2R1 已锁 4 个 TRAIN 与 4 个 sealed DEVELOPMENT identity。Phase-C 的下一步只对这 8 个
identity 的 `lowres_wide.zip` 发出 8 个 HEAD 请求，用于冻结 RGB body 的可用性和精确字节数。

本门不读取任何 RGB body，不重新请求 depth/confidence/intrinsics，不运行 base model 或 task
head，不训练，也不打开 D2 DEVELOPMENT 或 R2。HEAD 通过后仍需另立带精确 body 上限的
Phase-C source/training activation；本次 HEAD 权限不能自动继承为 GET 或训练权限。
