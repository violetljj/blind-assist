# DepthART D2 Phase-C source materialization result

状态：`PASS / EXACT_EIGHT_SOURCE_MATERIALIZED / DEVELOPMENT_SEALED / NO_MODEL_OUTPUT`

Phase-C 对 D2R1 锁定的 4 TRAIN + 4 sealed DEVELOPMENT identity 下载了精确 32 个 ZIP，
总 body `5,281,655,713` bytes。每个 ZIP 的安全路径和全成员 CRC 已验证，再按冻结的各 300 个
frame stems 提取 RGB/intrinsics/depth/confidence，共 9,600 个文件、`255,202,648` bytes。

独立 validator 重算了 8 个 checkpoint 和全部 9,600 个文件的 bytes/SHA；TRAIN 与
`development_sealed` 路径完全分离。整个过程没有解码图像、派生 truth、运行模型、训练或访问 R2。

本结果只关闭 source materialization。唯一下一步是显式授权仅 4 个 TRAIN identity 的源文件解码、
冻结 saved-context DepthART base output 和 step-500 单 head 训练；4 个 DEVELOPMENT identity 必须继续
sealed，直到唯一 head hash 已锁。该训练 PASS 仍不等于 Development quality、R2、性能或产品权限。
