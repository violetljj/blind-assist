# P1-A4 online TAPIR outcome-blind selection

状态：`PYTORCH_ONLINE_BOOTSTAPIR_SELECTED / PRIVATE_TRUTH_UNREAD / PERFORMANCE_NOT_RUN`。

冻结协议 commit `611aa2dc` 之后才执行 selection。第一候选 official Google DeepMind PyTorch Online
BootsTAPIR 满足 sequential recurrent API、任意 frame-0 points、tracks + occlusion + expected-distance、官方
checkpoint、Apache-2.0 license 与本机 CUDA mechanics canary，因此按冻结顺序立即停止 selection；JAX fallback、
CoTracker、TAPNext/TAPNext++ 与 Cutie 均未下载、未 smoke、未形成 performance arm。

固定身份见 [`machine-readable receipt`](P1_A4_ONLINE_TAPIR_SELECTION_RECEIPT_2026-08-21.json)：upstream commit
`c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`，source manifest SHA-256
`927750992e5c5b6306154b891b1e47dcca47b6fa3ed18236a2c81f28a3fccb15`，checkpoint SHA-256
`87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`。Constructor 与 official live-demo
postprocessing 已逐字绑定。机械 canary 的 25-point track/occlusion/expected-distance 均为 finite，峰值 CUDA allocation
`325,528,576 bytes`，wall time `2.767 s`。

本记录只建立 interface/runtime capability，不是 ADT performance evidence。下一步只允许使用同一 source、checkpoint、
runtime、固定 aggregation 与 frozen evaluator 进行一个 P1-A4 capability probe；不得再选模或调 threshold。
