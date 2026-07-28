# RCLE rotation-compensation mechanism audit R1

状态：`complete / standalone rotation-compensation route stop supported`

本模块只回答 ADVIO 首轮补偿未改善的实现与机制原因。冻结协议是
[`RCLE_ROTATION_COMPENSATION_MECHANISM_AUDIT_R1_CONTRACT_2026-07-28.json`](../../../../docs/research/rcle/RCLE_ROTATION_COMPENSATION_MECHANISM_AUDIT_R1_CONTRACT_2026-07-28.json)。

稳定执行顺序：

1. 运行 `synthetic_direction_audit.py`，覆盖 yaw/pitch/roll、正负号及
   raw/correct/reverse；
2. 分别以 `runner.py --arm raw` 和 `--arm undistorted` 对同一 600 pair 运行；
3. 两个 arm 均为单进程连续执行，只允许一个 `PairState` 和连续三-pair streak；
4. 用 `analyze.py` 按冻结的 pose-only high/low 窗汇总。

新实现显式按 ADVIO 官方 `wxyz` 解码 pose。去畸变 arm 使用官方固定版本
`iphone-03.yaml` 的 pinhole radial-tangential 参数、相同 K 与原 720×1280 canvas，
先 remap、传播有效区域掩膜，再缩放至 0.5；越界 remap 区域不进入 support。raw arm
保持原图再缩放。最终版本是 `ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3`，变更见
[`IMPLEMENTATION_AMENDMENT_R3`](../../../../docs/research/rcle/RCLE_ROTATION_COMPENSATION_MECHANISM_AUDIT_R1_IMPLEMENTATION_AMENDMENT_R3_2026-07-28.json)。
两臂都不改 `0.01/s` 或三连续 pair 规则。

输出只能写入
`artifacts.local/evidence/rcle_rotation_compensation_mechanism_audit_r1/`。
本模块不计算 AUROC/F1，不产生性能、泛化、Android、产品或安全结论，也不访问
已预留的 ADVIO sequence 16 sealed session。

最终实现、A/B、pose-only 分层、连续状态和负结果见
[`RESULT_2026-07-28.md`](RESULT_2026-07-28.md)。
