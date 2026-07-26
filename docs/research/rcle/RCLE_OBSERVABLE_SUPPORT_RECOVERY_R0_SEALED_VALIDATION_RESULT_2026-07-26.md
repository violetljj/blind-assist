# RCLE Observable Support Recovery R0 Sealed Validation 结果

状态：`PASS / VALID / INDEPENDENT_SYNTHETIC_EVIDENCE_ONLY`

候选：`OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0`

## 冻结链

- 设计锁：`3fcc21e28ba84e18d10b1c236a9a0df167d2a6464ea5ebefcb52ce4395152bac`
- implementation lock：`a1dc1388ea6b6cb8ff7e7541da407cb827b6384678cdf46a097426a2111a5497`
- development receipt：`93b4c9244e9ef3bd11e8ab3557bfda0ad6dd6cd324116dbd05898f71b5214e3c`
- validation lock：`edf2bb62c1771b61fd39d0843f4b3201b57e38a6bc38f4e1b397e8ef74f054cf`

独立上下文在确认 19/19 candidate sources、6/6 validation controls、锁定环境、
development `PASS / VALID` 和唯一输出位置无漂移后，才物化 sealed validation。

## 唯一执行

- seeds：`3000–3019`
- matrix：原完整 2520 trials，仅替换 seed
- clean/stress：`1680/840`
- started/finished：`2026-07-26T14:45:20+08:00` /
  `2026-07-26T15:29:07+08:00`
- 实际运行次数：`1`
- patch、失败 seed 替换和 rerun：均未发生

初始命令观察窗口超时后，原父进程与 workers 继续运行；独立上下文只监视同一
run-state，没有启动第二次执行，也没有读取 partial metrics。

## 结果

- planned/actual：`2520/2520`
- evaluable/not evaluable：`2519/1`
- clean coverage：`1680/1680`，point `1.0`，
  cluster bootstrap 95% `[1.0, 1.0]`，worst cell `1.0`
- stress coverage：`839/840`，point `0.9988095238095238`，
  cluster bootstrap 95% `[0.9964285714285714, 1.0]`，
  worst cell `0.95`
- `clean_rotation_yaw_pitch`：`PASS`
- `clean_rotation_roll`：`PASS`
- `clean_closing`：`PASS`
- `fps_consistency`：`PASS`
- `stress`：`PASS`
- `coverage`：`PASS`
- scientific verdict：`PASS`

## Receipt

- sealed receipt SHA-256：
  `d10afb25cbe6bd8104b842adbd128b229804d5bdda0e8ff03d3954386806365c`
- receipt-validation 文件 SHA-256：
  `b3d4b26c06a06449b1368185b730c466e489589fa49286532fbefa2e59c23cf9`
- 独立重算：`VALID`，2520 trials

固定输出：
`artifacts.local/evidence/rcle_observable_support_recovery_r0/sealed_validation_gate_r0/`

## 终态与权限

终态：
`INDEPENDENT_SYNTHETIC_PASS_ONLY_PHASE_B_REMAINS_CLOSED_PENDING_SEPARATE_DECISION`

该结果只说明冻结的 synthetic observable support recovery 假设在 development
和独立 sealed validation 上均通过原门。它不证明真实 pose、rolling shutter、
真实遮挡、真实相机或人体有效性，也不授权真实数据、Phase B、Replay、
Android、告警、安全或生产路径。是否讨论 Phase B 必须成为新的独立决策；
本结果本身不自动扩权。
