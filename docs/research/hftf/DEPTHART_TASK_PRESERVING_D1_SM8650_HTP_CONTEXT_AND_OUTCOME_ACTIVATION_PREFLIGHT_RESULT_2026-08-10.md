# DepthART task-preserving D1 SM8650/v75 device preflight result

状态：`PASS / DEVELOPMENT_OUTCOME_ACTIVATION_AUTHORIZED_NOT_STARTED / NO_TASK_OUTCOME`

本次使用 fresh ADB identity，而不是继承旧附件：`R5CX10M8Y8X / SM-S9280 / e3q /
SM8650 / Android 16 API 36 / 2026-07-05`，build fingerprint 为
`samsung/e3qzcx/e3q:16/BP4A.251205.006/S9280ZCS6DZG1:user/release-keys`。

exact SHA 的 `608×448` candidate DLC 已在 QAIRT `2.47.0.260601114230`、HTP v75
上完成 context 与执行前门：

- Attempt 1 因 HTP op-package registration 使用了错误的 `arm64/` 相对路径而 exit 13；
  日志保留，candidate/runtime/input/tolerance 均未修改。
- Attempt 2 使用同名 package 的 host `LD_LIBRARY_PATH` 与 DSP `ADSP_LIBRARY_PATH`
  解析，CPU/HTP package registration PASS，graph finalize status `0x0`，生成
  22,552,576-byte context（SHA-256 `FA5DC9DC...FC3D`）。
- direct DLC 和 saved context 各完成一次 inference；输出均为 finite float32
  `[1,608,448]`、1,089,536 bytes、SHA-256 `0A464746...8EF9`，两者 bit-exact。

PyTorch reference 与 HTP direct 的 raw-depth diagnostic 仍不一致：`max_abs=1.42328m`、
`mean_abs=1.06983m`、`rmse=1.08726m`。这保留并强化 strict G4-D 的既有负终态，
不能改写成 raw-depth parity PASS；同时 D1 是 task-preserving successor，最终判定必须由
冻结的 clearance、false-clear、false-block、temporal 与 geometry task gates 完成，不能用
该 diagnostic 直接替代 D1 task-quality 结果。

本次没有打开 ARKitScenes D1 task truth、reference/candidate outcome 或 R2 cohort，也没有
测量性能。唯一下一门是显式激活 D1 Development task-quality screen；该激活会首次消费
冻结的 8-session Development outcome，必须另有 activation receipt 后才能执行。
