# RCLE Observable Support Recovery R0 Development Gate 结果

状态：`PASS / VALID / VALIDATION_REMAINS_UNAUTHORIZED`

设计锁：
`3fcc21e28ba84e18d10b1c236a9a0df167d2a6464ea5ebefcb52ce4395152bac`

候选：`OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0`

## 执行边界

本轮只实现设计锁中的唯一三帧 observable support manager，并在代码、16 项
测试、Python/OpenCV 环境、receipt schema 和固定输出位置锁定后，对
development seeds `2000–2019` 完整运行一次原 2520-trial matrix。未物化或
运行 sealed validation `3000–3019`，未抓取真实数据，未进入 Phase B、
Replay、Android、人体、安全或生产路径。

实现锁 SHA-256：
`a1dc1388ea6b6cb8ff7e7541da407cb827b6384678cdf46a097426a2111a5497`

固定输出：
`artifacts.local/evidence/rcle_observable_support_recovery_r0/development_gate_r0/`

## 锁前验证

- 16 项单元/边界测试全部通过；
- 覆盖完整候选双跑确定性、poison oracle 字段防火墙、完整 7×7 双线性
  median-centered photometric、field-exit 优先级、新生失败不冒充遮挡、
  4×4 确定性补点；
- support `<12` 与 hull `<0.10` 继续保持 `NOT_EVALUABLE` 负回归；
- development inventory 固定为 clean 1680、stress 840、合计 2520。

## Development 结果

- planned/actual：`2520/2520`；
- evaluable/not evaluable：`2518/2`；
- clean coverage：point `1.0`，cluster bootstrap lower `1.0`，
  worst cell `1.0`；
- stress coverage：point `0.997619`，cluster bootstrap lower `0.992857`，
  worst cell `0.95`；
- clean rotation yaw/pitch、clean rotation roll、clean closing、FPS、
  stress 和 coverage 六个原组件全部 `PASS`；
- 三个 stress profile 均 `PASS`；
- 原汇总 verdict：`PASS`，`scientific_gate_pass=true`。

运行时间仅为当前 host、4-worker contention 下的分析性记录，不是 Android
或 Kill Gate：total median `196.938 ms/pair`，p95 `354.229 ms/pair`。

## Receipt

receipt SHA-256：
`93b4c9244e9ef3bd11e8ab3557bfda0ad6dd6cd324116dbd05898f71b5214e3c`

独立重算：
`VALID`，2520 trials，终态
`DEVELOPMENT_GATE_PASS_VALIDATION_REMAINS_UNAUTHORIZED`。

## 权限与下一边界

该 PASS 只产生 synthetic development evidence，不是 sealed validation、
真实数据、Phase B 或生产权限。候选代码、环境、tests、schema、输出和
development receipt 现保持冻结。任何 sealed validation 动作都必须成为
另立的明确授权，并在独立上下文对 `3000–3019` 一次性完整运行；在此之前
不得物化验证帧或读取验证结果。
