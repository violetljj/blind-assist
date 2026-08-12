# DepthART-S D3R1 Phase-B execution stop

状态：`INVALID_INCOMPLETE / NO_SCIENTIFIC_TERMINAL / NO_SUCCESSOR`

本次已按用户授权启动冻结的 exact-32 / exact-64 Phase-B body 与 source-truth-support
执行，但在第 `2/32` 个身份、写入第二个 checkpoint 之前按 source-integrity hard gate
停止。冻结的 300-frame plan 包含 `42898216_694900.389`；producer 在
`lowres_depth.zip` 的 frame inventory 中找不到该 stem，抛出
`ValueError: missing selected depth frames: ['42898216_694900.389']`。

这不是科学 FAIL。协议要求 exact 300 stem coverage，且明确规定 transport、CRC、schema、
decode 或 coverage 错误属于 execution invalid/incomplete，不能记成 source-support
ineligible。只有完整处理 32 个身份后才允许计算 first-16。因此本次：

- 计划 `32` 个身份、`64` 个 source assets、`9,600` 帧；只形成 `1` 个完整 checkpoint；
- 第一个 checkpoint 为未通过 support，但不能把 partial `0/1` 提升成整体合格率或科学终态；
- 没有生成 scientific manifest/validation，没有评价 selection，也没有 first-16 lock；
- `scientific_terminal = null`，`selected_phase_b = null`，`next_gate = null`。

停止后的未绑定 operator 诊断只用于描述报告的 member-name 缺失：depth 与 confidence 两个 ZIP 都观察到
`16,106` 个 PNG stem、CRC probe 无坏成员，目标 stem 在两种 modality 中都不存在，而相邻
`...694900.373` 与 `...694900.406` 均存在。该观察没有绑定 probe implementation、stdout/log、
failure receipt 或 identity-2 body SHA，只作为 `UNBOUND_POST_STOP_OPERATOR_DIAGNOSTIC` 记录；
metadata-only inventory auditor 没有重新打开 archive 或读取正文，因此它不冒充独立科学结果。

当前 r0 root 保留且不可修改。因为第二个身份已有两个 source body 但无 checkpoint，resume
会被 orphan-inventory gate 拒绝；不得删 orphan 强行恢复、替换邻帧、重搜窗口、换用其余 21
个 identity、覆盖或同版本重跑。任何未来恢复都必须另行授权并另立版本/协议/新 root，且不是
本版本的自动 successor。

RGB、模型输出、TRAIN/DEVELOPMENT 角色、训练、Development outcome、R2、性能、默认 App、
production 和 safety 权限均未打开。

机器回执：[JSON](DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_EXECUTION_STOP_2026-08-12.json)。
