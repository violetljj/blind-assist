# DEPTHART_ADMISSION_R1 A3：ONNX/QNN deployment preflight

状态：`G3-A_EXPORT_PASS / G3-B_PARTIAL_PASS / G3-C_PRIMITIVE_REFERENCE_CONVERTIBLE / HTP_V73_AND_AARCH64_PACKAGE_COMPILED / RUNTIME_NOT_EVALUATED / G4_NOT_EVALUATED`

本轮检查导出链路、图改写与 QAIRT converter reachability；不产生完整 ONNX runtime parity、HTP 执行、Android 或生产证据。R0 结论与数据角色不变。

## 观察

- 安装 MSVC 2022、CUDA 12.8.93，并为 RTX 5060 / SM 12.0 编译官方 Selective Scan CUDA extension；核心测试 `9/9` 通过。
- 使用 PyTorch legacy custom-symbolic exporter 成功生成 31,985,722-byte metric S448 ONNX：输入 `image,K`、输出 `depth`、3555 nodes、5 个 `com.depthart::SelectiveScan`，SHA-256 `06A0C059...78C`。
- QAIRT `2.47.0.260601` 实际已存在于 `E:\codex-tools\qairt`，此前只是未加入 PATH。原图转换首先在 10 个固定 batched-linear Einsum 上失败。
- 将两种 Einsum 逐项等价改写为 reversed-input MatMul 后，QAIRT 继续前进，但在 metric Camera Embedder 的 ONNX `Acos` 上因没有 translation 而停止。
- 将 Camera Embedder 外提为 host 计算的四级 `camera_prompt_*` 输入；PyTorch prompt parity `max_abs=0.0`，外提图 2823 nodes、`Acos=0`、`SelectiveScan=5`。再应用 Einsum→MatMul 后，QAIRT 已真正触达 5 个 SelectiveScan。
- 正常转换的首个明确停止点仍是 `onnx_selectivescan`：`No translation registered for op type onnx_selectivescan`。随后 dry-run 还枚举出 `Erf`、`LayerNormalization`、`Resize`、`ConstantOfShape`、`Expand`、`Where`、`Mod` 及若干 unsupported attributes；这些是待清理/复核候选，不等同于已经证明的转换停止点。
- 第一档 Graph Hygiene Pass 已完成：移除 123 个 `BatchNormalization.training_mode=0`、108 个 `Reshape.allowzero=0`，以及零 padding AveragePool 的 4+4 个默认属性；节点数保持 2823。重跑 normal conversion 后 frontier 未漂移，仍首先停在 5 个 `onnx_selectivescan`。
- Fixed-S448 static-shape pass 已将 6 个目标全为 `[1,…]` 的 no-op Expand 旁路，并把 4 个常量 `2 mod 3` 折叠为 Constant；死代码清理后节点数从 2823 降至 2723。normal frontier 仍为 SelectiveScan；新 dry-run 候选收敛为 `Erf×27 / LayerNorm×23 / Resize×13 / SelectiveScan×5`。LayerNorm、Resize、Erf 继续保持原图。
- 为 `com.depthart::SelectiveScan` 建立 QAIRT OpDef XML，并编译仅提供 shape/type inference 的 converter library。它验证 7 输入、1 输出、2 个 BOOL 参数和 rank-3 输出合同，严格复制输入 `u` 的 shape 与 dtype；它不是 QNN/HTP runtime kernel。
- QAIRT `--op_package_config` + `--converter_op_package_lib` 已完成全图转换与 DLC 写出。优化后 QNN IR 共 850 ops，5 个 SelectiveScan 位于 op id `66/194/340/435/587`，输出形状依次为 `[1,48,196] / [1,128,196] / [1,336,196] / [1,336,196] / [1,672,196]`，并保留 `delta_softplus=1 / out_float=0`；最终输出为 `depth [1,448,448]`。
- 跨过 SelectiveScan mapping 后没有出现新的 normal-converter hard blocker。最终图包含 LayerNorm 与 Resize，`Erf` 不再存在，说明此前三类 dry-run candidate 没有成为此次转换的停止点；这只证明 converter acceptance，不证明它们或 SelectiveScan 能在目标 HTP runtime 执行。
- 生成 DLC 为 32,003,812 bytes，SHA-256 `6ACD65D82FF3C0ABC7E1BC4787FCBA881D7E5CC4F5D48722F00F814D897DC680`。DLC info 报告的 `7090M` MAC、`664.2 MiB` steady-state memory 与 `A/D/G/C` runtime 列均为静态诊断，其中 runtime 列明确假设 Snapdragon 855，不得当作 Snapdragon 8 Gen 2 HTP 支持或性能证据。
- exact primitive feasibility 将每个长度 196 的 recurrence 展开为 3,730 个标准 ONNX 节点；对真实冻结合同的 `C=48/128/336/672, G=4, N=8, L=196` 随机输入，ORT 与 reference recurrence 在 `rtol=3e-5 / atol=3e-6` 下全部通过。完整图从 2,723 膨胀至 21,368 ONNX nodes，5 个 custom SelectiveScan 降为 0。
- QAIRT 仍成功转换该 primitive 图并写出 DLC，证明它 technically convertible；但优化后 QNN IR 为 21,440 ops，而 custom-mapping 图只有 850 ops（25.2×）。DLC 为 47,687,076 bytes（custom mapping 的 1.49×），转换约 789 秒，且每个 scan 保留 196 级串行 recurrence。故它被保留为 parity oracle/upper bound，不选为当前移动端实现。
- 已按冻结 `G=4/N=8/L=196` 合同落盘 float32 HTP scalar reference kernel：逐 channel 保持 8-float stack state、无 heap、实现 stable softplus/transition/input/B/C/D 完整 recurrence；源码合同测试通过。它是 correctness-first spike，不是 HVX 性能 kernel，也尚未编译。
- QAIRT 2.47 的本地官方文档与 makefile 确认 SM8550/v73 需要 Hexagon SDK 5.5.5 + Tools 8.7.06；本机缺少 QPM3 与该 SDK。普通 clang probe 因官方 HTP headers 缺 `HVX_Vector`/intrinsics 停止，说明必须先从需 Qualcomm 登录的 QPM3 补工具链。公开依赖 Android NDK r26c（26.1.10909125）已安装并写入 `depthart-deploy-env.ps1`。
- 后续已通过 Qualcomm QPM3 安装并登记 Hexagon SDK `5.5.5.0`（约 5.20 GiB），Tools `8.7.06` 的 `hexagon-clang++` 与 v73 runtime libraries 验证可用；QPM license 保持 `ACTIVE`。该事实关闭旧的 `HEXAGON_SDK_AUTH_BLOCKED`，不产生 runtime authority。
- QAIRT 官方 generator 从冻结 XML 生成 `DepthArtSelectiveScanPackageInterfaceProvider`，并由仓库构建脚本将 correctness-first kernel 编译链接为 v73 `elf32-hexagon` package（65,616 bytes，SHA-256 `8A8E7B07C54276511212CE46E47B6081286AD8369EAF62A937D4C3B4391AE662`）与 Android prepare-side `ELF64 AArch64` package（892,448 bytes，SHA-256 `289D7001419775D31DEB84DC0B8C7072295E9F184A04B446AFD60B06ADBF1103`）。aarch64 binary 明确依赖 `libQnnHtp.so` 与 `libQnnHtpPrepare.so`；两端均导出同一 interface provider。
- 本轮只完成 compile/link/ELF/symbol 验证；未执行 package load、QNN context、算子输入输出 parity、HTP graph、Snapdragon 实机、partition、CPU fallback、latency 或 thermal。官方 Windows/Gow Makefile还存在无关 v85 preflight 与 source wildcard 问题，构建脚本因此复用 generator source，并以相同 flags 显式编译 v73/aarch64。

机器可读 receipt：[`a3-onnx-qnn-preflight.json`](../../../artifacts.local/evidence/hftf/depthart-admission-r1/a3-onnx-qnn-preflight.json)、[`selective-scan-converter-mapping-receipt.json`](../../../artifacts.local/evidence/hftf/depthart-admission-r1/qairt/selective-scan-converter-mapping-receipt.json)、[`selective-scan-primitive-lowering-receipt.json`](../../../artifacts.local/evidence/hftf/depthart-admission-r1/qairt/selective-scan-primitive-lowering-receipt.json)、[`selective-scan-htp-kernel-preflight-receipt.json`](../../../artifacts.local/evidence/hftf/depthart-admission-r1/qairt/selective-scan-htp-kernel-preflight-receipt.json)

## 结论与边界

A3 已证明 ONNX static graph 可以生成，并通过 converter-only custom mapping 完成整图 QAIRT conversion；外围 normal-converter blocker 空间已收敛。exact primitive lowering technically feasible，但 25.2× QNN IR op 膨胀与 196 级串行链使它只保留为 parity oracle。correctness-first kernel 的 v73 与 aarch64 package 已实际编译链接，旧工具链阻塞关闭；当前核心缺口转为 package load/execute 与 primitive-oracle parity。canonical end-to-end parity、kernel runtime parity、QNN graph/context、partition、Snapdragon 实机与 latency/thermal 均未完成。因此 G4 仍为 `NOT_EVALUATED`，不是 HTP PASS 或 FAIL。reference 合同继续保持 `image,K→depth`；mobile graph 的 prompt 输入只代表硬件感知分区，不是固定 K 冒充动态 metric conditioning。
