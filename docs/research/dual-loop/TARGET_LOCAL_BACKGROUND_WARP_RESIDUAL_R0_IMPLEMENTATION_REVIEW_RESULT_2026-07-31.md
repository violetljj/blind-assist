# TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0 B Development 实现复核结果

状态：`PASS / B_IMPLEMENTATION_READY / SYNTHETIC_ONLY / C1_C2_NOT_AUTHORIZED`

日期：2026-07-31（Asia/Hong_Kong）

绑定设计合同：[TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0 设计合同](TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_DESIGN_CONTRACT.md)

合同 SHA-256：
`caca9e4aeeddc3b557403b3dbcf427d600504ad6cfb89d203e33d48a9e556bde`

实现 Module：[dual_loop_target_local_background_warp_residual_r0](../../../scripts/research/dual_loop_target_local_background_warp_residual_r0/README.md)

## 结论

B Development 的离线实现复核通过。producer、truth-late evaluator、R1–R4 选择、
固定事件中位数/分母、弃权优先级、truth firewall、输出 hash 和 implementation lock
已经形成可运行的隔离接口。

本 PASS 只表示实现可进入 B 的下一步输入/输出审计；当前没有读取 CrowdBot、Matoaka、
Shiraz 或任何真实候选 output，没有生成 C1/C2 结果，也没有 Android、shadow、active、
产品或安全权限。

## 复核范围

- producer 只接受 frame/detection identity、原生 luma 图像、bbox、严格相邻帧和 image
  SHA-256；truth、event label、pose/Vicon、oracle、旧 decision 和后验输出字段在图像
  打开前拒绝；
- similarity RANSAC 使用固定 2-point 枚举、1000 次上限、2 px 门、内点/重投影/空间
  支持/condition number 门、闭式 Procrustes refit 与确定性 tie-break；
- evaluator 只在 producer receipt 校验后读取独立 truth-late manifest，以
  truth-eligible adjacent pair 为分母，按 parent event 计算 median、coverage、
  wrong-signed 和 paired gain；
- implementation lock 绑定当前合同 SHA、Module `.py` 文件 SHA、Python/OpenCV/NumPy
  版本、模型身份和 B Development 边界。lock artifact SHA-256：
  `8e5b417e0a86a2f25543142b12e9cf1426e8bcfcf323d1dc4fc37aa49d86f9b4`
  （`artifacts.local/evidence/dual-loop/target-local-background-warp-residual-r0/b-implementation-lock-v2.json`）。

## 验证证据

- root adapter synthetic suite：`14/14 OK`；覆盖 zero-motion、RANSAC determinism、
  dynamic mask、输入 gate priority、truth firewall、event median/coverage/deadband、
  ring tie-break 和 truth-late end-to-end；
- `py_compile` 通过；implementation lock create/validate 均为 `VALID`；
- output/evaluator 均拒绝覆盖已有文件；生产行为与 Android 文件未修改。

## 后继边界

下一步若继续，只能先冻结 B Development 的 burned-source metadata/input manifest，
再在候选输出访问前执行一次 producer。C1 metadata-only admission、C2 复现、Android、
active policy、Confirmation、产品和安全结论仍需另外明确授权。
