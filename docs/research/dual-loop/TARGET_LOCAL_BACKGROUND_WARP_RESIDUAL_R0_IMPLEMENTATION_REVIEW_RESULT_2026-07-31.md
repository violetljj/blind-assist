# TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0 B Development 实现复核结果

状态：`PASS / B_TERMINAL_CLOSED / C1_C2_NOT_AUTHORIZED`

日期：2026-07-31（Asia/Hong_Kong）

绑定设计合同：[TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0 设计合同](TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_DESIGN_CONTRACT.md)

合同 SHA-256：
`caca9e4aeeddc3b557403b3dbcf427d600504ad6cfb89d203e33d48a9e556bde`

实现 Module：[dual_loop_target_local_background_warp_residual_r0](../../../scripts/research/dual_loop_target_local_background_warp_residual_r0/README.md)

## 结论

B Development 的离线实现复核通过，burned REveL 输入已冻结，且唯一一次 producer 与 truth-late
evaluator 已完成。R1–R4 选择、固定事件中位数/分母、弃权优先级、truth firewall、输出 hash 和
implementation lock 均通过校验；按合同终点关闭本候选。

本结果仍只表示 B Development 诊断；不读取 CrowdBot、Matoaka、Shiraz 或任何其他候选
source，不生成 C1/C2 结果，也没有 Android、shadow、active、产品或安全权限。

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
  `3511da1178af9acd90fbe2ad84c1e93bf208ca4b33cca713a2f4973e735689a2`
  （`artifacts.local/evidence/dual-loop/target-local-background-warp-residual-r0/b-implementation-lock-v3.json`）。

## 验证证据

- root adapter synthetic suite：`15/15 OK`；覆盖 zero-motion、RANSAC determinism、
  dynamic mask、输入 gate priority、truth firewall、event median/coverage/deadband、
  ring tie-break 和 truth-late end-to-end；
- burned REveL input freeze：`FROZEN`，`12,876` 个固定 pair、`8,363` 个图像、`32`
  个原生 shape mismatch pair 均保留在分母；receipt 明确 `truth_read=false`、
  `candidate_output_read=false`；
- B producer receipt：`COMPLETE`，固定输入 `12,876` rows，四个 ring 共 `51,504` rows；
  `truth_read=false`、`forbidden_outcome_fields_rejected=true`，producer output SHA-256 为
  `65e72dcd136249057302facc7232a1194a331cdd43a8e2a981ca865429066fbb`；按
  `(session_id,target_id,current_source_frame_id,previous_source_frame_id,ring_config_id)`
  检查无重复 pair identity；
- truth-late join：`FROZEN_TRUTH_LATE`，`6,538` rows、`960` 个 parent events、两个 target；
  truth-late SHA-256 为 `b109b72e8a5938c33c57683892a1ca077871185035b43b09b41743c4c97116f1`；
  producer receipt 完成前未读取 truth；
- evaluator：`VALID / UNIQUE_SELECTION`，唯一选择 `R1`；唯一 session
  `REVEL_DYNAMIC_SINGLE_CAPTURE` 的 raw coverage 为 `0.4947917`、residual coverage 为
  `0.4020833`，paired event gain 为 `-34/960 = -0.0354167`，positive contribution
  fraction 为 `0.0385417`，因此 `development_gate_passed=false`，合同终点为
  `NO_DEVELOPMENT_INCREMENT / CLOSE_CANDIDATE`；
- `py_compile` 通过；implementation lock create/validate 均为 `VALID`；
- output/evaluator 均拒绝覆盖已有文件；生产行为与 Android 文件未修改。

## 后继边界

B 候选已按唯一终点关闭；不得重跑、调参、缩小分母或事后修复 burned output。C1
metadata-only admission、C2 复现、Android、active policy、Confirmation、产品和安全结论仍需另外
明确授权；本终点不提供任何自动升级路径。
