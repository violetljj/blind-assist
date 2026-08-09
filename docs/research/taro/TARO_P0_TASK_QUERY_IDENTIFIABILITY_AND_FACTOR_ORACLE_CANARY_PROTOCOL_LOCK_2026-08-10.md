# TARO P0 task-query identifiability 与 factor-oracle canary 协议锁

状态：`P0_PROTOCOL_AND_SCHEMA_FROZEN / STATIC_VALIDATION_ONLY / SCIENTIFIC_STATUS_NOT_RUN / O0M_EXECUTION_NOT_AUTHORIZED`

日期：2026-08-10

机器合同：[JSON](TARO_P0_TASK_QUERY_IDENTIFIABILITY_AND_FACTOR_ORACLE_CANARY_PROTOCOL_LOCK_2026-08-10.json)

Schema：[TARO P0 schema bundle](TARO_P0_SCHEMA_BUNDLE_2026-08-10.json)

解析期望：[TARO P0 analytic fixture spec](TARO_P0_ANALYTIC_FIXTURE_SPEC_2026-08-10.json)

## P0 结论先行

P0 已把 TARO 的对象、数值判据、factorial、负控、数据角色、失败范围与权限机器化；本阶段没有
运行 solver、oracle 或真实数据。真实 O0 当前仍为：

`TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE`

原因不是“还没多跑几帧”，而是 complete factor truth、truth-clear factor bundle、连续 boundary/
uncertainty truth、显式 target timestamp/pose、deterministic injection adapter 和 fresh paired outcome
均未满足。

## 四个冻结 schema

- `TaroFrameReceipt`：source/session/parent/frame、capture/site/device/mount、timestamp ceiling、
  Camera/K/crop/rotation/resize、pose/IMU/tracks、metric anchor、factor identity、query/action budget、
  data role 与 provenance；
- `TaroTaskQuery`：body profile、path、horizon、swept volume、linearization state、所有 active-contact
  query Jacobian branch、clear/occupied margin 与 deterministic reducer identity；
- `TaroFactorPosterior`：factor reference、四维 residual state/covariance、measurement singular values、
  observable mask、update/freeze/abstain reason、query interval/identifiability/state 与 causal timestamp；
- `TaroObservationCandidate`：冻结 frame/query/cutoff/provenance、动作类型、camera-only delta、
  predicted/realized baseline、body-motion filter、预测信息/风险价值、realized receipt、失败原因与停止条件。

`UNKNOWN` 不是 negative；计划动作不是已执行观测；第一版 `requires_body_motion=true` 一律拒绝。

## 可识别性判据

第一版只开放：

```text
g = [log_scale, support_tangent_x, support_tangent_y, support_offset_m]
one normalized unit = [0.10, 0.05 rad, 0.05 rad, 0.10 m]
```

`delta K / delta pose / delta time` 固定为有效 receipt 或 corruption control，不与 GaugeFix 联合求解。
Boundary 是 O0 treatment block，不是第一版 GaugeFix state。

信息矩阵只来自去重后的 measurement residual：

```text
A = Sigma_r^(-1/2) * dr/dg
```

prior、LM damping、正则化和 learned covariance 不得补秩。强子空间规则固定为：

```text
sigma_i >= max(1.0, 0.001 * sigma_max)
```

`Null(A) subset Null(J_C)` 只作零噪声极限诊断；正式 gate 使用弱子空间单位 trust region 内完整
body/path query 的最大歧义半径，要求 `R_weak <= 0.02 m`。强子空间的冻结 95% whitened
measurement-noise budget 经伪逆形成独立 `H_meas`，只扩宽最终 clearance interval，绝不补秩、降秩
或进入 2 cm 可识别性 gate。接触竞争距离在 `0.01 m` 内、分支
clearance spread 超过 `0.02 m` 时返回 `UNKNOWN_NONSMOOTH_CONTACT_SWITCH`。

最终状态仍由 deterministic interval reducer 产生：95% interval lower `> +0.05 m` 才可 clear，
upper `<= 0 m` 才可 occupied，其余为 `UNKNOWN`。

## Factorial 与 oracle 语义

三块 `SCALE / SUPPORT / BOUNDARY` 冻结为完整 `2^3` 八臂。K/receipt corruption 是独立负控，
不属于 factorial。

- Primary：`VALUE_ONLY_COMMON_SUPPORT`，只换 value，mask/validity/uncertainty/common-support 完全不变；
- Diagnostic ceiling：`FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY`，单独报告 value、coverage 与 sigma 变化；
- 六个 mechanics case 均冻结八臂 × 两 mode 的逐臂 payload/output/common-support SHA-256；validator
  从 factor value/validity/uncertainty 和 receipt 重算 96 份收据，不接受手填 best-arm 标签；
- 真实 O0 的主比较预定为 `ALL_ORACLE vs NONE`；单 factor/interaction 只能诊断，不得 outcome 后
  挑一个“受支持组合”作为通过结果。

## O0M 与 O0R 边界

未来唯一 successor 只允许再冻结一个 synthetic O0M 协议。O0M 的十个 gate 覆盖 binding、oracle
positive control、discriminating opportunity、identifiability truth、退化 fail-closed、干预纯度、
factor specificity、compound closure、单调/确定性与未来信息泄漏。即使全部通过，也只证明解析
fixture 上的 mechanics。

真实 O0R 必须另有 fresh parent/session/site-disjoint paired contract，最低需要 complete factor/query
truth、clear/occupied 双侧支持、timestamp/pose/camera/anchor receipt、deterministic adapter，以及
prospective power 与 `delta_min`。这些条件未满足前，不得读取 B1 consumed Selection 来替代。

## 路由与失败范围

- P0 PASS：只允许冻结 O0M protocol，不自动授权实现或执行；
- O0M PASS：只建立 synthetic mechanics，可另开 O0R source-readiness 或 synthetic G0 diagnostic；
- O0R `NOT_EVALUABLE`：保留数据缺口，不把 synthetic PASS 写成真实 headroom；
- A0 active oracle 失败：关闭 PARA；passive continuation 必须另立新路线版本，不能进入原 joint J0；
- A1 scorer 失败：关闭 learned scorer；analytic/passive continuation 同样不能隐式进入原 joint J0。

P0 的失败只关闭协议/validator 版本。它不改 Assistive Geometry、DepthART、默认 App、产品或 safety
权限。

## 当前唯一 successor

`TARO_O0M_SYNTHETIC_IDENTIFIABILITY_AND_FACTORIAL_MECHANICS_PROTOCOL_LOCK`

该 successor 仍为 non-execution：只冻结独立 scene/family/seed、implementation hashes、numeric
tolerance、十项 gate、timeout 与 exclusive artifact root；不得读真实数据、运行 canary、创建
GaugeFix 或宣称真实 factor causal headroom。
