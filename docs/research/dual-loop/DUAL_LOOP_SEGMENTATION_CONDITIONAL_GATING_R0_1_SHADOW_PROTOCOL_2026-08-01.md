# Conditional Segmentation Gating R0.1 Post-primary Shadow Protocol

状态：`PROTOCOL_FROZEN / EXECUTION_COMPLETE / VALID /
POST_TERMINAL_SHADOW_ABLATION_COMPLETE_DIAGNOSTIC_ONLY /
POST_PRIMARY_SHADOW_ABLATION_ONLY / NO_SELECTION_AUTHORITY /
FINAL_CONFIRMATION_NOT_ACTIVATED / DEFAULT_APP_UNCHANGED`

日期：2026-08-01（Asia/Hong_Kong）

实现记录：初始冻结 Git `6ef3014dbea24b24ca31fadd1c9c9eda829d2481` 的首次
activation 在 raw shadow frame/component 文件读取前停止。原因是 runner 将 input
binding list 传给了 single-binding loader，触发 `TypeError`；没有创建 output root，
没有计算 mask、component decision 或任何 shadow 指标。V2 只把两组 input list 路由到
既有 multi-file bound loader，并让 preflight 先完整验证 520/11,757 membership。
科学定义、阈值、角色、material/heterogeneity 规则和 output contract 均不变。

V2 implementation 已在 Git `827dcda976394cd4d2a0c6f5bc29993ada9d9d5d` 完成一次
shadow execution，并写出 520 行 frame 与 23,514 行 component decision。初始独立
validator 在任何 aggregation 检查前以 `reported primary binding drifted` 停止：
runner 的 primary 摘要包含 `reference_only/terminal_unchanged`，validator 却期待
不存在的 `protocol_id` 字段。validation recovery 只让 validator 精确匹配 runner 已
冻结的 primary-summary schema；不改 result、frame/component 输出、算法或解释规则。
recovery validator 现已通过 167,327 项检查、错误数 0。详见
[R0.1 result](DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0_1_SHADOW_RESULT_2026-08-01.md)；
两条 shadow 均无 material，且 `H_min/H_cross` 全部为 false。有限三臂静态手工门
家族终态为 `TWO_SHADOWS_WEAK_FIXED_HANDCRAFTED_GATING_FAMILY_STOP`。

## 纠正与研究问题

R0 只冻结并执行了一个 primary：
`CLASS_CONDITIONED_MULTI_NEGATIVE`。它的负结果能够关闭“在 R0 内用未执行候选救援
primary”的自由度，却不能证明另外两个已提出但未执行的条件门同样失败。因此历史
machine terminal
`CONDITIONAL_GATING_NO_ROBUST_INCREMENT_STOP_GATING_ROUTE` 保持不可变，但自本协议起
其 claim scope 明确收窄为 `PRIMARY_CANDIDATE_ONLY`。

两个 shadow 的算法在 R0 结果产生前已经作为概念提案出现，但没有进入 repo config、
definition hash、runner 或执行授权。准确状态是：

```text
PREVIOUSLY_ARTICULATED_BEFORE_PRIMARY_RESULT
BUT_NOT_HASH_BOUND_OR_EXECUTION_AUTHORIZED
POST_R0_FORWARD_FREEZE_REQUIRED
```

R0.1 只回答：在不选择、不救援 primary 的前提下，两个前向冻结的 shadow 是否构成
“有限三臂门控家族已经全部无信号”这一宽结论的反例或机制证据。

## 角色冻结

| arm | 角色 | 能否改变 R0 primary terminal |
|---|---|---|
| `CLASS_CONDITIONED_MULTI_NEGATIVE` | `FROZEN_R0_PRIMARY_REFERENCE` | 否 |
| `CLASS_CONDITIONAL_TEMPORAL` | `SHADOW_DIAGNOSTIC_ONLY` | 否 |
| `MULTI_NEGATIVE` | `SHADOW_DIAGNOSTIC_ONLY` | 否 |

R0.1 一次执行两个 shadow，全量公开，不排序、不选优、不按 session 选 arm。即使某个
shadow 通过全部门，它也只有 post-primary burned-Development counterexample 身份；
必须另立新单候选和新 Development 数据，才可能获得后续执行权限。

## 共同合同

两个 shadow 原样复用 R0 的 `0.65 / 63 / 0.35 / connectivity=8`，以及：

- `T[k,t] = M[k,t] AND (M[k,t-1] OR M[k,t-2])`；
- same-class raw mask history，按 session/sequence 重置；
- pixel temporal unit；
- confidence、area 与 upper membership 只从 raw current component 计算一次；
- upper 为纯几何 `y < 0.35H` any-pixel intersection；
- missing confidence 保留；
- 门后 fragment 不重算特征；
- predicate 禁止 truth、residual、mechanism、session、scene、role 和 YOLO attribution。

十个 source sessions、520 帧与 11,757 raw components 仍全部是 consumed Development；
held-out 仍只作 fit-free session stress，不是独立验证。

## 两个 shadow

### `CLASS_CONDITIONAL_TEMPORAL`

```text
obstacle:
    keep same-class causal 2-of-3 supported pixels only

boundary_step_curb:
    reject whole raw component iff
        confidence is known below 0.65
        AND raw area <= 63
```

它相对 primary 只改变 obstacle 分支，诊断更激进 temporal rejection 的 FP/recall 交换。

### `MULTI_NEGATIVE`

两个 predicted class 都使用：

```text
reject pixels iff
    noncausal
    AND confidence is known below 0.65
    AND (raw area <= 63 OR intersects geometric upper band)
```

它相对 primary 只改变 boundary 分支，诊断 boundary protection 是否限制 FP reduction。
两个 shadow 彼此同时改变两个分支，不能作为单变量因果对照。

## 结果解释预冻结

R0.1 的 protocol ID 为 `DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0_1`，mode 为
`POST_R0_FORWARD_SHADOW_DIAGNOSTIC`。execution terminal 固定为：

```text
POST_TERMINAL_SHADOW_ABLATION_COMPLETE_DIAGNOSTIC_ONLY
```

它不随数值改变。每个 shadow 分别与固定集合
`{REFERENCE_CAUSAL_2_OF_3_UNION, REFERENCE_CONFIDENCE_GE_0_65,
CLASS_CONDITIONED_MULTI_NEGATIVE}` 比较；两个 shadow 不互相挤出 frontier。冻结：

```text
R4 =
    overall recall retention >= 0.90
    AND minimum-session recall retention >= 0.80
    AND boundary_step_curb recall retention >= 0.80
    AND obstacle recall retention >= 0.80

F = FP reduction >= 0.30

N =
    shadow is a frontier member in fixed-reference-set + itself
    AND strictly Pareto-dominates at least one fixed reference

MATERIAL = R4 AND (F OR N)
```

若任一 shadow 为 `MATERIAL`，family assessment 为
`ALTERNATIVE_GATING_SIGNAL_OBSERVED_NO_SELECTION_AUTHORITY`；五门全过另加
`POST_PRIMARY_SHADOW_FAMILY_COUNTEREXAMPLE_DEVELOPMENT_ONLY` annotation。

若没有 material，则检查无新阈值的 heterogeneity：

- `H_min`：某 shadow 只失败 minimum-session 门，其余四门全部通过；
- `H_cross`：存在一个 session 由 C1 严格 Pareto 支配 C2，另一个 session 由 C2
  严格 Pareto 支配 C1。

有 heterogeneity 时只记
`SESSION_HETEROGENEITY_OBSERVED_STATIC_GATING_FAMILY_STOP /
LEARNED_QUALITY_AWARE_HYPOTHESIS_ONLY`；否则记
`TWO_SHADOWS_WEAK_FIXED_HANDCRAFTED_GATING_FAMILY_STOP`。

后两项只关闭这三个精确定义的静态手工门，不得扩大成“所有 conditional gating、
learned gating 或所有 postprocess 均失败”。无论哪一分支，都停止继续手工堆门；
有 material 时可在新 Development 数据上冻结至多一个新单候选，也可转 residual-aware
训练；无 material 时 residual-aware DDRNet Development 设计成为下一主边界。

## 证据与输出边界

新 config：
[`configs/dual_loop_segmentation_conditional_gating_r0_1/shadow.json`](../../../configs/dual_loop_segmentation_conditional_gating_r0_1/shadow.json)

R0.1 绑定 R0 frozen Git、primary definition hash、result/frame/component/validation
四个 SHA，并写入独立
`artifacts.local/evidence/dual-loop-segmentation-conditional-gating-r0-1-shadow/`。
禁止覆盖 R0 evidence root。执行前只允许 synthetic tests、config preflight 和仓库检查，
不得读取 shadow outcome。

本协议不训练模型、不访问 fresh holdout，不接 Android、QNN/A568、risk/feedback、
TTS、振动、提醒或默认 App。协议冻结时 residual-aware DDRNet 只为 eligible 方向；
R0.1 的 `VALID` 负结果现已使其 Development 设计成为下一主边界，但不由本协议自动
授权训练或扩大证据范围。
