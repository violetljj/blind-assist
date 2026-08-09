# TARO P0 protocol lock result

状态：`TARO_P0_PROTOCOL_AND_SCHEMA_LOCK_PASS / SCIENTIFIC_STATUS_NOT_RUN / O0M_EXECUTION_NOT_AUTHORIZED`

日期：2026-08-10

机器结果：[JSON](TARO_P0_PROTOCOL_LOCK_RESULT_2026-08-10.json)

## 结论

P0 非执行协议锁通过。四个 schema、十个 identifiability fixture、两个动作过滤反例、六个
factor-oracle mechanics case、八臂 × 两种 oracle mode 的 96 份逐臂 payload/output/common-support
hash，以及十项未来 O0M gate 已机器冻结；33 个静态/mutation tests 全部通过。专项 validator
为 `VALID`；通用治理 validator 为 `VALID_ZERO_ERRORS_TWO_WARNINGS`，披露的 early-stage warning 为
`SEALED_DATA_ASSIGNED_NONCONFIRMATION:1/2`，只对应尚未授权执行的未来 O0M/O0R 分区。

这不是 TARO 科学结果：没有 solver、canary、真实数据读取、factor injection、训练、主动提示或
设备执行。真实 O0 仍为：

`TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE`

## 本轮修正的关键风险

- 信息只来自 measurement-only whitened residual Jacobian；prior/LM/regularizer 不得补秩；
- `Null(F) subset Null(J_C)` 降为极限诊断，finite weak-subspace task ambiguity 才是主 gate；
- K corruption 独立于 `S/P/B` 的 `2^3` factorial；
- value-only common-support 与 full-block oracle 分开，避免把 coverage/sigma 收缩混入 value headroom；
- identifiability truth、非光滑分支与 96 份逐臂 receipt hash 均由 validator 从数值输入重算；
- weak-subspace task ambiguity `R_weak` 与 strong-subspace measurement interval `H_meas` 分字段；
  只有前者进入 2 cm identifiability gate，后者只扩宽最终 decision interval；
- future timestamp、重复 evidence、anchor/outcome 共享、矛盾 posterior、非有限 truth、空 gate 与
  未支持 schema keyword 全部 fail-closed；
- 非空 gate 正文漂移与 causal-factor 标签/数值不一致同样 fail-closed；
- ObservationCandidate 的 predicted/realized baseline、frame/query/cutoff/provenance 与 body-motion/
  realized-receipt 交叉语义已冻结；posterior covariance 必须对称半正定；P0 Module 采用文件 allowlist；
- FrameReceipt、TaskQuery、FactorPosterior、ObservationCandidate 的 frame/query/body/path、factor identity、
  timestamp cutoff 与 provenance 跨对象一致性全部 fail-closed；
- FrameReceipt `max_source_timestamp_ns` 是 anchor、posterior、candidate 的唯一因果水位；
- 增加独立 `TaroTaskQuery` schema；
- active branch 失败后的 passive 继续必须另立版本，不得隐式进入 joint J0。

## 唯一 successor

`TARO_O0M_SYNTHETIC_IDENTIFIABILITY_AND_FACTORIAL_MECHANICS_PROTOCOL_LOCK`

它仍只允许协议冻结，不授权 O0M 实现或执行。P0 PASS 的 claim ceiling 仅为静态合同自洽，默认
App、Assistive Geometry、DepthART 与所有产品/安全权限均不变。
