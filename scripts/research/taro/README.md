# TARO research scripts

状态：`P0_PROTOCOL_AND_SCHEMA_FROZEN / PAIR_SUPPORT_AUDIT_AVAILABLE / R13_ORACLE_HEADROOM_PASS / R27_DUAL_SOURCE_FULL_GATE_PASS / R27_OPENLORIS_FRESH_SOURCE_FAIL / R31_RELIABILITY_CONSISTENCY_SUCCESSOR`

## 研究问题与版本

本 Module 服务于独立 `WILD_LAB / CANARY_LITE` 路线 TARO。当前版本只静态验证
`TARO_P0_TASK_QUERY_IDENTIFIABILITY_AND_FACTOR_ORACLE_CANARY_PROTOCOL_LOCK`：冻结四个
machine-readable schema、measurement-only observability、有限弱子空间 task ambiguity、
解析 fixture、三因子八臂 factorial、负控、数据角色和权限边界。

动态状态、唯一 successor 与 claim ceiling 以
[`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 为准。

## 稳定 Interface

- `validate_taro_p0_protocol.py`：只读 P0 JSON protocol、schema bundle 与 analytic fixture spec，
  校验 binding SHA、schema examples、factorial/identifiability 不变量、禁止扩权和 O0 runtime 缺席；
- `test_validate_taro_p0_protocol.py`：mutation tests，覆盖执行扩权、TaskQuery 缺失、prior 补秩、
  K 混入 factorial、missing-anchor 变 clear、body-motion 放行等恶意漂移。
- `audit_observation_pair_support.py`：只读 disclosed candidate-input JSON，按 exact parent/video 与 sensor
  timestamp 审计相邻帧、pose validity、passive/extended window pair count 和 identity digest；不打开 depth
  blob、FARO/highres、truth/label/outcome，也不运行模型或网络。
- `test_audit_observation_pair_support.py`：3 个 focused tests，覆盖 passive/extended window 计数、无效 pose
  fail-closed 与非 candidate-input schema 拒绝；结果只回答 source 是否支持下一 observability canary。
- `validate_rgb_visual_evidence_backend_preflight.py`：验证 RGB pair shadow preflight 的唯一 backend 数量、
  冻结 YOLO/labels 大小与 SHA-256、同预算/scene/UNKNOWN 门以及零 pre-lock live model reads。
- `aggregate_yolo_positive_evidence_shadow.py`：从 instrumentation test log 提取不含图像/box 的 opaque-scene
  聚合 receipt，并按冻结的协议/模型/identity、scene/reference、runtime 与 parent-macro gate 产生唯一 terminal；
- `test_aggregate_yolo_positive_evidence_shadow.py`：覆盖冻结四场 PASS、模型 identity 漂移和 exact payload
  lookup 不完整时的 fail-closed 聚合。
- positive-oracle 与 R13–R30 task-evidence runtime、测试和运行说明位于
  [`taro_o1r_r12_clear_observability_runtime`](../taro_o1r_r12_clear_observability_runtime/README.md)；当前 Bonn
  R1 因 recovery/CLEAR parent 分母不足而 `NOT_EVALUABLE`；R27 已在 ARKit/TUM 完整过门，但 Bonn dynamic
  与 fresh OpenLORIS 均发生 macro 回归，现已拒绝。当前只授权 consumed R31 reliability/occlusion-consistency
  Development，以及后续另立 source/parent-disjoint confirmation。

运行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest scripts.research.taro.test_validate_taro_p0_protocol scripts.research.taro.test_audit_observation_pair_support scripts.research.taro.test_aggregate_yolo_positive_evidence_shadow
```

本基础 Module 自身没有 solver、数据 materializer、模型、trainer 或 action scorer；pair-support audit 只是
Development source-capability precheck。隔离 runtime 中的 R23–R30 scorer/trainer 及其结果不改变本 Module 的 P0
权限，也不授权 Android、产品或安全结论。

## 输出

P0 validator 与 pair-support audit 都向 stdout 输出短 JSON；pair audit 可选写入独占的
`artifacts.local/experiments/` 路径。未来经独立协议授权的 TARO
执行只允许写入 `artifacts.local/evidence/taro/`、`artifacts.local/work/taro/` 或
`artifacts.local/models/taro/`，不得写入其他研究路线的 namespace。

## 安全边界

- P0 PASS 只证明静态合同自洽，不证明 task-query identifiability、factor causal headroom、
  GaugeFix/PARA 有效、真实数据质量、设备性能、产品或安全；
- `prior`、LM damping、regularizer 和 learned covariance 不得进入 measurement-only rank；
- K/pose/time 只作冻结 receipt 或 corruption control，不是第一版 GaugeFix state；
- `UNKNOWN`、缺 anchor、无基线、wrong-K/time、非光滑 contact conflict 不得转成 negative/clear；
- B1 consumed Selection、Calibration、Confirmation 以及现有其他路线 artifact 均不可读取或改写；
- 不授权用户提示、Android、QNN/HTP、默认 App、真实用户或独立助行主张。

## 停止条件

任一 binding、schema、权限、factorial purity、degenerate expectation 或 route isolation 漂移，P0
静态验证即失败；失败只关闭该协议/validator 版本。P0 通过也不会自动授权 O0M。真实 O0R 在
complete factor truth、fresh paired outcome、timestamp/pose receipt 和 deterministic injection
interface 缺失时必须保持 `NOT_EVALUABLE_DATA_AND_INTERFACE`。

## 假设与规则质疑

当前 finite task-ambiguity threshold、状态 normalization 与 numeric tolerance 只服务于解析 mechanics，
它们可以在 outcome 前以新协议版本挑战；一旦 O0M outcome 产生，任何修改必须建立新 evidence
version。不得用 B1 outcome 或 O0M 结果反向调当前 P0。

## 失败资产复用

失败的 P0 schema、fixture 或 validator 可保留为 regression、negative evidence、counterexample 和
协议 mutation corpus；不得包装为真实 causal headroom、unseen Confirmation 或产品证据。
