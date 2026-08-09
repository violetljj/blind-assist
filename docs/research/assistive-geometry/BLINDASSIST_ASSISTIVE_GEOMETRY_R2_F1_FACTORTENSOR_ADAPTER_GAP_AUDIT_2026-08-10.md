# Assistive Geometry R2 F1 FactorTensorAdapter 接口缺口审计

终态：`R2_F1_EXECUTION_BLOCKED_FACTORTENSOR_ADAPTER_ABSENT`

## 结论

独立审计成立。byte-frozen F1 factor schema 与 byte-frozen F0 reducer 之间没有一个经版本化、
hash-bound、位于 learned graph 之外的 deterministic `FactorTensorAdapter`。这不是字段改名问题，
而是四组尚未定义的计算语义：

1. `depth_log_sigma_hw` 没有确定性校准/聚合为 reducer 所需的 scalar
   `depth_scale.scale_sigma_m`；
2. F1 只有 `support_residual_sigma_m`，没有 reducer 必需的 `normal_sigma_rad` 与
   `height_sigma_m`，也没有冻结的推导；
3. dense depth/boundary/evidence tensors 没有确定性变换为 ordered metric
   `boundary.obstacles[]`，缺少 component extraction、split/merge、depth/lateral interval、
   boundary/evidence sigma 和 canonical order；
4. camera receipt SHA 没有定义 adapter 对 K、transform、gravity、orientation 与
   boundary coverage 的逐字段验证/构造。

因此 F0 PASS 与 F1-P protocol lock 都保持有效，但二者不能直接组合，F1 execution 仍为
`false`。当前没有实现 adapter、没有运行 mutation canary、没有物化标签、没有定义模型、
没有 optimizer step，也没有读取 task outcome。

## 冻结边界

- F1 factor schema SHA-256：
  `8016430D639EC78199432F55ABB8EBDC847A4073C24F84A17E429A07D1BB5F7E`
- F1-P lock result SHA-256：
  `55BD2566A7EBBF1C16A481A2981195E5F62F928BFF78DEBA6B4281A020405F67`
- F0 reducer SHA-256：
  `2D6C26AD75B98911FD610FE0428D47584C877BA9AC7F091F768D98C035932092`

本审计不改写上述文件。F1-P 原来指向 supervision source/label contract 的 successor routing
被这份更晚的接口审计取代，但 F1-P 协议字节、结论和监督前门 blocker 均不被撤销。数据监督
合同仍然是 adapter 前门闭合后的独立必要条件。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_SCHEMA_AND_MUTATION_CANARY_LOCK`

execution authority 为 `false`。它只允许冻结 outside-graph、zero-parameter、deterministic adapter
的 schema、17 个必需操作、frame contract、合成 fixture 与 mutation canary gates，并绑定 F1 schema
和 F0 reducer SHA。不得实现/执行 adapter，不得物化或训练，不得创建 checkpoint、读取 task outcome、
进入 F1/F2，或启动 teacher、temporal、mobile。

未来 mutation canary 至少必须覆盖：逐字段 coverage 1.0、确定性 replay、scale/support uncertainty
腐化、obstacle split/merge/order、portrait/landscape frame equivariance、missing depth fail-closed、
不确定性增加不得在没有新 positive obstacle evidence 时触发 `CLEAR→OCCUPIED`、无 learned shortcut，
以及 F0 reducer SHA 不变。

机器可读真源：
[adapter gap audit](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_GAP_AUDIT_2026-08-10.json)。

## Claim ceiling

本结果只证明 F1 execution 缺少 deterministic tensor-to-reducer 接口。它不定义 adapter mechanics，
不证明 adapter 正确，不建立 factor supervision/learnability，也不是 task、deployment 或 safety evidence。
