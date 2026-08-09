# Assistive Geometry R2 F1 FactorTensorAdapter Protocol Lock

状态：`R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_FROZEN / CANARY_NOT_RUN / F1_EXECUTION_NOT_AUTHORIZED`

## 冻结内容

本锁只定义 F1 factor tensor 到 byte-frozen F0 reducer frame 的确定性 seam：

- `14/14` F1 prediction fields 均有显式 consumer；
- F0 frame、support、scale、boundary 与 obstacle 的全部必需字段均有唯一 producer；
- 冻结 17 个零参数操作，包括 receipt/K/gravity 绑定、scale 与 uncertainty、support uncertainty、
  8-connected dense component、metric interval、evidence 聚合和 canonical ordering；
- 冻结 8 个 tiny synthetic cases 与 `A01..A10` mutation canary gates；
- adapter 必须在 learned graph 外，不能读取 task outcome、输出 final task shortcut 或进行任何训练。

Machine contract：
[`BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_LOCK_2026-08-10.json`](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_LOCK_2026-08-10.json)

Synthetic fixture：
[`BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_FIXTURE_2026-08-10.json`](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_FIXTURE_2026-08-10.json)

## Fail-closed 语义

- receipt、sample、K、transform、gravity、shape 或 schema 不一致时，输出完整但 invalid 的 F0 frame，
  让 reducer 保持 `UNKNOWN`；不得猜测或补 K/pose/factor。
- local depth 缺失保留为 `depth_valid=false`，不能转成 occupied。
- 增大 uncertainty 或减少 validity/positive evidence，不能增强 occupied，也不能把 UNKNOWN 变 clear。
- support normal/height uncertainty 来自独立 fit-only calibration receipt，不从 task outcome 或 reducer state
  反推。

## 权限

本锁没有实现或运行 adapter，也没有读取真实数据、标签、模型输出或 task outcome。协议 PASS 只允许
下一步创建零参数实现并运行冻结 synthetic canary；即使 canary PASS，F1 supervision frontdoor、模型、
训练、F2、设备、产品和 safety 仍不自动授权。

## Claim ceiling

只建立机器接口与未来 synthetic canary 的事前合同，不建立 adapter mechanics PASS、factor
learnability、真实 factor headroom 或 task utility。
