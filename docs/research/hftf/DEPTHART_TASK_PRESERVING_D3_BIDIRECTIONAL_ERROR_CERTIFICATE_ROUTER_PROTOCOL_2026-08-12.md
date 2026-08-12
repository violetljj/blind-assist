# DepthART task-preserving D3 bidirectional error-certificate router

状态：`PRE_OUTCOME / PROTOCOL_FROZEN / SYNTHETIC_MECHANICS_PASS / SOURCE_SCOPE_NOT_ACTIVATED`

D3 是一个新版本，不是 D2 的续跑或结果回救。D1、D2 的一次性 Development outcome 均已
消费并保持 FAIL；不得从中重新拟合 checkpoint、阈值、数据、postprocess、denominator 或 gate。
历史 archive 中的 `Stage-C D3-Q0`、`D39` 也与本协议无关。

## 可证伪主张

D2 暴露的核心矛盾不是“head 不够大”，而是一个直接状态 head 在降低 false-clear 时把
false-block 推高。D3 因而改写学习任务：不重新预测最终状态，只学习两种相互独立的纠错证书。

- `CLEAR release`：只有存在强 free-corridor 证书时，才把 baseline 的 OCCUPIED/UNKNOWN
  纠正为 CLEAR；
- `OCCUPIED veto`：只有存在强 connected-intrusion 证书时，才把 baseline 的 CLEAR/UNKNOWN
  纠正为 OCCUPIED；
- 两种证书都强时输出 `UNKNOWN_GROUND`；证据弱、含糊或缺少 hard evidence 时保持 baseline。

三 horizon 上，occupied 证书做 cumulative-max，clear-through 证书做 cumulative-min；较近
horizon 已经 OCCUPIED/UNKNOWN 后，较远 horizon 不能重新变成 CLEAR。阈值在数据访问前固定为
strong `0.9`、opposite maximum `0.1`。这使假设可被明确反证：若同一个冻结 router 不能在
fresh Development 同时降低 false-clear 与 false-block，并通过原有绝对质量门，D3 即 FAIL。

## mechanics 与候选边界

纯 CPU synthetic canary 已验证 baseline identity composition、双向纠错、冲突转 UNKNOWN、
hard-evidence veto、UNKNOWN promotion、horizon 单调与确定性 replay。该 PASS 只证明组合逻辑，
不证明数据支持、可学习性、准确率、设备性能或候选资格。

未来唯一候选固定为两个不共享权重的 `Linear(16,16)-SiLU-Linear(16,1)` certificate heads，
共 578 参数。输入单位是 band × horizon，来自 baseline 状态/clearance、ground/support、
connected intrusion 与 free-corridor 因子；运行时禁止 canonical reference。完整 feature extractor、
TRAIN standardization、两个 head 权重和 router policy 必须封成一个 hash，才能打开 Development。

## fresh data 防火墙

D3 要求 ARKitScenes official Training fold 中至少 16 个全新 parent/session，最终严格分成
8 TRAIN + 8 DEVELOPMENT，每身份 300 帧。它们必须与 D1 的 primary/reserve/selected、D2 的
全部 source/TRAIN/Development、sealed R2 roster，以及其他 outcome-bearing research roles
完全不重叠。

顺序先冻结 48 身份 metadata pool，再 label-blind 取得前 32 个满足 portrait/pose continuity
的身份；随后只用 depth/confidence 检查 source-truth support，取冻结顺序前 16 个合格身份，
前 8 个为 TRAIN、后 8 个封存为 DEVELOPMENT。每个 band × horizon 都必须同时有至少 30 个
truth CLEAR 与 30 个 truth OCCUPIED cells，避免 D2 出现的空 denominator。少于 16 个合格身份
直接 `D3_DATA_SUPPORT_NOT_EVALUABLE`，不得在看到 outcome 后替换。

## 训练与一次性评价

训练只允许 D3 TRAIN：固定 seed 31、TRAIN-only standardization、CPU float64、AdamW、
1000 full-batch steps、只取 step-1000，无 early stopping。Development 不得用于训练、校准、
阈值、checkpoint 或候选选择。

Development 保留 D1/D2 的绝对门；除此以外，候选还必须相对同帧 baseline 同时把 pooled
false-clear 和 false-block 各降低至少 `0.01`，且 coverage、MAE、temporal、transition 与
valid-to-unknown 不得越过冻结的 noninferiority 边界。pooled、parent/session macro、worst parent
和 9 个 grid 都必须 finite；任何 required denominator 为空即 fail-closed。

即使 D3 PASS，也只产生 identity-disjoint Development feasibility，不自动锁 R2 candidate。
strict G4-D 终态不重开，R2 继续 sealed；性能、默认 App、production 与 safety 均未授权。

## 当前唯一 successor

`EXPLICIT_D3_FRESH_SOURCE_SCOPE_AND_PARENT_DISJOINT_METADATA_ROSTER_LOCK`

该门只允许冻结排除集合、48 身份 metadata pool 和精确 source-use scope；当前没有媒体 HEAD、
body、source truth、训练、Development outcome 或 R2 权限。
