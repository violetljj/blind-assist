# BlindAssist Assistive Geometry QSF

状态：`current / WILD_LAB / H1_IMPLEMENTED / ATTEMPT_03_PERFORMANCE_QUALIFIED / H1_NOT_EVALUABLE_EVAL_RIGHT_CENSOR_ZERO / H2_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

## 主张与边界

AG-QSF（Assistive Geometry Queryable Survival Field）研究：

> RGB 与显式身体/安全 profile 查询能否预测 body-swept robust q-contact 的删失生存分布，
> 并由同一分布得到 horizon-consistent occupancy 与有界 clearance 查询？

它是现行 Assistive Geometry 单帧 direct-head 主线的并行 `WILD_LAB` 路线，不是 B1 的
下一阶段，也不改变 B1、C0、D0、M0 或 DepthART 的 successor、训练进程和终态。

当前唯一真源是本页。准备阶段边界保留在
[`BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0_PREPARATION_PROTOCOL_2026-08-09.json`](BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0_PREPARATION_PROTOCOL_2026-08-09.json)；
当前 H1 实现、输入、parent roster、资源门和 TRAIN-only canary gate 由
[`BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_TRAIN_CANARY_PROTOCOL_2026-08-09.json`](BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_TRAIN_CANARY_PROTOCOL_2026-08-09.json)
冻结。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_PARENT_LEVEL_TRAIN_SUPPORT_AUDIT_AND_RELOCK`

H1-only 的 target compiler、censor/UNKNOWN mask、四桶 hazard decoder、right-censored
likelihood、参数精确匹配 head、checkpoint/flip/zero-support/gradient 测试和 parent-disjoint
TRAIN-only canary 已冻结。H1 protocol 同时作为 exact-three-input 的 embedded shared-resource
manifest：TRAIN target 明确登记为 `CONTENT_INSPECTED / TRAIN_TARGET_INPUT_ONLY`，runner 在使用前
逐个复算所选 RGB/NPZ 的 size 与 SHA-256；fit/eval 的 event、right-censor、known-occupied 或
clearance-event 任一支持为零即形成 `H1_TRAIN_CANARY_NOT_EVALUABLE_DATA_SUPPORT`，不得用伪分母
产生科学结论。Attempt 01 的 16-frame performance-only pilot 得到 finite feature、`388 MiB`
峰值和 `577.126 s` 全量投影，但冻结 conservative maximum 为 `1214.252 s > 900 s`，因此形成
`H1_TRAIN_CANARY_PERFORMANCE_NOT_QUALIFIED`，未训练 head、未生成 checkpoint。Attempt 02 把
feature-extraction batch 从 4 重锁为 16，但 combined timing 反而为 `9.995 s`，同样不合格。
两次 pilot 随后定位到 estimator 把一次性 model load 乘了 `full/pilot=64` 的缺陷。Attempt 03
恢复 batch 4，只将估算器改为“fixed model load + scaled variable extraction + 30 s”，conservative
maximum 为该投影的 2 倍；模型、loss、roster、frame selection 和科学 gate 均不变，并使用全新
输出 namespace。Attempt 03 pilot 以 `349.621 s` 投影、`699.242 s` conservative maximum 合格；
full run 随后逐 SHA 复核 1024 个 RGB/target 并完成 frozen feature，但科学支持前门发现 fit
`event/censor/occupied=1213/18/3162`、eval `262/0/784`，因此在 head 初始化/训练前形成
`H1_TRAIN_CANARY_NOT_EVALUABLE_DATA_SUPPORT`，没有 checkpoint。下一步只允许无模型、TRAIN-only
的 parent-level support audit；它必须在任何模型 outcome 外预冻结具备非零支持的新 split，且
披露 support-based roster selection。H1 形成有效科学终态后，H2 才能成为新的唯一
successor；只有 H1、H2 各自通过后才能另立组合协议。

该 audit 已由
[`BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_PARENT_SUPPORT_AUDIT_PROTOCOL_2026-08-09.json`](BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_PARENT_SUPPORT_AUDIT_PROTOCOL_2026-08-09.json)
冻结：扫描同一 16-parent TRAIN target roster 的 parent-level support，只读取 target NPZ；eval
固定取 manifest 顺序中前 4 个 selected-64 四类支持均非零的 parent，不加载模型或 feature。

## 与其他路线并行

- AG-QSF 使用独立代码 Module、协议、run identity、checkpoint、target cache、进度和 artifact root。
- 其他路线无需等待 AG-QSF；AG-QSF 也不得等待或修改正在运行的 B1 seed、scheduler、GPU
  进程、checkpoint、progress、Development gate 或候选集合。
- 并行允许共享不可变数据、冻结初始化、相机/几何合同、truth-reader 约定、synthetic fixture、
  文献、测试工具和明确标注的 operational lesson。
- 每项共享输入必须由 QSF resource manifest 记录 producer、logical path、版本身份、provenance、
  license scope、数据角色、outcome access 和 selection influence；共享访问一律只读。
- 共享 TRAIN 或已消费 Development 信息不会获得 fresh/Confirmation 身份。跨路线 outcome 若仅作
  诊断，必须标为 `DIAGNOSTIC_ONLY`；若影响算法或门，则该证据只保留 Development 权限。
- B1 Development/Confirmation artifact 与 consumed role 由 producer、role 和 path 共同硬拒绝；
  tracked Development/Confirmation protocol 仅可作为 `SCHEMA_ONLY` 阅读。混合角色 B0 source root
  不得作为内容输入，必须先由逐文件 hash 的 TRAIN-only role-filtered manifest 收窄。

## 独立输出

- evidence/report：`artifacts.local/evidence/assistive-geometry-qsf/`
- checkpoint/model：`artifacts.local/models/assistive-geometry-qsf/`
- target/cache/临时工作：`artifacts.local/work/assistive-geometry-qsf/`

禁止向其他路线的 artifact root 写入任何文件；禁止把别的路线 active run 目录登记成共享输入。
当 B1 正式 seed 正在运行时，QSF 只做 CPU/synthetic/文档与轻 I/O 工作，不启动竞争 GPU、
显存或重 I/O 的任务。H1 validator 会把 B1 formal runner 进程或 `<5000 MiB` 空闲显存返回为
`H1_CANARY_DEFERRED_RESOURCE_ISOLATION`，这只是调度延后，不是 H1 科学失败。

## 当前允许

- QSF 协议、schema、validator、synthetic mechanics、测试和冻结的 H1-only 实现；
- 在 foreign formal GPU 完全空闲、runtime preflight 通过后运行 H1 performance pilot 与
  TRAIN-only canary；
- H2-only 的非可执行 schema/接口占位；
- 通过逐项 manifest 只读复用冻结资源；
- 不访问受保护 outcome 的 TRAIN-only 数据审计和 canary 设计。

## 当前禁止

- 与 B1 formal GPU/重 I/O 重叠运行 H1 pilot/canary，读取 B1 Development/Confirmation outcome
  或抢跑 candidate comparison；
- 读取或复用其他路线 active checkpoint、progress、optimizer/scheduler/RNG state；
- 把 invalid/support 缺失当 right-censored clear，或把 `UNKNOWN` 当 negative；
- 训练 `H1+H2`、启动 C0/D0/M0、接 Android/HTP、替换 DA2 或修改默认 App；
- H1 canary 形成有效科学终态前实现、物化或训练 H2；
- 声称 learnability、优于 B1、论文 novelty、Confirmation、部署、产品或助盲安全。

## Claim ceiling

当前只证明 H1 target/mask/decoder/loss/head mechanics、精确参数匹配、parent-disjoint canary
合同与资源隔离门已实现并通过专项测试；Attempt 01/02 是性能负终态，Attempt 03 性能合格但
held-out right-censor 支持为零，故科学结果为 `NOT_EVALUABLE`。尚未证明 H1/H2
可学、真实数据有效、优于 direct head、可部署或安全。
