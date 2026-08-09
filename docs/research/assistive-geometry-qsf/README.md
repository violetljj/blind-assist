# BlindAssist Assistive Geometry QSF

状态：`current / WILD_LAB / H1_IMPLEMENTED / ATTEMPT_01_PERFORMANCE_NOT_QUALIFIED / ATTEMPT_02_BATCH16_RELOCKED_NOT_RUN / DEFAULT_APP_UNCHANGED`

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

`BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_TRAIN_CANARY_ATTEMPT_02_BATCH16_PERFORMANCE_PILOT`

H1-only 的 target compiler、censor/UNKNOWN mask、四桶 hazard decoder、right-censored
likelihood、参数精确匹配 head、checkpoint/flip/zero-support/gradient 测试和 parent-disjoint
TRAIN-only canary 已冻结。H1 protocol 同时作为 exact-three-input 的 embedded shared-resource
manifest：TRAIN target 明确登记为 `CONTENT_INSPECTED / TRAIN_TARGET_INPUT_ONLY`，runner 在使用前
逐个复算所选 RGB/NPZ 的 size 与 SHA-256；fit/eval 的 event、right-censor、known-occupied 或
clearance-event 任一支持为零即形成 `H1_TRAIN_CANARY_NOT_EVALUABLE_DATA_SUPPORT`，不得用伪分母
产生科学结论。Attempt 01 的 16-frame performance-only pilot 得到 finite feature、`388 MiB`
峰值和 `577.126 s` 全量投影，但冻结 conservative maximum 为 `1214.252 s > 900 s`，因此形成
`H1_TRAIN_CANARY_PERFORMANCE_NOT_QUALIFIED`，未训练 head、未生成 checkpoint。Attempt 02 只把
feature-extraction batch 从 4 重锁为 16；模型、loss、roster、frame selection 和科学 gate 均不变，
使用全新输出 namespace。Attempt 02 pilot 在 `900 s` 上界内合格后，才运行 `12 fit parent / 4 eval parent ×
64 frame` 的 frozen-encoder H1 canary。H1 形成有效科学终态后，H2 才能成为新的唯一
successor；只有 H1、H2 各自通过后才能另立组合协议。

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
合同与资源隔离门已实现并通过专项测试；Attempt 01 仅形成性能负终态，Attempt 02 尚未运行。
真实 TRAIN canary 尚未运行，因此尚未证明 H1/H2
可学、真实数据有效、优于 direct head、可部署或安全。
