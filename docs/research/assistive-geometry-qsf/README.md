# BlindAssist Assistive Geometry QSF

状态：`current / WILD_LAB / PARALLEL_ROUTE_PREPARED / H1_IMPLEMENTATION_NOT_STARTED / TRAIN_CANARY_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

## 主张与边界

AG-QSF（Assistive Geometry Queryable Survival Field）研究：

> RGB 与显式身体/安全 profile 查询能否预测 body-swept robust q-contact 的删失生存分布，
> 并由同一分布得到 horizon-consistent occupancy 与有界 clearance 查询？

它是现行 Assistive Geometry 单帧 direct-head 主线的并行 `WILD_LAB` 路线，不是 B1 的
下一阶段，也不改变 B1、C0、D0、M0 或 DepthART 的 successor、训练进程和终态。

当前唯一真源是本页；机器边界见
[`BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0_PREPARATION_PROTOCOL_2026-08-09.json`](BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0_PREPARATION_PROTOCOL_2026-08-09.json)。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_ONLY_IMPLEMENTATION_AND_TRAIN_CANARY_LOCK`

只允许先实现并冻结 H1-only 的 target compiler、censor/UNKNOWN mask、hazard decoder、
right-censored likelihood、参数预算匹配 head、checkpoint/flip/zero-support/gradient 测试与
TRAIN-only canary 协议。H2-only 当前只保留 schema/接口占位，不实现、不物化、不训练；
H1 canary 形成终态后，H2 才能成为新的唯一 successor。只有 H1、H2 各自 canary 通过后
才能另立组合协议。

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

## 独立输出

- evidence/report：`artifacts.local/evidence/assistive-geometry-qsf/`
- checkpoint/model：`artifacts.local/models/assistive-geometry-qsf/`
- target/cache/临时工作：`artifacts.local/work/assistive-geometry-qsf/`

禁止向其他路线的 artifact root 写入任何文件；禁止把别的路线 active run 目录登记成共享输入。
当 B1 正式 seed 正在运行时，QSF 只做 CPU/synthetic/文档与轻 I/O 工作，不启动竞争 GPU、
显存或重 I/O 的任务；未来 GPU canary 必须在独立协议中重新做资源预检和排期。

## 当前允许

- QSF 协议、schema、validator、synthetic mechanics、测试和 H1-only 实现；
- H2-only 的非可执行 schema/接口占位；
- 通过逐项 manifest 只读复用冻结资源；
- 不访问受保护 outcome 的 TRAIN-only 数据审计和 canary 设计。

## 当前禁止

- 运行真实 TRAIN canary、读取 B1 Development/Confirmation outcome 或抢跑 candidate comparison；
- 读取或复用其他路线 active checkpoint、progress、optimizer/scheduler/RNG state；
- 把 invalid/support 缺失当 right-censored clear，或把 `UNKNOWN` 当 negative；
- 训练 `H1+H2`、启动 C0/D0/M0、接 Android/HTP、替换 DA2 或修改默认 App；
- H1 canary 形成终态前实现、物化或训练 H2；
- 声称 learnability、优于 B1、论文 novelty、Confirmation、部署、产品或助盲安全。

## Claim ceiling

当前只证明并行路线的代码、状态和资源隔离合同已准备完成。尚未证明 H1/H2 可学、真实数据有效、
优于 direct head、可部署或安全。
