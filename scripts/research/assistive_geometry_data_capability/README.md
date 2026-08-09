# Assistive Geometry Data Capability Atlas Module

状态：`current / AG-DCA_R0_COMPLETE / REUSABLE_CHECKER / NO_ALGORITHM_AUTHORITY`

本 Module 是数据能力基础设施，不是算法路线。它逐 bytes/SHA 扫描冻结的 16-parent / 4,800-frame
TRAIN target，只统计 truth/source capability；随后把版本化 hypothesis requirements 映射为
`SUPPORTED_FOR_PROTOCOL_LOCK` 或 `NOT_SUPPORTED_*`。任何 PASS 只允许编写下一份协议，不授权算法执行。

## 稳定 Interface

- [`build_capability_atlas.py`](build_capability_atlas.py)：全量 atlas producer、parent/orientation
  聚合、joint parent-disjoint checker 与 authority checker；
- [`check_hypothesis_requirements.py`](check_hypothesis_requirements.py)：把已有 atlas 与新的版本化
  requirements 合同快速重放；PASS 仍只允许另锁协议；
- [`test_build_capability_atlas.py`](test_build_capability_atlas.py)：时序、聚合、joint support 与
  data/authority 分离回归；
- 机器 protocol 与 hypothesis requirements 位于
  `docs/research/assistive-geometry-data-capability/`。

R0 full-TRAIN atlas 已执行完成，三条冻结 hypothesis 均为 `NOT_SUPPORTED_*`。当前无活动 DCA
execution successor；新 hypothesis 只能提供新的版本化 requirements 并重放不可变 atlas，不得修改
R0 requirements/gate。

## 输出

- `artifacts.local/evidence/assistive-geometry-data-capability/r0/atlas.json`
- `artifacts.local/evidence/assistive-geometry-data-capability/r0/hypothesis-decisions.json`
- governed summary：
  `docs/research/assistive-geometry-data-capability/BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_RESULT_2026-08-10.{json,md}`

## 安全边界

只读 TRAIN truth/source 字段；不读 RGB、模型、feature、checkpoint、B1 consumed Development、
Calibration 或 Confirmation。`UNKNOWN` 不计作 clear/occupied/support。checker 不能授予执行、选择、
晋级、R2 mainline、默认 App、产品或 safety authority。

## 停止条件

- manifest、target receipt、implementation 或 requirements hash 漂移：停止且不产出 atlas；
- capability 不足：对应 hypothesis `NOT_SUPPORTED_DATA`；
- reducer/interface/fresh-outcome 等 authority 不足：`NOT_SUPPORTED_AUTHORITY`；
- 两者均不足：`NOT_SUPPORTED_DATA_AND_AUTHORITY`；
- 只有全部通过才返回 `SUPPORTED_FOR_PROTOCOL_LOCK`，仍不得直接执行 hypothesis。
