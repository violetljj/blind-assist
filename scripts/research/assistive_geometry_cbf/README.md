# Assistive Geometry CBF research Module

状态：`current / WILD_LAB / R0_CLOSED_DATA_SUPPORT_NOT_EVALUABLE / ORACLE_NEVER_AUTHORIZED / MODEL_AND_TRAINING_NEVER_AUTHORIZED`

本 Module 是独立的 `BLINDASSIST_ASSISTIVE_GEOMETRY_CBF_R0` 路线。它先核验现有 TRAIN-only
source geometry 能否支撑 ground-aligned `32×31` 2.5D 网格；通过后才可另锁
body-profile configuration-space inflation、connected free space 与 maximum-bottleneck corridor
oracle。审计只有 `44/1024` 帧 evaluable、0/16 parent 过门，故 R0 已在 oracle 前关闭；它不继承
A0、AG-QSF 或合成 H3 的科学 authority。

## 稳定 Interface

- [`audit_grid_support.py`](audit_grid_support.py)：逐 bytes/SHA 复核固定 TRAIN target，读取且只读取
  depth/valid/ground/intrinsics/gravity/camera-height，执行 parent、orientation 和纵横覆盖门；
- [`test_audit_grid_support.py`](test_audit_grid_support.py)：冻结坐标约定、UNKNOWN、边界门和抽样顺序；
- [机器协议](../../../docs/research/assistive-geometry-cbf/BLINDASSIST_ASSISTIVE_GEOMETRY_CBF_R0_DATA_SUPPORT_AUDIT_PROTOCOL_2026-08-09.json)：
  冻结输入、实现、阈值、terminal 与 claim ceiling。
- [governed result](../../../docs/research/assistive-geometry-cbf/BLINDASSIST_ASSISTIVE_GEOMETRY_CBF_R0_DATA_SUPPORT_AUDIT_RESULT_2026-08-09.md)：
  数据支撑 `NOT_EVALUABLE` 与路线关闭收据。

历史复现命令（现有输出路径已占用，禁止覆盖）：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.assistive_geometry_cbf.audit_grid_support
```

## 输出

- `artifacts.local/evidence/assistive-geometry-cbf/r0-data-support-audit/result.json`（完整结果，SHA-256
  `C6D151BC...DA34`）；
- 本 R0 不得创建后续 oracle/model/checkpoint 输出。

## 安全边界

只读使用固定 B1 TRAIN target cache；硬拒绝 Development/Confirmation 路径与自定义协议扩权。
`UNKNOWN` 不是 free、occupied 或 negative。A0 consumed Development Selection 不得读取、复用、
重标或参与阈值选择。当前不授权 oracle outcome、模型、训练、Android/HTP、默认 App、产品或助盲安全主张。

## 停止条件

- 输入 manifest、target bytes/SHA、实现 SHA、字段或路径漂移：停止，不产出科学 terminal；
- 数据支撑 gate 未过：签署 `AG_CBF_R0_DATA_SUPPORT_NOT_EVALUABLE_ROUTE_CLOSE`，路线关闭；
- 数据支撑 gate 已失败：R0 无 successor，不得事后降低门或用 UNKNOWN 填补覆盖；
- 任一受保护 outcome 被打开：本 evidence version 无效并停止。
