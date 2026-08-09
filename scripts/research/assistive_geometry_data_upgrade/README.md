# Assistive Geometry Data Upgrade Engine Module

状态：`current / AG-DUE_R0_GAP_PROTOCOL_LOCKED / STATIC_VALIDATOR_READY / REAL_SOURCE_PRESCREEN_NOT_AUTHORIZED`

本 Module 提供 AG-DUE 的 metadata-only source admission 合同。它不下载或打开 payload，不运行
Teacher，不生成 pseudo-label，不物化训练集，也不训练模型。

## 稳定 Interface

- [`validate_due_r0.py`](validate_due_r0.py)：校验冻结 protocol/hash，并对已另行锁定的 source
  manifest 计算 capability、orientation、per-parent 与 joint-parent gate；
- [`test_validate_due_r0.py`](test_validate_due_r0.py)：覆盖 `PRESCREEN_ADMIT`、PARTIAL、license/privacy
  REJECT、claim-bound receipt、source-native 非自动 GT、Teacher candidate、protected-role firewall、
  DCA threshold drift、UNKNOWN 和 joint-parent 负控；
- 机器 protocol、gap contract 与 source schema 位于
  `docs/research/assistive-geometry-data-upgrade/`。

只验证冻结 protocol：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.assistive_geometry_data_upgrade.validate_due_r0
```

source manifest 尚未获得 execution authority。未来锁定后才能增加 `--source-manifest`；输出必须写入
`artifacts.local/evidence/assistive-geometry-data-upgrade/`，且 `PRESCREEN_ADMIT` 仍只允许另锁
source-specific integrity/payload-audit protocol；它不建立 source data support 或 DCA PASS。

## 输出

- 无 manifest 参数时只向 stdout 输出静态 protocol `VALID/INVALID`；
- 未来锁定的真实 prescreen receipt 只能写入
  `artifacts.local/evidence/assistive-geometry-data-upgrade/<source-id>/`；
- governed protocol、gap contract 与 source schema 保持在
  `docs/research/assistive-geometry-data-upgrade/`。

## 安全边界

只允许 metadata-only、claim-bound prescreen；不得打开 source payload、下载数据、运行 Teacher、
生成 pseudo-label、物化训练集或访问 protected outcome。任何 `PRESCREEN_ADMIT` 都不能建立
source truth、DCA PASS、F1 execution、默认 App、产品或 safety authority。

## 停止条件

- protocol、DCA requirements、source schema、validator/test 或 protected-roster binding 漂移：停止；
- license/privacy/access、ancestry/identity、角色冲突或 UNKNOWN-as-negative 失败：`REJECT`；
- claim-bound receipt、joint parent、orientation 或 capability gate 不足：`PARTIAL`；
- 只有 governance 与至少一个 gap screen 同时匹配才可 `PRESCREEN_ADMIT`，随后仍必须停止并另锁
  source-specific integrity/payload-audit protocol。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_INITIAL_SOURCE_MANIFEST_LOCK`

禁止在该 successor 前运行真实 source prescreen、下载数据、读取现有 payload、调用 Teacher 或
启动任何模型/训练/Development/Confirmation。
