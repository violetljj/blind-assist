# Assistive Geometry Data Upgrade Engine Module

状态：`current / AG-DUE_R0_SANPO_INITIAL_STATIC_PRESCREEN_COMPLETE_BOTH_PARTIAL / SOURCE_SUPPORT_FALSE / PAYLOAD_NOT_AUTHORIZED`

本 Module 提供 AG-DUE 的 metadata-only source admission 合同。它不下载或打开 payload，不运行
Teacher，不生成 pseudo-label，不物化训练集，也不训练模型。

## 稳定 Interface

- [`validate_due_r0.py`](validate_due_r0.py)：校验冻结 protocol/hash，并对已另行锁定的 source
  manifest 计算 capability、orientation、per-parent 与 joint-parent gate；
- [`test_validate_due_r0.py`](test_validate_due_r0.py)：覆盖 `PRESCREEN_ADMIT`、PARTIAL、license/privacy
  REJECT、claim-bound receipt、source-native 非自动 GT、Teacher candidate、protected-role firewall、
  DCA threshold drift、UNKNOWN 和 joint-parent 负控；
- [`validate_due_sanpo_manifest_lock.py`](validate_due_sanpo_manifest_lock.py)：只校验 SANPO metadata
  bootstrap 与两份零计数 manifest 的 byte/hash/identity/authority lock，不调用 prescreen evaluator；
- [`test_validate_due_sanpo_manifest_lock.py`](test_validate_due_sanpo_manifest_lock.py)：7 项 mutation
  覆盖 payload access、split SHA、非零 frame assertion、metadata→truth、identity 与 authority 漂移；
- [`validate_due_sanpo_prescreen_result.py`](validate_due_sanpo_prescreen_result.py)：固定两份 locked
  manifest 的 exact path/hash，重新计算完整 R0 结果并验证 governed terminal、权限与 successor；
- [`test_validate_due_sanpo_prescreen_result.py`](test_validate_due_sanpo_prescreen_result.py)：7 项
  replay/mutation 覆盖 decision、source support、hard rejection、payload authority 与 successor 漂移；
- 机器 protocol、gap contract 与 source schema 位于
  `docs/research/assistive-geometry-data-upgrade/`。

验证冻结 protocol 与 governed result：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.assistive_geometry_data_upgrade.validate_due_r0
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.assistive_geometry_data_upgrade.validate_due_sanpo_prescreen_result
```

两份 SANPO source manifest 已按冻结 successor 完成一次 static prescreen，均为 `PARTIAL` 且 hard
rejection 为空。通用 evaluator 只用于 mechanics/diagnostic；tracked governed terminal 必须由专用
result validator 对 exact locked path/hash 重放，不能由任意 `--source-manifest` 或 `--output` 签署。

## 输出

- 无 manifest 参数时只向 stdout 输出静态 protocol `VALID/INVALID`；
- 本轮完整 prescreen 输出位于
  `artifacts.local/evidence/assistive-geometry-data-upgrade/sanpo-initial-static-prescreen-r0/`；
- tracked governed result 保存确定性摘要与 canonical replay SHA；validator 从锁定 manifest 重算，
  不要求本机 artifact 存在，也不把 artifact 当 source truth；
- governed protocol、gap contract、source schema 与 result 保持在
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

`BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_SOURCE_SPECIFIC_INTEGRITY_AND_CAPABILITY_AUDIT_PROTOCOL_LOCK`

只允许另锁 Synthetic source-specific integrity/capability audit 协议；禁止提前执行审计、联网刷新
metadata、下载数据、读取现有 payload、调用 Teacher 或启动任何模型/训练/Development/Confirmation。
