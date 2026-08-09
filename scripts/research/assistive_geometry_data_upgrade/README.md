# Assistive Geometry Data Upgrade Engine Module

状态：`current / AG-DUE_R1_SANPO_SYNTHETIC_PREFLIGHT_NOT_EVALUABLE / EXACT_METADATA_OBJECT_MISSING / ROUTE_CLOSED`

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
- [`validate_due_sanpo_synthetic_r1_protocol.py`](validate_due_sanpo_synthetic_r1_protocol.py)：验证 exact
  Synthetic session/object paths、三阶段读取权限、factor claim ceiling、单-parent F1 门与 successor；
- [`test_validate_due_sanpo_synthetic_r1_protocol.py`](test_validate_due_sanpo_synthetic_r1_protocol.py)：13 项
  mutation 覆盖 session/body、pose/timestamp、depth/support、panoptic/boundary、parent、输出与训练扩权；
- [`run_due_sanpo_synthetic_r1_metadata_preflight.py`](run_due_sanpo_synthetic_r1_metadata_preflight.py)：
  只允许四个 exact metadata object HEAD/body 和三个 exact frame-prefix LIST；分离 metadata-local-SHA
  与 frame-provider-only receipt，锁定预算、六位 numeric index、lowest-25 和 frame-body byte=0；
- [`test_run_due_sanpo_synthetic_r1_metadata_preflight.py`](test_run_due_sanpo_synthetic_r1_metadata_preflight.py)：
  20 项离线 canary 覆盖 path/body escape、receipt、suffix、alias/duplicate、camera/K/fps、pose/timestamp、
  count-to-capability 污染、output 与 observed-404 fail-close；
- [`validate_due_sanpo_synthetic_r1_preflight_result.py`](validate_due_sanpo_synthetic_r1_preflight_result.py)
  与对应 tests：绑定 exact 404、attempt/artifact SHA、零 inventory/capability count、权限与无 successor 终态；
- 机器 protocol、gap contract 与 source schema 位于
  `docs/research/assistive-geometry-data-upgrade/`。

验证冻结 protocol 与 governed result：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.assistive_geometry_data_upgrade.validate_due_r0
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.assistive_geometry_data_upgrade.validate_due_sanpo_prescreen_result
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.assistive_geometry_data_upgrade.validate_due_sanpo_synthetic_r1_protocol
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.assistive_geometry_data_upgrade.validate_due_sanpo_synthetic_r1_preflight_result
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
- R1 preflight 输出位于
  `artifacts.local/evidence/assistive-geometry-data-upgrade/sanpo-synthetic-r1-metadata-preflight/`；
  保存 attempt/retry、fail-closed inventory/schema 与 result receipt，不保存 raw metadata 或 frame body；
- governed protocol、gap contract、source schema 与 result 保持在
  `docs/research/assistive-geometry-data-upgrade/`。

## 安全边界

R1 preflight 已 `NOT_EVALUABLE` 关闭。frame body、Teacher、pseudo-label、物化、训练与 protected outcome
继续禁止；404 不得通过替代路径、fallback session 或扩大 LIST 救援。任何 inventory 或 metadata access
都不能建立 source truth、DCA PASS、F1 execution、默认 App、产品或 safety authority。

## 停止条件

- protocol、DCA requirements、source schema、validator/test 或 protected-roster binding 漂移：停止；
- license/privacy/access、ancestry/identity、角色冲突或 UNKNOWN-as-negative 失败：`REJECT`；
- claim-bound receipt、joint parent、orientation 或 capability gate 不足：`PARTIAL`；
- 只有 governance 与至少一个 gap screen 同时匹配才可 `PRESCREEN_ADMIT`，随后仍必须停止并另锁
  source-specific integrity/payload-audit protocol。
- R1 发现 session/split/camera/lens 漂移、metadata schema 不完整、pose/timestamp 推断、factor truth
  升格、单 parent 冒充 12-parent 门或输出越界时立即停止。

## 当前终态

`NONE_STOP_AT_PREFLIGHT_TERMINAL`

本 exact source/session/path 无继续执行 authority。未来若提出新 source/session/path，必须另行版本化
R0 source manifest 与 source-specific protocol；禁止从本 result 直接进入 body canary。
