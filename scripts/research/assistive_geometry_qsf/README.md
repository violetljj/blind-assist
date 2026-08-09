# Assistive Geometry QSF research Module

状态：`current / WILD_LAB / H1_IMPLEMENTED / ATTEMPT_03_PERFORMANCE_QUALIFIED / H1_NOT_EVALUABLE_EVAL_RIGHT_CENSOR_ZERO / H2_NOT_AUTHORIZED`

## 研究问题与版本

- route：`BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0`
- protocol：`BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_TRAIN_CANARY_PROTOCOL_2026-08-09`
- stage/profile：`CANARY / CANARY_LITE`
- current：[Assistive Geometry QSF](../../../docs/research/assistive-geometry-qsf/README.md)

本 Module 当前实现 H1 censored robust-contact survival：四个 hazard bins 结构化派生
`1.0/1.5/2.0 m` occupancy，并保留独立 confidence；H2 profile-conditioned swept
configuration clearance 仍只保留非可执行 schema/接口占位。H1 canary 形成有效科学终态前
不得实现、物化或训练 H2。

## 稳定 Interface

H1 mechanics 与参数匹配实现：

- [`h1_survival.py`](h1_survival.py)：target compiler、event/censor/UNKNOWN 分离、hazard
  distribution/decoder、loss、parameter-matched head；
- [`run_h1_train_canary.py`](run_h1_train_canary.py)：只读复用 TRAIN targets 与官方
  DepthART 初始化，逐文件复核所选 RGB/target 的 producer size/SHA，冻结 encoder 后提取 pooled
  band feature，再执行有非零科学分母前门的 parent-disjoint head canary；
- [`validate_h1_train_canary.py`](validate_h1_train_canary.py)：实现/input/roster/gate/resource
  lock 与 foreign GPU 隔离预检。

验证 tracked preparation protocol：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.assistive_geometry_qsf.validate_qsf_preparation
```

验证某轮共享输入 manifest 和计划输出路径：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.assistive_geometry_qsf.validate_qsf_preparation `
  --resource-manifest artifacts.local/work/assistive-geometry-qsf/<run>/shared-resources.json `
  --planned-output artifacts.local/evidence/assistive-geometry-qsf/<run>
```

共享 manifest 必须符合 [`shared_resource_manifest.template.json`](shared_resource_manifest.template.json)
的字段合同。validator fail closed：只读、immutable identity、provenance、license scope、数据角色、
outcome access、selection influence 或 owned output root 任一不合格即拒绝。
文件 identity 使用实算 `SHA256` 或与当前消费内容一致的 pinned `GIT_COMMIT` blob；目录必须使用
`MANIFEST_SHA256`，且 `manifest_path` 位于该目录内、声明 `complete=true`，逐文件绑定相对路径、
size 与 SHA-256；validator 会复算全部成员并拒绝遗漏、额外文件、symlink 或 Windows junction。最小格式见
[`fixtures/directory_identity/directory.manifest.json`](fixtures/directory_identity/directory.manifest.json)。
路径不存在、hash/blob 不匹配或自定义协议扩权都会拒绝。
producer/path-specific 门另行拒绝 B1 consumed Development/Confirmation 与对应 artifact；相关
tracked protocol 仅允许 schema-only，混合角色 B0 source 只有收窄成逐文件 hash 的 TRAIN-only
manifest 后才可作为内容输入。当前 H1 protocol 自身就是 exact-three-input embedded manifest，
并把实际读取的 target 诚实登记为 `CONTENT_INSPECTED / TRAIN_TARGET_INPUT_ONLY`。

验证当前 H1 lock：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.assistive_geometry_qsf.validate_h1_train_canary
```

资源预检：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.assistive_geometry_qsf.validate_h1_train_canary `
  --runtime-preflight
```

返回 `H1_CANARY_DEFERRED_RESOURCE_ISOLATION` 时不得运行 pilot；它不表示科学失败。Attempt 01
因 conservative maximum `1214.252 s > 900 s` 形成 performance-only 负终态；Attempt 02 的
batch 16 combined timing 没有改善，并暴露 estimator 将一次性 model load 随帧数放大的缺陷。
Attempt 03 恢复 batch 4，分离 fixed setup 与 scaled extraction，使用
`h1-train-canary-attempt-03-r0` namespace。资源 READY 后唯一执行顺序仍是 `pilot-r0`，再由同一
Attempt-03 protocol/hash 运行 `run-r0`。

Attempt 03 pilot 已合格；full run 在训练 head 前由 nonzero-denominator frontdoor 停止：fit
`event/censor/occupied=1213/18/3162`，eval `262/0/784`。它是 data-support `NOT_EVALUABLE`，
不是 H1 科学 PASS/FAIL。没有模型 checkpoint 被物化。当前唯一 successor 是无模型、TRAIN-only
parent-level support audit 与新 split relock；H2 继续不可实现、不可物化、不可训练。

## 输出

- `artifacts.local/evidence/assistive-geometry-qsf/`
- `artifacts.local/models/assistive-geometry-qsf/`
- `artifacts.local/work/assistive-geometry-qsf/`（含可重建 derived targets/cache）

所有其他路线目录均为 foreign-owned；本 Module 不得写入。
当 B1 正式 seed 运行时只允许 CPU/synthetic/light-I/O 工作；H1 GPU feature extraction
必须通过当前 runtime preflight，不能与 foreign formal runner 重叠。

## 安全边界

`UNKNOWN` 不是 negative；invalid/support 缺失不是 right-censored clear。共享资源不会传播
Confirmation、部署、产品或安全 authority。不得控制其他路线进程、读取 active checkpoint/progress、
读取 B1 Development/Confirmation outcome、接 Android/HTP 或修改默认 App。

## 停止条件

- manifest 或输出所有权验证失败：只停止该准备/run identity；
- H1/H2 mechanics 测试失败：只停止对应实现版本；
- 资源 version/provenance/license 不明：该资源局部 `DO_NOT_REUSE`，不关闭路线；
- H1 与 H2 未分别过 canary：`H1+H2` 保持不可训练。

## 假设与规则质疑

H1/H2 的 causal difference、falsifier 和 kill gate 由 route current 与 machine protocol 持有。
门槛若被证据证明错位，必须建立版本化 challenge；不得静默更改 mask、coverage 或 outcome 角色。

## 失败资产复用

失败实现和合成反例可转为 `REGRESSION_FIXTURE`、`COUNTEREXAMPLE` 或 `DIAGNOSTIC`；共享的
TRAIN/consumed 信息必须保留原角色，不能重新包装为 unseen Confirmation。
