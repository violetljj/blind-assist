# Assistive Geometry QSF research Module

状态：`preparation / WILD_LAB / PARALLEL_ROUTE_PREPARED / TRAIN_CANARY_NOT_AUTHORIZED`

## 研究问题与版本

- route：`BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0`
- protocol：`BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0_PREPARATION_PROTOCOL_2026-08-09`
- stage/profile：`PREPARATION / CANARY_LITE`
- current：[Assistive Geometry QSF](../../../docs/research/assistive-geometry-qsf/README.md)

本 Module 当前只为 H1 censored robust-contact survival 提供独立并行实现面；H2
profile-conditioned swept configuration clearance 仅保留非可执行 schema/接口占位。
H1 canary 形成终态前不得实现、物化或训练 H2；当前也不授权真实 TRAIN canary。

## 稳定 Interface

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

## 输出

- `artifacts.local/evidence/assistive-geometry-qsf/`
- `artifacts.local/models/assistive-geometry-qsf/`
- `artifacts.local/work/assistive-geometry-qsf/`（含可重建 derived targets/cache）

所有其他路线目录均为 foreign-owned；本 Module 不得写入。
当 B1 正式 seed 运行时只允许 CPU/synthetic/light-I/O 工作；未来 GPU canary 必须另做预检。

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
