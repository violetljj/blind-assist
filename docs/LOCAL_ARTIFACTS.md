# 本地产物目录说明

BlindAssist 的源码根目录只保留代码、文档、配置和明确需要版本管理的材料。所有下载、数据集、设备证据、训练输出、模型源文件和临时工作区统一收敛到：

```text
artifacts.local/
```

该目录被 `.gitignore` 忽略，默认不提交 Git。

## 物理存储位置（2026-08-05 起）

`artifacts.local/` 是项目内的稳定逻辑路径，但其物理存储位于：

```text
F:\ba-data\blindassist-artifacts-20260805\
```

仓库中的 `artifacts.local` 是指向该位置的 Windows junction。因此，新的大下载、数据集、模型、解压缓存、训练/benchmark 输出和临时物化必须继续写入 `artifacts.local/<category>/`，不要新建 E: 下的旁路数据目录；这些写入会自动落到 F:。需要直接指定绝对下载根目录的工具，使用上述 F: 路径，并仍须在 `artifacts.local/` 下保留可发现的 manifest、receipt、hash 与任务记录。

CARLA 算法资产库的兼容路径 `E:\linnan\CARLA` 也是 junction，物理目标为 `artifacts.local/runtime/carla-asset-library/`。已登记 Git worktree 的本地 `artifacts.local/` 必须分别指向 `artifacts.local/worktrees/<worktree>/`；不得在 worktree 内重新创建实体数据目录。

当本机 `artifacts.local/` 是外置 junction 时，Gradle 的 root/module build directory 自动改写到 `artifacts.local/work/gradle-build/`。CI 或没有外置 artifact junction 的普通 clone 仍使用标准 `*/build/` 路径，避免改变发布工作流。

下载、迁移或容量报告必须同时说明逻辑路径和物理目标盘：当目标是 `artifacts.local/...` 时，报告 F: 的可用空间，不能把工作目录位于 E: 或 E: 的剩余空间误报为实际落盘位置。

## Canonical 结构

```text
artifacts.local/
  downloads/       # 可重新获取且具备 URL/hash/receipt 的下载缓存
  evidence/        # 不可从结论反推的证据、manifest、receipt、ledger 与结果
  models/          # ONNX/PT/SavedModel、校准数据和本地候选导出
  presentations/   # 本地演示稿备份
  work/            # 可重建的工作目录
  tmp/             # 短期文件；只有完成引用/authority 检查后才可按任务清理
```

六个顶层类别不变；跨实验复用由 [Resource fabric](RESOURCE_FABRIC.md) 在这些
类别内部提供：

```text
downloads/resource-store/          # 可重获原始数据的内容寻址对象
models/resource-store/             # 模型内容寻址对象
evidence/resource-store/           # 不可替代或 sealed 原始对象
work/resource-cache/normalized/    # 共享规范化数据
work/resource-cache/features/      # 共享特征
evidence/resource-fabric/          # catalog、lifecycle、难例、薄实验和实时报告
```

新数据/模型优先通过 `tools/data/resource_fabric.py ingest` 登记；同一内容只产生
一个 resource id。Development 数据消费后不得重新取得 fresh authority，但应以
`development_consumed` 证据状态和 `shared` 存储状态继续服务训练、诊断、
特征缓存、难例挖掘、回归和明确披露的 Development replay。新实验目录只保留
manifest、参数、小型结果和证据边界，重型媒体、张量和权重通过 resource/cache
id 引用。

所有稳定历史资产（包括尚未迁入六个 canonical 顶层类别的兼容现场）由
[资产管理体系](asset-management/README.md) 零拷贝登记。`asset_catalog.py discover`
只建立 locator、文件清单、分类和生命周期总账，不因建账移动或复制 payload；
新调用方使用 `resolve --consumer --purpose` 获取路径并自动留下消费记录。

新调用方只能写入上述六个顶层类别，并在其下使用清楚的领域/协议/version 子目录。`tmp/` 不是“随时可清空”的同义词：当前任务可能暂存尚未登记的 lock、audit 或迁移指针；清理必须按精确子目录确认所有者和持久化去向。

## 本地构建与编译产物

- Android/CMake 生成的 module-local `.cxx/` 是机器绑定缓存，必须被 Git 忽略且永不提交。其物理清理属于构建现场操作，需由当前任务所有者确认；不得因为治理优化而默认删除。
- 仓库根目录不得物理留存 `.obj/.o/.lib/.a/.exp/.dll/.so/.pdb/.ilk` 等编译器或链接器输出。它们即使被 `.gitignore` 匹配，也必须迁入明确命名的 `artifacts.local/work/<task>/` 目录，并在迁移时保留原路径、字节数和 SHA-256 回执。
- `scripts/check_repo_hygiene.ps1` 会拒绝任何出现在变更面的 `.cxx/` 路径，也会扫描根目录中已被忽略的上述 native 产物；单纯增加忽略规则不构成治理完成。

## 现存兼容/待迁移顶层目录

2026-07-28 现场还存在以下非 canonical 顶层目录：

```text
cache/ caches/ calibration/ camera-source-prescreen-r1/
counterfactual-capture-smoke/ counterfactual-capture-smoke-v1r2/
crash-diagnosis/ datasets/ experiments/ gradle-home/ logs/
r2a1/ r2a2/ r2a3/ r2a4/ synthetic/ tests/ tools/
```

这些目录是历史调用方或在途研究形成的兼容现场，不等于可删除项，也不得继续作为新脚本的默认输出：

- 数据、研究结果和复现材料在确认 manifest、引用、hash、证据角色后，分别迁入 `evidence/<domain>/` 或 `work/<domain>/`。
- 可重新获取的压缩包/源 payload 只有在 URL、SHA256、receipt 与清理记录齐全时才迁入或清理 `downloads/`。
- `cache/` 与 `caches/` 先逐调用方合并；名称相似不能证明内容可重建。
- `gradle-home/` 与 `tools/` 应最终迁往 `E:\codex-tools\projects\blindassist\state\` 或 `toolchain\`，但必须先审计本地及未跟踪调用。
- `r2a1`–`r2a4`、当前 RCLE/SANPO、Bonn/Looming 和任何被 lock/receipt 引用的目录，在协议所有者完成分类前保持原位。

禁止按文件扩展名做批量“保留 JSON、删除其余”清理：大型 JSON 可能是可重建 payload，小型二进制也可能是唯一证据。清理单位必须是带 authority 与复现信息的命名目录或 manifest 条目。

## 兼容入口

为避免一次性破坏已有脚本和历史命令，迁移阶段保留下列目录 junction：

```text
test-artifacts.local -> artifacts.local/evidence
.downloads           -> artifacts.local/downloads
work                 -> artifacts.local/work
tmp                  -> artifacts.local/tmp
.pytest_cache         -> artifacts.local/cache/pytest
.ruff_cache           -> artifacts.local/cache/ruff
.python311            -> artifacts.local/runtime/legacy-python311
.venv-export          -> artifacts.local/runtime/venv-export
.venv-export312       -> artifacts.local/runtime/venv-export312
python311-local       -> artifacts.local/runtime/python311-local
captures              -> artifacts.local/evidence/captures
output                -> artifacts.local/work/output
E:\linnan\CARLA       -> artifacts.local/runtime/carla-asset-library
E:\linnan\artifacts.local -> artifacts.local/legacy/outer-workspace-artifacts
E:\linnan\tmp         -> artifacts.local/tmp/outer-workspace
```

新脚本和新文档必须直接使用 `artifacts.local/`。旧入口将在对应脚本完成分批迁移和验证后移除。

## evidence 目录职责

- `datasets/`：评测集、manifest、标签、QA 预览和来源证明。
- `device-regression/`：真机安装、启动、截图、性能和 summary 证据。
- `detector-benchmark/`：本机检测器 smoke benchmark。
- `detector-ab-device-benchmark/`：同设备检测器 A/B 证据。
- `depth-fusion-benchmark/`：深度融合候选证据。
- `sanpo-*`：SANPO 数据、训练、门禁和 benchmark 证据。
- `legacy/`：历史迁移和早期备份，只用于追溯。

## 工具与状态不属于产物

JDK、Android SDK、Python 运行时和构建缓存位于：

```text
E:\codex-tools\projects\blindassist\toolchain\
E:\codex-tools\projects\blindassist\state\
```

仓库内 `.jdk`、`.android-sdk` 等旧路径在迁移期是 junction，不是工具的真实存储位置。`.python311` 暂保留原位；导出环境需要时应在 `E:\codex-tools` 重建，不应把虚拟环境重新提交或长期复制到仓库。

## 2026-07-26 可重建 payload 清理记录

因 E 盘容量不足，对 `E:\linnan\linnan` 做了两轮限定清理。以下是当时的历史收据，不表示这些路径在未来任意时刻仍可按同样方式清理：

- 第一轮清空 `artifacts.local/tmp`，删除两个导出虚拟环境、仓库内 Gradle 缓存、benchmark/实验构建产物、Python 安装包，以及 `ustrf-r12d` 和旧 `ustrf-sensor-replay-r2` 下载缓存；目标逻辑大小约 `23.38 GiB`。
- 第二轮只在已结束或已收口的 A/B 级实验目录中删除可重新下载或重建的 archive、bag、视频、图像帧、点云、模型和 APK 等 payload，共 `160,571` 个文件、逻辑大小 `78.718 GiB`，删除失败数为 `0`。
- 两轮操作后 E 盘可用空间由约 `8.89 GiB` 增至 `91.89 GiB`。

清理保留了 Markdown、JSON/JSONL、CSV、YAML、TXT、SHA/hash、manifest、receipt、ledger、脚本和 handoff。`sanpo-v4-real-canonical-r3-20260713`、Bonn/Looming R1、当前 RCLE 工作及 `artifacts.local/work/codex-handoffs/INDEX.md` 未删除。后续复现必须按保留的 URL、hash、manifest、receipt、ledger 和脚本重新下载或重建 payload，不应假定历史本地路径仍含原始数据。

## 维护规则

- 不提交 `artifacts.local/` 下的原图、数据集、日志、模型源文件或设备证据。
- 新脚本不得在仓库根目录创建新的数据、模型、下载或临时目录。
- `scripts/check_repo_hygiene.ps1` 在 Windows 本机检查 canonical artifact root、CARLA、兼容别名和所有已登记 worktree；实体目录或逃逸到 artifact volume 之外的 reparse target 会直接失败。
- 文档引用本地证据时使用 `artifacts.local/evidence/...`。
- 正式 APK 仍按 [APK_ARCHIVE.md](APK_ARCHIVE.md) 归档，不放入本目录充当发布物。
- 清理前确认不存在唯一验证证据；目录迁移应先核对文件数量和失败数。
- 删除前再次检查活动进程、任务 handoff、tracked/local 引用、manifest/receipt 与目标绝对路径；只删除明确命名且可恢复的范围。
- 清理可重新下载的大 payload 时，至少保留来源 URL、SHA256、manifest、receipt、ledger、执行脚本、结果摘要和 cleanup record。
