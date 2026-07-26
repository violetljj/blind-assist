# 本地产物目录说明

BlindAssist 的源码根目录只保留代码、文档、配置和明确需要版本管理的材料。所有下载、数据集、设备证据、训练输出、模型源文件和临时工作区统一收敛到：

```text
artifacts.local/
```

该目录被 `.gitignore` 忽略，默认不提交 Git。

## 当前结构

```text
artifacts.local/
  downloads/       # 下载缓存、候选模型和外部数据
  evidence/        # 数据集、benchmark、真机回归和训练证据
  models/          # ONNX/PT/SavedModel、校准数据和本地导出
  presentations/   # 本地演示稿备份
  work/            # 可重建的工作目录
  tmp/             # 短期临时文件
```

## 兼容入口

为避免一次性破坏已有脚本和历史命令，迁移阶段保留下列目录 junction：

```text
test-artifacts.local -> artifacts.local/evidence
.downloads           -> artifacts.local/downloads
work                 -> artifacts.local/work
tmp                  -> artifacts.local/tmp
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

因 E 盘容量不足，对 `E:\linnan\linnan` 做了两轮限定清理：

- 第一轮清空 `artifacts.local/tmp`，删除两个导出虚拟环境、仓库内 Gradle 缓存、benchmark/实验构建产物、Python 安装包，以及 `ustrf-r12d` 和旧 `ustrf-sensor-replay-r2` 下载缓存；目标逻辑大小约 `23.38 GiB`。
- 第二轮只在已结束或已收口的 A/B 级实验目录中删除可重新下载或重建的 archive、bag、视频、图像帧、点云、模型和 APK 等 payload，共 `160,571` 个文件、逻辑大小 `78.718 GiB`，删除失败数为 `0`。
- 两轮操作后 E 盘可用空间由约 `8.89 GiB` 增至 `91.89 GiB`。

清理保留了 Markdown、JSON/JSONL、CSV、YAML、TXT、SHA/hash、manifest、receipt、ledger、脚本和 handoff。`sanpo-v4-real-canonical-r3-20260713`、Bonn/Looming R1、当前 RCLE 工作及 `artifacts.local/work/codex-handoffs/INDEX.md` 未删除。后续复现必须按保留的 URL、hash、manifest、receipt、ledger 和脚本重新下载或重建 payload，不应假定历史本地路径仍含原始数据。

## 维护规则

- 不提交 `artifacts.local/` 下的原图、数据集、日志、模型源文件或设备证据。
- 新脚本不得在仓库根目录创建新的数据、模型、下载或临时目录。
- 文档引用本地证据时使用 `artifacts.local/evidence/...`。
- 正式 APK 仍按 [APK_ARCHIVE.md](APK_ARCHIVE.md) 归档，不放入本目录充当发布物。
- 清理前确认不存在唯一验证证据；目录迁移应先核对文件数量和失败数。
