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

仓库内 `.jdk`、`.android-sdk`、`.gradle-local` 等旧路径在迁移期是 junction，不是工具的真实存储位置。`.python311` 与 `.venv-export312` 暂保留原位：Windows 对基础 DLL 的占用阻止了安全删除，而 venv 内含绝对解释器路径，需在 `E:\codex-tools` 重建并验证后再切换。旧 `.venv-export` 已迁移为兼容 junction，但它原先依赖已不存在的 `G:\Python`，不作为可用验证环境。

## 维护规则

- 不提交 `artifacts.local/` 下的原图、数据集、日志、模型源文件或设备证据。
- 新脚本不得在仓库根目录创建新的数据、模型、下载或临时目录。
- 文档引用本地证据时使用 `artifacts.local/evidence/...`。
- 正式 APK 仍按 [APK_ARCHIVE.md](APK_ARCHIVE.md) 归档，不放入本目录充当发布物。
- 清理前确认不存在唯一验证证据；目录迁移应先核对文件数量和失败数。
