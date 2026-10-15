# 项目脚本入口

本目录收纳构建、设备、发布和仓库维护入口，以及仍在使用的
[研究辅助模块](research/README.md)。命令均从仓库根目录执行。
[统一命令与研究工具](../tools/README.md) · [代码地图](../docs/CODE_MAP.md)

## 按任务选择

| 任务 | 入口 | 说明 |
| --- | --- | --- |
| Android 构建与单元测试 | [run_android_gradle.ps1](run_android_gradle.ps1) | 包装 Gradle、SDK/JDK 选择及构建环境 |
| 设备回归 | [run_device_regression.ps1](run_device_regression.ps1) | 显式设备矩阵；见[设备回归说明](../docs/DEVICE_REGRESSION.md) |
| 融合实验资源安装 | [install_mobile_fusion_assets.ps1](install_mobile_fusion_assets.ps1) | 将已准备的模型与标定文件安装到指定设备，并核对哈希 |
| 模型接口检查 | [inspect_tflite.py](inspect_tflite.py) | 检查 TFLite 输入、输出形状和类型 |
| 发布验证 | [verify_release_apk.ps1](verify_release_apk.ps1)、[verify_apk_16kb.ps1](verify_apk_16kb.ps1) | 见[发布与验证](../docs/RELEASE_AND_VERIFICATION.md) |
| 发布记录与归档 | [generate_release_manifest.ps1](generate_release_manifest.ps1)、[archive_apk.ps1](archive_apk.ps1) | 生成清单，归档里程碑 APK |
| 开源发布元数据 | [check_open_source_readiness.ps1](check_open_source_readiness.ps1) | 检查公开发布所需的仓库说明 |
| 目录结构与文档链接 | [check_project_structure.ps1](check_project_structure.ps1)、[check_docs_index.ps1](check_docs_index.ps1) | 结构调整或导航更新后的定向检查 |
| 生成文件、密钥与依赖卫生 | [check_repo_hygiene.ps1](check_repo_hygiene.ps1) | 范围和选项见脚本参数 |
| 工作区归属 | [show_worktree_scope.ps1](show_worktree_scope.ps1) | 只读查看分支、工作区与未提交研究文件 |
| 知识索引维护 | [refresh_knowledge.ps1](refresh_knowledge.ps1) | 知识库源记录变更时使用；日常查询见 [knowledge.py](../tools/knowledge.py) |

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:assembleDebug
pwsh -NoProfile -File scripts/check_docs_index.ps1
```

[project.ps1](project.ps1) 保留旧调用兼容；新调用使用 [tools/ba.ps1](../tools/ba.ps1)。
`test_*.ps1` 验证相应维护脚本；研究模块测试与模块代码放在一起。

## 研究代码在哪里

具体研究路线的状态、协议、实现与结果从 [research/active](../research/active/)
各路线 `CURRENT.md` 进入；`scripts/research/` 中现存的辅助模块继续保留。
已归档代码按[历史索引](../docs/history-index.md)查找，不根据目录名判断代码是否过时。
