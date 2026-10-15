# 新电脑交接说明

新电脑从[快速开始](QUICKSTART_ZH.md)恢复源码与 Android 开发环境；研究工具、数据和
设备只在对应任务需要时恢复。先阅读 [AGENTS.md](../AGENTS.md)，当前工作入口见
[项目状态](PROJECT_STATE.md)与[文档索引](README.md)。

## 源码与 Android 环境

准备 Git、PowerShell 7、JDK 17、Android SDK Platform 37 与 Platform Tools。工具版本以
快速开始和 Gradle 配置为准，不复制旧电脑的 SDK、虚拟环境或机器绑定缓存。

```powershell
git clone https://github.com/violetljj/blind-assist.git
cd blind-assist
git status --short --branch
```

按新电脑的实际 SDK 位置创建未跟踪的 `local.properties`，例如：

```properties
sdk.dir=C\:\\Users\\<your-user>\\AppData\\Local\\Android\\Sdk
```

将 `JAVA_HOME` 指向实际 JDK 17 目录，再运行统一入口定位环境缺项：

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 -PreflightOnly
```

需要验证 APK 装配时运行：

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:assembleDebug
```

普通 clone 输出位于 `app/build/outputs/apk/debug/app-debug.apk`。已配置外置 artifact
junction 时，Gradle 输出改为
`artifacts.local/work/gradle-build/app/outputs/apk/debug/app-debug.apk`；映射规则见
[本地产物目录说明](LOCAL_ARTIFACTS.md)。Android 构建不需要先恢复研究数据或 Python 导出环境。

## 本地工具与研究资料

仓库不包含可直接恢复的 Codex skills 压缩包或恢复脚本；使用当前电脑已配置的工具即可。
项目规则与操作入口保存在仓库中，Android 构建不依赖个人 skills 快照。

需要研究 Python 或其他本地工具时，按示例创建本机配置：

```powershell
Copy-Item config/local.example.toml config/local.toml
```

编辑 `config/local.toml` 中实际需要的工具路径，再按对应 profile 使用 `tools/ba.ps1`。
例如继续 DTR 研究时：

```powershell
pwsh -NoProfile -File tools/ba.ps1 setup research-dtr-r0
pwsh -NoProfile -File tools/ba.ps1 doctor research-dtr-r0
```

`config/local.toml` 与 `local.properties` 都只保存本机配置，不提交 Git。远程研究环境见
[主机与研究计算](HOST_RESEARCH_COMPUTE.md)，恢复既有证据见
[关键证据备份与恢复](operations/CRITICAL_EVIDENCE_BACKUP.md)。

数据、模型、下载、证据和临时工作统一经 `artifacts.local/` 分类保存，遵循
[本地产物目录说明](LOCAL_ARTIFACTS.md)。现有主力机的盘符与 junction 是该机器的配置，
新电脑应按自己的存储布局设置；不要把旧机器路径当成已恢复的数据。历史结果的
manifest、hash 和来源记录用于确认身份，不因复制目录而获得新的评价资格。

## 手机与交付

使用新电脑 SDK 的 `platform-tools/adb` 检查设备，将构建输出的实际 APK 路径交给
`adb install -r`。普通 debug 安装与检查见[设备回归](DEVICE_REGRESSION.md)；实时融合
演示另按[硬件指南](HARDWARE_OBSTACLE_DEMO.md)准备 demo 包、实验资产与匹配硬件。

演示、交付候选和里程碑按[发布与验证](RELEASE_AND_VERIFICATION.md)、
[APK 归档策略](APK_ARCHIVE.md)留存。`releases/apk/` 保存小型收据，原始 APK 放外部归档。
后续修改前检查 `git status --short`，按实际变更运行相关验证并记录结果；无需为环境交接
重新运行历史研究实验。
