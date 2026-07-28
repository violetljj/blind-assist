# 新电脑交接说明

本文用于在新的 Windows 电脑上继续开发 BlindAssist。开始前先阅读 `AGENTS.md` 和 [文档索引](README.md)，再按下面清单恢复项目、Codex skills 和 Android 构建环境。

## 1. 克隆仓库

```powershell
git clone git@github.com:violetljj/blind-assist.git
cd blind-assist
```

如果新电脑还没有配置好 SSH，先添加 GitHub SSH key；也可以临时使用 HTTPS 远端克隆，后续再切回 SSH。

## 2. 恢复 Codex skills

仓库内包含 skills 快照：

```text
codex/skills-snapshot/codex-skills-20260522.zip
```

在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\restore_codex_skills.ps1
```

恢复脚本会把压缩包解压到：

```text
%USERPROFILE%\.codex\skills
```

如果该目录已经存在，脚本会先创建带时间戳的备份，再恢复快照。恢复后重启 Codex，让新会话重新发现 skills。

## 3. 安装 Android 开发工具

新电脑需要安装：

- Android Studio 及其自带 JDK 17。
- Android SDK Platform 35。
- Android SDK Build Tools。
- Android Platform Tools，用于 `adb`。
- Git for Windows。

然后在仓库根目录创建 `local.properties`：

```properties
sdk.dir=C\:\\Users\\<your-user>\\AppData\\Local\\Android\\Sdk
```

请把路径改成新电脑实际的 Android SDK 目录。

## 4. 验证项目

运行仓库验证命令：

```powershell
$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
$env:PATH="$env:JAVA_HOME\bin;$env:PATH"
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --no-daemon
```

debug APK 应生成在：

```text
app/build/outputs/apk/debug/app-debug.apk
```

检查随包模型资产。Python 工具统一使用 `E:\codex-tools\bin\blindassist-python.cmd`；仓库内旧 `.venv-export312` 已不存在：

```powershell
& E:\codex-tools\bin\blindassist-python.cmd scripts\inspect_tflite.py
```

期望模型形状：

```text
input shape=[1, 320, 320, 3] dtype=float32
output shape=[1, 84, 2100] dtype=float32
```

如果 Python 导出环境尚未恢复，只要已跟踪的 TFLite 资产存在，Android 构建验证仍可先运行。下载、数据集、训练和 benchmark 输出统一放在 `artifacts.local/`。

## 5. 可选手机安装

连接已开启 USB 调试或无线调试的手机后运行：

```powershell
E:\codex-tools\tools\android-sdk\platform-tools\adb.exe devices
E:\codex-tools\tools\android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
```

在新电脑上，SDK 路径可能是系统 Android SDK，而不是仓库本地 `.android-sdk`。这种情况下请使用新 SDK 的 `platform-tools` 目录中的 `adb.exe`。

## 6. 需要保留的开发规则

- `DEVELOPMENT_LOG.md` 中的执行者名称继续使用 `violjjet`。
- 修改前先运行 `git status --short`。
- 通用工具安装到 `E:\codex-tools`；项目下载、数据集和实验证据进入 `artifacts.local/`。不要提交 SDK、Gradle 缓存、虚拟环境或机器特定生成文件。
- 代码、配置、模型、测试或已采纳技术决策变更时记录到 `DEVELOPMENT_LOG.md`；纯只读排查和对话无需写日志。
- 仅在用户可见状态、用法或前置条件变化时更新 `README.md`；发布事实写入 `CHANGELOG.md`，详细验证按 [发布与验证工作流](RELEASE_AND_VERIFICATION.md) 记录。
- 演示、老师查看、交付候选、里程碑或用户明确要求的 APK，应先保存到完整本地归档目录 `E:\linnan\blind-assist-apk-archive\apks`；普通 debug 构建无需归档。
- 只有当 APK 是已记录的 Git 里程碑，或用户明确要求提交该 APK 时，才把它提交到 `releases/apk/`。
