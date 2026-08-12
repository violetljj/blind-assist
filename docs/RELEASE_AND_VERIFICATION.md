# 发布与验证工作流

本文件定义 BlindAssist 的验证深度、版本判断和交付闭环。它补充项目 `AGENTS.md`，但不替代专项脚本或测试文档。

## 验证分级

| 变更类型 | 最低验证 | 何时升级 |
| --- | --- | --- |
| 文档、注释、协作规则 | 链接、路径和文本自检 | 文档改变构建或使用说明时，按相应影响范围验证 |
| 单模块 Kotlin 逻辑 | 相关模块单测或静态检查 | 影响跨模块接口、状态流或 Android 资源时，增加构建 |
| 相机、检测、风险、反馈、权限、模型资产 | 相关单测/检查 + `:app:assembleDebug` | 需要交付、风险规则或真实设备行为变化时，增加真机回归 |
| 交付候选、演示包、里程碑 | 完整相关测试与 lint、debug APK、按脚本参数运行 `scripts/verify_release_apk.ps1` | 有在线设备时运行 `docs/DEVICE_REGRESSION.md` 的真机回归 |

先读取各模块和脚本的现有说明；不要为了满足流程而无差别运行全部 Gradle 任务。验证失败或未运行时，记录原因、影响和后续动作。

## 版本与文档

- 版本号服务于交付与可追溯性，不按任务次数递增，也不追溯性重命名历史版本。
- 准备交付，或用户可见行为、兼容性、安全边界、模型、权限或核心架构发生实质变化时，评估并记录版本变更。
- `README.md` 维护当前用户可见状态和用法；`CHANGELOG.md` 记录实际发布版本；`DEVELOPMENT_LOG.md` 记录实现过程、验证和未决风险。

## 交付闭环

1. 检查 `git status --short`，确认交付范围没有夹带无关改动。
2. 运行与风险匹配的测试、lint 和构建；对 debug 交付 APK 至少运行：

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\verify_release_apk.ps1 -ApkPath .\app\build\outputs\apk\debug\app-debug.apk
   ```

   如需锁定版本断言，再补充 `-ExpectedVersionCode` 和 `-ExpectedVersionName`；其他包型或 SDK 路径按 `Get-Help .\scripts\verify_release_apk.ps1 -Full` 调整。
3. 仅在演示、老师查看、交付候选、里程碑或用户明确要求时，运行 `scripts/archive_apk.ps1`；归档与 Git 准入规则见 [APK_ARCHIVE.md](APK_ARCHIVE.md)。
4. 写入必要的 `CHANGELOG.md`、`DEVELOPMENT_LOG.md` 和用户可见 README 更新，并说明版本判断。
5. 推送前再次检查 staged diff、分支、upstream 和远端；只有普通非强推到已授权的 `origin` 才能继续。

## GitHub Release 自动化

推送与 `app/build.gradle.kts` 中 `versionName` 完全匹配的 `v*` tag 时，
`.github/workflows/release.yml` 会执行以下 fail-closed 流程：

1. 构建 debug-signed evaluation APK；
2. 校验 package、versionCode、versionName、签名可解析性和 16KB 静态兼容性；
3. 生成 `SHA256SUMS`、`release-manifest.json`、`apk-verification.json` 和证据边界说明；
4. 仅在同名 Release 尚不存在时创建 GitHub Release，拒绝覆盖不可变资产。

这条自动化不产生生产签名，也不构成真机准确率、用户效果、部署或安全证据。创建 tag 前仍须完成本页与 `DEVICE_REGRESSION.md` 中适用于该版本的验证，并更新 `CHANGELOG.md`。

## 常用环境和专项入口

- 开发环境与新电脑恢复：[NEW_COMPUTER_HANDOFF.md](NEW_COMPUTER_HANDOFF.md)
- 真机回归：[DEVICE_REGRESSION.md](DEVICE_REGRESSION.md)
- 本地产物：[LOCAL_ARTIFACTS.md](LOCAL_ARTIFACTS.md)
- APK 归档：[APK_ARCHIVE.md](APK_ARCHIVE.md)
