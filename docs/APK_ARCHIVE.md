# APK 归档策略

BlindAssist 使用两层 APK 归档，既方便课堂/演示交付，也避免把每一次本地构建都变成仓库历史：

- GitHub 里程碑 APK：`releases/apk/` 只保留累计 `versionName` 差值达到 `>= 0.5` 的版本，或用户明确标记为 Git 里程碑的 APK。
- 完整本地归档：所有生成过的历史 APK 都保留在 `E:\linnan\blind-assist-apk-archive\apks`。

完整本地归档创建于 2026-05-22，包含历史 APK 集合以及后续本地 debug 归档。归档清单位于：

```text
E:\linnan\blind-assist-apk-archive\APK_ARCHIVE_MANIFEST.csv
```

校验任意已归档 APK：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "E:\linnan\blind-assist-apk-archive\apks\<apk-name>"
```

发布交付、老师查看或演示前，先把新 APK 归档到本地：

```powershell
.\scripts\archive_apk.ps1
```

只有当该 APK 符合下方里程碑规则，或用户明确批准同步到 `releases/apk/` 时，才使用 `-Milestone`。普通 debug 构建应保留在完整本地归档中；只有证据确实有价值时，才同步写入 `README.md`、`CHANGELOG.md` 或 `DEVELOPMENT_LOG.md`。

## GitHub 里程碑 APK

| APK | 大小 bytes | SHA256 |
| --- | ---: | --- |
| BlindAssist-v0.1.0-debug-958d5a9-rebuilt.apk | 32214207 | 00EE49D8142C8B35F19A36786F77BA348652684A7C2E7B602EDE3E766C3A09F7 |
| BlindAssist-v0.7.0-debug-d948f6b-rebuilt.apk | 32230591 | 8C97206267FB97CD5A56579A073553F9A96E94A0B289787B8F6277C4F94755ED |
| BlindAssist-v1.3.0-debug-e29b99a-rebuilt.apk | 32230591 | 26E9F01A326EBE90E3852C37F0FFD69D1B955BFCBFA74B4F6886FCC9B9D8998C |
| BlindAssist-v2.0.0-debug-52a0c93-rebuilt.apk | 32247033 | 769D19AEC89689C5CA566E685575F582A0C95CAE96F7009B616697A4402657A1 |
| BlindAssist-v2.6.0-debug-71f921d-current.apk | 32263443 | D937E7BDA2DC798847EB2BB4A1C67F573B12FA7338024D834B91064810899AB6 |
| BlindAssist-v3.2.0-debug-20260518-152635.apk | 46948985 | DA48B4C71BA2DE2F41F010F0ADBCC9E4B64CB2E31359B2E0D1B3327379A9CFC2 |
| BlindAssist-v4.1.0-debug-20260518-233751.apk | 47184730 | 6BB935CAB1FFC23A45FCAA003165DADB7E59A36456416E91A3891DAD977811E3 |
| BlindAssist-v4.8.0-debug-20260519-005155.apk | 47031020 | 0755CFA323BCD2BE0834850527DAF962A330B0AFE5259B0F44239CDDB0611F0D |
| BlindAssist-v5.3.0-debug-20260519-113731.apk | 47052096 | 263181516D3BFEECB4E00A6B5FE9310E282932E33EA981069B095ADEC8534D72 |
| BlindAssist-v5.8.0-debug-20260519-170622.apk | 47068480 | 290F8FAB30446C0E38024D766B7D7E63661BACCE4D9E869DE652CE79C0DEAF56 |
| BlindAssist-v6.9.0-debug-20260522-204908.apk | 47205843 | 8E29DB53AB6FA5E2AA257740E910027D96AA8F9A66501F4985535A3C788A862A |
| BlindAssist-v7.0.0-debug-20260523-000649.apk | 47205851 | E3D7F9DC265E1A173D26378AAC6D1C95429E303B245DFDBA206B95BD7063B9D5 |
| BlindAssist-v7.1.0-debug-20260524-162936.apk | 47205843 | 5ADA5DC82A71AABDA3438C76CA7E7AA9341C15FDD11308C2861E9695AD75F323 |
| BlindAssist-v7.6.0-debug-20260525-004833.apk | 47222231 | 2DE5CE894D0C46A8D099000B6E2624DA4B102E61A0E31BE8F501252D102520DC |

## 本地测试证据归档

历史 `test-artifacts/` 证据已复制到：

```text
E:\linnan\blind-assist-apk-archive\test-artifacts\test-artifacts-20260525-001501
```

该归档包含 145 个文件，总计 103,677,703 bytes。仓库后续不再跟踪新的 `test-artifacts/` 内容；未来设备回归证据应保留在 `test-artifacts.local/` 的分组目录中，或在用户明确要求时作为 release 附件处理。

## 后续规则

创建新 APK 时，先本地归档。只有当前 `versionName` 比 `releases/apk/` 中最新已提交 APK 至少高 `0.5`，或用户明确要求提交 Git 里程碑 APK 时，才把 APK 提交到 `releases/apk/`。更小更新产生的 APK 留在完整本地归档。

新的测试截图、原始设备日志、临时 APK、zip 快照、PPT 导出、ONNX/PT/NPY 模型转换中间产物和机器本地缓存默认不进入 Git。`scripts/run_device_regression.ps1` 生成的设备回归证据写入 `test-artifacts.local/device-regression/<timestamp>/` 目录，只用于后续本地对比。CI 通过 `scripts/check_repo_hygiene.ps1` 执行这条面向未来的规则；既有历史产物不会因该策略被重写或删除。
