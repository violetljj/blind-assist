# APK Archive Policy

BlindAssist keeps APK access in two layers:

- GitHub milestone APKs: `releases/apk/` keeps only versions whose cumulative `versionName` delta from the last committed APK is `>= 0.5`, or APKs explicitly marked by the user as Git milestones.
- Complete local archive: every generated historical APK is kept under `E:\linnan\blind-assist-apk-archive\apks`.

The complete local archive was created on 2026-05-22 and contains 32 APK files. Its manifest is:

```text
E:\linnan\blind-assist-apk-archive\APK_ARCHIVE_MANIFEST.csv
```

Verify any archived APK with:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "E:\linnan\blind-assist-apk-archive\apks\<apk-name>"
```

## GitHub Milestone APKs

| APK | Size bytes | SHA256 |
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

## Future Rule

When creating a new APK, archive it locally first. Commit it to `releases/apk/` only when the current `versionName` is at least `0.5` higher than the newest APK already committed in this directory, or when the user explicitly asks for a Git milestone APK. Smaller update APKs stay in the local archive and should be documented in the README or changelog only when relevant.

New test screenshots, raw device logs, temporary APKs, zip snapshots, PPT exports, ONNX/PT/NPY model conversion artifacts, and machine-local caches should stay outside Git by default. CI enforces this future-facing rule through `scripts/check_repo_hygiene.ps1`; existing historical artifacts are not removed or rewritten by this policy.
