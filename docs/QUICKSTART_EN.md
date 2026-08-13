# BlindAssist three-minute quick start

Status: `current`

Last verified: 2026-08-12

This page gets a new contributor from clone to one honest, reviewable result. The
three minutes cover orientation and starting a check; the first Android build can
take longer while Gradle downloads dependencies.

> BlindAssist is a research and accessibility prototype, not a certified mobility
> or safety device. A successful build does not prove perception quality, user
> outcomes, or safe navigation.

## Before the timer

Install Git, PowerShell 7, JDK 17, and Android SDK Platform 35. On Windows, the
maintained wrapper discovers and validates the local toolchain. On Linux, the
commands below mirror the public GitHub Actions path; clean-host verification is
still tracked as a contributor task.

Do not clone datasets, download research payloads, connect a device, or configure
secrets for this quick start.

## 0:00 — clone and enter the repository

```bash
git clone https://github.com/violetljj/blind-assist.git
cd blind-assist
git status --short --branch
```

Expected result: branch `master` and no changed files.

## 0:45 — choose the smallest useful path

### Documentation or community contribution

These checks require PowerShell 7 but no Android device:

```powershell
pwsh -NoProfile -File scripts/check_docs_index.ps1
pwsh -NoProfile -File scripts/check_open_source_readiness.ps1
```

### Android contribution on Windows

Run the bounded preflight before starting Gradle:

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 -PreflightOnly
```

Expected result: the command reaches a successful preflight. If it reports
`ENV_BLOCKED`, fix the named JDK or SDK requirement instead of bypassing the
wrapper.

### Android contribution on Linux

Use the same focused tasks as CI:

```bash
chmod +x ./gradlew
./gradlew --max-workers=2 \
  "-Dorg.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8" \
  :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
```

This is a CI-validated path, not yet a promise that every Linux workstation is
supported. If a clean machine differs, report the distribution, architecture,
JDK, Android SDK packages, exact command, and complete error in the setup issue.

## 2:00 — produce one verifiable result

For a focused Android check on Windows:

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 `
  :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
```

The APK is written to:

```text
app/build/outputs/apk/debug/app-debug.apk
```

For a documentation-only change, finish with the checks that own that surface:

```powershell
git diff --check
pwsh -NoProfile -File scripts/check_docs_index.ps1
```

Run structure, release, permission, default-app, or shared-infrastructure gates
only when the change touches that risk. A push by itself does not require an
unrelated full-repository gate.

## 3:00 — pick a bounded first contribution

- Browse the [`good first issue` queue](https://github.com/violetljj/blind-assist/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22).
- Comment on the issue before starting so scope and duplicate work stay visible.
- Change only the files named in the issue and run its copyable verification
  commands.
- Open one focused pull request and report commands, results, remaining gaps,
  and whether the default app is affected.

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before editing and
[GOVERNANCE.md](../GOVERNANCE.md) before volunteering for review or maintenance.
Questions and early ideas belong in
[GitHub Discussions](https://github.com/violetljj/blind-assist/discussions); security
or privacy reports must follow [SECURITY.md](../SECURITY.md).

## What not to do in a first contribution

- Do not change the packaged model, permissions, feedback policy, release signing,
  research gates, or default-app authority unless the issue explicitly permits it.
- Do not commit raw camera footage, device logs, private data, credentials,
  restricted datasets, SDK payloads, generated APKs, or `artifacts.local/` files.
- Do not convert `UNKNOWN`, synthetic evidence, a build result, or a benchmark into
  a product or safety claim.

The stable module map is in [CODE_MAP.md](CODE_MAP.md). The full evidence boundary
is in [RESEARCH_GOVERNANCE.md](RESEARCH_GOVERNANCE.md).
