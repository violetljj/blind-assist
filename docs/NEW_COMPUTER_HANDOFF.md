# New Computer Handoff

This note is for continuing BlindAssist development on a new Windows computer.
Read `AGENTS.md` first, then use this checklist to restore the project, Codex
skills, and the Android build environment.

## 1. Clone the repository

```powershell
git clone git@github.com:violetljj/blind-assist.git
cd blind-assist
```

If SSH is not ready yet, add a GitHub SSH key on the new computer before
cloning, or temporarily use the HTTPS remote and switch back to SSH later.

## 2. Restore Codex skills

The repository includes a skills snapshot at:

```text
codex/skills-snapshot/codex-skills-20260522.zip
```

From the repository root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\restore_codex_skills.ps1
```

The restore script extracts the archive into:

```text
%USERPROFILE%\.codex\skills
```

If that folder already exists, the script creates a timestamped backup before
restoring the snapshot. Restart Codex after restoring skills so the new session
can discover them.

## 3. Install Android development tools

Install these on the new computer:

- Android Studio with bundled JDK 17.
- Android SDK Platform 35.
- Android SDK Build Tools.
- Android Platform Tools for `adb`.
- Git for Windows.

Then create `local.properties` in the repository root:

```properties
sdk.dir=C\:\\Users\\<your-user>\\AppData\\Local\\Android\\Sdk
```

Adjust the path to the actual SDK directory on the new computer.

## 4. Verify the project

Use the repository validation command:

```powershell
$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
$env:PATH="$env:JAVA_HOME\bin;$env:PATH"
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --no-daemon
```

The debug APK should be generated at:

```text
app/build/outputs/apk/debug/app-debug.apk
```

To check the bundled model asset:

```powershell
.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py
```

Expected model shapes:

```text
input shape=[1, 320, 320, 3] dtype=float32
output shape=[1, 84, 2100] dtype=float32
```

If the Python export environment is not restored yet, Android build validation
can still run as long as the tracked TFLite asset is present.

## 5. Optional phone install

Connect a phone with USB or wireless debugging, then run:

```powershell
.\.android-sdk\platform-tools\adb.exe devices
.\.android-sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
```

On a fresh computer, the SDK path may be the system Android SDK path instead of
the repository-local `.android-sdk` folder. In that case use `adb.exe` from the
new SDK's `platform-tools` directory.

## 6. Development rules to keep

- Keep using `violjjet` as the executor name in `DEVELOPMENT_LOG.md`.
- Run `git status --short` before editing.
- Do not commit local SDKs, Gradle caches, virtual environments, downloads, or
  generated machine-specific files.
- Record every implementation, configuration change, analysis-only pass, and
  validation result in `DEVELOPMENT_LOG.md`.
- Update `README.md` when project state, usage, build flow, test conclusions,
  model assets, or important decisions change.
- Preserve versioned APKs under `releases/apk/` when a build is meant for demo,
  testing, or teacher review.
