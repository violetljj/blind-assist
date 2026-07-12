# BlindAssist agent rules

## Project boundaries

- BlindAssist is a native Android Kotlin multi-module assistive prototype: CameraX captures frames, local TFLite YOLO detects objects, and a rule layer drives speech, vibration, and Compose feedback.
- Keep module responsibilities stable: `:app` is the shell and assets; `:feature:assist` coordinates runtime; `:core:assist` owns pure risk logic; `:core:vision` owns detection; `:core:device` owns Android device adapters; `:core:ui` owns UI state and rendering.
- Do not casually add large frameworks, replace/remove model assets, or change CameraX, TFLite, coordinate mapping, risk rules, or feedback behavior without focused tests and documented evidence.
- Describe the product cautiously: it is an assistive prototype and must not be represented as a substitute for human safety judgment.

## Change hygiene and documentation

- Before editing, run `git status --short`; preserve unrelated changes and do not revert code you do not understand.
- Update `DEVELOPMENT_LOG.md` for code, configuration, model, test, or adopted technical-decision changes. Record scope, rationale, verification, and remaining risk. Pure read-only investigation, review, or conversation does not require a log entry.
- Use `violjjet` as the executor name in new development-log entries.
- Update `README.md` only when user-visible capabilities, usage, prerequisites, current project status, or important public decisions change. Put release history in `CHANGELOG.md` and detailed evidence in the relevant `docs/` page.
- Add an item to `idea.md` only when the user asks to retain it or the team explicitly defers a non-trivial proposal. Mark implemented items `【已完成】` and partial items `【部分完成】`.

## Verification and release discipline

- Match verification to risk: documentation-only changes normally need no Gradle run; a module change needs relevant tests or lint; changes to runtime, vision, risk, feedback, permissions, or assets need focused tests plus an Android build; a delivery candidate needs the release checklist and final-APK verification.
- If required verification cannot run, record the reason and impact in the final report and, when the work changed the project, in `DEVELOPMENT_LOG.md`.
- Release procedure, command matrix, version decision, and final APK verification live in [docs/RELEASE_AND_VERIFICATION.md](docs/RELEASE_AND_VERIFICATION.md). APK retention rules live in [docs/APK_ARCHIVE.md](docs/APK_ARCHIVE.md).
- Do not change a version merely because a task happened. Change it for a planned delivery or a user-visible, compatibility, safety, model, permission, or substantial architecture change, and record the rationale.
- Archive an APK only for a delivery candidate, demonstration, teacher review, milestone, or user request; an ordinary debug build is not an archive event.

## Local artifacts, hardware, and external operations

- Do not commit SDKs, Gradle caches, virtual environments, downloads, local datasets, device logs, screenshots, or other machine-generated artifacts. Follow [docs/LOCAL_ARTIFACTS.md](docs/LOCAL_ARTIFACTS.md).
- The in-app glasses connection is currently a placeholder: it does not imply Bluetooth, network, ESP32, or real-device integration. For legacy-source paths, migration boundaries, and safe fallback requirements, read [docs/GLASSES_HARDWARE_ROUTE.md](docs/GLASSES_HARDWARE_ROUTE.md) before hardware work.
- For known sandbox restrictions, use the workspace-level elevation policy. Keep the command narrow and state why elevation is needed.
- Before any push, inspect the working tree, current branch, upstream, and exact remote. A regular non-force push to `git@github.com:violetljj/blind-assist.git` is pre-authorized, but never assume `master` or change another remote, rewrite history, delete a branch, or push local data/ignored artifacts without explicit approval.
- GitHub CLI login and PATH state are runtime facts: verify them with `E:\linnan\tools\gh\bin\gh.exe auth status` when needed rather than assuming a stored login remains valid.
