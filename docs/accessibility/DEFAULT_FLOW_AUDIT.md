# Default-flow accessibility audit

This checklist records the bounded implementation for issue [#9](https://github.com/violetljj/blind-assist/issues/9): the onboarding-to-home flow, settings selectors, and the controls used to enter the phone-camera assist flow.

## What changed

- Home assist modes reflow into a vertical, readable control at large font scales instead of forcing labels into a fixed horizontal rail.
- The primary assist action permits its title and summary to wrap when text is enlarged.
- Settings segmented selectors stack at large font scales and allow up to two lines per option.
- Scenario titles and descriptions can wrap instead of being clipped to one line.
- Redundant selector-container announcements were removed; each actionable option remains an individually labelled radio control.
- Home mode options now expose an explicit selected/unselected state description.

## Automated evidence

The focused Android Compose coverage is:

- `BlindAssistComposeTest.defaultFlowKeepsModeAndSettingsOptionsAsIndependentAccessibleControls`
  - verifies the home mode is clickable, a radio button, explicitly unselected, and at least 48dp high;
  - verifies the settings selector container has no duplicate content description;
  - verifies the settings profile option is clickable, a radio button, and at least 48dp high.
- `FeatureScreenAccessibilityTest.defaultFlowReflowsHomeControlsForLargeFont`
  - renders the default flow at `fontScale = 1.9` in a 360dp-wide window;
  - verifies the primary assist action and Sensitive mode remain displayed, clickable, and at least 48dp high.

Run the focused surface with:

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:connectedDebugAndroidTest '-Pandroid.testInstrumentationRunnerArguments.class=com.linnan.blindassist.ui.compose.FeatureScreenAccessibilityTest,com.linnan.blindassist.ui.compose.BlindAssistComposeTest'
```

The connected tests have not been run for this contribution: local validation is blocked by the missing JDK 17, and the upstream Android workflow awaits maintainer approval. Passing results would establish the asserted Compose semantics and layout reachability only; TalkBack usability requires the manual checks below.

## Regression matrix

| Surface | Result | Evidence / owner |
| --- | --- | --- |
| Compose semantics, roles, state, clickability | NOT_TESTED; regression assertions added | Android test suite; contributor/CI |
| 48dp touch-target floor for changed controls | NOT_TESTED; regression assertions added | Android test suite; contributor/CI |
| Enlarged text at `fontScale = 1.9` | NOT_TESTED; regression assertions added | Android test suite; contributor/CI |
| TalkBack cold-start through camera entry | NOT_TESTED | Requires a physical or emulator device with TalkBack; maintainer/device owner |
| Switch Access, keyboard, or D-pad traversal | NOT_TESTED | Requires the corresponding input service/device; maintainer/device owner |
| Display-size scaling and contrast measurement | NOT_TESTED | Requires device settings and a contrast measurement pass; maintainer/device owner |

For each device run, record the tested commit, Android version, device or emulator profile, font/display scale, TalkBack version, command, and observed result. Keep a row `NOT_TESTED` until its evidence is recorded.

No safety, deployment, or user-outcome claim is made by this audit. The manual rows remain explicit evidence gaps rather than being inferred from Compose tests.
