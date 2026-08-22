# P1 prospective first-person Goal Contract capture

状态：`HOST_VALIDATOR_AND_DEVICE_RECORDER_IMPLEMENTED / REAL_DEVICE_COHORT_NOT_CAPTURED / PA3_INFERENCE_NOT_AUTHORIZED`

This surface closes the provenance seam between a C0 public Goal Contract and physical first-person observations.
It only accepts a complete frozen roster of at least five episodes whose forward-camera recordings start after each
goal timestamp and before truth or provider execution. Every media file must be inside the declared capture root and
match its frozen SHA-256, dimensions, duration, source role, and continuous-capture receipt.

Before the first device recording, `arm_capture` freezes the full episode roster, deterministic media filenames,
capture instruction, source role, and extraction offsets. The later device receipt must bind that plan hash and use the
same filenames; the validator enforces `goal < arm < physical capture < receipt` for every episode and rejects partial
rosters or reused media. The post-capture receipt must declare
`recorder_authority=DEVICE_OWNED_CONTINUOUS_VIDEO_RECORDER`; replay/public/synthetic sources are not admitted.

The capture instruction is global and fixed:

```text
APPROACH_NAMED_BUILDING_AND_STOP_AFTER_ENTRANCE_IS_IN_VIEW_V1
```

Frame selection is likewise global: extract exactly three frames per episode at `2.5`, `1.5`, and `0.5` seconds before
the end of the continuous recording. No pixel, truth, proposal, model output, visibility, or per-episode frame choice is
accepted by the capture receipt. The materialized manifest remains provider-public, records zero model calls, and does
not authorize PA3. Private truth is created only afterward and the existing PA3 materializer still applies the frozen
visible-episode and visible-frame gate.

The receipt establishes machine-validated local provenance, not external attestation that a device clock or recorder
was honest. A real cohort therefore still needs device-owned sidecars and source custody; pre-existing public videos,
Mapillary resampling, replay, and synthetic video cannot claim this physical post-goal role.

The standalone CameraX producer lives at `apps/demos/goal-capture-app`. It is a separate application ID and runtime from
the default BlindAssist app. It consumes the frozen plan, records rear-camera video without audio, waits for asynchronous
`Finalize`, validates duration/dimensions/SHA-256, and emits a receipt only after the complete roster. Its canonical JSON
and full sample receipt hash are cross-checked against the Python contract. No real-device run has occurred yet.

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.goal_copilot_bridge.p1_prospective_capture.arm_capture `
  --c0-receipt artifacts.local/<run>/goal_contract_receipt.json `
  --output artifacts.local/<run>/capture_plan.json

E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.goal_copilot_bridge.p1_prospective_capture.materialize_capture `
  --c0-receipt artifacts.local/<run>/goal_contract_receipt.json `
  --capture-plan artifacts.local/<run>/capture_plan.json `
  --capture-receipt artifacts.local/<run>/physical_capture_receipt.json `
  --capture-root artifacts.local/<run>/device_captures `
  --output-dir artifacts.local/<run>/materialized_capture
```

Focused mechanics check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.p1_prospective_capture.test_arm_capture `
  scripts.research.goal_copilot_bridge.p1_prospective_capture.test_materialize_capture
```
