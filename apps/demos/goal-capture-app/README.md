# Goal Capture research app

Standalone CameraX recorder for the P1 prospective first-person Goal Contract cohort. It is a separate Gradle module
and application ID; it does not change or share a runtime path with the default BlindAssist app.

Use the in-app system file picker to select the host-generated `capture_plan.json`. After validation, the app stores it
atomically under its app-specific external files directory at:

```text
prospective-goal-capture/inbox/capture_plan.json
```

The app validates the plan and C0-binding hashes, records every episode in frozen order with the rear-facing camera and
no audio, and treats CameraX `Finalize` as the recording boundary. Cancellation, lifecycle interruption, recorder error,
partial completion, reused media, or timeline/media disagreement produces a non-authorizing `capture_hold.json`; only a
complete roster can produce `physical_capture_receipt.json`.

The exported ZIP contains `capture_plan.json`, `device_captures/*.mp4`, and the receipt. Extract it on the host and pass
the extracted `device_captures` directory as `--capture-root` to
`p1_prospective_capture.materialize_capture`. The host validator independently rechecks hashes, metadata, chronology,
and fixed frame extraction before private truth can be created.

This is research capture infrastructure only. It does not authorize PA3, identity verification, default-App behavior,
product use, or safety claims.
