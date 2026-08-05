# P3 Temporal Development Screen R0

This is a development-only screen. Its maximum conclusion is `DEVELOPMENT_SIGNAL_ONLY`; it is not a new P1 and cannot support a generalization, safety, deployment, A5S, QNN/HTP, Android, or cadence claim.

The historical identity scope was 16 train and 4 validation parents. Under the frozen four-frame / 500 ms rule, only 13 train and 3 validation parents contribute clips. `422841`, `382112`, `472626`, and `382124` remain recorded identities but contribute zero evidence and must not be counted.

The screen keeps A2-392, 392 input, four-frame clips, seed `20260805`, AdamW `2e-5`, three epochs, and the R0.1 temporal evidence/loss semantics. It compares the selected A2-392 student with one P3 temporal student using parent-wise development descriptions rather than treating clips or frames as independent samples.

Truth is bound to the existing spatial development cache. Each band is valid exactly when `truth_valid` is true. An invalid band is `UNKNOWN_GROUND`; otherwise clearance `<= 1.5 m` is `OCCUPIED` and greater clearance is `CLEAR`. Geometry transition labels remain valid for every identity-selected band, including explicit `UNKNOWN_GROUND`; sensor invalidity is not silently removed from the nine-class transition distribution. The producer records both per-band independent-metric-sensor validity and its `any` reduction; the latter is the frozen R0.1 `tof_valid` compatibility alias. Teacher timestamps equal sample timestamps and `teacher_valid=true`.

The Bonn sealing Attempt 02 cohort remains encrypted archival material. Its bundle, private targets, and outcomes are forbidden inputs to this route. The A2 model may be loaded only after all static bindings and absent-output checks pass, solely to materialize frozen A2 disagreement for the 13/3 development identities.

The immutable protocol binds only known inputs and code. It never self-binds future manifest or candidate-checkpoint hashes. Materialization emits a separate immutable activation binding receipt for the train/validation manifests, weights and disagreement cache; training consumes that receipt. Evaluation is bound after training from the training receipt, without rewriting this protocol.
