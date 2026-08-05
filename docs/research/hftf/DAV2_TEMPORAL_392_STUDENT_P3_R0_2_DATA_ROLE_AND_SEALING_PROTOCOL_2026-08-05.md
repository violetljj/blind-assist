# P3 R0.2 data-role and sealing protocol

This successor does not reopen P3 R0.1 or rewrite the closed local-universe R0.2 audit. It uses the separately frozen public RGB-D admission result to lock a new source universe before any private Bonn target is read.

## Frozen roles

- Train: the existing 16 ARKitScenes visit parents.
- Validation: the existing 4 disjoint ARKitScenes visit parents.
- Public holdout: all 11 ancestry-clean Bonn sequence parents admitted by the public-source audit.

The four attempted R0.1 holdout parents, four consumed legacy-P1 parents and fifteen previously used Bonn sequences remain permanently excluded. Train, validation and holdout have zero parent overlap.

## Label-blind clip lock

Clip construction is deterministic and cannot inspect geometry or transition labels. Train and validation retain every complete non-overlapping four-frame window. Holdout takes the first twelve complete non-overlapping RGB-D windows from each of the eleven frozen Bonn parents, giving 132 public clips. Adjacent RGB frames must be at most 500 ms apart; the independent depth sample must be present and within 50 ms. Missing depth references never enter the public clip identity.

The public manifest contains only the exact frozen allowlist: frame, video and parent identity; source-native timestamp; sealed-target identity; RGB identity and SHA256.

## Sealing boundary

The role freeze does not read private depth targets and does not create a sealed target bundle. Before sealing, a separate private-target producer must be implemented, tested, committed and SHA-bound. It may derive targets only for the already locked frames and may not replace a clip or parent.

The public coverage receipt may expose aggregate coverage only. It must prove at least 32 evaluable clips, 8 video parents, at least 8 examples of each key transition and all nine geometry-transition counts. If the locked cohort is insufficient, the terminal is:

```text
P3_R0_2_SEALED_HOLDOUT_NOT_EVALUABLE_NO_COHORT_SUBSTITUTION
```

No target-driven replacement is permitted.

## Authority

Successful role freezing authorizes only implementation and precommitment of the private sealing producer. It does not authorize A2 checkpoint loading, disagreement materialization, P3 construction, optimizer construction, training or holdout opening. The A2-392 model, four supervision families, seed, three epochs, evaluator, seventeen numeric gates and serial promotion order remain unchanged.

For successor datasets, the evidence field is `independent_metric_sensor_valid`; the historical R0.1 `tof_valid` field is not silently reused or rewritten.
