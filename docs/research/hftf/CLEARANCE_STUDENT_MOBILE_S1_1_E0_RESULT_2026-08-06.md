# Clearance-Student Mobile S1.1 E0 result

Terminal: `S1_1_E0_TRAINING_COMPLETE_DEVELOPMENT_ONLY`.

The single authorized S1.1 mechanism-correction sanity run completed on the
A4 train/validation stream. It used the official torchvision MobileNetV3-Large
ImageNet weights, confidence-masked metric depth, and the materialized
geometry-target cache. The consumed 120-frame model-variant gate was not read.

Fixed run facts:

- parameter count: `5,047,194`;
- encoder pre/post binding: equal;
- encoder digest: `BBC261917B54693257E2AC3E694BDF9D01AECB2880F9867216B0172D5055D023`;
- train total: `0.554664` (epoch 1) → `0.314878` (epoch 2);
- validation total: `0.387633` (epoch 1) → `0.379273` (epoch 2);
- training time: `3302.24s`.

This is only an E0 optimization/binding sanity result. The current E0 runner
did not yet emit the protocol-required depth quantiles/saturation report or
finite geometry-output coverage report. Therefore E0 is not promoted to E1:

- no new E1 cohort was opened;
- no Canonical/S0/S1.1 gate comparison was run;
- no scale-aligned AbsRel claim is made for S1.1;
- no QNN, QAT, Android, production, or safety authority is granted.

The missing E0 diagnostics are a tooling-completeness gap, not evidence that
S1.1 passed or failed the mobile geometry question. The one-experiment limit
and all consumed-cohort prohibitions remain in force.
