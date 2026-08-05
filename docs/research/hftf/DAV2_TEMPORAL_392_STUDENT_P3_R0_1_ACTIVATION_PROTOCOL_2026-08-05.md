# P3 R0.1 Activation Protocol

This protocol freezes a one-shot, static and read-only activation binding check for the A2-392 temporal student. It does not authorize training.

The check hashes opaque checkpoint, cache and sealed-bundle bytes without deserializing them. It must not import an ML runtime, load a checkpoint, construct a model or optimizer, parse sealed outcomes, create training caches, create the candidate output directory, or start training.

Every required binding is an AND gate: R0.1 code and governance files; current Git commit; the unique A2 checkpoint and its receipt, protocol and result; frozen parent-A2 disagreement assets; legacy-P1 exclusions; three role manifests; zero parent overlap; nine transition counts and frozen class weights; sealed bundle and aggregate coverage identities; unopened/runtime flags; and candidate-directory absence.

Any mismatch terminates as:

```text
P3_R0_1_ACTIVATION_BINDING_INVALID_NO_MODEL_LOAD
```

Only a complete pass terminates as:

```text
P3_R0_1_ACTIVATION_READY_HOLDOUT_UNOPENED_MODEL_UNLOADED
```

Training activation remains a later, separate command bound to an exactly reproducible READY receipt.
