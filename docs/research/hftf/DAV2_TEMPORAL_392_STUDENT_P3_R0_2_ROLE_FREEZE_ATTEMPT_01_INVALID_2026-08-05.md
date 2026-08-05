# P3 R0.2 role-freeze Attempt 01 invalid result

Attempt 01 is invalid and cannot enter sealing or Activation.

The two executions were byte-identical, and the Bonn holdout correctly materialized all 11 frozen parents into 132 label-blind clips. However, the ARKitScenes manifests contained four frozen parents with no complete four-frame clip under the immutable 500 ms adjacent-gap rule. Actual clip-parent coverage was therefore `13/3/11`, not the declared `16/4/11` for train, validation and holdout.

The original producer incorrectly copied roster counts into its receipt without comparing them to the parent IDs actually present in the clip manifests. A fail-closed parent-coverage invariant and regression test have now been added. Attempt 01 outputs remain retained but invalid.

No label, transition, holdout outcome or model output was read. No sealed target, checkpoint, P3 model, optimizer or training run was created. This result does not reject the Bonn holdout or the temporal model; it shows that the existing scoped ARKit validation identities provide only three clip-capable parents.

The next permitted action is a separately frozen, label-blind ARKitScenes validation-identity capacity extension. Existing train parents and the four permanently excluded R0.1 holdout parents cannot be reassigned to rescue validation.

```text
P3_R0_2_ROLE_FREEZE_INVALID_VALIDATION_CLIP_PARENT_COVERAGE_INSUFFICIENT
```
