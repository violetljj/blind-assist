# P3 R0.2.1 role-freeze Attempt 02 result

Attempt 02 successfully closed the three data identities without reading labels or model output.

- Train: 13 parents, 481 four-frame clips.
- Validation: 11 parents, 699 four-frame clips.
- Public holdout: 11 Bonn parents, 132 four-frame clips.
- Every frozen parent contributes at least one clip.
- Parent roles are disjoint and no frame is reused.
- All five output hashes replay exactly.
- The public holdout manifest retains identity fields only and records `outcomes_opened=false`.

No sealed target was produced. No checkpoint, model or optimizer was loaded or constructed, and training did not start. This result authorizes only the next producer-freeze phase for private sealing and aggregate coverage; it does not authorize Activation or training.

```text
P3_R0_2_DATA_ROLES_FROZEN_HOLDOUT_IDENTITIES_LOCKED_TARGETS_UNOPENED
```
