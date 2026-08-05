# P3 R0.2.1 ARKit validation extension protocol

Attempt 01 proved that the existing scoped validation identities contain only three parents with a complete four-frame clip under the frozen 500 ms adjacent-gap rule. This extension is limited to repairing validation identity capacity; it does not alter the 11-parent Bonn holdout, model, losses, evaluator or gates.

Before candidate visit IDs are read, the protocol binds the official ARKitScenes raw split CSV and excludes 28 visit parents already assigned or inspected by prior project work plus the one cross-fold visit. It selects exactly eight reserve parents from the official Validation fold by ascending SHA256 of the protocol ID and visit ID. One video is fixed per visit by the lowest numeric video ID.

After roster locking, a separate committed media preflight may read only archive identity, RGB-D/confidence/intrinsics headers, timestamps, continuity and hashes. It may not read task labels, transition coverage or model output. A failed reserve parent cannot be replaced under this protocol.

```text
P3_R0_2_1_ARKIT_VALIDATION_EXTENSION_PROTOCOL_FROZEN_ROSTER_NOT_YET_SELECTED
```
