# P3 R0.2.1 ARKit role-manifest protocol

This protocol freezes a label-blind merge of the original ARKitScenes identity manifest and the locked eight-video validation extension. It does not inspect clearance, geometry states, transitions, model output or performance.

Attempt 01 already established that three train parents and one validation parent cannot provide a four-frame clip under the immutable 500 ms adjacent-gap rule. They are removed rather than counted as usable. The remaining 13 train parents are retained; validation contains the three original clip-capable parents plus all eight precommitted extension parents. The 11-parent Bonn holdout is not changed by this producer.

The producer and tests are SHA-bound before materialization. Its output only carries identity assets and the assigned train/validation role.

```text
P3_R0_2_1_ARKIT_ROLE_MANIFEST_PROTOCOL_FROZEN_NOT_MATERIALIZED
```
