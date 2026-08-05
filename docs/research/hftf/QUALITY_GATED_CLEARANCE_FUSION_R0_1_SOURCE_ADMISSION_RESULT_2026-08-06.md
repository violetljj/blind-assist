# Quality-gated clearance fusion R0.1 source admission result

Terminal:

```text
QUALITY_GATED_CLEARANCE_FUSION_R0_1_ARKit_LICENSE_SCOPE_REQUIRED
```

The label-blind admission producer completed without reading RGB/depth image
bodies, clearance, geometry labels, transitions, model outputs or metrics.

Admitted source identities:

- TUM `rgbd_dataset_freiburg3_sitting_halfsphere`
  - archive SHA256: `BA9F0FAB0D07E22F04FBFAE16EB4E3FB44088A32C920AD36C782B5024ED4B767`
  - 651,422,497 bytes
- TUM `rgbd_dataset_freiburg3_sitting_rpy`
  - archive SHA256: `2AC397FFF9E21CBFAD707D549BC07B83D27FE02F59F3678072E9A7BBA684A67E`
  - 485,314,017 bytes

The deterministic ARKitScenes identity rule selected visit `381644` with
videos `41069048`, `41069050`, and `41069051`. Its media was not downloaded:
the existing license extension only covers the previously locked eight
validation videos, not this new visit.

Available new parents: `2`.
Required minimum: `3`.

No model was loaded, no optimizer was constructed, no training started, and no
holdout outcome was opened. The route cannot proceed to development replay
until the ARKitScenes license scope is explicitly extended or a separately
frozen, identity-only source admission supplies a third new parent.
