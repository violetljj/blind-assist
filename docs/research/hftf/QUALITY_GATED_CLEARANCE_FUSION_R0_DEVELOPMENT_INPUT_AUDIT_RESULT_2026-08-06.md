# Quality-gated clearance fusion R0 development input audit

Terminal:

```text
QUALITY_GATED_CLEARANCE_FUSION_R0_DEVELOPMENT_INPUT_NOT_READY
```

The three-parent source admission and clip-capacity gates passed, but the
required replay input stream is not materialized. The frozen development
contract requires, for every frame, the same precomputed raw geometry stream:

- raw three-band clearance;
- raw geometry validity and state;
- ToF or independent metric-sensor validity;
- teacher age;
- frozen parent-A2 disagreement;
- source-native frame identity and timestamp.

The newly admitted ARKitScenes visit currently has only the authorized RGB,
depth, confidence, intrinsics and trajectory media manifest. Existing TUM
Metric3D manifests/reports are from a different consumed route and do not carry
the frozen R0 input schema or the same three-parent source contract. They are
not silently rewrapped as quality-fusion inputs.

No model was loaded, no checkpoint was loaded, no optimizer was constructed,
no PNG/depth body was decoded for this audit, no labels or outcomes were read,
and no replay was started. The source admission, media integrity and clip
capacity evidence remain valid and are preserved.

To continue, a new producer must be frozen and committed that materializes the
exact raw geometry stream for these three parents under the development
protocol. That producer must separately bind any runtime/model it is permitted
to load and must not reuse consumed P3 outputs or alter the frozen filter.
