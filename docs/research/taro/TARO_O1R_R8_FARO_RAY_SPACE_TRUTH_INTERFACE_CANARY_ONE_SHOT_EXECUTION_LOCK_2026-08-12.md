# TARO O1R R8 FARO ray-space truth-interface canary lock

This one-shot reuses exactly the already-selected eight R8 parents and 133 FARO
frames. It tests whether a FARO-owned 3x3 path-query frame can recover definite
clear observability without inheriting source-model query availability.

The implementation samples a fixed 36-point swept volume per query, retains
fail-closed `UNKNOWN`, and must preserve every old definite-clear label, never
turn an old definite-occupied label into clear, and observe zero frozen positive
occupancy predictions on a new definite-clear label. At least 50 clear queries
across four parents are required for the interface canary to pass.

The run is post-hoc interface evidence only. It reads no unselected FARO, changes
no source selection or threshold, performs no training, and cannot promote R8 or
support deployment, product, or safety claims. A pass can only freeze the label
interface before a separate fresh-parent confirmation.
