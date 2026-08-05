# P3 R0.2 Identity Capacity Audit Protocol

This protocol freezes a finite, local source universe before the identity audit runs. It admits the already scoped ARKitScenes identity manifest and two Bonn RGB-D video parents; legacy TUM P1, the four R0.1 attempted holdout parents, paused FRESH-TF, RGB-only public video and synthetic sources cannot enter the candidate pool.

The audit may read only parent and RGB identities, source-native timestamps, four-frame continuity, raw truth-asset availability and ancestry overlap. Clearance, geometry, transitions, model outputs, disagreement, performance and outcomes are forbidden.

Capacity requires at least eight existing train parents, four existing validation parents and eight new sealed-holdout parents, all video-parent disjoint. Twelve holdout parents are preferred but are not the minimum gate. An insufficient result closes the current temporal-data route as `P3_TEMPORAL_ROUTE_DATA_NOT_READY`; it does not authorize source search or protocol expansion.
