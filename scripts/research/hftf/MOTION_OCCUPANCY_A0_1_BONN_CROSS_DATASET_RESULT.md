# Motion occupancy A0.1 Bonn cross-dataset result

Date: 2026-08-03

Terminal: `BONN_CROSS_DATASET_REFERENCE_NOT_EVALUABLE`

No frozen A0.1 probability outcome was produced.

The original z-up reference yielded 140/150 candidate-valid frames but zero
known sensor clearance opportunities. After the pre-outcome correction to a
general calibrated world plane, only 41/150 frames had both candidate and
reference clearance. The supplied Bonn `groundtruth` trajectory therefore
cannot be treated as a directly calibrated RGB optical camera-to-world pose for
this fixed-plane reference. Its implied camera height diverged from the
per-frame sensor-depth ground estimate during the sequence.

The paired-valid requirement was 90%; the corrected first source reached only
27.33%. The second source and pooled frozen probability evaluator were not run.
Changing to a third reference after these observations would be protocol
rescue, so this cross-dataset attempt stops here.

This terminal is a reference/protocol incompatibility, not evidence that the
frozen occupancy model does or does not transfer to Bonn. A valid cross-camera
test now requires either source-native camera extrinsics and floor geometry, a
separately validated reference construction, or the final camera's controlled
distance/geometry capture.

Corrected first-source clearance report SHA-256:
`FB474D7C99DFF0A22D8066DBBCF2022D9C9111AC23FC9BC2AF9DBFC3023C2894`.
