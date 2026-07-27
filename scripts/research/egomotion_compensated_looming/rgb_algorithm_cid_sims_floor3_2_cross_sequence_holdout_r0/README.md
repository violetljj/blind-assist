# CID-SIMS floor3_2 cross-sequence holdout R0

This module implements the pre-access-frozen, geometry-first development
holdout for the different official `floor3_2` run of the same CID-SIMS Floor3
scene.

The formal runner creates its scientific claim before the first ZIP open. It
then uses only depth member names, `groundtruth.txt`, and previous-frame depth
bytes to classify every complete 10-second window. Exactly two positive and two
below-reference windows, with starts at least 20 seconds apart, are required.
If the set is unavailable, execution stops before any RGB member byte is read.

If selection succeeds, the exact ordered RGB member identity is fsynced before
the frozen development-canary `evaluate_window` function is called unchanged.
The validator is a separate implementation that does not import either the
formal runner or the RGB producer.

The maximum authority is
`CROSS_SEQUENCE_SAME_SOURCE_DEVELOPMENT_HOLDOUT_ONLY`: this is not independent
confirmation, cross-source generalization, performance qualification, or
product/safety evidence.
