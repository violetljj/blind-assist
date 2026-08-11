# TARO O1R R7 fresh source inventory and frame-plan execution lock

Status: `AUTHORIZED_UNCONSUMED`.

This one-shot authorizes only local container inventory, ZIP CRC validation,
trajectory parsing, intrinsics indexing, and construction of the exact
pose-bounded frame plan for the already downloaded eight-parent fresh cohort.

Frozen expectations are 170 frames with per-parent counts
`25/20/25/6/11/16/56/11` and 599,589,047 materialized bytes. Pixel arrays,
model execution, truth scoring, and training remain forbidden.

The output root is
`artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-inventory-r0`; its
creation consumes this authority even if execution later fails. A PASS admits
only the separately locked Phase A source/model successor.
