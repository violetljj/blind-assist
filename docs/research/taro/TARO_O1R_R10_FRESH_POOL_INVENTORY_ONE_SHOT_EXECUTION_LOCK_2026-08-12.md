# TARO O1R R10 fresh-pool inventory lock

The exact 96 source assets passed byte-integrity validation. A deterministic,
non-writing container scan established 32 parent frame counts totaling 710 exact
pose-bounded frames and `2,388,664,781` uncompressed/materialized bytes.

This one-shot lock permits only ZIP inventory/CRC validation, trajectory parsing,
and exact frame-plan sealing. It does not permit pixel-array decode, model
execution, FARO access, truth scoring, or training.

The output root is consumed on exclusive creation. The frozen per-parent counts,
total frame count, and materialized-byte ceiling must match exactly; failure does
not restore authority and no overwrite, resume, repair in place, or rerun is
allowed. A pass permits only a separately locked source-only Phase A.
