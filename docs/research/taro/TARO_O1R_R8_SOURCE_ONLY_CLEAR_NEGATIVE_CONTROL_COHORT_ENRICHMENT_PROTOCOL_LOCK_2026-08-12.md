# TARO O1R R8 source-only clear-negative-control enrichment protocol

Status: `FROZEN_DATA_USE_AUTHORIZED`.

R7's positive gates were strong, but the fresh cohort contained only one
definite-clear query. R8 therefore freezes a new 24-parent metadata-only pool,
runs source-only Phase A across that pool, and selects the final eight parents
before any FARO payload is opened.

The selector scores only query slots with no frozen positive evidence, an
available query frame, no weak obstacle veto, at least nine visible far
anchors, and at least 80% candidate visibility beyond 2.5 m. It does not emit
clear and has no FARO, truth, label, or outcome input. On the consumed R7 data,
it gave a nonzero score only to the sole parent containing a definite-clear
label; this is explicitly post-hoc development motivation, not confirmation.

The 24-parent pool and all 72 asset URLs are deterministic and frozen. On
2026-08-12 the user explicitly replied `授权` to the pending request to authorize
this exact R8 pool. That authority covers bounded HEAD/download/inventory and
the locked source-only Phase A. FARO remains closed until all 24 parent scores
and the final top-eight identities are sealed.
