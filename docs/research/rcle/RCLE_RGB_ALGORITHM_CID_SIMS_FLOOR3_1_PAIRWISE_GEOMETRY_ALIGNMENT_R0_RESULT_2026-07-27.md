# RCLE RGB algorithm / CID-SIMS floor3_1 pairwise geometry alignment R0

Date: 2026-07-27

Protocol: `RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_PAIRWISE_GEOMETRY_ALIGNMENT_R0`

Terminal: `POSTHOC_PAIRWISE_ALIGNMENT_COMPUTED / VALID`

Authority: `POSTHOC_REAL_DATA_MECHANISM_ALIGNMENT_ONLY`

## Answer

The immutable RGB output aligns strongly, pair by pair, with source-native
depth-plus-pose radial geometry in these two adjacent windows.

The positive window is a stable approach episode: all 299 pairs are in the
frozen `POSITIVE_APPROACH_GEOMETRY` band, all 299 RGB pairs trigger, their
first onsets are identical at 0.037610 s, and the full 9.976715 s run overlaps.

The nominal weak-motion control window is not a pure negative control. It
contains a transition into a source-native positive-approach episode:

- the RGB trigger run starts at pair 151, 5.068813 s after window start;
- the geometry-positive run starts at pair 156, 5.235547 s after window start;
- RGB therefore leads the frozen 0.05/s geometry-band crossing by 0.166734 s;
- all 143 geometry-positive pairs trigger, and the 143-pair geometry run is
  fully contained in the 148-pair RGB trigger run.

This explains the previously surprising 49.5% trigger rate in window 0 much
better than a persistent weak-motion false trigger explanation: the latter
half of that window contains real positive radial geometry.

## Fixed-band results

| Scope | Geometry band | Pairs | RGB triggers | Trigger fraction |
| --- | --- | ---: | ---: | ---: |
| Window 0 | `BELOW_TRIGGER_REFERENCE` | 139 | 0 | 0.0000 |
| Window 0 | `WEAK_POSITIVE_RADIAL` | 17 | 5 | 0.2941 |
| Window 0 | `POSITIVE_APPROACH_GEOMETRY` | 143 | 143 | 1.0000 |
| Window 1 | `POSITIVE_APPROACH_GEOMETRY` | 299 | 299 | 1.0000 |
| Combined | `BELOW_TRIGGER_REFERENCE` | 139 | 0 | 0.0000 |
| Combined | `WEAK_POSITIVE_RADIAL` | 17 | 5 | 0.2941 |
| Combined | `POSITIVE_APPROACH_GEOMETRY` | 442 | 442 | 1.0000 |

All 598 pairs were geometry-evaluable. Geometry coverage was 1.0 in each
window and there were no geometry abstentions.

## Alignment diagnostics

| Metric | Window 0 | Window 1 | Combined diagnostic |
| --- | ---: | ---: | ---: |
| Pearson, geometry vs RGB | 0.9364 | 0.6790 | 0.8806 |
| Spearman, geometry vs RGB | 0.8337 | 0.6443 | 0.8606 |
| Positive-geometry trigger coverage | 1.0000 | 1.0000 | 1.0000 |
| Trigger pairs in positive geometry | 0.9662 | 1.0000 | 0.9888 |
| Positive/trigger Jaccard | 0.9662 | 1.0000 | 0.9888 |

These are descriptive diagnostics over temporally dependent adjacent pairs.
They are not independent-sample performance estimates.

## Validation

The producer computed 598 geometry rows with 8 CPU workers. The validator did
not import the alignment producer: it independently reconstructed the ZIP
timestamp pairs, repeated the frozen geometry computation, compared ledger
rows with IEEE-754 `float.hex()` equality, and recomputed all aggregates.

Validation returned `VALID` with `errors=[]`.

The producer and validator share the already-frozen CID geometry helper.
Therefore this is an independent orchestration and aggregation audit, not an
independent geometry implementation confirmation.

## Authority limits

- The RGB algorithm was not re-executed; its immutable R0 ledger was read.
- No threshold was tuned.
- The analysis is outcome-aware and posthoc.
- The two windows are adjacent and belong to one evolving motion episode.
- Pair rows are temporally autocorrelated and must not be treated as 598
  independent trials.
- The global depth-plus-pose radial estimator and local RGB feature-flow
  estimator are related but not identically calibrated measurements.
- The original R0 evidence status remains `INVALID_R0_EVIDENCE / INVALID`.
- This result is not performance qualification, independent confirmation, or
  a product/safety claim.

## Execution receipt

The first foreground launcher was terminated by the tool timeout during
pre-output work. Read-only inspection found no `run_r0` directory and no live
Python worker. The same hash-locked implementation was then launched in the
background and exclusively materialized `run_r0`. No implementation file was
changed after the lock.

- Contract SHA-256:
  `ef0e6696e86559d0c5c6d22b2fd1f9bedcd33550381059e860d3433b06df3b97`
- Implementation lock SHA-256:
  `e61f00c88f8d4fa7301c13e130ef9b5ffbf7d0e8b8ba54ad8c83507b34067310`
- Activation SHA-256:
  `1e96fa73cf57b839a457c79e1fe38fa22f9e5337451065fe182eac57fab3ba71`
- Alignment ledger SHA-256:
  `f8a67659931aac705faa0f63ca1506935eca1b76fe982da79c933c535db0cff8`
- Result SHA-256:
  `d42f1b9dbdcd85692d7abf717bfb276c824003f0f26443f89de8c3afa5960f30`
- Validation SHA-256:
  `fd780c9ef6632e55e8640d9db305848362c1c6ac2aa8122f1a11bbcf2674f49b`

## Recommended successor

Do not tune the threshold from these rows. The highest-value next step is a
new, versioned, RGB-outcome-blind geometry-stratified holdout using disjoint
windows already present in the local CID-SIMS cache. Freeze deterministic
selection rules and window identities first, then run the unchanged RGB
algorithm once on those windows.

That successor would test whether the observed band separation and onset
alignment recur outside this already-inspected adjacent episode. It would
remain a development/generalization canary, not performance qualification.
