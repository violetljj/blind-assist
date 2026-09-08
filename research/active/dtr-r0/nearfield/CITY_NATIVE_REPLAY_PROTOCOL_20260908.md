# Native thin-positive sampling with old-scene rehearsal

Pre-outcome EXPLORE protocol, 2026-09-08. The previous native-only300-step fit
revived City support but kept bollard alerts0/2 and reduced Willow BODY/HEAD
support IoU0.4428/0.2037 to0.1345/0.0266. It is not reopened or extended.
Hypothesis: mixing original TRAIN supervision while increasing reliable native
thin-positive exposure may preserve localization and reduce that thin miss.
This is one combined data-recipe test, not independent identification of its
two ingredients. One new fit; no new acquisition or architecture/loss change.

Reuse corrected native TRAIN45, DEV45 and original-route40 labels-v2 from
`city-native-adapt-20260908`; preserve same-instance collision disagreement as
UNKNOWN. Never restore the withdrawn meter collision-verification claims.
The reliable TRAIN bollard-positive pool consists of sample indices6,8,11,13,
selected from TRAIN-only positive masks. Native global visible-surface labels
remain authoritative for losses; targets must not be inferred from sample names.

Replay corpus: first64 original G10/Willow samples with split=train in saved
dataset order (16 groups g2000..g2015). Verify original checkpoint-source receipt,
dataset/RGB/label/support hashes. They must have no sample/group/RGB overlap with
Willow VAL16 or City DEV/route RGB. Original VAL/TEST data never enter fitting.

Start from original G13-D seed17, not the last fitted checkpoint. Exactly300
steps, batch16: four draws with replacement from verified native thin positives,
four uniform draws from all45 native TRAIN frames (may include the thin pool),
eight uniform draws from64 replay TRAIN frames. Seed17 RNG, fixed source ordering,
save complete schedule and per-source exposure. Keep AdamW headLR1e-4,
backboneLR1e-6, weight_decay1e-4, frozen BN running buffers, masked near BCE plus
0.25 existing globally balanced UNKNOWN-aware support BCE. Same network,
resolution, preprocessing and finalstep300 checkpoint. No early stopping,
extra steps, augmentation, source enlargement, or alternate candidates.

Compare original, cached previous native-only fit, and this one new fit. Reuse
the fixed DEV FPR<=10% operating-point rule; lock choices before opening route
regression labels. Show unchanged historical cutoffs as well. DEV and route
have already informed development: all findings are consumed Development,
never fresh confirmation. Only TRAIN inputs determine gradient updates.

Retain at most a Development challenger if all hold: both DEV head recalls>=50%
at FPR<=10% and positive support IoU>0; at least1/2 corrected route bollard
opportunities has both an alert and support overlap; no increase in other
verified target misses and no more than2 additional route false alerts/head
versus initial under the same evaluation policy; Willow VAL16 loses at most2
correct head decisions and keeps at least80% of initial positive support IoU
in each head. Primary operating policy is DEV-selected, historical cutoffs are
reported separately. These small, correlated regression conditions diagnose
useful retention, not real-world reliability. Missing coverage stays unknown.

Success preserves a diagnostic challenger only. Failure retains the original
model, records the remaining failure, and stops this run without a rescue fit.
Complete scoped delivery and process release after the one comparison.
