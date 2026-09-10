# MZ15: learned shared return support

EXPLORE, consumed Development,2026-09-10. User authorized the small model
comparison after MZ14. No collection, encoder finetuning, sensor-interface
change, temporal restart, EVAL access or default-App promotion.

Hypothesis: directly learning a shared within-zone return-support field can
suppress unsupported winning locations while retaining useful far evidence.
MZ14 identified17 MZ13 added false bits without actual source at the winner;
its source oracle also lost6 recovered far TPs. This motivates learning both
correct support and discrimination, not installing that oracle or another gate.

Two matched new fits: SHARED predicts one source-support value per return/cell,
then applies fixed angular/range BODY/HEAD geometry; QUERY predicts four local
query-specific values with the same feature/context architecture and geometry.
Both use a64->16 3x3 convolution on frozen normalized18x32 features, sample the
existing7x7 angular cells per zone, concatenate local and zone-mean16D context,
absolute/relative2D position, both return ranges/validity, current range and echo
slot (42 inputs), and a42->32->1 or4 MLP. QUERY begins with repeated SHARED final
weights and identical shared layers. Multiple supported positions/returns remain
possible; no one-pixel assignment or mutual exclusion across BODY/HEAD.

Supervision is evaluator-only actual selected-bin contributor presence for
SHARED, and actual per-query contributor presence for QUERY. Local BCE balances
positive/negative cells separately within each frame with known pixels and a
valid return. Query BCE has weight0.25; local BCE weight1. Query loss is masked
where no geometric candidate exists, or where a full-image positive has no
correct eligible return contributor, to avoid forcing support outside coverage.
The inference signature takes only frozen dense RGB features and range/valid
packets. Native labels and actual source locations never enter it.

Exactly1200 Adam steps per arm, lr0.001, seed115, batch16: four samples each from
old TRAIN2500, relation TRAIN5000, distance TRAIN2500, and consumed MZ6clean200.
Same sampled batch IDs in both arms, last checkpoint only, no parameter sweep.
This deliberately gives sequence25% sampling weight; results on it are training
regression. Existing MZ9 frozen readout is a historical component comparator,
not an equal-exposure baseline. QUERY is the matched learned comparator.

MZ5 baseline is immutable for this bounded augmentation test. For each query,
choose the lowest cutoff strictly above the maximum model score on oldDEV
negative labels where MZ5 is negative and a candidate is available. This permits
zero added oldDEV false bits by calibration; it is not independent validation.
Accept new positive support above cutoff, otherwise retain MZ5. This composition
preserves baseline coverage and every positive, including its existing false
and premature-near judgments; it cannot solve those errors by construction.

Freeze cutoffs on oldDEV1000. Evaluate relationDEV2000 and distanceDEV1000
separately, MZ6clean/stress200 each, wrong-zone RGB control for both arms, and
the oldDEV regression. New3000 are already consumed diagnostics, excluded from
gradients/cutoff fitting, never independent confirmation. Report per-query TP/FP,
exact frames, family and group/site outcomes, unsupported/UNKNOWN, local winning
source consistency, and retention of the6 MZ14 oracle-lost far additions.

Useful effect requires no added FP per query in either newDEV cohort or sequence
condition, strictly more far TPs than MZ5 in each newDEV cohort, clean and stress
pole recall at least48/49, and no loss of baseline near TPs. Shared attribution
is supported as a mechanism only if wrong correspondence reduces its recovered
far evidence and the matched QUERY does not explain an equal/better effect.
Judge effect separately from mechanism; retain a useful simpler/matched method
honestly if it wins. Failure ends these two fits; no cutoff rescue or additional
training/capture. Finish audit, report, registration, push and resource release.
