# MZ47: native local enrichment under missing close returns

2026-09-11 EXPLORE. MZ45/MZ46 expose failure of global decisions on richer
forms and support removal. Test whether adding actual native local supervision
improves the existing spatial readout, with an equal-budget old-data replay
control. Retain MZ20/MZ28/MZ35/MZ37 and the two fixed failure cohorts.

Both arms start from the unchanged MZ20 BODY_RANK checkpoint, with its saved
crop grid, frozen backbone/normalization and architecture. Fit300 steps of
Adam0.001 with the existing balanced_local +0.25*body_rank_query_loss.
Both use the first300 original MZ15 batches of16 TRAIN frames. Each step adds
four frames: REPLAY samples four old TRAIN frames matching the four extra
event-label patterns; ENRICH uses one complete near-state quartet from pipe,
ladder or pouch in MZ42, cycling the three families. Selection seed147 is fixed.
Only those12 richer frames enter enrichment fitting; no new evaluation
outcome controls sampling, checkpoints, thresholds or stopping.

Training and old-DEV calibration use the fixed DROP_CLOSE observation.
Delete the dropped return slots and their local contributor supervision;
surviving slots retain their original native labels. Never apply IDEAL slot
contributor labels to a MERGE_CLOSE midpoint. MERGE_CLOSE and IDEAL remain
fixed evaluation conditions only. Report positive events with no candidate
or no eligible native witness; they remain in the denominator and are not
converted to negative/CLEAR supervision.

Use the unchanged cutoff_zero_added rule once per arm on the original1000
DEV frames under DROP_CLOSE, with the corresponding frozen MZ5 baseline.
The availability bank and MZ35/MZ37 selector/restoration parameters stay fixed.
Report local scores before/after the bank and the composed final decisions.
No alternative cutoff, longer schedule, additional fit or outcome-driven
rescue is included.

Keep distinct fit and evaluation masks: MZ42 birch near plus MZ44 birch far
(8 form-transfer frames); MZ44 far pipe/ladder/pouch(12 distance-transfer);
retained rod controls and MZ46 unsupported rod(12 context/control frames).
All44 unique rich/context frames remain visible, including the12 fit frames.
Check the old DEV and relation/distance DEV cohorts and all admitted MZ36
frames, preserving its UNKNOWN. These are consumed Development checks, not
fresh confirmation or independent hardware tests. The separately collected
kilotier source is excluded from this fixed comparison.

Report TP/FP/FN, exact frames, gains/losses versus REPLAY and frozen strong
comparators, and old-data regression. Evidence for enrichment requires added
true events over REPLAY on the predeclared non-fit transfer groups without
added false events there; legacy costs remain explicit and block an automatic
replacement claim. No gain distinguishes missing evidence from a failed
training hypothesis. Retain a useful challenger only within measured scope.

Reuse the existing2.32GB MZ16 mmap; compute new RGB/crop features once in memory
and store only compact predictions, local labels and small checkpoints.
Budget: one preparation, two300-step fits, fixed scoring and meaningful
source/decision audits. Preserve failures and source/checkpoint identities;
repair mechanical failures without changing this comparison. Finish records,
scoped delivery and owned resource release after this bounded run.
