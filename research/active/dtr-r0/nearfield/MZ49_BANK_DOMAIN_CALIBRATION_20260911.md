# MZ49: calibrate over the candidates that reach inference

2026-09-11 EXPLORE. MZ47's saved-output diagnosis identifies a candidate-domain
mismatch: all four ENRICH old-DEV calibration maxima have no banked candidate.
Test whether applying the same zero-added calibration rule to the final
candidate domain recovers useful events without additional false alerts.

Freeze both MZ47 REPLAY/ENRICH checkpoints and every saved inference output.
For each arm, calculate exactly one new cutoff on the same original1000 DEV
frames under DROP_CLOSE, replacing raw/support with restricted_raw/support.
Keep the same MZ5 baseline and native event truth. This changes only the
candidate domain of cutoff_zero_added; no new percentile, optimization,
threshold search, training, inference or outcome-selected rule is allowed.
Original MZ37 remains an unchanged comparator, because its original calibration
condition also differs; do not conflate that change with domain alignment.

Recompose each arm with the unchanged availability bank, MZ35 selector and
MZ37 restoration. Score the original1000 DEV, relation2000 DEV, distance1000
DEV, all44 rich frames with their12fit/32nonfit split, and all400 MZ36 attempts
with20 excluded frames/80UNKNOWN. Report all three fixed observation profiles,
although only DROP_CLOSE calibrates. MZ48 data remains excluded.

Compare each recalibrated arm with its own unchanged MZ47 counterpart, then
compare the two recalibrated arms to keep the data contribution visible.
A useful restricted-input component must recover at least one true event
outside calibration without adding any false event across the noncalibration
cohorts, retain prior positives and state IDEAL/MERGE costs. If false alerts
increase, retain the tradeoff only as a challenger/diagnostic, without claiming
zero-cost improvement. This cannot resolve MZ47's missing native witnesses.

Budget: two single cutoff calculations and one fixed CPU saved-output replay,
independent scalar recount, mechanical composition parity, scoped delivery.
No GPU job, dense cache, new source, additional fit or threshold rescue.
All evidence is consumed controlled Development, not hardware validation.
