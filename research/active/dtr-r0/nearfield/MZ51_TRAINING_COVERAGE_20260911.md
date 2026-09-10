# MZ51: matched new versus old negative replay

2026-09-11 EXPLORE. MZ50's richer native spatial learning adds71 true events
and1 false event on1,024 nonfit frames, but adds19 false events on older
noncalibration cohorts. Its fit uses only1,280 new-source frames. Test whether
training-negative coverage reduces that cost without erasing the useful new
local evidence. Preserve MZ50's fixed outputs and same-weight OPEN/GATED
comparison; do not retune its thresholds.

Start two arms from the exact same completed MZ50 checkpoint, without optimizer
state. Keep the frozen visual backbone, spatial head architecture, packet
profiles and all MZ48 partitions. Both arms use600 Adam0.001 steps, each with
the same8 new fit frames and the unchanged MZ50 local-plus-query loss. Add8
query-negative presentations, two for each query, with0.25 times mean softplus
of the corresponding OPEN maximum. This is frame-event negative supervision,
not a local-negative label or evidence that unknown space is clear. Other
queries in the chosen frame are not assigned a negative target.

NEW_NEG draws these extras only from MZ50's1,280 fit frames. OLD_NEG draws
them only from the existing7,562 old TRAIN IDs with an available old detail
map. Use their existing native event truth to require the selected query be
negative. Neither model scores nor calibration/heldout outcomes select training
IDs. NumPy seed151 fixes shared batches and both negative schedules before
fitting; sample with replacement. Cycle IDEAL, MERGE_CLOSE, DROP_CLOSE by step,
identically in both arms. Record actual unique frames and query presentations.
This compares negative-source coverage against equal-budget continued learning,
not a larger training budget or an outcome-mined validation replay.

Fit/local supervision remains on the1,280 new frames. Keep256 rich calibration,
640 held-out-site and384 withheld-birch frames unchanged and pair-disjoint.
Old TRAIN IDs must be disjoint from original1,000 DEV, relation2,000 DEV and
distance1,000 DEV. Never relabel a merged echo; local native surface labels
are independent of its bins. Retain MZ36's400 attempts/80UNKNOWN and older44
rich frames as consumed Development transfer checks. No protected blind data,
fresh-confirmation or hardware claim is authorized by this experiment.

After each arm, calculate exactly one cutoff per OPEN/GATED readout on the
same original1,000 DEV plus256 rich calibration frames under DROP_CLOSE,
using unchanged MZ37 as additive baseline and the same zero-added rule.
Four fixed cutoff vectors total; no search, checkpoint selection, additional
fit, epoch extension or threshold rescue. Report every original comparator,
both new arms/readouts, all three observation profiles, per-query/group TP/FP/FN,
paired gains/losses versus MZ50 and equal-budget NEW_NEG, local winning-native
evidence and missing-candidate recoveries. Baseline MZ37 positives must remain.

Primary decision: OLD_NEG GATED should reduce old noncalibration false alerts
relative to both MZ50 and NEW_NEG while retaining at least MZ50's453 nonfit
DROP true events with no more than21 false events. Report the full tradeoff
if this fails. A remaining false-alert cost keeps the result a challenger;
no default promotion. OPEN provides the parallel test of whether training
separation helps its previously severe negative maxima; candidate restriction
and calibration domains remain unchanged.

Reuse MZ50 saved comparator outputs and the existing old detail mmap. Decode
each new RGB once, compute only the required frozen detail features into RAM,
and check a16-frame pre-fit replay against MZ50. Save only small checkpoints,
schedules, predictions and evidence, without a new permanent dense cache.
Run GPU work after confirming prior capture/model processes have released.
Stop after these two fixed fits, fixed scoring, a decision-changing saved-output
diagnosis if needed, scoped delivery and resource release. The separate CPU
audit of80 existing native samples may assess the45-degree crop visibility gap;
it must not change this run's inputs or fit schedule.
