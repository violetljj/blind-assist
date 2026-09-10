# MZ50: local intrusion evidence without mandatory echo candidates

2026-09-11 EXPLORE. MZ47 finds only5 native-supported positives among12
near-object fit positives under DROP_CLOSE; hard echo eligibility excludes
the others before learning. MZ49 improves recall by changing the calibration
domain but adds old-source false alerts. Test a shared spatial query head
whose RGB evidence can survive missing returns, using MZ48's richer source
and training-only native cell labels independent of selected echo bins.

Keep the original frozen visual backbone, detail normalization and MZ37
predictor unchanged. Initialize the local16-channel convolution from MZ20
BODY_RANK. A40->32->4 MLP consumes16 local and16 zone-mean visual channels,
absolute/within-zone coordinates, two ranges divided by4 and two validity
bits. Initialize its first40 input weights and all output weights from the
MZ20 query head; omit the old current-echo range and echo index channels.
Fine-tune this small convolution/MLP once. Native geometry, cell labels,
family/site/context IDs and intended relations never enter inference.

Compare two readouts of exactly the same fitted cell logits: OPEN pools all
angular cells; GATED pools only cells eligible under at least one currently
valid return and the unchanged body/head query geometry. Thus OPEN versus
GATED isolates mandatory echo eligibility without differing fit budgets or
weights. Compare both additive candidates against unchanged MZ37, MZ5 and
the restricted-training MZ43 ensemble. Negative RGB logits or absent echo
eligibility are not evidence of free space.

Use all2,560 admitted MZ48 frames. Preserve the frozen640 held-out-site
frames. Among the1,920 TRAIN_CANDIDATE frames, hold all384 birch frames out
of fitting and calibration. Of the other four families, region06-site002
provides256 calibration frames; the other five sites provide1,280 fit frames.
Pairs remain intact. All are controlled Development on consumed source sites;
the new family and site partitions are withheld from this fit, not fresh blind
or foundation-model-independent tests. Keep the original44 rich frames and
all400 MZ36 attempts with their20 excluded frames/80UNKNOWN for transfer checks.

Budget: one seed150 fit,600 steps,batch16,Adam0.001. Draw1,280 fit IDs with
replacement from a fixed NumPy schedule. Cycle IDEAL, MERGE_CLOSE and DROP_CLOSE
by step. The3 profiles remain limited-sensitivity proxies, not a calibrated
VL53L8CX simulator. Native angular cell labels are unchanged across profiles;
they describe actual surface/query membership and do not label a merged echo.
Use per-frame positive/negative balanced local BCE, plus0.25 frame-query BCE.
The frame term skips positive events with no native witness in the45-degree
crop. Local unknown cells are masked. No early stop, resampling, extra seed,
checkpoint selection or outcome-dependent retraining is included.

After this single fit, calibrate each readout exactly once with the existing
zero-added cutoff rule over original1,000 DEV plus256 rich calibration frames,
using DROP_CLOSE, native event truth and MZ37 as the additive baseline. No
bank/reject/restore changes or threshold search. Score all three profiles on
the old DEV, relation2,000 DEV, distance1,000 DEV, MZ48's four partitions,
44 older rich frames and all400 MZ36 attempts. Report TP/FP/FN, per-query and
family/site/context/range tradeoffs, known/UNKNOWN, paired contexts, and whether
new positives have matching native local evidence. Preserve every prior MZ37
positive. Calibration frames are reported separately from transfer evidence.

A useful component adds at least one true event on the640 held-out-site
or384 nonfit-family frames under DROP_CLOSE, with no additional false event
across noncalibration cohorts; report all other profile costs. Any added false
alert retains the result only as a challenger with its measured tradeoff.
If OPEN cannot beat GATED on missing-echo positives, the proposed gate removal
has not earned its extra candidate freedom. Do not promote the default app or
claim natural-scene/hardware/safety performance from this experiment.

Prepare code and CPU metadata while both machines collect. Start GPU work
only after primary collection/auxiliary jobs release their resources. Extract
new visual features once into RAM, reuse the existing old detail mmap, and
save only compact predictions, the small checkpoint, schedule and receipts.
Do not create another dense cache or recompress original RGB. Release GPU
processes on success/failure, retain scientific evidence, and stop after this
fixed experiment, diagnosis if it changes the decision, and scoped delivery.
