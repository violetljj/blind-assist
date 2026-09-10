# MZ56: legitimate in-field metric context for full-RGB hypotheses

2026-09-11 EXPLORE. MZ54's FULL_RASTER gains only2 nonfit events over MZ37;
its outside45 cells are deliberately invariant to all ToF packets. Test
whether measured in-field scene context helps near-body hypotheses outside
the measured field, rather than treating the local packet restriction as a
physical prohibition on multimodal reasoning. Wider RGB alone is insufficient
in MZ54; this does not establish that global context will help.

Start two matched600-step continuations from the exact completed MZ54 FULL
checkpoint, without optimizer state: LOCAL_ONLY and GLOBAL_ANCHOR. Preserve
its full raster, encoder, local convolution and41 existing head inputs.
Both get an identical49->32->4 head, copying first41 columns and zeroing8 new
columns. Both instantiate the same small6->8 ReLU packet encoder. Only
GLOBAL_ANCHOR receives its pooled vector; LOCAL_ONLY receives8 zeros. Same
initial model logits and parameter budget, identical training rows/rules.

Encode the64 actual zone tokens from two cleaned range/4 slots, two validity
bits and two static zone angles. Mask zones without any valid slot and average
the8-dimensional tokens over valid zones. All-missing packets produce exactly
zero context even after learning. This vector is legitimate in-field context,
not outside-object ranging, calibrated uncertainty or invented signal/sigma.
Outside local range/validity/sensor-coverage stay zero/false/false. Source truth,
camera pose from UE, object identity and observation-profile names never enter
prediction. Static original calibration and observed packets remain allowed.

Reuse MZ51's exact saved shared/OLD_NEG/query600x8 schedule, Adam0.001,
seed151 and IDEAL/MERGE_CLOSE/DROP_CLOSE cycle. Preserve MZ54 full native local
and witnessed-frame losses plus selected old-negative penalty and all UNKNOWN.
Fit only the same1280 eligible new frames and original TRAIN IDs; retain256
rich calibration,640 held-out-site,384 nonfit-family, original old DEV and
MZ36's400 attempts/80UNKNOWN. These are consumed Development, not fresh tests.

Each arm gets one unchanged zero-added cutoff vector from original1000 DEV
plus256 rich calibration under DROP, with MZ37 additive baseline. Two vectors
total. Report every previous MZ54/MZ53/MZ51 comparator, all query/profile errors
and paired changes. Same GLOBAL weights with only its global vector suppressed
use the existing GLOBAL cutoff as a third readout; no third fit/cutoff. This
isolates inference use of the new path without hiding effects of learning.

Primary check: more nonfit BODY_NEAR true events with a native winning cell
outside45 than LOCAL_ONLY, with no higher total nonfit or old noncalibration
FP. Also report total BODY_NEAR TP, all other query losses, each winning
cell's native and sensor coverage, and original-positive retention. Validate
all-missing behavior and exact initial logits with the new columns zero.
A useful gain keeps a challenger with costs; failure rejects this design,
not all metric context. No threshold rescue, epoch extension or model selection.

Reuse the exact already computed MZ54 training features via a source-bound,
read-only task-owned hard link, with hashes and original frame-index manifest.
Transfer ownership before removing MZ54 scratch. No duplicate physical dense
allocation or cache mutation. Re-extract only uncached full evaluation RGB in
fixed batches, with original SHA/encoder/normalization. Preserve original
training inputs and frozen baseline outputs; no new native-depth access.

Budget: two600-step fits, two cutoffs, one saved-output score and focused
geometry/initialization/UNKNOWN checks. Run GPU work only without primary UE.
Record actual costs and release all processes/handles; remove this task's last
temporary feature link after scoring, retaining checkpoint/output/source receipts.
Stop after scoped result/disposition/delivery. MZ55 capture is separate source
work and its outcomes cannot tune this experiment. No physical-sensor,
natural-scene, clearance or default-App promotion claim.
