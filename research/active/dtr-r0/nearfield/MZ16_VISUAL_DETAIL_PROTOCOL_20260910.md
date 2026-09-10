# MZ16: controlled visual detail at fixed readout

EXPLORE, consumed Development,2026-09-10. User authorized testing visual spatial
detail after MZ15. No capture, EVAL access, encoder finetuning, temporal module,
threshold rescue or baseline promotion without a measured effect.

Question: does retaining original local RGB detail improve actual return/query
localization and far detection at the fixed false-alert budget? MZ15's proposed
shared readout failed. Keep its QUERY architecture, weights at initialization,
optimizer,1200steps and saved16-frame training batches fixed between two new arms.

HIGH_DETAIL crops native640x360 RGB at(208,68,432,292), yielding224x224 pixels
covering the ToF field. LOW_DETAIL first applies the old BOX resize256x144, then
BILINEAR640x360, and takes exactly the same crop. Both arms have identical crop,
FoV,224input, frozen encoder,64x28x28 feature grid, old frozen normalization and
10740-parameter QUERY readout. LOW does not regain discarded detail by upsampling.
This paired blur/control isolates available detail within this new ROI pipeline;
comparison to historical MZ15 additionally changes crop/context/grid and cannot
be attributed to resolution alone. Image resampling is the intended intervention.

Dynamic frozen extraction preserves the backbone's last stride4 feature and
stride32 output, then the existing deep projection/detail layers. Check the
dynamic256 path against legacy extraction before caching. The original frozen
model is not edited. Recalibrate angular sampling to the ROI; keep physical rays,
ToF ranges/validity, query masks and native labels identical. Normalized absolute
coordinates now refer to the common ROI in both arms. Missing packet support
remains UNKNOWN, never CLEAR.

Reuse MZ15 batch IDs:19200 draws/arm from the existing10200 pool,7562 unique
training frames. Cache only their union with oldDEV1000, MZ6clean200 and newDEV3000:
11562 distinct RGB frames. Native labels/packets are reused from verified caches,
not newly inferred model inputs. MZ6 is trained regression; newDEV remains consumed
Development and is excluded from gradients and cutoff calibration.

Local evaluation precedes task interpretation. On known valid geometrically
eligible return/cells, measure average precision for actual query contributor
presence, per query and macro over queries with positives. Compute local
precision/recall at query-specific cutoffs calibrated to at most1% negative-cell
FPR on oldDEV. These diagnostic local cutoffs are separate from alert cutoffs and
never select the deployed/composed output. Save per-frame local TP/FP/FN and
denominators; source-presence consistency of alert winners is a separate check.
Cell labels signify at least one actual contributing pixel, not exact segmentation.

Alert cutoffs use MZ15's zero-added-oldDEV-FP rule, then freeze. Compose with MZ5
by adding positives only, retaining baseline coverage and existing false alerts.
Compare per-query TP/FP, exact frames, sign/cabinet and group/site outcomes, pole
clean/stress, and wrong-zone RGB control. Correct stress contributor slot alignment
is mandatory and does not alter inference packets.

Evidence for better localization requires HIGH macro local AP > LOW on each
newDEV cohort, with local precision/recall and FPR reported rather than claiming
equal FPR transfers automatically. A usable HIGH augmentation additionally needs
zero added FP per query on both newDEV cohorts and both sequence conditions,
strictly positive far gains on both newDEV cohorts, clean/stress pole>=48/49,
and immutable baseline near positives. Report localization gains separately from
task admission. One seed and paired1200step fits only; stop afterward regardless
of outcome and finish audit, delivery and resource release.
