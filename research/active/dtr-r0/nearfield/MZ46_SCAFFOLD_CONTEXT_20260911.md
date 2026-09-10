# MZ46: remove ancillary supports at fixed target geometry

2026-09-11 EXPLORE. MZ45 found failure on four new mesh forms, but those
forms also lack the retained rod's12 ancillary supports and differ in size
and placement. Test whether support context contributes to the retained
rod's apparent advantage before changing another model.

Freeze the four far oblique-rod states from MZ44, with its camera, site,
lighting, renderer, native materials and complete adjustable_cross_member
object dictionary unchanged. Remove only the other12 objects per case.
Record original case identities and exact source/runtime hashes. This is
one four-frame source intervention, not another20-frame acquisition.
The existing four supported frames and saved MZ45 predictions are the
comparator; the MZ45 new-form failure remains a negative comparison.

Evaluate all eight old/new frames with unchanged RGB, ToF, MZ5, MZ28,
MZ35, MZ37, MZ43 ToF/ensemble and ideal/mixed joint FUSION readouts under
the fixed IDEAL, MERGE_CLOSE and DROP_CLOSE observation conditions.
IDEAL is a comparator, not a VL53L8CX capability assertion. Missing and
mixed observations retain their existing semantics; no native truth,
object identity or support mask enters model input. No fit, calibration,
threshold selection or checkpoint change is allowed.

First replay the four supported frames against their MZ45 decisions and
saved arrays. Independently reconstruct native event masks/counts for all
eight frames. Report native labels, UNKNOWN, changed mask pixels and
depth differences on event pixels, not merely intended labels. Attribute
an effect to ancillary context only on pairs whose target dictionary,
native event labels and masks are unchanged; report any remaining native
depth difference explicitly. Changed visibility is a confound, not CLEAR.

Report paired TP/FP/FN and logits for each state and condition. Loss of a
previous correct positive on an eligible pair supports context dependency
for that case; improvements show a harmful context effect. Unchanged task
decisions do not support context removal as an explanation of MZ45's
failure, even if logits move. Four states on one site do not isolate every
shape/size effect or establish broad generalization.

A prespecified2x2 diagnostic crosses old/new RGB with old/new ToF packets
using the same fixed models and cached visual computation. These hybrid
inputs distinguish branch contributions; they are synthetic diagnostic
combinations, not additional physical observations or a trained method.

Budget: one preparation, one four-frame capture, one fixed inference and
paired scoring, plus necessary source/label/compact checks. Mechanical
failures may be diagnosed and repaired with retained evidence, never by
outcome-driven recapture. Inspect all four new RGB/native previews. Reuse
extraction-free lossless sources, retain raw evidence on the worker and
return a byte-verified compact package. Release only owned UE/Zen and
temporary resources. Stop this experiment after its diagnosis, record and
scoped delivery; no automatic model training or larger capture is included.
