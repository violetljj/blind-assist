# MZ52: full-RGB native cell supervision on the existing source

2026-09-11 source derivation for controlled Development. MZ51's CPU audit
finds352 of512 MZ48 BODY_NEAR events outside the45-degree local field. In the
80 prescribed native samples, all10 such gaps are below the224crop despite
hundreds of corresponding native pixels in the original RGB frame. Prepare
spatial supervision for a subsequent wider-RGB local model without increasing
ToF observability, recapturing scenes or returning complete raw depth.

Use exactly the existing2,560 MZ48 native depth files and immutable source
bindings, with their original640x360 RGB/ToF packets and1920/640 source split.
Derive full-frame cell_event_presence[N,45,80,4] by partitioning original pixels
into fixed nonoverlapping8x8 blocks. A cell is positive if any valid native
point belongs to the original BODY_NEAR/BODY_FAR/HEAD_NEAR/HEAD_FAR query;
cell_known[N,45,80] is true if any point is valid. Validity is the unchanged
finite, axial>0, axial<100, radial<=4m rule and original level-camera geometry.
Store per-cell event counts and valid-point counts as uint8 (maximum64) so
full-frame query counts and the>=3-pixel event rule can be recovered exactly.
Empty/unknown cells are not negative-space or clearance evidence.

These arrays are training/evaluator-only. Keep original ToF45-degree packets,
validity and sensitivity proxies unchanged; neither full-frame depth nor these
cell labels become predictor inputs. Preserve all site/family/pair identities,
including the MZ50/MZ51 fit/calibration/nonfit partitions. The evidence proves
label coverage and a possible RGB learning opportunity, not successful RGB
metric ranging or a physical sensor's ability outside its field of view.

Process existing raw files on their owning primary/secondary machine and
return only compressed cell labels and exact input/output hashes. Add new
versioned files under the MZ52 artifact tree; do not alter MZ48 source ZIPs,
raw data, spec, auxiliary or receipts. No UE capture, training, inference,
model/threshold selection or permanent dense feature cache is included.

Require all2,560 input native hashes and exact original full-event counts/
labels to match. Validate all8x8 cell counts against pixel membership, unknown
mask consistency,1280 paired event-cell invariances and unique frame identity.
Independently recount the80 already returned native samples using a separate
block-loop path and verify their serialized labels. Report all full-frame
witness opportunities, old45-degree gaps, output/transfer bytes and actual CPU
or CUDA backend/time. Choose existing NumPy CPU aggregation when it avoids
GPU contention; this is fixed pixel/bin bookkeeping, not neural computation.
Release task-owned workers on success/failure, retain evidence, and stop after
one complete derivation/verification and scoped source delivery.
