# MZ54 execution and disposition

The [matched result](MZ54_FULL_RGB_RESULTS_20260911.md) fails its primary
near-body improvement gate. FULL adds2 native-grounded events below the crop
but loses19 other true events relative to matched CROP. Preserve this as a
NEGATIVE_CONTROL for wider-field learning with local-only ToF conditioning;
it does not rule out full-RGB models with other legitimate metric context.

The actual CUDA pipeline took258.778 seconds:97.088 for training features,
11.448 for CROP fitting,14.863 for FULL fitting,123.953 for streamed evaluation,
with remaining time for source/model initialization and output binding.
CPU scoring took2.478 seconds. Each head has10,708 trainable parameters.
10,517 full and3,000 crop feature extractions include16 old TRAIN crop replay
frames; their802,816 values match the existing cache exactly. Model geometry,
packet sanitization, masked context and UNKNOWN gradients passed10 synthetic
CPU tests. Source preparation took2.744 seconds with47 bound files; its
initial receipt-layout handling failure is preserved alongside the correction.

The runtime verified4,236,880,570 RGB bytes across10,533 loads, reused existing
old crop features and generated no new global-model predictions. Only1,250 new
training crops were held in RAM (250,880,000 bytes). A4,408,012,928-byte temporary
full-feature NPY provided4,783 repeat-access training rows; evaluation streamed
the remaining frames. No second permanent dense cache was created.

After run/score checks and process exit, the exact feature file was handed to
the separately registered MZ56 metric-context comparison through an NTFS hard
link. File identity, hash and row-index manifest were checked, then the MZ54
scratch tree was removed. The sole remaining link belongs to MZ56 and must be
removed at that experiment's terminal cleanup. This preserves computation
without a second physical4.41GB allocation; it is not disk reclamation yet.
All MZ54 checkpoints, outputs, source bindings and failure logs remain.

MZ56 will test measured in-field context with outside sensor coverage still
false. That separate comparison cannot retroactively rescue MZ54's gate.
Neither experiment changes ToF's physical field or supplies native truth to
prediction. Evidence remains consumed controlled Development.
