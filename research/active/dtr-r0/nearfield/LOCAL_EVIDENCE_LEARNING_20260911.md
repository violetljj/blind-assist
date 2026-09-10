# Apply restricted-sensing research to the observed local-evidence gap

2026-09-11. Applied reading during MZ48 capture; proposals below are not measured
algorithm improvements. The immediate evidence is MZ45/MZ46 context and shape
failure plus the matched MZ47 enrichment comparison. Its two300-step fits used
only12 new near examples and cannot adjudicate richer-data training in general.

## Useful ideas and the input limits that travel with them

- [DELTAR, ECCV2022](https://zju3dv.github.io/deltar/) treats a lightweight ToF
  zone as a depth distribution over an area. Its RGB/ToF fusion and calibration
  are relevant; interpreting one return as one RGB pixel would discard that
  distinction. Its depth-distribution inputs and dense reconstruction task are
  different from the current distance/validity obstacle-event experiment.
- [LiteSense, CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_LiteSense_Lifting_Lightweight_ToF_with_RGB_for_High-Resolution_Metric_Depth_CVPR_2026_paper.pdf)
  uses local cross-modal fusion with compact normalized histograms, including
  VL53L8CH hardware in its collection. Local fusion is worth testing. CNH is
  additional evidence, so its reported results do not establish what our
  VL53L8CX distance/validity-only contract can achieve. Do not synthesize a rich
  histogram and quietly give it to the deployed student.
- [Depth Privileged Object Detection, AAAI2021](https://ojs.aaai.org/index.php/AAAI/article/download/16459/16266)
  studies depth-derived supervision with RGB-only inference. This supports a
  training/inference distinction, rather than a requirement to feed dense depth
  at deployment. Its object-detection results do not prove metric intrusion
  detection with missing returns.
- [Distill What RGB Can Recover, 2026 preprint](https://arxiv.org/html/2608.00110)
  emphasizes selective transfer of privileged geometric evidence when the RGB
  student can recover it. Its VLM setting is distant from this small readout;
  the applicable caution is that training truth cannot make visually
  unobservable metric facts observable. Keep uncertain cases explicit.

The [VL53L8CX datasheet](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf)
also qualifies detection volume by target distance, reflectance, ambient light,
resolution, sharpener, mode and integration time. A nominal zone count or
maximum range is not a promise to resolve every small object. The present
MERGE_CLOSE/DROP_CLOSE conditions remain sensitivity proxies, not a calibrated
device simulator.

## Next mechanism to test, based on our failures

The current local readout requires a geometrically eligible measured return.
When DROP_CLOSE deletes that return, the corresponding local supervision and
candidate disappear together. More examples may improve surviving candidates,
but cannot by themselves remove this architectural constraint. MZ41 already
shows that blindly protecting missing-observation queries adds false alerts;
MZ46 shows that a global RGB fallback can depend on surrounding supports.

A concrete alternative is a small spatial intrusion head over the existing
RGB crop features. During training, supervise it with native event membership
per angular cell, independently of whether a ToF echo survives. At inference,
provide only RGB, range/validity packets and fixed calibration. Use packet
evidence to condition confidence, without deleting all local RGB candidates
when a return is absent. These would be predicted hypotheses, not measured
free-space or clearance facts. Compare with the fixed original pipeline and
an equal-budget replay arm; assess false-alert costs as well as recovered
events. This is an untested proposal, not a new retained baseline.

MZ48's paired support-present/absent targets can test whether local predictions
remain stable when the same native intrusion is preserved. An invariant loss
is a separate candidate mechanism: first establish an ordinary supervised
spatial-head comparator before attributing benefit to pair consistency. Keep
site-held-out Development, shape/distance transfer and visible nonintruding
objects visible in the results. Additional collection should expand missing
combinations and evidence coverage rather than duplicate a few RGB scenes.

Native slot-independent labels are compact supervision and belong in the
evaluator package. They do not belong in predictor manifests. Preserve full
native source on its capture host for later independent checks; transmit the
training packet, original RGB bytes, labels and source hashes. This makes the
larger collection useful for several mechanisms without another raw-data
transfer or another permanent dense-feature cache.
