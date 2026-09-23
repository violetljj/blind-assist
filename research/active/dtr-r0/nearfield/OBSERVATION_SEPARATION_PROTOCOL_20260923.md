# Matched observation separation against repeated-pose variation

2026-09-23 EXPLORE. User authorized the proposed input-distinguishability check.
Reuse consumed LOCAL stability and rescue-fresh sources. No new capture, training,
alert change or threshold search. The question is whether opposite corridor truths
still have measurable public-input differences, not an information-theoretic ceiling.

Use all eight geometry groups in each source at approach_dwell_return frames4..8.
Verify identical camera and objects within each five-image repeat, and identical
camera/background/target sizes/materials across INSIDE/BOUNDARY/OUTSIDE. Only target
lateral position may differ. Main pairs are BOUNDARY versus OUTSIDE; INSIDE versus
OUTSIDE is the easier reference. These are32 pairs from16groups and48poses, not
independent draws or new scenes. Do not exclude failed validity/truth pairs.
Distance is the saved dwell depth (2.55m in stability; verify rescue metadata).

Public RGB distances are full-image absolute difference mean and maximum fixed
20x20 patch mean, over all channels at original320x180 resolution. No target mask,
crop selection, labels or learned features. For each pair retain the20within-pose
distances and25cross-pose distances. Record cross median / within95th percentile,
fraction of cross distances exceeding that percentile, and distance-separation AUC
(not obstacle classification AUC). Constant-zero cases have explicit null ratios.
Repeat after excluding arrival frame4 as a predeclared sensitivity; no selection
between the two summaries. RGB variation includes rendering history/exposure and
is not a calibrated physical camera-noise model.

ToF: reproduce original saved observations for all240 selected frames. At frame4
of each of48poses generate16independent draws with new pose-specific seeds under
the unchanged single-return proxy. Save only public64x6tokens as observations;
native depth is a source-construction input, never a predictor input. Compare all
120within-pose draw pairs per pose and256cross-pose pairs per matched contrast.
Use RMS distance difference in metres on common-valid zones, RMS standardized by
sqrt((.01+.02*zA)^2+(.01+.02*zB)^2), and validity-mask disagreement fraction.
Report missing comparisons and common-zone counts; no missing value becomes free
space. The proxy is uncalibrated:5%conditional dropout,10cm dominant bins and
Gaussian range noise. Noise repetitions are not independent scene evidence.

No combined score or metric selection. Seal all distance matrices before joining
evaluator labels. Check opposite valid union BODY/HEAD truths at all repeated frames;
report source UNKNOWN unchanged. For each fixed metric a pair is separated only
when >=90%cross distances exceed its within-pose95th percentile and no comparison
is missing. Report every pair, each cohort/relation, weakest groups and hard pairs.
This is a descriptive evidence criterion, not an alert or deployment gate.

If public RGB or ToF differences survive variation broadly, retain a diagnostic
component: input information is not shown exhausted in these matches; the next
question is transferable task readout. Partial separation points to concrete hard
groups. No separation means only these metrics failed, never that all decoders or
hardware cannot succeed. Same-pose templates, pose identities and paired negatives
are evaluator aids and cannot be promoted to a deployable classifier.

One governed extraction/analysis with bounded16draws per pose; no sweep or automatic
successor. Repair mechanical defects with separate receipts and unchanged scientific
criteria. Compare equivalent CPU/GPU RGB matrix work including transfers; fixed
NumPy sensor reconstruction remains CPU. Preserve all input hashes, public matrices,
draws, source snapshot and summaries. A/LOCAL/UNKNOWN and old negative controls stay.
