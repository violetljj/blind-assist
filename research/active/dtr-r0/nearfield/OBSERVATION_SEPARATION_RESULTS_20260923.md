# Matched RGB differences survive repeated-pose variation

2026-09-23 EXPLORE. The fixed [protocol](OBSERVATION_SEPARATION_PROTOCOL_20260923.md)
completed without capture or training. Decision:
**MATCHED_RGB_DIFFERENCES_SURVIVE_VARIATION**. Retain a diagnostic COMPONENT:
these paired observations still contain measurable spatial differences. This
does not establish a transferable obstacle decoder or disprove information
limitations in other scenes. A, LOCAL, UNKNOWN and App behavior are unchanged.

## Matched evidence

Two consumed sources provide16geometry groups,48poses and240saved frames.
Every pose has five identical-camera/objects repeats (dwell indices4..8).
Each group supplies BOUNDARY versus OUTSIDE and INSIDE versus OUTSIDE:32pairs.
Camera, background, target sizes/materials and target forward/vertical positions
are identical within a contrast; only target lateral position changes. All32
pairs have valid opposite union BODY/HEAD labels for all five repeats.
The stability dwell depth is2.55m and rescue depth2.38m. These are not coverage
of the full0.3-3m range, a3m entrance test or independent natural scenes.

RGB uses the original320x180 public image: full-image L1 and maximum L1 over
fixed20x20patches, without target masks. Each pair has20within-pose and25cross-pose
comparisons. A fixed metric separates a pair when at least90%cross distances
exceed its within-pose95th percentile, with no missing comparisons. This is a
pair-distance diagnostic criterion, not an obstacle alert threshold or accuracy.

| Fixed metric: separated pairs | Stability boundary /8 | Rescue boundary /8 | Stability inside /8 | Rescue inside /8 |
|---|---:|---:|---:|---:|
| RGB full-image L1 |8|7|3|4|
| RGB maximum fixed-patch L1 |8|8|8|8|
| ToF common-valid range RMS |2|5|8|8|
| ToF noise-standardized range RMS |3|5|8|8|
| ToF validity-mask disagreement |0|0|0|0|

For RGB local patches, median cross/noise ratios are3.18/3.63for boundary
and3.92/9.94for inside (stability/rescue). The weakest pair is the stability
small horizontal head obstacle at the boundary: ratio1.16. Thus not every
pair has a large margin. Full-image averaging fails on several pairs where
the fixed local metric succeeds; this is evidence about these distance
metrics, not proof that pooling caused the earlier model failures.

The predeclared sensitivity excludes arrival frame4 on BOTH sides. All32local
RGB pairs remain separated; full-image totals change22/32to23/32. This removes
one arrival-history concern but does not make the remaining repeats independent
camera noise: rendering history, exposure and temporal antialiasing can contribute.

## ToF and coverage

All240saved ToF observations reproduce exactly from the original simulation
identities. For frame4of each of48poses,16new pose-specific noise draws give
768public observations. The unchanged proxy has a dominant10cmrange bin,
conditional5%dropout and Gaussian sigma=.01+.02*range. It is not calibrated
hardware noise. Native depth only constructs those public draws; it never
enters the distance metric or a predictor.

Each ToF contrast includes240within-pose and256cross-pose comparisons. Repeated
draws do not multiply the scene denominator. Missing returns are excluded only
from common-valid range coordinates, with overlap reported separately; they
are never zero-distance or distant/free-space returns. No whole comparison
was missing. Boundary cross comparisons retain at least48common zones.

Standardized ToF separates8/16boundary pairs versus16/16inside pairs. Among
the weakest boundary cases, the fraction exceeding same-pose noise is2.7%
for stability small protruding plane,3.5%for stability large hanging plane,
and10.9%for rescue small hanging plane. Other failures are close to the90%
criterion; retain raw fractions rather than treating the cutoff as a physical
information limit. Validity-mask disagreement alone separates none.

Original UNKNOWN remains105/120selected stability frames and101/120rescue
frames;90and85respectively are silent. No clear-space authority is added.

## Decision and next question

The evidence does not justify saying the current input budget is exhausted:
fixed local RGB differences survive repeated-pose variation for all tested
opposite-label pairs. It also does not justify saying a learned model can
generalize those differences. Pair identity and a matched alternative scene
are available to this evaluator, not to an online obstacle detector.

Prioritize a narrowly defined, transferable lateral-position readout that
preserves local RGB evidence and uses ToF only with explicit correspondence
uncertainty. A useful next test would hold out complete geometry groups and
compare preserved spatial layout with a spatially shuffled control, measuring
misses, false alerts and localization separately. This is a proposal, not a
new fitted successor, and the failed coarse regional pairing remains negative.
The present result gives no reason by itself to buy different hardware or
claim that added temporal views are necessary. Range-entry observability and
other distances remain open.

## Execution and retained evidence

Primary governed run succeeded on its first attempt in7.38s of diagnostic
execution, excluding runner overhead. Equivalent10-image RGB distance probes,
including transfers, measured CPU70.37ms versus CUDA1.86ms; the actual matrix
backend was CUDA on RTX5060Laptop. Fixed NumPy sensor reconstruction ran on CPU.
These are diagnostic timings, not application latency or an algorithm speedup.
Six focused tests passed, including missing returns, known noise units, local
signal dilution, ties and arrival exclusion. Public matrices were sealed before
the evaluator label join; hashes, source snapshot and draws are preserved.

Independent audit passed8,519assertions. It replays all32RGB matrices in CPU
float64 and all ToF matrices from saved public draws, independently recounts
metrics/decisions, and checks source geometry, opposite labels, UNKNOWN and
seals. Maximum matrix discrepancies are6.34e-8for RGB and3.56e-15for ToF.
It does not independently regenerate sensor noise or visible-valid labels;
original240-frame ToF replay belongs to the primary run. Audit receipt:
`artifacts.local/evidence/ba-observation-separation-audit-20260923-run/result.json`.
The audit execution succeeded on its first attempt. A duplicate preparation
request stopped at exclusive-create because the prepared plan already existed;
the existing sealed plan was used unchanged, with no scientific rerun.

Durable evidence: `artifacts.local/evidence/ba-observation-separation-20260923/`
and sibling `-run/`, physically under the canonical F:-backed artifact tree.
No capture process, paid allocation or background worker was started. The
governed execution process exited successfully. No observation/predictor
cutoff, model checkpoint or failed historical receipt was changed.
