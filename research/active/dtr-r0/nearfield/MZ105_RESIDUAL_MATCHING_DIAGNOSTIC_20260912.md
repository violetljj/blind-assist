# MZ105: residual SGBM matching separability

2026-09-12. EXPLORE, consumed rendered Development; diagnostic only.

Question: which of the original SGBM+ToF 62 raw false query-frames are far
surfaces estimated near, despite existing uniqueness, left/right consistency
and component checks? Can observable local matching distinguish residual false
support while retaining thin and small true support?

Keep SGBM+ToF and the original alert state unchanged. MZ104 is a system replacement
comparison including unequal external reliability validation, not a pure model
accuracy comparison. A deletion-only gate cannot recover its missing small_head
raw TP: 13/15 cannot reach 95%. Per-panel FP bounds require removing at least
90+123=213 false query-frames; MZ101 can lose at most two TP, MZ102 none. The
original adapted model's processing cost is also unchanged by a downstream gate.

One fixed diagnostic, no training, model replay, capture or method/window sweep:
all 576 cached MZ101/MZ102 frames. Compute grayscale 5x5 zero-mean normalized
cross-correlation (ZNCC) at integer disparities 0..95, with complete in-image
patches only. Round cached SGBM disparity to the nearest integer. Features are
assigned ZNCC, assigned minus best far ZNCC (integer disparity corresponding to
axial depth >4m, including zero), and assigned minus best competing ZNCC outside
the assigned +/-1 pixel basin. Zero-variance/invalid patches have unavailable
scores, not evidence of clear space. No patch-size or score combination search.

Produce all features from RGB, cached SGBM depth, observable pose/calibration and
original ToF support before evaluator access. Verify source RGB capture hashes,
cached-depth identity and original support parity. Seal features before
reading native depth, task truth or family annotations. Native depth is only
used to partition false-support origins: far, eligible near outside query,
eligible near inside query, below range, or unavailable. Report stereo-only,
ToF-only and shared false support; ToF-supported errors cannot be removed by
stereo verification alone. No subsampling of candidate query-support pixels.

For each feature, query score is the maximum over its stereo support pixels,
matching the existing existential union readout. Independent ToF support is
always retained. Report score overlap and the attainable deletion envelope on
consumed data: strict preservation of every original raw TP on both panels,
plus a separately labeled relaxed diagnostic allowing 95% overall and 95% in
each critical family on each panel. The latter is not an accepted improvement.
Thresholds in the envelope are evaluator-derived descriptions, not selected
production parameters. Include per-panel raw and final effects, exact TP/FP
transitions, critical-family counts, detected-event losses, timing changes and
UNKNOWN counts. Integer/patch matching is a limited signal, not calibrated
confidence or a claim about every reliability estimator.

If strict retention leaves no FP reduction, stop this signal without tightening
away true small objects. If separable, retain diagnostic evidence only; fixing a
method and testing complete new scenes is a separate successor decision. No
automatic integration or fresh collection. Budget: one feature pass, one
evaluation; mechanical corrections preserve logs and sealed inputs. Use measured
backend selection for the fixed tensor workload. Preserve artifacts under
artifacts.local/work/mz105-residual-matching-20260912 and release task processes.

Context: [confidence survey](https://arxiv.org/abs/2101.00431), inspected abstract:
evaluates confidence across stereo algorithms and datasets, including strengths
and limits; does not establish effectiveness on this panel or novelty here.

Engineering correction before any features or outcomes: float32 local variance
cancellation failed CPU/GPU equivalence on the unlabeled first pair (maximum
ZNCC difference 0.665). Preserve failed source/log/backend. Compute the same
formula using centered raw grayscale and float64; require CPU/GPU agreement
within 1e-7. This changes numerical implementation, not the matching method.
Original summary receipts omit cached-depth payload hashes. Verify their identity
by exact array replay of the unchanged OpenCV SGBM, including validity masks,
on all frames. This CPU replay is FROZEN_PROTOCOL_CPU_ONLY; GPU ZNCC selection is
measured separately. Preserve the failed pre-payload assertion log.
