# Motion consistency did not establish public return attribution

2026-09-23 EXPLORE. Decision: `MOTION_RETURN_ATTRIBUTION_NOT_ESTABLISHED`.
The frozen two-frame motion/range residual fails the public attribution gate
on both consumed controlled cohorts. Retain this exact recipe as a
NEGATIVE_CONTROL. No ownership cutoff, student, alert filter or App change
follows; A, LOCAL and UNKNOWN keep their existing roles.

## Effect and coverage

The test uses 1,152 saved RGB/ToF frames, 96 clips and 16 geometry groups.
All 1,152 saved ToF observations reproduce under the original proxy simulator
identities. Each frame keeps a fixed 3,072-point grid, with 638 inside the ToF
footprint. Public scores are sealed before evaluator attribution is opened.

Primary AUC is the macro mean over frame/zone subsets containing both actual
winning-bin contributors and noncontributors. All three scores use exactly the
same available points in each zone. AUC .5 means random ordering, not 50%
classification accuracy. Nonmixed/missing zones are not counted as successful.

| Quantity | Stability | Rescue |
| --- | ---: | ---: |
| Motion/range residual AUC | 0.60311 | 0.59615 |
| Static-coordinate control AUC | 0.63651 | 0.63334 |
| Wrong previous-zone control AUC | 0.58061 | 0.56756 |
| Evaluable mixed zones / all frame-zones | 7,588 / 36,864 | 6,810 / 36,864 |
| Common points within evaluable zones | 63,842 | 55,221 |
| Matched contributor points / all current contributors | 143,498 / 215,193 | 133,366 / 205,778 |
| Matched contributor coverage | 66.68% | 64.81% |
| Common-control contributor points | 132,974 | 123,408 |
| All matched points / all FoV points | 204,363 / 367,488 | 186,290 / 367,488 |

Both cohorts cover eight evaluable geometry groups and exceed the 50% matched
contributor coverage requirement, but neither reaches AUC .70 or a .05 gain
over both controls. Motion scores are 0.03340/0.03720 below the static control
and only 0.02250/0.02859 above wrong-zone scores. The static control retains the
same motion-selected previous zone and replaces its pixel coordinate by the
current coordinate: this isolates the displacement term on the common subset,
not an independently deployed no-RGB algorithm.

First frames have no causal history: 48 per cohort, 30,624 FoV points each are
explicitly missing and included in coverage. Additional first-failure counts
are tracking 59,498/69,896, current return 35,933/44,118, previous footprint
9,463/9,284 and previous return 27,607/27,276. These mutually exclusive reasons
sum with available and first-frame rows to the complete FoV denominator;
independent flags preserve overlapping causes. Missing never means clear.

## What this says about target ownership

Observed-return contributor membership includes background and is weaker than
target ownership. Evaluator-only target-in-corridor contributor AUC is
0.60429/0.51207, based on only 187 mixed zones per cohort. The complete fixed
grid contains only 426/479 relevant current contributor points, of which
330/324 have matched scores. This sparse auxiliary population cannot establish
general corridor-target localization or override the failed primary gate.

Projection consistency assumes a current/prior range pair refers to a
trackable surface during pure camera translation. Dominant-bin switching,
low texture, aperture effects, noise and jointly moving background can violate
that assumption. The experiment does not independently prove which of those
causes dominates. It does show that this fixed public two-frame residual does
not provide the required incremental attribution evidence. It does not rule
out learned temporal association, other observations or RGB information.

The full source `phase=dwell` label includes the arrival frame, so its phase
statistics are not an exact same-pose control. Source phases and all group,
shape and relation results are retained; exact identical-pose accounting is
reported separately by the independent audit.

The exact same-pose subset contains 96 frames per cohort, checked by identical
camera and object configurations. Motion/static/wrong-zone contributor AUC is
0.60826/0.60389/0.61144 on stability (2,158 mixed zones), and
0.58266/0.59485/0.59177 on rescue (2,224). Above-chance scores can therefore
also occur without actual pose change; they are not unique motion evidence.

## Validation and evidence

Seven focused tests pass: projection ratio/center, missing-return semantics,
stationary texture, exact native-to-public sampling/zone membership, tied AUC,
and empty-population handling. Main governed extraction/evaluation completes
once in 60.870 s. OpenCV CPU LK is used; the backend receipt's tiny scalar
bookkeeping probe is not an optical-flow benchmark or a CPU/GPU speed claim.
No new capture, fitted parameter, selected ownership cutoff or noise resampling
occurs. Controlled reused sequences are not fresh or hardware evidence.

Independent audit passes 756,511 counted assertions across all 73,728 zone
rows and 1,152 frames. It recomputes AUC by pairwise positive/negative comparison,
all residuals from saved coordinates/ranges, coverage, strata, missingness,
first-frame handling, seals and both failed gates. It does not independently
regenerate optical flow or native contributor lineage.

Audit v1 was prepared but not executed. v2 was stopped after a confirmed
implementation performance defect: repeated NPZ access decompressed entire
arrays inside the frame loop. Its owned process was terminated and failed
receipt retained. v3 loads arrays once and completes the same checks under a
new sealed identity and output path in about 9.5 s; no scientific score or
criterion changes. Main execution was not repeated. All owned processes end.

Protocol: `MOTION_RETURN_PROTOCOL_20260923.md`; public API:
`motion_return_public.py`; runner: `motion_return_diagnostic.py`.
Run through `tools/ba.ps1 run research-ue -RunSpec <saved-spec>`; a replay must
use a new identity/output and preserve the existing terminal receipts.

Durable roots under `artifacts.local/evidence/` are
`ba-motion-return-20260923` (plan/seal, RunSpec, console), sibling `-run`
(public arrays/seals, evaluator labels, complete zone rows/strata, result,
backend and output seal), and `ba-motion-return-audit-20260923` with sibling
`-run-v3` (completed independent audit; v2 receipt retained). All payloads use
the canonical F:-backed junction.
