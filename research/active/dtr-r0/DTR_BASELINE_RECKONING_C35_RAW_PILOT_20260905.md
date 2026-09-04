# DTR Baseline Reckoning: C35 Raw-Input Pilot

Status: `PRELIMINARY_CONSUMED_RAW_INPUT_RECKONING / FULL_11_BLOCKED_BY_REMOVED_PAYLOADS`

Date: 2026-09-05

## Outcome

The strongest simple raw-input baseline is already a serious challenger on the
only retained cohort that can still be replayed fairly.  On C35, raw detector
and depth measurements followed by a class-aware constant-velocity Kalman
tracker, the shared route tube, and a bounded 0.60 s event hold tied X94 on
Event F1 and false alert segments while fragmenting substantially less.

This is a consumed single-cohort Development pilot.  It is not a full
eleven-cohort reckoning, statistical evidence, fresh confirmation, or a paper
identity decision.

| Arm | Event P | Event R | Event F1 | False segments | Median lead | p10 lead | Fragment false runs | Median CLEAR | Frame F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Kalman radial TTC | 50.00% | 100.00% | 66.67% | 4 | 2.60 s | 2.60 s | 8 | 0.00 s | 77.48% |
| Raw Kalman CV + route tube | 50.00% | 100.00% | 66.67% | 4 | 2.65 s | 2.60 s | 8 | 0.00 s | 78.29% |
| **Raw Kalman CV + route tube + 0.60 s hold** | **50.00%** | **100.00%** | **66.67%** | **4** | 2.65 s | 2.60 s | **2** | 0.20 s | 81.52% |
| X24 retained core | 44.44% | 100.00% | 61.54% | 5 | 2.60 s | 2.50 s | 10 | 0.00 s | 60.63% |
| X94 complete mechanism | 50.00% | 100.00% | 66.67% | 4 | 2.80 s | 2.66 s | 10 | 0.00 s | **84.62%** |

Counts: 4 contact events.  The strongest simple arm and X94 each produced 4
matched detections and 4 false alert segments.

## Interpretation

The pilot does not justify saying that X94 loses.  X94 retains a 3.09 pp Frame
F1 advantage, 0.15 s more median lead, 0.06 s more p10 lead, and immediate
CLEAR.  The simple arm instead removes 8 fragmentation runs while preserving
the same event hit/false-segment outcome.  That is exactly the tradeoff the
full reckoning must resolve: framewise collision qualification and event
emission are not the same objective.

The 0.60 s hold improved the no-hysteresis raw route arm's Frame F1 from 78.29%
to 81.52% and reduced fragment false runs from 8 to 2, without changing Event
F1 or false segments.  This supports treating event emission as a distinct
layer, but does not promote X95 or select a final architecture.

## Fair-input boundary

The new baseline starts from the frozen YOLO candidate masks and dense depth
payload.  It does not consume X24, X73, or X94 tracks.  It shares X24's issued
plan admission/fallback and route-tube geometry so that the comparison does not
confound tracking with a different route contract.

CTRV is `NOT_EVALUABLE` because the shared raw measurement contract has no
frozen causal target yaw-rate.  A tiny learned predictor remains pending a
training-group freeze.  X73 remains pending a raw reproduction or a verified
sealed export.

## Why the eleven-cohort run cannot currently execute

C35 is the only one of the eleven frozen cohort roots whose dense model and
evaluator payloads remain present.  For C26, C27, C28, C32, C34, C36, C37,
C39, C40, and C41, the 2026-09-03 closed-payload cleanup records both `model`
and `evaluator` as `DELETED`.  Frozen manifests, result summaries, candidate
masks, and prior prediction evidence remain, but they are insufficient to
reconstruct raw metric measurements or score a new arm under the same pixels.

Using X24/X94 derived tracks would violate the reckoning input contract.
Recapturing the CARLA scenarios would create new pixels rather than restore the
consumed eleven-cohort panel.  Therefore the honest state is
`FULL_11_RAW_INPUT_NOT_REPLAYABLE_FROM_RETAINED_PAYLOADS`, not a baseline loss
or win.

## Evidence

- Result: `artifacts.local/evidence/dtr-baseline-reckoning/raw-kalman-c35-pilot-20260905-01/summary.json`
- Result SHA-256: `875B05F9DD2059B03B4E2AA9543B2B9CEA275F3096F831584BB56C1AAC361BA4`
- Prediction manifest SHA-256: `F6F0BDF940923025208B4EB4B6553601EEAAD819B1510008B572B2EEFF9EB6CA`
- New predictions were sealed before the runner opened C35 evaluator rows.
- The current metric implementation is compatible with the existing consumed
  CARLA scorer.  Final paper claims still require the preregistered shared
  maximum one-to-one event matcher and paired cluster uncertainty.

## Next decision

Do not start X97.  The next admissible full experiment is a newly retained raw
CARLA roster on which every classical arm, X24, X73, X94, and the separately
frozen X95 challenger can be sealed before truth opens.  The C35 pilot is strong
enough to justify that run; it is not strong enough to choose the paper identity.
