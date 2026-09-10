# MZ49: candidate-domain alignment recovers events with false-alert costs

2026-09-11 EXPLORE. [Frozen comparison](MZ49_BANK_DOMAIN_CALIBRATION_20260911.md).
Applying the same calibration rule to candidates that actually reach inference
recovers useful events, but fails the predeclared no-added-false-event condition.
Retain the calibrated outputs as CHALLENGER with explicit costs; keep the
original MZ37 and MZ47 cutoffs. No model training or new inference was performed.

Only REPLAY and ENRICH are recalibrated, each once on the original1000 DEV
frames under DROP_CLOSE. Their weights, scores, bank, selector and restoration
are unchanged. The only change to cutoff_zero_added is using restricted_raw
and restricted_support. The original MZ37 comparator is left untouched.

## Fixed non-fit rich cohort

Each cell is TP/FP/FN over32 positive and96 negative event bits in32 frames.

| Condition | REPLAY before -> after | ENRICH before -> after |
|---|---:|---:|
| IDEAL | 22/1/10 -> 29/1/3 | 18/1/14 -> 26/1/6 |
| Close echoes merged | 24/1/8 -> 31/1/1 | 19/1/13 -> 27/1/5 |
| Close echoes missing | 18/0/14 -> 24/0/8 | 14/0/18 -> 22/0/10 |

The12 fit frames remain4TP/0FP/8FN under DROP_CLOSE in both arms. Changing the
cutoff does not restore deleted local evidence. ENRICH still trails REPLAY
after the same domain correction, so this does not establish an enrichment
advantage or rescue the MZ47 data-contribution hypothesis.

## Regression cost remains material

Under DROP_CLOSE outside calibration and the12 rich fit frames, each arm gains
42 true events and loses none. REPLAY adds37 false events; ENRICH adds27.

| Cohort | REPLAY added TP / added FP | ENRICH added TP / added FP |
|---|---:|---:|
| Relation DEV,2000frames | 21 / 31 | 21 / 23 |
| Distance DEV,1000frames | 14 / 1 | 10 / 1 |
| Rich non-fit,32frames | 6 / 0 | 8 / 0 |
| MZ36,400attempts | 1 / 5 | 3 / 3 |

Old-DEV DROP_CLOSE has zero added false events by the fixed calibration rule,
with12/15 recovered true events. This does not generalize to the other cohorts
or to the IDEAL/MERGE profiles: full per-query and paired costs remain in the
result. A consistent candidate domain removes an unnecessary restriction, but
does not provide sufficient separation of true and false local evidence.
Future learning should target that separation with the richer source and
native spatial supervision; further cutoff search was not performed here.

## Verification and scope

Original composition reproduces318,528 saved scalar values exactly before
the new cutoffs are applied. All old positive decisions are retained under
every condition. Independent scalar scoring recounts212,352 known decisions.
MZ36 retains400 attempts and80 UNKNOWN bits; unknown frames are not credited
as correct negatives. There are exactly two new cutoff vectors and no GPU
job, model forward, fit or permanent feature cache. Saved-output processing
took0.34s on CPU, excluding interpreter/import startup.

The calibration input is consumed Development. This is a selective recall/
false-alert tradeoff, not a new default or physical-sensor claim. Evidence is
under `artifacts.local/work/mz49-bank-domain-calibration-20260911/run-v1/`:
`result.json`, `predictions.npz`, both cutoff vectors, `audit.json`, and the
SHA-bound execution receipt.

The original registration referenced MZ47's still-finalizing diagnostic
receipt; its later addition of explanatory report hashes changed registration
metadata. `registration-amendment.json` preserves the original record and
binds a fixed copy of the final diagnostic receipt. The actual prediction
arrays, checkpoints, protocol and calculation inputs pass unchanged-hash
verification; the calculation never consumed the diagnostic receipt itself.
No experimental data or result was replaced to make this metadata correction.
