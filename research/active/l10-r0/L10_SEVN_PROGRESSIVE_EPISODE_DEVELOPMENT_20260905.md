# L10 SEVN progressive episode Development result

Date: 2026-09-05

Decision: `L10_SEVN_TRIGGERED_PROGRESSIVE_EPISODE_DEVELOPMENT_GATE_NOT_MET`

## Question

Can a truth-free deficit-triggered observation policy retain the correct
bindings of a fixed sweep, add no wrong bindings, and use fewer observations on
the consumed 24-episode SEVN action panel?

## Result

| Arm | Correct | Wrong | UNKNOWN | Coverage | Binding precision | Extra observations | Correct / extra observation |
|---|---:|---:|---:|---:|---:|---:|---:|
| PASSIVE | 0 | 3 | 21 | 12.5% | 0.0% | 0 | n/a |
| FIXED SWEEP | 11 | 4 | 9 | 62.5% | 73.3% | 72 | 0.153 |
| TRIGGERED ACTIVE | 12 | 5 | 7 | 70.8% | 70.6% | 49 | 0.245 |

Triggered action distribution was `14` SWEEP, `5` APPROACH, `3` HOLD, and `2`
PAN_LEFT. Compared with FIXED SWEEP it added one correct binding and reduced
extra observations by `23/72` (`31.9%`), but it also added one wrong binding.
The frozen no-added-wrong-binding gate therefore failed.

## Failure localization

- All `12` triggered correct bindings came from the retained V2 global
  text-to-mask topology branch, not the portal-private witness branch.
- Four triggered wrong bindings also came from that V2 branch. Three were
  already present in the initial observation, so the frozen UNKNOWN-only
  trigger chose HOLD and could not repair them.
- The fifth wrong binding was SEVN020 after APPROACH. It came from a unique
  portal-private exact-token witness at confidence `0.14467365`, but its selected
  mask had zero intersection with the evaluator target door.
- SEVN023 was the symmetric positive: APPROACH changed FIXED SWEEP's UNKNOWN to
  a correct binding at confidence `0.89943356`.

The observed issue is not lack of active-perception headroom. The trigger
improves observation efficiency and exposes an additional correct case. The
blocking mechanism is commitment authority: current single-observation scores
cannot distinguish an informative recovery from an incorrect portal binding.

## Decision boundary

Freeze the runner, policy, 24 episodes, and result. Do not select a confidence
threshold, special-case APPROACH, or modify the UNKNOWN trigger on this consumed
panel. The next decision-changing experiment requires independent evidence for
physical-instance/facade continuity, such as a learned physical-instance
verifier, multiple references, or a new action-labelled source with exact
facade/entrance ownership truth.

This run is consumed same-source SEVN Development mechanism and demo evidence.
It is not fresh confirmation, calibration or conformal evidence, independent
facade/entrance ownership, arrival, handoff, product-benefit, reliability, or
safety evidence.

Authoritative machine-readable result:
`l10_sevn_progressive_episode_result_v1.json`.
