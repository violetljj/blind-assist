# DTR Baseline Reckoning

Status: `PREREGISTERED_TASK_CONTRACT / EXECUTION_NOT_COMPLETE`

Date: 2026-09-05

## Decision

Stop mechanism successors after X96. Do not start X97. The next DTR question is
whether the accumulated mechanism is materially better than strong classical
baselines under one event contract.

The paper task is now **event-level route-risk detection**. Primary outcomes
are Event Precision, Event Recall, Event F1, false alert segments, first-alert
lead time, fragmentation, and CLEAR delay. Frame Precision/Recall/F1 are
secondary diagnostics and must still be reported.

This declaration does not retroactively promote X95. Existing X95 results are
consumed post-hoc Development diagnostics. X95 can regain promotion eligibility
only on a newly selected, preregistered, held-out source whose truth is unopened
until every arm is sealed.

## Two panels, one metric contract

The final result has two panels. Their scores must not be pooled because their
observation contracts differ.

### CARLA detector-to-risk panel

Use the eleven already consumed cohorts only for retrospective mechanism
reckoning: C26, C27, C28, C32, C34, C35, C36, C37, C39, C40, and C41. They may
determine framing and eliminate successors, but cannot confirm or promote an
arm.

A later confirmation panel must use a new frozen CARLA source. Replaying,
reseeding, resampling, or relabeling the eleven consumed cohorts cannot make
them fresh.

### JRDB public-real panel

Use the six frozen X21 sequences for detector-derived comparisons when all arms
consume the same causal observation ledger. Keep the 27-sequence native-track
ceiling as a separate privileged-input diagnostic. A native-label trajectory
arm and a detector-derived X21 arm are not head-to-head results even if their
metric columns have the same names.

## Arms

Every comparable arm within a panel must receive identical current-and-past
observations, route representation, horizon, event roster, and evaluator.

1. Distance / radial TTC.
2. Causal CV extrapolation plus route tube, without event persistence.
3. Kalman/SORT-style CV tracker plus route tube plus a 0.60 s bounded evidence
   horizon.
4. CTRV / constant-turn route with the same target observation contract. If
   causal yaw-rate is unavailable, report `NOT_EVALUABLE`; do not substitute a
   privileged pose source.
5. One tiny learned predictor trained only on training groups, with fixed
   features and leave-one-group-out or source-held-out inference.
6. X24 retained core.
7. X73 geometric/parent representative.
8. X94 complete mechanism.
9. X95 event-oriented challenger.

X24 already includes a robust-CV metric tracker, route tube, and a 0.60 s track
hold. It must not be relabeled as an independent Kalman/SORT baseline. The
classical tracker arm starts from the shared raw measurement stream rather than
from X73/X94 derived tracks.

## Shared evaluator

- `UNKNOWN` and `NOT_EVALUABLE` frames are excluded from both positive and
  negative denominators. They are never CLEAR.
- Truth events are maximal contiguous future-route-contact intervals under the
  frozen 3.0 s horizon.
- Predicted alert segments are maximal contiguous ACTIVE intervals.
- Event matching is maximum one-to-one matching between alert segments and
  truth events. A match must begin no later than contact and overlap the event's
  eligible warning window.
- Event Precision = matched alert segments / evaluable alert segments.
- Event Recall = matched truth events / evaluable truth events.
- False segments are evaluable unmatched alert segments.
- First-alert lead is contact time minus the first matched alert onset. Report
  median and p10 over matched events.
- Fragmentation is the number of extra matched-window alert onsets beyond one
  per recalled event; also report fragmented-event rate.
- CLEAR delay is the time from the first known-negative frame after an event to
  the first CLEAR output. Report the median and right-censored count.
- Frame Precision/Recall/F1 use only evaluator-known frames and are secondary.
- Counts and exposure-normalized false burden must accompany every ratio.

Statistical uncertainty is clustered by episode/sequence, never by frame. A
claim that X94 "significantly wins" requires the 95% paired cluster-bootstrap
interval for Event F1 delta versus the strongest simple baseline to remain
above zero, plus no material regression in false-segment burden or p10 lead.
Raw per-cluster outcomes remain authoritative if the interval is unstable.

## Four mechanism principles

Historical X numbers are implementation history, not the paper abstraction.
The ablation is rebuilt as these four cumulative principles:

1. **Collision authority separation** — existence does not confer collision
   authority.
2. **Bounded evidence transport** — historical evidence can persist only
   within an explicit authority horizon.
3. **Relational consistency** — derived proxies cannot count as independent
   corroboration.
4. **Shape-qualified occupancy** — motion evidence must remain compatible with
   object spatial support.

Report each cumulative row with all primary event metrics and secondary frame
metrics. Do not choose the order after observing the ablation.

## X95 fresh-admission rule

Before any new X95 prediction:

1. select and hash the held-out source roster without evaluator truth access;
2. freeze the event contract, every arm, tiny-model training groups, and the
   one-to-one matcher;
3. seal predictions for all arms;
4. open truth once and score every arm together;
5. report Event F1 gain, false-event burden, lead, fragmentation, CLEAR, and
   frame F1 even when they disagree.

An event gain with modest frame-F1 loss is a valid positive finding only if it
survives this fresh procedure. Reinterpreting the consumed X95 result is not.

## Paper identity decision

- X94 wins the strongest simple baseline on event quality and the paired
  uncertainty interval excludes zero: DTR is an algorithm method.
- X94/X95 win primarily on dropout, fragmentation, false segments, or CLEAR:
  DTR is evidence-qualified collision reasoning under partial observability.
- Kalman/SORT plus bounded hysteresis is practically tied with the complete
  mechanism: DTR is an event contract and failure-aware framework with a strong
  simple-baseline analysis, not a collection of rules presented as the main
  algorithm.

No successor work starts until this reckoning is complete or a named source
limitation makes a required arm `NOT_EVALUABLE`.
