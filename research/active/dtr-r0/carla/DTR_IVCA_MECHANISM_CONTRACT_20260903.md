# DTR-IVCA mechanism contract

Date: 2026-09-03

Status: `MECHANISM_KERNEL_READY_SOURCE_NOT_OPENED`

## Question

Can interval-born collision authority reduce false ONSET births without losing
critical events, lead, or X94's one-observation full-dropout rescue?

## Mechanism

IVCA means **Interval-Valued Collision Authority**, implemented as
**Interval-Born, Transport-Sustained Collision Authority**.

For every current authorized carrier, compute all continuous components of

`wearer issued-route tube x transported target occupancy`

over the unchanged X24 horizon and tube radius. Each component records entry,
exit, overlap duration, minimum clearance, and the time of minimum clearance.
Multiple separated components are retained; the earliest currently authorized
component owns ONSET.

Only a current measured carrier with current authority, no explicit
contradiction, and the current plan receipt may sign an ONSET receipt. Missing
surface, rigid, or metric representations are not contradictions. X94 may renew
one such receipt only across its unchanged one-observation full dropout, with
the same current held parent and plan receipt. A transport renewal cannot renew
itself.

## Frozen implementation

- Interval authority SHA-256:
  `45E836511375EA31FF7267795BBB28FE68C3F8CA197E50691F9DD09C94F8F6C3`
- Evaluator SHA-256:
  `C4B5DFD459860FED0D78657D596E020A1943BB72BC14EE626046D31375D567CD`
- Focused test SHA-256:
  `386825CBD0AA52F51644888D6396827F608621E17996D77BA29370666E102D40`

The kernel uses bounded minimization and bracketed roots on each
constant-velocity X24 route segment. This recovers a continuous contact
interval even when it is narrower than X25's former 50 ms sample step.

Evaluator truth is constructed only from source-native evaluator rows. Entry
and exit remain interval-censored between adjacent negative and positive
samples; prediction output never defines or interpolates evaluator truth.

## Frozen comparison arms

1. X73: latest complete source-disjoint confirmed baseline.
2. X93: precision-oriented Development arm.
3. X94: X93 plus one-observation full-dropout persistence.
4. IVCA: X94 plus interval-valued ONSET authority.

X95 is an already closed learned state-model challenger and X96 is an already
closed bounded-survival challenger. IVCA receives neither identifier and is not
a longer dropout bridge.

## Required source roles

- sustained true conflict;
- future-only true conflict;
- near miss without contact;
- crossing target that exits before wearer arrival;
- lateral-only non-conflict;
- receding non-conflict;
- current overlap with closing;
- one complete detector-plus-metric dropout;
- dropout while conflict remains and dropout after conflict clears.

Roles and trajectories must be frozen before pixels, predictions, and
evaluator outcomes. A role not realized by source-native truth is
`NOT_EVALUABLE`; it cannot be moved or repaired after truth opens.

## Primary adjudication

IVCA advances over X94 only if all of the following hold:

- false event births decrease;
- no critical event is lost;
- lead has no material regression under a preregistered margin;
- X94 continuity rescue is retained;
- event fragmentation does not regress;
- the gain is attributable to current interval birth, not changed HOLD or
  extended persistence.

Report event precision/recall/F1, false ONSET births, event fragmentation,
first-alert lead, midpoint interval IoU with temporal resolution, censored
entry/exit error, minimum-clearance error, X94 continuity rescue TP, and false
persistence duration. Frame F1 remains diagnostic. A frame-only gain is not an
IVCA success.

## Claim boundary

The current result proves only the deterministic geometry, truth-separation,
and authority-reducer contracts on focused synthetic tests. No new source has
been opened, no X73/X93/X94/IVCA comparative result exists, and no
generalization, real-world, deployment, reliability, user-benefit, or safety
claim follows.
