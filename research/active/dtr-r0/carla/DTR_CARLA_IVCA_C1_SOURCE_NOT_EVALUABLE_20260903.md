# DTR CARLA IVCA-C1 source terminal

Date: 2026-09-03

Decision: `DTR_CARLA_IVCA_C1_SOURCE_NOT_EVALUABLE`

## Frozen question

Could one new trajectory-disjoint, role-balanced CARLA panel jointly evaluate
byte-frozen X73, X93, X94, and IVCA while exercising positive and negative
one-frame full-dropout roles?

## Source execution

The protocol was frozen and committed before capture. The first 8 GiB default
lease was refused before a run directory existed because the experiment unique
byte cap would be exceeded. A read-only audit showed that the same-specification
C41 run used `3,270,755,950` unique bytes and that a 4 GiB reservation remained
inside the unchanged global cap. The generic C2 runner was therefore given an
explicit reservation parameter whose default remains 8 GiB. A 4 GiB guard
passed at projected unique bytes `118,468,778,912 < 120,259,084,288`.

The sole IVCA-C1 invocation captured the complete 1280x720 instance shard:

- 8 episodes, 91 frames each, 728 payloads;
- 73 unique actual blueprints;
- zero blueprint fallbacks;
- all scripted poses within `1.573e-5 m` of authority;
- all expected episodes, frames, payloads, calibration, and alignment present.

No wearable, depth, witness, model, X73, X93, X94, or IVCA output was produced.

## Terminal source failure

Seven of eight episode outcomes and responsible-asset sets matched exactly:

| Episode | Frozen role | Expected | Observed | First contact |
|---|---|---:|---:|---:|
| ep_01 | sustained conflict + positive dropout | CONTACT | CONTACT | 3.9 s |
| ep_02 | near miss | SAFE | SAFE | none |
| ep_03 | future-only lateral conflict | CONTACT | CONTACT | 3.3 s |
| ep_04 | crossing exits early + negative dropout | SAFE | SAFE | none |
| ep_05 | current overlap + closing | CONTACT | CONTACT | 0.0 s |
| ep_06 | receding | SAFE | SAFE | none |
| ep_07 | lateral-only non-conflict | SAFE | **CONTACT** | **6.3 s** |
| ep_08 | stationary future conflict + positive dropout | CONTACT | CONTACT | 2.6 s |

`ep_07` contacted `c8_l04_target`; its minimum center-to-polygon distance was
`0.276922 m`, inside the frozen `0.45 m` wearer radius. Therefore both
`all_expected_outcomes_match` and `all_expected_responsible_sets_match` failed.
The protocol preregistered source-native role mismatch as `NOT_EVALUABLE` with
no replacement, so this cannot be repaired by relabeling CONTACT, increasing
the lateral offset, changing duration, moving the score window, or retrying the
same cohort.

## Evidence

- Frozen protocol SHA-256:
  `548A58F8493FD0C0900D62C40F39CD59B8993C7BF996E6DFDBBD797CE33B9CF9`
- Instance result SHA-256:
  `5CDAFB735ACE03156D7E474F9E9DD10469A3B2FA8F31831A69C52F6C7C6C2E18`
- Immutable local source root:
  `E:/linnan/CARLA/experiments/dtr-carla-c2-rich-scene/evidence/ivca-c1-20260903-01`
- Files retained: `777`
- CARLA/Python task processes after termination: `0`
- RPC listeners on 2000/2001/2002 after termination: `0`
- Storage lease after termination: released

## Decision and claim boundary

IVCA-C1 is terminal source-not-evaluable before prediction. It supplies source
design information only. It is not a negative result for X73, X93, X94, IVCA,
interval birth, transport persistence, event precision, lead, or dropout
recovery. No four-arm comparison, generalization, real-world, deployment,
reliability, user-benefit, or safety claim follows.
