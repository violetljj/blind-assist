# DTR-CARLA N3/N4 multi-town native dynamics and one frozen replay

Date: 2026-08-31

## Outcome

The N3 source expansion completed on the exact requested three-scene roster:

- Town01 crowded pedestrians;
- Town04 bus stop;
- Town05 parking lot.

Every scene materialized 401 native-policy trace frames. Heavy vehicles and
two-wheelers both moved under CARLA-native behavior in every scene, and all four
authored long-tail effects were observed in every scene (`12/12`).

The sole frozen N4 four-modal replay invocation was then consumed. Town01
completed all 401 frames and all four event actors entered the wearable view.
Town04 stopped at its N2 preflight because free physical memory was `3.83 GiB`,
below the frozen `4.00 GiB` floor. This occurred before a Town04 child directory
or any Town04 pixels were created; Town05 never started. The invocation was not
retried and must not be reported as a three-scene replay success.

## Frozen N3 source

Source root:

`E:\linnan\CARLA\experiments\dtr-carla-n3-multitown-native-dynamics\evidence\n3-multitown-native-v3-20260831-0045`

Outer status: `DTR_CARLA_N3_MULTITOWN_NATIVE_DYNAMICS_MATERIALIZED`

| Scene | Walkers | Vehicles | Frozen long-tail actors | Result |
|---|---:|---:|---|---|
| Town01 crowded pedestrians | 24 | 5 | nearby pedestrian + Mercedes Sprinter | 401 frames; heavy/two-wheel motion; `4/4` effects |
| Town04 bus stop | 15 | 5 | nearby pedestrian + Mitsubishi Fusorosa | 401 frames; heavy/two-wheel motion; `4/4` effects |
| Town05 parking lot | 13 | 5 | nearby pedestrian + CarlaCola truck | 401 frames; heavy/two-wheel motion; `4/4` effects |

Totals: 1,203 trace frames, 67 dynamic actors, 12 observed long-tail effects.
The three vehicle events in each scene were frozen to one nearby native heavy
vehicle after the N3 v2 route-feasibility diagnostic. Two-wheelers remained
independent native Traffic Manager actors and were required to move. This is an
explicit same-source Development authoring choice, not a natural-distribution or
source-disjoint result.

## Frozen wearer routes

The route builder uses the frozen evaluator-side trace before replay, centers
the active event actor horizontally, and caps planar wearer motion at `1.8 m/s`.
The exact formal bundle audits were:

| Scene | Max route speed | Max event range | Route length | Protocol SHA-256 |
|---|---:|---:|---:|---|
| Town01 | 1.800019 m/s | 6.304 m | 19.200 m | `79AB0BEBC2D6D95A6BDCBC0DA6C6823F30A53B6D971251465362131A148668C6` |
| Town04 | 1.800020 m/s | 6.950 m | 22.428 m | `EECA799598F0793B820DA9FC5AF8D4B7F8CE84455966BD4AAC038BBF7FCCC41C` |
| Town05 | 1.800017 m/s | 7.573 m | 22.394 m | `C687770C99E4D04DC4AFFD7FEE92A32D02E2940CD414856BC3E19D2D700087BC` |

The small excess above `1.8` is six-decimal coordinate serialization error and
is inside the frozen `1e-4 m/s` audit tolerance. All route audits passed their
four checks before the replay-attempt receipt was written.

## Sole N4 replay attempt

Run root:

`E:\linnan\CARLA\experiments\dtr-carla-n4-multitown-frozen-replay\evidence\n4-multitown-frozen-v1-20260831-0055`

- Attempt authority: `SOLE_FROZEN_FOUR_MODAL_REPLAY_ATTEMPT_CONSUMED`
- Bundle SHA-256: `A17DE41545EA4200DCF12ACE5351857E880A1807E3A354DF4030ECF1101D5C78`
- Attempt receipt SHA-256: `F0352E88DC88A74C67E87CA6A2C3175FE41B5606DFC8DB71A511622B21A03166`
- Post-run failure receipt SHA-256: `D04DBBEB2EE2D0B3AB413CD068E7A04B78F0C76F1B3DEBE333A5F46A9EC7808F`
- Final status: `DTR_CARLA_N4_REPLAY_ATTEMPT_CONSUMED_INCOMPLETE`

Town01 completed 401 aligned frames and 1,604 payloads in the fixed order
`instance / wearable RGB / metric depth / witness RGB`. All 15 child checks
passed. Event visibility was:

| Event | Visible frames | Maximum instance pixels |
|---|---:|---:|
| occluded jaywalk | 33 | 13,705 |
| sudden brake | 27 | 451,690 |
| reverse pullout | 33 | 449,306 |
| door open | 21 | 328,209 |

The Town01 result SHA-256 is
`8FAC08C2D68862C03F1EE6988E83B84BC0D6E7ED6822EF647742D20C1DFF026F`.
Its contact sheet SHA-256 is
`1A98B9724CBE90C2C9CDAC4F993B606CE9A161EAAB0C2DF731002973DD3BAFF2`.

The failure boundary is exact: Town04 and Town05 have no replay evidence. No
route, source, threshold, memory floor, or sensor setting was changed after
pixels were observed, and no replay was restarted. CARLA processes and ports
`26300`--`26302` were absent after termination.

## Claim boundary and decision

This work establishes the requested multi-town native source mechanisms and a
Town01 four-modal Development visibility result. It does not establish a
complete three-town four-modal result, natural wearer motion, traffic
distribution validity, source-disjoint confirmation, real-world benefit,
product readiness, or safety.

N4 v1 is closed as a consumed incomplete attempt. Completing Town04/Town05 would
require explicit authority for a new versioned protocol and a new replay; it is
not part of this result.
