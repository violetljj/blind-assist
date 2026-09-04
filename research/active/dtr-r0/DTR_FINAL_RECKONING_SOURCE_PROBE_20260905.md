# DTR Final Reckoning R1 source-probe adjudication

Status: `SOURCE_GATE_BLOCKED_NO_FINAL_CAPTURE_STARTED`

The Final Reckoning roster and its eleven arms remain frozen. No method was run,
fit, selected, or scored during this source-only work. The reserved `FIT_ONLY`,
`FINAL_A`, and `FINAL_B` seeds remain untouched, and no probe pixels may enter
those groups.

## What the probe established

- The ten-cell roster is physically materializable in Town10HD_Opt at 10 Hz and
  1280x720. Design 04 produced complete instance and independent witness shards:
  910 frames each, all expected outcomes and responsible sets matched, no
  blueprint fallback, and 73 actual blueprints.
- S08 now contains measured wearer/camera yaw rotation of 60 degrees while its
  target remains static and the source-native outcome remains SAFE.
- The source generator now supports optional, scenario-local wearer yaw and
  explicit per-asset trajectory overrides. Legacy protocols without those fields
  preserve their previous behavior.
- The remaining blocker is the occluder contract, not CARLA capacity or target
  dynamics. A collision-relevant pedestrian or vehicle can create the desired
  mask loss, but then correctly becomes an additional route-contact authority.
  It therefore cannot serve as the hidden foreground occluder for S09 or S10.

## Bounded design history

| Design | Capture outcome | Source-gate adjudication |
| --- | --- | --- |
| 04 | instance and witness complete | rejected: S08 yaw was 0 degrees; S09 contained six zero-pixel frames; S10 had only six post-reappearance frames |
| 05 | instance complete | rejected: yaw and reappearance repaired, but the template-owned shell trajectory ignored the scenario override |
| 06 | instance complete | rejected: override semantics worked, but S09 had no partial mask loss and S10 had only four contiguous zero-pixel frames |
| 07 | 910 instance frames written; capture returned not evaluable | rejected: S09 alias and S10 vehicle became additional responsible hazards |

Design 07 protocol SHA-256 is
`4B02AD5DBBC4DCA79A3DEF14D826A17D029A9C97D41A4B67AD1476820FC243C6`;
its instance result SHA-256 is
`F829FAF7282612154D268BBB2FD183051588A46C17EAD516C380431C9AA25B05`.
Design 04's complete instance and witness result SHA-256 values are
`B10685ECDD5DCCF4E20FE0F7B2E876EEA07563592F3D1C83E934BBC0247CA7B8`
and `7EE926D7F35F2CCE9E5381599C4F1C3ACCED41589475ED7FF7AE1BC9DC8554AF`.

## Decision

Do not start a design 08 by moving another collision-relevant actor across the
route. The next admissible source action is one bounded visual-only occluder
probe using a non-collision shell whose projected extent is tuned for:

- S09: target visibility ratio 0.05 through 0.45 for at least six frames, with
  no zero-pixel frame; and
- S10: at least six contiguous zero-pixel frames followed by at least eight
  visible frames, while the only responsible asset remains the target.

Until that single source gate passes and an independent witness agrees, final
raw materialization is unauthorized. This is a source-design block, not an
algorithm result and not evidence for or against X94, X95, or any baseline.
