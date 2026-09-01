# DTR CARLA C41 X82 fresh confirmation

Date: 2026-09-01

Decision: `DTR_CARLA_C41_X82_MECHANISM_NOT_EXERCISED`

## Protocol and source

C41 froze unchanged X82 before capture under protocol SHA-256
`67B806C47B9AA3B038C9CFD84E3BFF89C30D5944BBE84005CE269D2040BA08BE`.
It used seed `411082` and new layout/weather assignments `WetSunset`,
`ClearNoon`, `DustStorm`, and `HardRainNight`, while retaining the map, route,
camera, geometry, and trajectories.

The first `witness` server launch exited with an empty shard: zero files, zero
PNG frames, and no result. Its four attempt logs were preserved. The frozen
protocol permits one retry only for a failed server shard with zero durable
frames, so the same run ID resumed once, hash-verified and skipped the three
completed shards, and captured `witness`. The final source passed all checks
with four sensors, 728 frames per sensor, eight episodes, four layouts, and 73
unique blueprints. No CARLA process, port, or storage lease remained afterward.

## Result

All X25, X81, and X82 predictions were sealed before evaluator truth opened.

| Arm | TP / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| X24 | 87 / 33 / 85 | 72.50% | 50.58% | 59.59% |
| X81 | 135 / 15 / 37 | 90.00% | 78.49% | 83.85% |
| X82 | 135 / 15 / 37 | 90.00% | 78.49% | 83.85% |

X82 exercised zero held-proxy consensus releases, so its delta versus X81 was
`0 TP / 0 FP / 0.00 pp F1`. Full-arm precision, recall, F1, all four contact
recalls, and all safe-segment limits passed. The mechanism-exercise and
incremental-FP requirements did not pass. The inherited prediction also had
one confirmed non-rigid reference and one confirmed parent-identity mismatch,
both in `ep_07`, so the required authority-invariant gate did not pass.

## Evidence

- Source result SHA-256:
  `342F1DAB5D6E6D03B81E638A271B5907CD2985BD8F84D6B9DC277888BB9B4311`
- Model manifest SHA-256:
  `2159CADA93444805BBB18D2DB2EAC7A2F482E629E58E28E8FBCE43B05932C11E`
- Candidate manifest SHA-256:
  `E84203A1A5D55734BAAEC696F2378866AF7F49217D4B2F25706F7C06258E63F4`
- X24 freeze SHA-256:
  `C43D2CE687B2B050F5D555766C0CD6308195ED3780907B8CFF04EC7147FE6D34`
- X24 prediction SHA-256:
  `B7E144D2450CA9507F77202A637DC29EE805C88D9394AE13846E939D8C10537D`
- X81 prediction SHA-256:
  `4DF0D1760A97E0F2075E369BCB8811CFC8F73A55509CB372665F1243088A0F55`
- X82 prediction SHA-256:
  `B416DF49F3E8EBEB32C579C9F9BA43AD4651445565D0A846EBDD3FB458A13419`
- Summary SHA-256:
  `FC793AAD25E36B426D8E725C00BA24182FC51523CB8CC7C342DAC745E0535D08`
- Confirmation runner SHA-256:
  `81AFB93D69CB0B6D96D90E943ADF168FD7B4006EE7634C8863F29B1B0B8FEFF7`

## Claim boundary

C41 provides a strong full-arm synthetic result for the inherited X81/X82
pipeline, but no incremental positive or negative confirmation of X82 because
the frozen mechanism did not exercise. C41 is consumed and cannot be rerun as
confirmation. Its invariant failures may be used only for successor design.
X73 retains the latest complete source-disjoint confirmation authority. This is
not unseen-map, open-world traffic, real-sensor, deployment, reliability,
user-benefit, or safety evidence.
