# Metric3D clearance field A0 Development result

Date: 2026-08-03

Terminal: `METRIC3D_CLEARANCE_FIELD_A0_DEVELOPMENT_PASS`

## Result

The class-free clearance construction was evaluated on 120 unique RGB frames
from already consumed TUM Freiburg 3 `walking_static` and `walking_xyz`
windows. Metric3D received RGB and published intrinsics only. Registered sensor
depth generated the comparator field.

| Measure | Result | Gate |
|---|---:|---:|
| Paired valid fields | 119/120 (99.17%) | >=90% |
| Left/centre/right clearance MAE | 0.11579 m | <=0.25 m |
| Collision-envelope agreement | 93.35% | >=90% |
| False-clear rate | 4.03% | <=5% |
| Temporal clearance-delta MAE | 0.09031 m | <=0.15 m |
| Recovered camera-height MAE | 0.14255 m | diagnostic |
| CUDA mean latency | 165.96 ms/frame | diagnostic |

All five predeclared Development continuation gates passed. There were 356
known clearance comparisons and 1,068 known band-by-horizon collision
comparisons. `UNKNOWN` remained unknown rather than being counted as clear.

| Band | Clearance MAE | Collision agreement | False-clear rate |
|---|---:|---:|---:|
| Left | 0.07856 m | 95.80% | 4.20% |
| Centre | 0.14155 m | 91.53% | 6.78% |
| Right | 0.12748 m | 92.72% | 1.12% |

The centre band is the weakest Development region. Before downloading the
fresh archive, its dedicated false-clear cap was therefore fixed at 8%, while
the stricter pooled 5% cap remains unchanged.

## Interpretation

This is the first result in this branch where Metric3D is useful as a dense
collision-space representation rather than merely as a person-range source.
It supports opening one fresh sequence for A0. It does not yet establish
performance on a new sequence, final external lens, static-obstacle actionability,
alerts, Android, or safety.

The first runner invocation omitted the published TUM depth scale
(`5000 = 1 m`), which caused the sensor arm to be non-evaluable while the
Metric3D arm remained 120/120 valid. No collision outcome existed in that run.
The unit conversion was added with a regression test before the reported run.
