# Motion-conditioned occupancy A0.1 fresh result

Date: 2026-08-03

Terminal: `MOTION_CONDITIONED_OCCUPANCY_A0_1_FRESH_SUPPORTED_DEVELOPMENT_ONLY`

The frozen 18-feature, positive-weight-1.25 Logistic model passed all six
predeclared gates on the unopened TUM Freiburg 3 `walking_halfsphere` source.
No coefficient, feature, normalization, threshold, window, reference value, or
gate changed after the archive was opened.

## Source identity and execution

- Official archive bytes: `639559931`
- Official archive SHA-256:
  `C4984F4894724398590DCF18C3D9D54DBB76CCE925D51695F44AD8E697FC8458`
- Extracted source: 1,067 RGB frames and 1,029 depth frames
- Frozen manifest: seven 3-second windows at 10 FPS, 210 rows
- Manifest SHA-256:
  `409BD9FD838C2EC6222EE629ACF67A3DEC3B49067EB4D2F834111564A4BFAFEB`
- Known fresh band x horizon opportunities: 1,716
- Paired-valid frames: 198/210 (94.29%)

## Frozen-gate result

| Measure | Fresh result | Gate | Pass |
|---|---:|---:|:---:|
| Brier reduction vs deterministic | 40.45% | >=15% | yes |
| Log-loss reduction | 71.97% | >=20% | yes |
| Expected calibration error | 0.02909 | <=0.10 | yes |
| High-confidence-clear false-clear | 4.30% | <=5% | yes |
| High-confidence-clear coverage | 14.92% | >=10% | yes |
| Occupied recall at P>=0.50 | 88.43% | >=85% | yes |

The deterministic UniDepth clearance field still failed on the same frames:
clearance MAE 0.2586 m, collision agreement 84.97%, false-clear 7.63%, and
temporal clearance-delta MAE 0.2207 m. The positive result therefore supports
the motion-conditioned probabilistic formulation, not deterministic metric
depth accuracy by itself.

## Evidence boundary

This is fresh Development evidence from one additional sequence in the same
TUM sensor and annotation family. It authorizes an A1 static collision-risk
comparison and visualization. It does not establish external-camera transfer,
wearable operation, alert utility, Android latency, safety effectiveness, or
mainline promotion. The 4.30% false-clear result is also close enough to the 5%
gate that replication on an independent camera remains necessary.

Ignored machine reports:

- UniDepth field report SHA-256:
  `51018CA9576E4728EE76716C716F229DE231C27A9C32705BD9E12E98D953E3B2`
- Frozen probability report SHA-256:
  `F3D16ED71DAD3972F524234CCC3C4E41DF49474AB64D6C160328BC7FBA515336`
