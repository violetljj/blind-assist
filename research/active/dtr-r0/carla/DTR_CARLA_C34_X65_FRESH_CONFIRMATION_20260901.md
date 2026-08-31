# DTR CARLA C34 X65 fresh confirmation

Date: 2026-09-01

Decision: `DTR_CARLA_C34_X65_MECHANISM_NOT_EXERCISED`

## Frozen identity

- Cohort: `DTR_CARLA_C34_X65_FRESH_SOURCE_CONFIRMATION_V1`
- Seed: `341066`
- Frozen protocol SHA-256:
  `EF8DED7AFFF1699721730A341568CAC1CE56E4DC6C15CD6713063A15B097DC41`
- Confirmation runner SHA-256:
  `66E2C7F4DDD0409442D0DFED817F1AD0CCE337F52B096EEC764B2DA9BEBAB6D9`
- Unchanged X65 predictor SHA-256:
  `B87E444384CF6BE4A2B69A4B8536F9EA4CD10FE8A46DD9B5D0499A60AB94E4F1`
- Source result SHA-256:
  `B9FB1BCBF09F3F73E2D8DEE3A6719B04D8A3FA55CE7A79512A35DF603F7798C3`
- Formal summary SHA-256:
  `0BA9AE35BA9639E9990D06C1E1D2F033DA0DE644596630A10918E62EBB2584A4`

## Source and execution

C34 admitted genuinely new pixels under four new render assignments. Instance,
wearable, depth, and witness capture each completed with 728 PNGs. All eight
physical-occlusion contracts passed and 73 actual blueprint identities were
observed. X54, X64, and X65 predictions were sealed before the evaluator was
opened. This was the sole scored X65 invocation on C34 and must not be rerun.

## Result

| Arm | TP | FP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| X24 | 77 | 24 | 76.24% | 44.77% | 56.41% |
| X54 | 127 | 35 | 78.40% | 73.84% | 76.05% |
| X64 | 127 | 26 | 83.01% | 73.84% | 78.15% |
| X65 | 127 | 26 | 83.01% | 73.84% | 78.15% |

X65 improved over X54 by `-9 FP / +2.11 pp F1`, but tied X64 exactly at
`+0 TP / +0 FP / +0.00 pp F1`. Contact recall was `100.00/80.00/65.22/58.33%`
for episodes 01/03/05/07. Safe episodes produced `0/3/1/1` false-alert
segments, so the per-episode and total segment constraints passed. Every
required authority invariant remained zero.

The base quality floor was mixed: recall, F1, contact recall, safe segments,
and authority invariants passed; precision was `83.01%`, below the frozen
`85%` floor. The `90/75/82%` precision/recall/F1 stretch target was not met.

## Why this is not an X65 confirmation

The source contained 17 selected contact-loss ambiguity frames and 16
pre-conflict joint-credential frames, so the intended ambiguity existed. But
X65 recorded zero ancestry synchronization frames and zero conflict handback
frames. X65 therefore had no causal opportunity to differ from X64. Episode 07
also failed preserved-parent ancestry diagnosis and lost route-risk continuity
at the final two required frames.

The admissible conclusion is mechanism-not-exercised, not a positive or
negative estimate of X65's incremental value. C34 does confirm that the shared
X64/X65 line transfers at `83.01/73.84/78.15%` on this fresh scripted CARLA
source, within the stated same-map, route-layout, detector, and motion-profile
boundary.

## Next action and claim boundary

Do not rerun or tune C34. Use it as consumed Development evidence to isolate
the 26 false-positive frames and the unreachable ancestry handback. The next
algorithm should seek an observable, representation-independent route-conflict
credential that can suppress safe-frame persistence without reducing contact
recall. Any promotion claim still requires a new frozen source after that
mechanism first shows a falsifiable cross-cohort Development effect.

C34 is synthetic Development evidence only. It is not real-world, product
default, deployment, user-benefit, or safety authority.
