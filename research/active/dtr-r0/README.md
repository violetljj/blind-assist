# DTR: route-conditioned obstacle-risk events

Status: `DTR_R2_DYNAMIC_AND_CURVED_STATIC_CEILINGS_ESTABLISHED /
RGB_RUNTIME_DOWNSTREAM / HEAD_CLEARANCE_POSITIVE_NOT_EVALUABLE`

## Result first

DTR now answers a narrower and more useful question than object detection:

```text
wearer future route tube
        intersect
object future occupancy
        within 3 seconds
        -> ONSET / HOLD / ESCALATE / CLEAR
```

The current algorithm is R2, a robust early-warning branch plus one fixed
imminent guard. It was evaluated without detector tuning on three public,
real mobile-platform sources.

| Public-real dynamic cohort | Events | R0 route recall / false | R1 robust recall / false | **R2 guarded recall / false** |
| --- | ---: | ---: | ---: | ---: |
| THÖR-MAGNI, exact global route, 19 sessions | 10 | `9/10` / 55 | `9/10` / **37** | **`10/10` / 42** |
| JRDB test, robot-relative diagnostic, 27 sequences | 175 | `161/175` / 260 | **`164/175` / 255** | **`164/175` / 256** |
| CODa sequences 18+20, pose-authoritative development | 88 | `88/88` / 199 | `70/88` / **174** | **`86/88` / 190** |
| CODa sequence 16, rainy heavy-traffic holdout | 34 | `34/34` / **87** | `28/34` / 86 | **`33/34` / 95** |

R2 strictly dominates R0 on the route-authoritative THÖR ceiling: it recovers
the last event while deleting `13/55` false-alert segments (`23.6%`). On the
larger JRDB diagnostic it adds three recalled events and deletes four false
alerts. It essentially preserves R1 there, at one additional false segment.

CODa is the deliberate hard case, not a hidden win. Across its development and
holdout partitions, R2 recalls `119/122` events (`97.5%`) versus R0's `122/122`,
with nearly identical false alerts (`285` versus `286`). It recovers 21 of the
24 events lost by R1's robust filter, but it does not dominate R0 on CODa. This
is the retained cost of rejecting unstable early motion rather than a threshold
to tune away.

All 122 CODa dynamic critical events are pedestrians. Bicycle/micromobility and
vehicle tracks contribute `24,009` and `11,585` evaluated frames, respectively,
but zero countable positive critical events. They provide real class exposure
and false-alert evidence, not bicycle or vehicle recall.

## Static, wall, and temporary-obstacle ceiling

A second CODa ceiling keeps source-native static 3-D boxes fixed in the world,
uses current-and-past ego pose to extrapolate a constant-turn-rate-and-velocity
route, intersects that route with oriented obstacle footprints, and admits only
the lower-body or bounded head-clearance height bands.

Across CODa sequences 16, 18, and 20 it evaluates 182,274 known frames and 12
future path-contact events: six barrier/boundary, five fixed-structure, and one
temporary-obstacle event.

| Static arm | Event recall | False-alert segments | CLEAR | Median lead |
| --- | ---: | ---: | ---: | ---: |
| P0 current 3 m proximity | `12/12` | 104 | `3/4` | `2.70 s` |
| S1 straight route | `11/12` | 22 | `4/4` | `2.60 s` |
| S2 straight route + height | `11/12` | 22 | `4/4` | `2.60 s` |
| **S3 curved route + height** | **`12/12`** | **10** | **`4/4`** | **`3.00 s`** |

S3 preserves all proximity events while removing `94/104` false alerts
(`90.4%`). Relative to the straight route it recovers the turn event and cuts
false alerts from 22 to 10 (`54.5%`). Its breakdown is barrier/boundary `6/6`
with one false segment, fixed structure `5/5` with four, and temporary obstacle
`1/1` with five. All 12 recalled events also reached the one-shot `ESCALATE`
transition, with `1.40 s` median escalation lead.

The selected sequences contain no positive vegetation or head-clearance path
event. Vegetation has 33,954 evaluated frames and zero false alerts, which is
class exposure without positive recall authority. GOOSE was checked as a
possible compact supplement:
its 3-D validation archive contains `tree_crown`, `wire`, and `boom_barrier`
semantics, but only discrete point clouds/labels, not synchronized wearer-route
or ground-clearance truth. Treating semantic tree crown as a human head
collision would manufacture the answer, so positive hanging-branch performance
is `NOT_EVALUABLE` rather than a reason to download or tune another corpus.

## Frozen mechanics

R1 forms every causal pairwise velocity allowed by a 1.5-second target history,
uses the component-wise Theil-Sen median as target motion, and analytically
computes first entry into the 3-second, 0.65 m half-width route tube. The
fraction of velocity hypotheses entering the tube is diagnostic consensus
support, not a probability.

R2 leaves R1 unchanged. It admits R0's least-squares route intersection only
when its predicted entry is inside the already defined escalation half-horizon
(`1.5 s`). There is no new fitted threshold, support sweep, or event-label
calibration.

S3 estimates forward speed and yaw rate from the latest 0.5 seconds of causal
ego pose, samples a three-second curved route every 0.1 seconds, and intersects
it with source-native oriented boxes. The vertical contract is:

- lower-body occupancy: up to `1.35 m`;
- head-clearance occupancy: `1.35..2.10 m`;
- geometry wholly above `2.10 m`: non-actionable.

Known positive frames drive `ONSET`, sustained positives drive `HOLD`, and the
first entry inside half-horizon emits `ESCALATE` once. A known negative must
persist for 0.5 seconds before `CLEAR`; missing pose, track, or motion history
is `UNKNOWN` and cannot manufacture a clear.

Frozen fingerprints:

- R1: `741b815017297f64cb80f3f9d44282eb7fd16f79f60b04fe4f25ae8a9026f4b8`
- R2: `4142a575911e9d43508e996b0e0cf5062dc5c86d755dfe63d41279caf56302a8`

## Runtime bridge

The algorithm remains source-independent at the `CausalFrame` / metric-box
boundary. The Android USTRF core now has two adapters:

- metric-depth samples produce lower-body/head-clearance swept occupancy while
  ignoring geometry wholly above `2.10 m`;
- privileged native 3-D boxes rasterize oriented lower/head footprints into the
  same USTRF geometry packet for offline replay.

The earlier fixed-height JRDB RGB bridge remains the deployability hint: on one
curated 143-frame window, route intersection kept `3/3` recall and reduced
false alerts from 17 to 9. It is not a new generalization result, and the
current priority is the public-data algorithm ceiling rather than phone
recording or live-device promotion.

## Literature basis

The robust slope is grounded in [Sen's Theil-Sen estimator](https://doi.org/10.1080/01621459.1968.10480934),
and route-tube entry is the finite-horizon single-command case of
[Velocity Obstacles](https://doi.org/10.1177/027836499801700706). Component-wise
x/y medians are an engineering extension. R2's imminent fallback and the
one-shot `ESCALATE` are product-policy mechanisms, not claims from those papers.

`risk_score` and pairwise support remain ordinal diagnostics. A calibrated
collision probability would require explicit uncertainty and separate
validation, such as [continuous collision probability](https://arxiv.org/abs/2104.01659)
or a [Dynamic Lambda-Field](https://arxiv.org/abs/2103.04795). Extended or
nonholonomic vehicle prediction would require a model such as
[Generalized Velocity Obstacles](https://doi.org/10.1109/IROS.2009.5354175).

## Evidence receipts

- THÖR R2: `artifacts.local/evidence/dtr-r2/thor-magni-guarded/result.json`,
  SHA-256 `f1c54804b1d9218cb2bcb1d9bb1f8d5a545c189c8abb1bf3e9b409a220110de4`.
- JRDB test R2: `artifacts.local/evidence/dtr-r2/jrdb-test-guarded/result.json`,
  SHA-256 `20bebbdd2aee7d82206552644f6801d0ddd2df61b606747ee071f6da19c74e04`.
- CODa development R2: `artifacts.local/evidence/dtr-r2/coda-18-20-final/result.json`,
  SHA-256 `5882f2aaddf4d078f618b2ecf5c269f74f538bdfebfa5843f6205b18873c7c72`.
- CODa holdout R2: `artifacts.local/evidence/dtr-r2/coda-16-final/result.json`,
  SHA-256 `6b6b4ec8e37e77d1dc3aa60d07f383cabd8d00d6c29b7ebe14b30283c8fffe7c`.
- CODa curved static: `artifacts.local/evidence/dtr-static/coda-16-18-20-final/result.json`,
  SHA-256 `8bf8e11c95ad687e9771f0e2f6de871557044c7ecc7a9eafcadae6406bb409ec`.

## Reproduce

From the repository root:

```powershell
python research/active/dtr-r0/thor_magni_native_ceiling.py `
  --manifest-dir F:\ba-data\hftf-d7-public-real\manifests `
  --include-r1 --include-r2 `
  --output artifacts.local/evidence/dtr-r2/thor-magni-guarded/result.json

python research/active/dtr-r0/jrdb_native_ceiling.py `
  --labels-zip <jrdb-test-labels.zip> `
  --timestamps-zip <jrdb-test-timestamps.zip> `
  --include-r1 --include-r2 `
  --output artifacts.local/evidence/dtr-r2/jrdb-test-guarded/result.json

python research/active/dtr-r0/coda_native_ceiling.py `
  --sequence-root 20=<coda-sequence-20-native-root> `
  --sequence-root 18=<coda-sequence-18-native-root> `
  --output artifacts.local/evidence/dtr-r2/coda-18-20-final/result.json

python research/active/dtr-r0/coda_native_ceiling.py `
  --sequence-root 16=<coda-sequence-16-native-root> `
  --output artifacts.local/evidence/dtr-r2/coda-16-final/result.json

python research/active/dtr-r0/coda_static_ceiling.py `
  --sequence-root 20=<coda-sequence-20-native-root> `
  --sequence-root 18=<coda-sequence-18-native-root> `
  --sequence-root 16=<coda-sequence-16-native-root> `
  --output artifacts.local/evidence/dtr-static/coda-16-18-20-final/result.json
```

## Claim ceiling

These are privileged algorithm ceilings. THÖR uses exact controlled-lab global
tracks; JRDB test is a large robot-relative/interpolated diagnostic; CODa uses
source-native boxes, identities, pose, timestamp, and calibration. The future
path and future oriented-box contact are evaluator-only, but no RGB/LiDAR
detector, intent prediction, natural walking, Android runtime, user benefit, or
safety performance is established.

CODa establishes positive dynamic recall only for pedestrians and positive
static recall only for the 12 barrier/fixed/temporary events observed. It does
not establish positive bicycle, vehicle, vegetation, thin-branch, drop-off, or
head-clearance recall. `UNKNOWN` and `NOT_EVALUABLE` are never counted as safe.
