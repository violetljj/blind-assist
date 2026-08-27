# DTR-R0: Dynamic Travel Risk trajectory-to-route events

Status: `DTR_R0_ACTIVE / CAUSAL_DUAL_LIDAR_BRIDGE_DIRECTIONALLY_POSITIVE /
STRONG_EFFECT_NOT_REPRODUCED`

## Result first

The narrow R0 question is whether a causal short target track becomes more
actionable when it is intersected with the wearer's short route, instead of
warning merely because radial TTC is small.

The first exact public-real privileged ceiling is positive. On the 19 locally
available THÖR-MAGNI Pupil sessions, both arms received the same QTM global
camera-wearer and person centroids. Wearer route yaw was derived from only the
past 0.5 seconds of ego motion; future QTM positions were evaluator-only truth.

The run used 10 Hz samples from the 100 Hz source and covered 19 sessions,
461,182 source rows, 132 wearer-target identities, 357 evaluable contiguous
track segments, 520.0 target-track seconds, and 10 non-left-censored geometric
critical events.

| Metric | B2 radial TTC | C route intersection |
| --- | ---: | ---: |
| Critical-event recall | `80.0%` (`8/10`) | `90.0%` (`9/10`) |
| Lateral-crossing recall | `85.7%` (`6/7`) | `85.7%` (`6/7`) |
| Oncoming-corridor recall | `0.0%` (`0/1`) | `100.0%` (`1/1`) |
| False alert segments | `96` | `55` |
| False alert reduction | — | `42.7%` |
| Median first-alert lead | `1.85 s` | `2.90 s` |
| Mean alert segments / event | `1.3` | `1.5` |

C preserved the crossing partition, recovered the one oncoming event, raised
overall recall by 10 percentage points, and reduced target-level false alert
segments by 42.7%. It therefore crosses the frozen core line of non-decreasing
critical recall plus at least 40% false-alert reduction. The decision is
`PRIVILEGED_CEILING_GO_TO_RGB_DETECTOR_TRACKER`.

At that native stage this was an advancement decision, not obstacle-avoidance
completion. Only 10
critical events were evaluable, and no critical event retained enough
post-event track to score CLEAR. RGB perception, metric projection, product
utterances, natural walking, and safety remain unproved.

The durable result is
`artifacts.local/evidence/dtr-r0/thor-magni-native-ceiling-v1/result.json`,
SHA-256 `61d8d1993797cf13640c1e09f04337602d9874ad182a177c0a5fe6603e49fd69`.

## Fixed JRDB real-RGB bridge

The next smallest increment has also run. A fixed Development window from
JRDB train sequence `packard-poster-session-2019-03-20_1` uses stitched RGB
frames `115..257` (143 frames, 9.99 seconds). Before annotations are opened,
YOLO11n plus the existing causal tracker writes a truth-blind track ledger.
The evaluator then binds each current detector-track occurrence to a native
3-D center through frozen same-frame Hungarian IoU (`>=0.30`). Future native
geometry remains evaluator-only.

The bridge scored every evaluable target in the window, not only the two
targets used to select it. It produced 52 evaluable target segments and three
critical events: two lateral crossings and one oncoming event.

| Metric | B2 radial TTC | C route intersection |
| --- | ---: | ---: |
| Critical-event recall | `100%` (`3/3`) | `100%` (`3/3`) |
| False alert segments | `7` | `4` |
| False alert reduction | — | `42.9%` |
| Median first-alert lead | `3.94 s` | `2.20 s` |
| Mean alert segments / event | `1.00` | `1.33` |
| CLEAR on eligible events | `100%` (`2/2`) | `50%` (`1/2`) |

The two selection targets were detector-matched on `143/143` frames, although
their causal identities fragmented across two and four tracker IDs. Across all
targets, known prediction coverage was `45.97%`. The narrow positive effect
survived the detector/tracker bridge: critical recall did not fall and false
alert segments again dropped by more than 40%. The shorter lead and missed
CLEAR remain real defects; this one curated window is not a new advancement
gate.

The result is
`artifacts.local/evidence/dtr-r0/jrdb-rgb-bridge-v1/result.json`, SHA-256
`9dac2d1512cd21ddc2ae5d76d5785c822cd1997a0d7a3798e835fdbe3a73e175`.
The detector/tracker ledger was sealed before annotation access at
`result.tracks.jsonl`, SHA-256
`58873bca0fd120f5d71f18a960bccfc9289b42ac9f98d077996b7a82c90059fc`.

## Causal raw-sensor geometry bridge

The privileged current 3-D center has now been removed from the observation
path. The final fixed bridge selects the latest upper and lower JRDB Velodyne
scan whose header timestamp is no later than the image, motion-compensates each
scan through bag `odom -> base_link`, projects both with the official JRDB
cylindrical calibration, and keeps points inside a fixed YOLO11n-seg person
instance mask. The tracker, DTR arms, route horizon, event lifecycle, and
evaluator stayed unchanged. The sensor ledger was sealed before JRDB identity
and future event truth were opened. Current observations pair the raw sensor
center with a fixed `0.30 m` person radius; native body extent is used only by
the evaluator to define future event truth.

The dual-lidar mask bridge covered `4,363/4,826` detector-track occurrences
(`90.41%`). Among detector/native evaluator matches, geometry was available in
`3,174/3,319` cases (`95.63%`), with `0.106 m` median and `0.284 m` p90 planar
position error.

| Metric | B2 radial TTC | C route intersection |
| --- | ---: | ---: |
| Critical-event recall | `100%` (`3/3`) | `100%` (`3/3`) |
| False alert segments | `18` | `13` |
| False alert reduction | — | `27.8%` |
| Median first-alert lead | `3.67 s` | `1.91 s` |
| Mean alert segments / event | `1.00` | `1.33` |
| CLEAR on eligible events | `100%` (`2/2`) | `100%` (`2/2`) |

This is a real, causal, directionally positive observation: no critical event
was lost and five non-actionable ONSET segments disappeared. It does not
reproduce the frozen `>=40%` strong-effect line, so it is not an advancement or
Android-promotion result. The earlier full-box upper-lidar estimator also
remained below that line (`37 -> 33` false alert segments) and is closed rather
than tuned.

The final result is
`artifacts.local/evidence/dtr-r0/jrdb-dual-lidar-mask-bridge-v1/result.json`,
SHA-256
`bec37a541ff1fb20d0978a04f109d29f737c280f6b5bd023ca74532d787e489d`.
Its truth-blind sensor ledger is `result.sensor-tracks.jsonl`, SHA-256
`62909c877068bb36bc9f522229b8e6afdfc20958187279e0e3f8ad5ab7b4a506`.

## Why JRDB is diagnostic, not the route authority

The processed JRDB labels/timestamps split does not contain synchronized ego
pose. Its 3-D boxes are robot-relative and `890,153/890,932` used boxes are
source-marked interpolated. It can test relative closure at scale, but not the
exact wearer-route intersection that defines this question.

That diagnostic covered 27 sequences and 175 events. C raised recall from
75.43% to 92.00%, but false alert segments rose from 191 to 260 (`+36.1%`). This
is a useful generalization warning; it does not override the smaller
THÖR-MAGNI result because the two sources do not have the same route authority.
Its result is
`artifacts.local/evidence/dtr-r0/jrdb-native-ceiling-v1/result.json`, SHA-256
`c35724c1ba82c8f7956a27d8ac4e7493ac41d2dc9e9aa16181135a0adc579f92`.

## Frozen mechanics

All arms consume ordered causal frames and emit a shared
`ONSET / HOLD / CLEAR / UNKNOWN` lifecycle:

| Arm | Decision rule |
| --- | --- |
| `B0_detection_reminder` | Any current tracked detection requests a reminder. |
| `B1_distance_gate` | The nearest current detection requests a reminder inside a fixed distance. |
| `B2_radial_ttc` | A constant-velocity short track requests a reminder when radial closing time is inside 3 seconds. |
| `C_route_intersection` | Ego-compensated target occupancy intersects the time-aligned 3-second wearer tube. |

World coordinates are a metric 2-D ground plane. Detector observations are
`forward_m/left_m` in the sensor frame. `body_yaw_rad` owns wearer-route
direction and `sensor_yaw_rad` owns camera direction. Observations are
transformed into world coordinates before target velocity is fitted.

Missing current tracks, pose, or causal motion history produce `UNKNOWN`, never
`CLEAR`. All arms share a 0.50-second clear grace: one known negative cannot
fragment an alert, and UNKNOWN cannot complete a clear. Event counts are ONSET
segments, not positive-frame counts.

## Next obstacle-only step

Do not record the superseded 24/120 local clips, widen this one window, or tune
the route matcher, tracker, IoU, horizon, or lifecycle against these opened
outcomes. The detector/tracker and causal raw-geometry questions have both been
answered for this R0 window. The native ceiling is strong; the runnable raw
sensor bridge is positive but below the strong-effect line.

Do not rescue it with temporal-filter, threshold, or matcher sweeps. If DTR is
continued, change the metric-motion information source in a new versioned
increment; otherwise retain this bridge as the honest R0 ceiling-to-sensor
result. It is not another cohort-design task.

## Reproduce the ceilings and fixed bridge

From this directory or the repository root:

```powershell
python research/active/dtr-r0/thor_magni_native_ceiling.py `
  --manifest-dir F:\ba-data\hftf-d7-public-real\manifests `
  --output artifacts.local/evidence/dtr-r0/thor-magni-native-ceiling-v1/result.json

python research/active/dtr-r0/jrdb_native_ceiling.py `
  --labels-zip artifacts.local/datasets/ustrf-canonical-observation-source-authority-data-pack-r0/jrdb/test_labels.zip `
  --timestamps-zip artifacts.local/datasets/ustrf-canonical-observation-source-authority-data-pack-r0/jrdb/test_timestamps.zip `
  --output artifacts.local/evidence/dtr-r0/jrdb-native-ceiling-v1/result.json

python research/active/dtr-r0/jrdb_rgb_bridge.py `
  --labels-zip <jrdb-train-labels.zip> `
  --timestamps-zip <jrdb-train-timestamps.zip> `
  --bag <packard-poster-session-2019-03-20_1.bag> `
  --images-dir <ignored-image-directory> `
  --model <yolo11n.pt> `
  --calibration-defaults <jrdb-toolkit-calibration-defaults.yaml> `
  --sensor-setup-pdf <Sensor_setup_JRDB.pdf> `
  --output <ignored-evidence-directory>/result.json

python research/active/dtr-r0/jrdb_mask_lidar_bridge.py `
  --rgb-result <ignored-evidence-directory>/jrdb-rgb-bridge-v1/result.json `
  --rgb-tracks <ignored-evidence-directory>/jrdb-rgb-bridge-v1/result.tracks.jsonl `
  --labels-zip <jrdb-train-labels.zip> `
  --timestamps-zip <jrdb-train-timestamps.zip> `
  --bag <packard-poster-session-2019-03-20_1.bag> `
  --calibration-dir <jrdb-toolkit>/calibration `
  --segmentation-model <yolo11n-seg.pt> `
  --output <ignored-evidence-directory>/jrdb-dual-lidar-mask-bridge-v1/result.json
```

The synthetic generator and tests remain mechanism diagnostics only. They do
not contribute to either public-real decision.

## Claim ceiling

THÖR-MAGNI supplies real, synchronized global QTM geometry, but its run uses
privileged head/helmet centroids and a fixed 0.30 m person radius in a controlled
laboratory. It does not establish body collision truth, intended route, detector
quality, Android behavior, user benefit, natural-distribution performance, or
safety. The JRDB bridge adds real RGB detection and causal tracking plus exact
bag odometry. The final dual-lidar bridge also replaces current annotated range
with raw sensor geometry; annotations remain evaluator-only identity and future
event truth. The bag has no `tf_static`; the planar static chain is bound to
external official JRDB calibration and fixed Kinova URDF provenance. This is
one curated Development window, not RGB-only metric perception, Android
runtime, natural-distribution, product, user-benefit, or safety evidence.
