# DTR CARLA X64 consumed transfer Development

Date: 2026-08-31

Decision: `DTR_CARLA_X64_C26_C27_C28_CONSUMED_REFERENCE_TARGET_MET`

## Result

X64 combines two structural corrections around frozen X59:

1. X62 hands an issued-plan metric route entry back through a globally
   suppressed X44 frame only when the metric evidence and a suppressed surface
   component are synchronously measured. The metric vector must be
   longitudinally closing or lateral-dominant; the same metric identity alone
   may continue on HOLD.
2. X64 treats direction-only object permanence for a cross-route surface
   trajectory as existence memory, not independent route-risk authority.
   Longitudinal corridor memory and occupancy-peak-anchored crossing memory are
   retained.

No detector, weather/light label, duration, distance, absolute-speed cutoff,
or other numeric threshold was added.

| Consumed cohort | Arm | TP | FP | Precision | Recall | F1 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| C26 source-corrected | X54 | 122 | 20 | 85.92% | 70.52% | 77.46% | reference |
| C26 source-corrected | **X64** | **126** | **20** | **86.30%** | **72.83%** | **79.00%** | target met |
| C27 daylight | X54 | 119 | 25 | 82.64% | 68.79% | 75.08% | reference |
| C27 daylight | **X64** | **126** | **15** | **89.36%** | **72.83%** | **80.25%** | target met |
| C28 mixed lighting | X54 | 113 | 26 | 81.29% | 65.32% | 72.44% | reference |
| C28 mixed lighting | **X64** | **125** | **14** | **89.93%** | **72.25%** | **80.13%** | target met |

All three X64 replays passed the common reference targets: precision at least
85%, recall at least 70%, F1 at least 78%, every contact episode recall at
least 55%, bounded safe false segments, and zero required authority invariant
violations.

The incremental X64 effects versus X54 were:

- C26: `+4 TP / +0 FP`;
- C27: `+7 TP / -10 FP`;
- C28: `+12 TP / -12 FP`.

## Evidence identity

X64 predictor SHA-256:
`4A1B34C3CECF3635324DB909520AA4BE13578566FD3B9EDA28B8BE60364FE3DE`.

Local sealed summaries:

- C26:
  `artifacts.local/evidence/dtr-carla-x56-c26-confirmation/c26-x56-20260831-204200/x64-consumed-transfer-20260831-235900/summary.json`,
  SHA-256
  `2FBF9F1DB7155A7F0F97ADDC90868DA058205E7A85FE5626247ECC4A06A24CC5`;
- C27:
  `artifacts.local/evidence/dtr-carla-x57-c27-confirmation/c27-x57-20260831-220500/x64-consumed-transfer-20260831-235900/summary.json`,
  SHA-256
  `4A2BC90267911BF471E3A2EC73D0A9FAA1683AB0E5E4C54E21E874C8D7BBA323`;
- C28:
  `artifacts.local/evidence/dtr-carla-x59-c28-confirmation/c28-x59-20260831-225500/x64-consumed-transfer-20260831-235900/summary.json`,
  SHA-256
  `D85A710AA087DCDC19D23B3262F9E0E36F5582EA0755BBE5183E34DD7DE88973`.

## Claim and next decision

C26-C28 had already been opened before X64 was designed. These replays are
therefore post-hoc consumed synthetic Development evidence. They show a stable
mechanism effect across source-corrected, daylight, and mixed-lighting CARLA
cohorts, but they are not fresh confirmation, natural-distribution evidence,
Android runtime evidence, user-benefit evidence, deployment evidence, or
safety evidence.

Freeze X64 unchanged and run exactly one new source-disjoint CARLA cohort that
contains both longitudinal corridor occlusion and cross-route occlusion. Do not
tune X64 on C26-C28 or use those cohorts as fresh confirmation authority.
