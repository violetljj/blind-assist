# DTR CARLA C29-C32 X64 fresh confirmation

Date: 2026-09-01

Decision: `DTR_CARLA_C32_X64_MECHANISM_NOT_EXERCISED`

## Result

X64 remained byte-frozen at
`4A1B34C3CECF3635324DB909520AA4BE13578566FD3B9EDA28B8BE60364FE3DE`
through the complete C29-C32 source sequence. C29-C31 were terminal source
diagnostics and were never scored:

| Cohort | Source result | Occlusion contracts | Source result SHA-256 |
| --- | --- | ---: | --- |
| C29 sensor-topology | `SOURCE_NOT_EVALUABLE` | 6/8 | `B5208624C2C1A3B973758110584E458D85492D16E2C9BB3214AD5C91A08465F1` |
| C30 source-corrected | `SOURCE_NOT_EVALUABLE` | 6/8 | `344B8443F63918979937187A1AC6DF4B7004E90FA740793C4D385975F7EE967B` |
| C31 camera-corrected | `SOURCE_NOT_EVALUABLE` | 7/8 | `1230E255165A5059367E5EC21DC42C16F40C346366D412AEF75336FF0A5F641B` |
| C32 l03-restored | `SOURCE_COMPLETE` | 8/8 | `A81187F51568303ECC5C0C6C5DAD2954DE3F64D458188FC4959769E08A0B7BFC` |

C31 exposed a protocol-materialization defect rather than an algorithm result:
the C30/C31 chain reused a mutable parent object, so its nominal C28 l03
restoration still carried the C29 2.25x lateral amplification
(`+0.675/-0.675 m/s`) instead of the frozen C28 trajectory
(`+0.3/-0.3 m/s`). C32 read the frozen C28 protocol directly, changed only
that source trajectory plus seed/weather, and admitted all 8/8 physical
occlusion contracts. Its protocol SHA-256 is
`3535EB75466A57ABD688D5DE4A2330E6CB093C58C851A8F810070605A5949E98`.

The single C32 scored invocation produced:

| Arm | TP | FP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| X24 | 85 | 35 | 70.83% | 49.42% | 58.22% |
| X54 | 111 | 6 | 94.87% | 64.53% | 76.82% |
| **X64** | **117** | **7** | **94.35%** | **68.02%** | **79.05%** |

X64 improved on X54 by `+6 TP / +1 FP / +2.24 pp F1`, while preserving all
five required authority invariants at zero and keeping safe false-alert
segments bounded at `0/2/1/2` for ep_02/04/06/08. Contact recall was
`100% / 82.22% / 50% / 50%` for ep_01/03/05/07.

The confirmation gate did not pass because:

- recall was 68.02%, below the frozen 70% floor;
- ep_05 and ep_07 recall were 50%, below the frozen 55% per-contact floor;
- C32 exercised neither an X62 synchronized-conflict handback nor an X64
  unanchored-crossing release.

The authoritative terminal is therefore `MECHANISM_NOT_EXERCISED`, not
`GATE_MET` and not an algorithm failure inferred from an unexercised mechanism.

## Evidence identity

Local sealed confirmation root:
`artifacts.local/evidence/dtr-carla-x64-c32-confirmation/c32-x64-20260901-024500/`.

- X24 freeze:
  `511E923EBE1F72D9D2929CC643A058756864A5C6B14023D89EB67C6EFECFAB22`;
- X24 predictions:
  `946AA5C1C50C0ADF7BC214A62E63F0DB21F022BAD5289D9E8152626AE9002243`;
- X54 predictions:
  `C13CF730B0D25B50A01C5769DCC57D7EACE8B9A02865A754B11E12D6A19A4277`;
- X64 predictions:
  `041A7DB9CD223520756C5C4EF74AB0093A96619350EB77D9DEE3AC81EBE4EFFE`;
- summary:
  `ED8EB0AEDCE99ABC6BA075F5CA0FDD5F28E07CDA9386345B15A4EA4BE2A653D8`;
- confirmation runner:
  `5EC39B7490B842D890211B6CC89BC0225DE7357480279DC7264B35CF5ADC3AFD`.

The first C32 instance-server attempt exited before producing any durable
payload. Its four attempt logs were preserved. The same run ID used the one
pre-registered zero-frame retry, after which all four 728-frame sensor shards
and the join completed. No CARLA process, port, or storage lease remained.

## Claim and next decision

C32 is fresh, source-disjoint scripted-CARLA Development evidence. It shows
that the retained X54/X59/X62 stack transfers with high precision and a real
incremental recall effect, but it does not confirm X64's two new mechanisms
because neither was exercised. It is not unseen-map, open-world traffic,
natural-distribution, real-sensor, Android runtime, user-benefit, deployment,
reliability, or safety evidence.

C32 is now consumed. Use it only for diagnosis and structural development.
The next algorithm should target the ep_05/ep_07 visibility-to-route-risk gap
without using weather labels or weakening the zero-authority invariants. Any
later confirmation requires a new frozen source; C32 must never be rescored as
fresh authority.
