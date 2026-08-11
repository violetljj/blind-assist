# TARO O0R ARKitScenes direct Apple SUPPORT R4 full-cohort result

## Outcome

Terminal: `TARO_O0R_DIRECT_APPLE_SUPPORT_R4_FULL_COHORT_COMPLETE`.

The fixed R3 direct-Apple SUPPORT method has real full-cohort headroom, but it
must not replace the baseline unconditionally. Across all 171 existing O0R eval
truth frames / 1,539 queries, direct Apple SUPPORT improved both the
parent-macro support-height and support-normal errors. It also recovered 36
baseline extraction failures, but lost 108 baseline-evaluable queries when the
source plane was unavailable.

The result therefore supports a threshold-free baseline fallback, not an
unconditional direct branch and not a learned selector. This is a retrospective
WILD_LAB map on the already observed 16 eval parents. It is not fresh
confirmation, a formal O0R PASS, or deployment, product or safety evidence.

## Frozen method and source firewall

R4 replays the exact R3 factor ownership without fitting or thresholds:

- SCALE: the existing sealed Apple/candidate log-scale derivation;
- SUPPORT: the confidence-2 AppleDepth plane, independent of candidate range;
- BOUNDARY/query geometry: the source-scaled sealed DepthART raster;
- invalid Apple planes: explicit `UNKNOWN`, with no clipping or repair.

Phase A rebuilt source receipts from raw AppleDepth, confidence, exact
intrinsics and trajectory for all 171 physical frames. Its completion seal was
persisted before Phase B opened the bound R1 query records, compact truth or
FARO-derived outcomes. Geometry drift between the current and compact runtime
was preserved as evidence rather than silently normalized.

## Descriptive result

| Metric | Result |
|---|---:|
| physical frames / parents | 171 / 16 |
| queries | 1,539 |
| source SUPPORT frames | 158 |
| source `UNKNOWN` frames | 13 |
| direct extraction-evaluable queries | 1,422 |
| R1 baseline extraction-evaluable queries | 1,494 |
| extraction recovered / lost vs baseline | 36 / 108 |
| support height improved vs baseline | 1,212 queries |
| support normal improved vs baseline | 1,114 queries |
| height-and-normal no-regret vs baseline | 1,003 queries |
| direct boundary-evaluable queries | 121 |
| direct known point-clearance queries | 8 |

The median of parent-level median support-height error reductions was
`+0.318325443890 m`; the corresponding support-normal reduction was
`+0.026784068381 rad`. Both signs are improvements and all 16 parents
contributed to each metric. Compared with the R1 source-anchored branch, the
reductions were `+0.052054691799 m` and `+0.032141986337 rad`.

Direct source support coverage had a median parent coverage of
`0.909090909091`. Nine frames were rejected for implausible camera height and
four for excessive support slope. Those 13 frames remained `UNKNOWN`.

## Decision and next falsifiable step

Do not adopt direct Apple SUPPORT unconditionally. Retain its source-only plane
validity as the complete selection input and use the frozen R1 baseline when
the plane is unavailable. This yields the zero-parameter R4A policy
`DIRECT_WHEN_SOURCE_SUPPORT_AVAILABLE_ELSE_R1_BASELINE_V1`.

R4A must be evaluated only as a same-cohort retrospective replay. If it closes
the 108-query coverage regression without using truth-derived fields for
selection, the next scientific step is fresh confirmation on the separate
eight `ADAPTER_FIT` parents under a new pre-outcome role amendment. No selector
or threshold may be fitted on these 16 observed eval parents.

## Validation and reproducibility bindings

- implementation commit: `3303793f`;
- relevant focused tests: `27/27` PASS;
- summary content seal:
  `26A29C90137A1114CEABFC35F104D85785B04383F80D8853EC33E578694FBD25`;
- source-phase completion content seal:
  `037C5D8701D3A20701B4446F0C2BB8F5818EA65040558F9A36A5F2467B9D96E2`;
- evidence manifest file SHA-256:
  `EB87FC6141723D2B44DCB384DF594FE3BFD65436262B07BCF2C0E3809709F760`;
- result file SHA-256:
  `F8BDDCB58534D5C6436A40C4509F72E21B5CB7BBFEFC29522DBAFEBE12483C3C`;
- query blob SHA-256:
  `323FFB7F456517C4EAEAED301A18FC96A1B2813526D963230B3EB43B8D58F2A7`;
- query-record sequence SHA-256:
  `DA64D061734BEB384FA0DC80854775C835C73D5F43D8F830F8810D5FEB347DD0`;
- evidence root:
  `artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-support-r4-full-cohort`.

The manifest accounts for 518 pre-manifest files and 2,334,722 bytes. All file
hashes, 1,539 query seals and the exact canonical summary replay were verified.
