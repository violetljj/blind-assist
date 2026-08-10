# TARO O0R ARKitScenes source-anchored factor canary R1 result

## Outcome

Terminal: `TARO_O0R_SOURCE_ANCHORED_FACTOR_CANARY_R1A_RECONCILED_COMPLETE`.

The AppleDepth metric anchor remains useful, but unconditional application
*before* the current hard SUPPORT extractor is not adopted. On the locked 171
truth-frame cohort it substantially improved metric support-height and boundary
errors when both branches remained evaluable, while losing extraction for 112
queries across 14 physical frames and recovering none of the 45 baseline
failures. Every lost extraction remained unevaluable; no failure was converted
to a negative or known query.

This is a post-hoc, point-estimate-only WILD_LAB canary. It does not run the
formal uncertainty reducer or authorize O0R PASS, deployment, product or safety
claims.

## Execution and cohort

- `171/171` physical frames, `1,539/1,539` queries and all 16 eval parents were
  accounted for;
- the 239 previously sealed DepthART candidates and 239 source-scale records
  were reused; GPU inference, training and network requests were zero;
- each R0 source scale was exactly re-derived from its bound AppleDepth,
  confidence and sealed candidate before the scaled raster was constructed;
- scale was applied before the depth-range gate, support fit, boundary
  classification and point-clearance computation;
- elapsed CPU time was 2,435.735 seconds (40.60 minutes).

All 1,539 compact R3 query receipts reproduced exactly. The current runtime
reproduced the older R3 FARO geometry commitment for 72 queries; the other
1,467 used a new, internally bound reconstruction from the same source arrays.
The R1 runtime used NumPy 2.4.4, while the original factor execution was locked
to NumPy 2.1.3. Therefore this result is not described as an exact byte replay
of every old R3 factor frame.

## Descriptive result

Aggregation is query median within physical frame, then frame median within
parent, then median across parents. Separate baseline/anchored medians can use
different evaluable subsets; paired effect statistics use only queries where
both branches are evaluable.

| Metric | Baseline | Source anchored |
|---|---:|---:|
| SUPPORT evaluable queries | 1,494 | 1,382 |
| SUPPORT normal angular error, parent macro (rad) | 0.053962090827 | 0.051610488109 |
| SUPPORT height absolute error, parent macro (m) | 0.346181430640 | 0.108182872362 |
| BOUNDARY evaluable queries | 118 | 107 |
| BOUNDARY point-ID Jaccard, parent macro | 0.612632714210 | 0.852439501145 |
| BOUNDARY XYZ median error, parent macro (m) | 0.425752637094 | 0.109589634148 |
| point-clearance evaluable queries | 10 | 9 |
| point-clearance absolute error, parent macro (m) | 0.201631561403 | 0.166285932855 |

Paired-query results sharpen the interpretation:

- support height improved for `962/1,382` paired queries and worsened for 420;
  parent-macro paired reduction was `0.160397395705 m`;
- support normal improved for only `417/1,382` and worsened for 965;
  parent-macro paired change was `-0.005237116941 rad`, so scale corrected
  height much more reliably than support orientation;
- boundary XYZ error improved for `97/106` paired queries; paired parent-macro
  reduction was `0.294470756427 m`;
- boundary Jaccard improved for `73/106` paired queries and worsened for 33;
- point-clearance error improved for `7/9` paired known queries, but those nine
  queries cover only one parent at the paired parent-macro level. This is too
  little support for a query-effect claim.

## Failure and reliability diagnosis

Baseline failures were 36 `SUPPORT_SLOPE_EXCEEDED`, six
`SUPPORT_PLAUSIBLE_INSUFFICIENT` and three `SUPPORT_GATE_FAILED`. After source
scaling, failures were 110 `SUPPORT_SLOPE_EXCEEDED` and 47
`SUPPORT_GATE_FAILED`. The anchored branch lost 112 queries across 14 frames,
including 12 complete nine-query frames and two two-query partial losses; it
recovered zero baseline failures.

The source-only reliability diagnostics were not strong enough to justify an
abstention threshold on this same eval cohort. Spearman correlation with R0
source absolute log-scale error was 0.183 for log-ratio MAD, 0.056 for q95
absolute deviation and 0.209 for 4x4 tile-median IQR. Lost frames had somewhat
higher medians than other frames, but distributions were not separated:

| Reliability signal | Lost-frame median | Other-frame median |
|---|---:|---:|
| log-ratio MAD | 0.084592286726 | 0.078347297133 |
| q95 absolute deviation | 0.344112737349 | 0.275407526031 |
| 4x4 tile-median IQR | 0.157440740937 | 0.138735154914 |

No threshold was selected post hoc.

## Evidence reconciliation

The original R1 frame and aggregate query records all passed their content
seals and matched as a multiset. Its first derived summary was calculated from
in-memory floats before the evidence writer canonicalized records to 12 decimal
places. Recomputing from persisted records changed 11 derived numbers by at
most one canonical last-place unit (`1.0e-12`) and therefore changed the summary
hash, without changing any algorithm output, count, direction or decision.

R1A fixes the evidence-layer round-trip and supersedes only that first derived
summary. It did not reopen source arrays, recompute geometry or rerun the
algorithm. A regression test now requires summary hashes to survive a canonical
JSON round trip.

## Decision and next falsifiable step

Retain the R0 Apple scale estimator. Reject unconditional pre-extraction scale
injection into the present hard support fitter as the adopted route. The next
bounded successor is `APPLE_SEEDED_SUPPORT_RECOVERY_CANARY_R2`:

1. derive a confidence-bound AppleDepth support plane without FARO;
2. use it only to seed/recover support selection on the source-scaled candidate;
3. first test the frozen method on the 14 R1 lost frames;
4. keep any unrecovered or inconsistent query `UNKNOWN`;
5. expand to all 171 frames only if recovery does not trade away FARO support
   normal/height accuracy. No deployment threshold may be chosen from the same
   16 eval parents.

## Reproducibility bindings

- R1 implementation commit: `7b862b09814ffc04101dc1629bf6bb7b03480fb3`
- original R1 manifest SHA-256:
  `86DF7ECC39712BFAC11ED8C41A50147760704859D892A273FA00ABF35AE1C265`
- R1A reconciled summary seal:
  `0298BC831C611145150B115ED987C28E72C982859E3AC1BD4B0E682EA7E9B410`
- R1A validation seal:
  `5B2EF6BB9772932242AEE127E6A9E300ACBDE8DCA8D3845BA9E345CBE6AEEEDB`
- R1A diagnostics seal:
  `3B320964744D459F6A0577C44B89ABAB88A95A15688EDE53EC1A19DC5BE288AC`
- R1A result file SHA-256:
  `32EE579C4131C6D585FCB28A44B589A91553A21216402949C6120FA1536D999A`
- R1A manifest SHA-256:
  `136006B76EDFD7B89C0CAB722C9F3FC1D77AF0C7F7709903881FCF0A0B182C53`
- original R1 evidence root:
  `artifacts.local/evidence/taro/o0r-arkitscenes-source-factor-r1`
- authoritative reconciliation root:
  `artifacts.local/evidence/taro/o0r-arkitscenes-source-factor-r1a-reconciliation`
