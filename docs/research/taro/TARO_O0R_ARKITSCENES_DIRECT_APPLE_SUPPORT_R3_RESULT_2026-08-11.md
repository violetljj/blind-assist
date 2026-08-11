# TARO O0R ARKitScenes direct Apple SUPPORT R3 result

## Outcome

Terminal: `TARO_O0R_DIRECT_APPLE_SUPPORT_R3_COMPLETE`.

Direct sensor ownership of SUPPORT has real but partial headroom. On the exact
14 R1 extraction-lost frames, a confidence-2 AppleDepth plane produced a
physically admissible SUPPORT factor for 8 frames. This made 58 of 112 queries
support-evaluable; 20 queries across three frames were simultaneously no worse
than the R1 baseline in both support height and normal.

The route is not adopted unconditionally. Height improved more often, while
normal orientation worsened at the parent-macro level; six frames correctly
remained `UNKNOWN`. No point-clearance query became known because the bound
truth query itself failed its existing knownness gate on this selected lost
cohort.

This is a retrospective WILD_LAB factor canary. It is not a formal O0R PASS and
does not establish RGB-only, deployment, product or safety capability.

## Factor ownership and source firewall

The frozen R3 method separates factor ownership:

- SCALE: re-derive the existing R0 Apple/candidate log-scale commitment;
- SUPPORT: fit directly from registered AppleDepth with `confidence == 2` and
  Apple depth in `[0.25, 6.0] m`;
- BOUNDARY/query geometry: use the source-scaled sealed DepthART raster;
- candidate SUPPORT refit or veto: forbidden.

Candidate range participates only in the independently verified R0 scale-pair
mask. It cannot add, remove or veto Apple SUPPORT points. A mutation test
changes a valid candidate sample to 10 m and verifies that the Apple support
mask and complete plane result remain identical.

Phase A reconstructed its narrow source receipt from the bound
`upsampling.zip`, exact `lowres_wide_intrinsics.zip` member and raw
`lowres_wide.traj`. It opened only AppleDepth, confidence, intrinsics and
trajectory. R1 query records were not decoded until the source-phase completion
seal was written; compact truth and FARO were opened only in Phase B.

The refined plane was required to retain slope at most 20 degrees and final
camera height in `[0.45, 2.20] m`. Invalid values were not clipped or repaired.

## Descriptive result

| Metric | Result |
|---|---:|
| R1 lost physical frames accounted | 14/14 |
| R1 lost queries accounted | 112/112 |
| source SUPPORT frames | 8 |
| source `UNKNOWN` frames | 6 |
| post-hoc support-evaluable queries | 58 |
| height-and-normal no-regret queries | 20 |
| no-regret frames | 3 |
| height-improved queries | 38 |
| normal-improved queries | 20 |
| boundary-evaluable queries | 12 |
| known point-clearance queries | 0 |

Across the five parents with an evaluable metric, median-of-parent-medians
support-height error reduction was `+0.243541391322 m`. Support-normal error
reduction was `-0.039055106811 rad`, meaning normal accuracy worsened on the
parent-macro aggregate.

The 54 source-unavailable queries stayed unevaluable:

- `36` queries on four frames:
  `DIRECT_APPLE_SUPPORT_HEIGHT_IMPLAUSIBLE`;
- `18` queries on two frames: `SUPPORT_SLOPE_EXCEEDED`.

The zero known-query count is not evidence that direct SUPPORT cannot help a
valid query. All 58 support-evaluable records also carried
`TRUTH_QUERY_NOT_EVALUABLE` for the selected R1-lost truth cohort, so final
query error was structurally unavailable here.

## Decision and next falsifiable step

Reject the R2 candidate-refit design. Retain direct AppleDepth SUPPORT as the
only live support-recovery hypothesis, but do not apply it unconditionally.

The next bounded successor is `DIRECT_APPLE_SUPPORT_FULL_COHORT_R4`: replay the
same fixed, threshold-free R3 factor ownership across all 171 existing eval
truth frames / 1,539 queries. It must quantify full-cohort coverage and the
height-versus-normal tradeoff against both raw baseline and R1 source-anchored
extraction. Invalid source planes remain `UNKNOWN`; no selector or threshold
may be chosen from the same 16 eval parents. Only after this full-cohort map may
an Apple-only selector be fitted on the separate eight `ADAPTER_FIT` parents.

## Validation and reproducibility bindings

- implementation commit: `c4ad0429`;
- relevant focused tests: `21/21` PASS;
- summary content seal:
  `E678B53C496DD789EE12CC2EFDFB3981A89C886DAE88704D78264EE71EF39CF4`;
- source-phase completion content seal:
  `8CB1212FCB95A8401AA5ADE1B83E443E1535FE843FFE99138B89CB148359A471`;
- evidence manifest SHA-256:
  `6647C462D20DE8B1EC019D5A364F3DF3F357DBC5ADC85C381EC39C2BFB09ED6B`;
- result file SHA-256:
  `81A638486CF64F71060F422D128BD01DE68D97FDEF3771D3EB8447157B9180F0`;
- evidence root:
  `artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-support-r3`.

The manifest accounts for 47 pre-manifest files and 139,916 bytes. All file
hashes, 112 query-record seals, eight plane-record seals, six failure records,
the canonical summary replay and result binding were revalidated after the
one-shot execution.
