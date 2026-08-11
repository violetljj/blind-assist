# TARO O0R ARKitScenes Apple-seeded support recovery R2 result

## Outcome

Terminal: `TARO_O0R_APPLE_SEEDED_SUPPORT_RECOVERY_R2_COMPLETE`.

The source-scaled DepthART candidate must not refit or veto the AppleDepth
metric support plane. On the exact 14 physical frames / 112 queries where R1
lost extraction, the two-stage Apple-seeded candidate refit recovered only one
frame and made only two queries post-hoc evaluable. Neither query satisfied the
frozen height-and-normal no-regret comparison against the R1 baseline.

The candidate-refit method is therefore rejected. This is an algorithmic
negative result, not an execution failure. Every other query remained
unevaluable/`UNKNOWN`.

## Frozen method and execution

Phase A used AppleDepth, confidence, the R0 source scale, sealed DepthART
candidate and camera metadata to:

1. derive an AppleDepth metric support seed;
2. select candidate points around that seed;
3. refit the support plane on the source-scaled candidate;
4. require candidate support count/fraction, slope, height and normal
   consistency before sealing a source decision.

Only after all 14 source decisions were sealed did Phase B open FARO/query
truth for retrospective scoring. Training, GPU inference, network requests,
formal reducer execution and threshold selection were all zero.

## Descriptive result

- source-recovered frames: `1/14`;
- source-failure frames: `13/14`;
- post-hoc evaluable queries: `2/112`;
- height-and-normal no-regret queries: `0/112`;
- boundary-evaluable queries: `0/112`;
- known point-clearance queries: `0/112`.

On the sole evaluable frame, median support-height error improved by
`0.287995051826 m`, but support-normal error worsened by
`0.104267922983 rad`. The source failure distribution was:

| Failure | Queries |
|---|---:|
| `APPLE_SEEDED_SUPPORT_SLOPE_EXCEEDED` | 63 |
| `APPLE_SEEDED_SUPPORT_HEIGHT_DISAGREEMENT` | 20 |
| `SUPPORT_SLOPE_EXCEEDED` | 18 |
| `APPLE_SEEDED_SUPPORT_POINTS_INSUFFICIENT` | 9 |

These counts show that the monocular candidate cannot reliably inherit a
sensor-derived metric support seed through the current hard plane refit.

## Evidence-boundary note

R2 did not hydrate FARO arrays or read query fields during Phase A. It did,
however, parse the inline compact-truth package to extract its source metadata
receipt. Therefore R2 is not described as a strict byte/package-level
source-only execution. The R3 successor closes this seam by reconstructing a
narrow receipt directly from the raw AppleDepth/confidence, exact `.pincam`
member and raw trajectory, without opening compact truth in Phase A.

## Decision

Reject candidate refit/veto. Preserve the R0 Apple scale estimator, but give
SUPPORT ownership directly to the registered AppleDepth plane. Keep the
source-scaled DepthART raster only for dense boundary/query geometry. Any
invalid sensor plane remains `UNKNOWN`.

## Reproducibility bindings

- implementation commit: `9015ba22`;
- summary content seal:
  `348FF98ED1EA1532432D7F1827AC0F5BB6CB66E249C9E1963DCCE4DE5E636A33`;
- evidence manifest SHA-256:
  `13A441BC4F80C3E58B86A3F4422D6C1AEC2FF98E4D073134E6359354BC1B8653`;
- result file SHA-256:
  `4379E99CBBDB4C176BEC563B56E532ED4E6F60627D7B6CF987E7C4ADA49E5A33`;
- evidence root:
  `artifacts.local/evidence/taro/o0r-arkitscenes-apple-support-seed-r2`.

The manifest accounts for 33 pre-manifest files and 96,513 bytes; every file
size/hash and the result-to-summary binding were revalidated after execution.
