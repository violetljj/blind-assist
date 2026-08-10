# TARO O0R ARKitScenes Apple scale source canary R0 result

## Outcome

Terminal: `TARO_O0R_APPLE_SCALE_SOURCE_CANARY_COMPLETE`.

The source-visible, zero-parameter AppleDepth anchor recovered almost all of the
FARO oracle scale correction on the locked ARKitScenes landscape cohort. This
is the first TARO result in this route that turns a truth-only scale observation
into an executable source-side algorithm:

```text
log_scale = median(log(AppleDepth_m / sealed_DepthART_m))
```

The frozen mask used registered Apple pixel centres, `confidence == 2`, the
shared `[0.25, 6.0] m` range, and at least 256 valid pairs.

## Truth firewall and execution

- all `239/239` source-scale records were evaluable;
- all 239 records and the source-phase completion receipt were sealed before
  any existing FARO oracle record was opened;
- Phase A opened only `lowres_depth` and `confidence` members; FARO/RGB member
  opens, truth commitments, oracle reads, GPU inference, training and network
  requests were all zero;
- Phase B joined `1,494` sealed query oracle records covering 166 frames and all
  16 eval parents;
- elapsed time was 26.282 seconds; the new evidence root is about 475 KiB.

## Descriptive result

Aggregation is query median within physical frame, then frame median within
parent, then median across parents. No threshold or PASS/FAIL decision was
applied.

| Metric | Parent-macro value |
|---|---:|
| zero-correction absolute log error | 0.304976534252 |
| Apple-anchor absolute log error | 0.015609009266 |
| absolute log-error reduction | 0.291272458086 |
| Apple minus FARO signed log error | -0.007823366900 |
| oracle log metric scale | -0.304976534252 |
| Apple log metric scale | -0.314184955736 |

The remaining parent-macro absolute log error corresponds to about 1.57%
multiplicative scale error. All 16 parents and 163/166 paired physical frames
improved over the fixed metric-zero baseline. Three frames worsened; the main
outlier was `44796438:675.404`, where the source anchor applied `+0.125946` log
scale while the oracle was near zero, producing `0.143003` absolute log error.
This outlier motivates a source-only reliability/abstention diagnostic; it does
not negate the cohort-level result.

## Decision

Advance `SOURCE_ANCHORED_FACTOR_INJECTION_CANARY_R1`: apply the sealed Apple
scale before SUPPORT/BOUNDARY extraction and measure query effects while
preserving `UNKNOWN`. Include a source-only scale reliability signal, but do not
choose a deployment threshold from these same 16 eval parents.

This result is retrospective ARKitScenes WILD_LAB evidence. It does not repair
the R3 complete-query truth admission failure, authorize formal O0R PASS, prove
RGB-only operation, or support deployment/product/safety claims. The algorithm
requires AppleDepth-class metric depth at inference time.

## Reproducibility bindings

- implementation commit: `72d5bc22427d8a8bffcacbf582c646280ec8aacd`
- result SHA-256: `550A748F7FE2CA574146A4A2B09CD4AE017902839CD021FE4C768F034882BC51`
- summary SHA-256: `8A104F7D8795D0AA61F29BCF9A489D56BE63241E98486F12B6C1BA0C51DDA887`
- source-phase completion SHA-256: `5D3E881123A2918FF575CB319B8E41498B8CB49CAF6F6B6707A68D486686973F`
- manifest SHA-256: `A4DF67E029B80427D09E2FD269377703C6B17C7A0817DA2DDD9526526ACFDF14`
- summary content seal: `754FFBD166481029630A01026FF51D23C2CAE299126DD83D3126DC0DDB42367C`
- evidence root: `artifacts.local/evidence/taro/o0r-arkitscenes-candidate-scale-r0`
