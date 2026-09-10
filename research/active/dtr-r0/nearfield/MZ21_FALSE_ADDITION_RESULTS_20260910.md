# MZ21: false additions arise outside observed local support

2026-09-10 · consumed Development · saved-score and evaluator-geometry diagnosis.

All16 MZ16, all7 MZ18 and all8 MZ20 added placement false positives have
zero known valid geometrically eligible local candidates, zero actual contributors
to the mistaken query anywhere in the observed packet, and a winner with
`winning_query=false` and `winning_source=false`. MZ20 retains useful pole recovery,
but its remaining errors expose unsupported angular hypotheses receiving high
visual scores. A valid range and geometric eligibility do not establish a return
at the selected angular cell.

## Exact MZ20 errors

Frame is the zero-based cohort position; index is the selected source index.
Margins use each model's own frozen oldDEV cutoff, not a shared logit scale.

| Cohort / frame (index) | Site / family | Query | MZ16 margin | MZ18 margin | MZ20 margin |
|---|---|---|---:|---:|---:|
| distance /778 (3208) | big05_site_138 / hanging_sign | BODY_NEAR | +0.590756 | -0.370246 | +1.148852 |
| distance /819 (3249) | big05_site_110 / hanging_sign | BODY_FAR | -0.180759 | -0.170725 | +0.898589 |
| distance /939 (4939) | dense07_site_103 / hanging_sign | BODY_FAR | -0.096844 | -0.573016 | +1.346437 |
| relation /338 (1738) | big05_site_147 / hanging_sign | BODY_NEAR | +1.444920 | -1.055926 | +0.011604 |
| relation /473 (1873) | big05_site_165 / oblique_rod | HEAD_FAR | +1.053921 | +1.973155 | +1.572367 |
| relation /1488 (2888) | big05_site_159 / cabinet | HEAD_FAR | +1.368528 | +1.174690 | +0.439951 |
| relation /1538 (2938) | big05_site_154 / hanging_sign | BODY_NEAR | +0.079858 | -0.057652 | +0.027702 |
| relation /1677 (3077) | big05_site_106 / hanging_sign | BODY_FAR | +0.665136 | -1.353547 | +0.326792 |

The eight events span eight sites: seven big05 and one dense07. These are
correlated controlled output bits, not independent natural obstacle encounters.
MZ20 families are hanging_sign6, oblique_rod1 and cabinet1; MZ16's16 errors are
11/1/4 and MZ18's7 are3/2/2 in that family order. Placement denominators remain
relationDEV2000 and distanceDEV1000.

## Geometry and evidence availability

All eight MZ20 errors have geometric support and valid returns, but no known
eligible local samples. Each mistaken native full-image query also contains
exactly zero valid pixels, rather than one or two below the three-pixel event
criterion. Other query evidence remains visible:

- Five hanging-sign BODY errors have actual HEAD contributors. The closest valid
  native point is5.62-5.88cm above the BODY top boundary.
- The cabinet HEAD error's closest valid native point is6.93cm above the HEAD top.
- The oblique-rod HEAD error has actual BODY_FAR contributors. Its nearest valid
  native point is20.14cm from the HEAD box, with lateral and vertical separation.
- The remaining hanging-sign BODY_FAR error has only BODY_NEAR packet contributors;
  its nearest valid native point is1.89cm outside the target's lateral boundary.

These distances describe the closest actual valid native point to each query,
not the winning hypothesis. Saved predictions omit winner zone/echo/cell indices;
saved local scores cover known-valid candidates only. Exact winner coordinates,
their boundary distances and their immediate neighboring scores are therefore
unavailable without new inference. The diagnostic does not invent them.

The evidence motivates a separate learned evidence-availability or source-support
mechanism, or arbitration using observable evidence patterns. It does not select
an operating threshold or prove either corrective mechanism works. Native known
and contributor masks remain evaluator-only; using them as an inference veto
would leak truth. Missing visible support remains UNKNOWN, not a negative label
or CLEAR.

## Reproduction and scope

`mz21_false_addition_diagnostic.py` verifies saved receipts, reconstructs geometric
eligibility from the frozen rays and measured packets, and checks local-score
label ordering. It inspects the union of saved false additions, verifies selected
native file hashes, and independently reproduces the native query-event bits.
No learned inference, training, threshold selection, new source or protected EVAL
access occurred. NumPy CPU handles the bounded saved-score/geometry reductions.

```powershell
& E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe research/active/dtr-r0/nearfield/mz21_false_addition_diagnostic.py --root E:\linnan\linnan --output E:\linnan\linnan\artifacts.local\work\mz21-false-addition-diagnostic-20260910
```

The fresh-output-only runner completed with `receipt.json=PASS`. Detailed raw
scores, cutoffs, masks/counts, source paths, exact nearest native points and input
hashes are retained in `artifacts.local/work/mz21-false-addition-diagnostic-20260910/result.json`.
The matching receipt hashes the script and result. No process or reserved compute
remains. This diagnostic changes no model, frozen experiment or promotion claim.
