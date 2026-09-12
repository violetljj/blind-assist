# MZ106: short-history geometric contradiction fails useful retention

2026-09-12. EXPLORE,576 consumed rendered MZ101/MZ102 frames. **Retain unchanged
SGBM+ToF. Stop this fixed temporal verifier; no integration or automatic
successor.** The experiment executed successfully, but task gates failed.

## Fixed change and result

The [protocol](MZ106_TEMPORAL_GEOMETRY_PROTOCOL_20260912.md) adds one previous
left RGB image, pyramidal LK point tracking and relative camera pose. A stereo
point is removed only when tracked motion contradicts its near projection and
agrees with the4m..infinity ray segment. No history, <0.10m baseline, failed
tracking or inconclusive geometry leaves original support unchanged, with
verification UNKNOWN. Independent ToF support is always preserved. This is a
contradiction-only gate, not a requirement that every obstacle already have
history. No future image, native depth or actor label enters the verifier.

The identical gate runs with RGB-derived ORB/stereo/PnP temporal pose and, as
a separate diagnostic control, ideal simulator temporal pose. The RGB arm still
inherits the baseline's known camera/body orientation for spatial readout; only
its added relative temporal pose is estimated from RGB. Ideal poses never fill
invalid RGB estimates. No threshold or window was selected on these outcomes.

| Panel / stage | Baseline TP/FP/FN | Ideal temporal pose | RGB temporal pose |
| --- | ---: | ---: | ---: |
| MZ101 raw support | 219/43/5 | 218/43/6 | 217/41/7 |
| MZ102 raw support | 216/19/6 | 216/18/6 | 209/17/13 |
| MZ101 final alerts | 202/25/22 | 202/25/22 | 197/24/27 |
| MZ102 final alerts | 200/15/22 | 199/15/23 | 191/15/31 |

Descriptive pooling: raw435/62/11 becomes434/61/12 with ideal pose, or
426/58/20 with RGB pose. Final402/40/44 becomes401/40/45 with ideal pose,
or388/39/58 with RGB pose. These are strict deletion subsets: no new TP or FP.

- RGB pose removes4 raw FP but loses9 raw TP. MZ102 **small_head retains8/15**,
  failing95%; MZ101 thin_right retains33/34. Other present critical slices
  retain all their baseline raw TP. Final FP falls by only1 while14 TP are lost.
- Ideal pose removes1 raw FP and loses1 raw TP. It removes **zero final FP**
  and loses1 final TP. All critical slices retain their original raw TP.
- All56 baseline-detected events remain detected in both arms. Maximum extra
  first-correct delay is1.25s/1.75s on MZ101/MZ102 for RGB pose, and0/.25s for
  ideal pose. Captured-frame delays exclude processing cost.
- Fully false contiguous final alert segments remain2 in all arms. FP-only
  final segments23 become22 for RGB pose and remain23 for ideal pose.
  The historical metric called false_sessions is not distinct scene count.
- Raw no-support/UNKNOWN query-frames655 become668 with RGB pose or657 with
  ideal pose. This differs from verification-unavailable support that passes
  through unchanged; the latter is not certified correct or safe.

## Actual opportunity and pose limits

There are528 same-episode previous-frame pairs,48 episode starts.396 pairs
have real camera translation>=0.10m;132 have zero translation. MZ101's static
sequences advance0.175m per frame; MZ102's static cases use their declared
variable forward speed. The existing cross_hit/cross_miss/receding families
have no camera translation, although MZ102 includes camera pitch changes.
Their final predictions remain unchanged in both arms; no dynamic-scene gain
is established. No runtime family label is used to decide applicability.

The RGB-only pose producer was sealed before ideal camera comparison.495/528
pairs meet fixed internal pose validity;363 meet the verifier's >=0.10m
estimated-baseline condition. Internal validity is not an accuracy guarantee:
valid estimates have translation error median3.16cm, P958.75cm, max15.28cm;
rotation error median0.167deg, P950.495deg, max1.097deg.147 valid estimates exceed
5cm translation error. A separate pose-audit2cm opportunity statistic is not
the verifier's10cm gate and must not be used as its coverage count.

| Coverage across both panels | Ideal pose | RGB pose |
| --- | ---: | ---: |
| Candidate stereo query-support pixels | 1,464,519 | 1,464,519 |
| Passed forward/backward RGB tracking | 464,793 | 464,793 |
| Tracking + baseline + projected near/far separation usable | 221,427 | 260,533 |
| Rejected stereo pixels | 19,531 | 74,673 |
| Baseline raw TP queries with any usable pixel | 233/435 | 273/435 |
| Baseline raw FP queries with any usable pixel | 19/62 | 23/62 |

Thus even ideal pose leaves43/62 baseline false queries without any eligible
temporal contradiction check under this fixed gate. Repeated frames, weak
parallax or unavailable tracking do not independently refute near support.
Deleting many pixels does not imply removing a false query: the original
readout still accepts any retained support. RGB pose increases rejection,
including harmful true-support removals; it does not improve the tradeoff.
The ideal-pose result also fails, so pose error alone is not the full explanation.
This does not reject all temporal/multiview methods, wider trajectories or
different correspondence representations; none was tested here.

Independent native-depth attribution distinguishes lost task TP from deleted
true surface evidence. The ideal arm's one lost raw TP is
MZ101`head_turn_flat_11/HEAD`, originally supported entirely by25 erroneous
native-far pixels. Removing it corrects those pixels but cannot satisfy the
frozen task-retention criterion. Its one removed FP is
MZ102`thin_left_flat_02/HEAD` with14 native-far support pixels. Ideal geometry
therefore identifies some wrong support, but establishes no final alert benefit.

The RGB arm additionally loses MZ101`thin_right_textured_05/HEAD` and seven
consecutive MZ102`small_head_textured_03..09/HEAD` queries. Across its nine lost
raw TP, deleted support includes **3,324 native-in-query pixels** and850
native-far pixels. Its four removed FP delete51 native-far and324 near-outside
pixels. Thus the RGB arm's retention failure includes substantial actual small
obstacle evidence loss, not just removal of coincidental false-support TP.

## Verification, cost and disposition

Three focused tests check physical translation and coordinate inversion,
known near/far reprojection, unavailable geometry, and independently generated
optical-flow direction/cycle. A separate review builds camera axes directly
from capture.basis for all528 pairs; the inverse-transform discrepancy is at
most5.55e-17. Source boundary, relative-pose inversion and ray-segment checks pass.
All baseline support and final predictions match original caches.
The [independent reconstruction audit](../../../../artifacts.local/work/mz106-temporal-geometry-20260912/pose-audit/verifier-audit-summary.json)
verifies581 sealed hashes and rebuilds all576 frames in three arms directly
from original depths and saved rejection masks, recomputing independent ToF
support and final state. It confirms small_head8/15 and the native attribution
above. Audit sources and details are retained beside that receipt.

The pose producer uses OpenCV CPU ORB/PnP, mean21.7ms/frame and P9528.5ms.
The verifier uses CPU LK because this OpenCV build has no CUDA SparsePyrLK
backend. Mean complete verifier frame43.8ms, P95149.6ms includes cached I/O and
both pose arms. It excludes original SGBM computation and pose estimation;
adding these separately measured averages is not an integrated latency test.
No device or real-time warning claim follows from these cached timings.

Intended terminal: `MZ106_TEMPORAL_CONTRADICTION_RETENTION_FAILED`,
NEGATIVE_CONTROL for the fixed one-past LK near/far contradiction gate with
these pose sources and thresholds. The registry still rejects historical
`experiments/index.jsonl:303`input fingerprint; pending structured metadata is
preserved without editing the old ledger. All task processes have exited.
Source RGB/depth, pose estimates, masks, failure attribution and receipts remain
under the canonical artifact junction for reproducibility. No fresh collection,
pose refit, threshold change, model retry or successor experiment was launched.

Evidence: [summary](../../../../artifacts.local/work/mz106-temporal-geometry-20260912/evaluation-v1/summary.json),
[sealed predictions](../../../../artifacts.local/work/mz106-temporal-geometry-20260912/verifier-v1/seal.json),
[coverage](../../../../artifacts.local/work/mz106-temporal-geometry-20260912/verifier-v1/pixel-summary.json),
[pose estimates](../../../../artifacts.local/work/mz106-temporal-geometry-20260912/pose-audit/estimates.json),
[pose accuracy audit](../../../../artifacts.local/work/mz106-temporal-geometry-20260912/pose-audit/evaluator-summary.json),
[pose producer](../../../../artifacts.local/work/mz106-temporal-geometry-20260912/pose-audit/pose_audit.py),
[verifier](mz106_temporal_geometry.py), [tests](test_mz106_temporal_geometry.py).
