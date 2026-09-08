# Approach target uncertainty audit

2026-09-08. Engineering diagnosis of the immutable approach-v1 capture and labels.
No new model run, labels, threshold selection, or additional capture.

## Result

All 14 active UNKNOWN target frames pass the existing render-agreement gate:
enough matching pixels and zero unexplained nearer-clone pixels. Independent
collision identity is the limiting check. There are **15 UNKNOWN rays across
14 frames**; frame 43 has two. The other 70 active target frames remain evaluable.

| Ray diagnosis | Rays | Frame indices |
| --- | ---: | --- |
| Other component closer than target, within 3 cm render tolerance | 12 | 7, 10, 21, 22, 26, 43, 45, 61, 91, 92, 93, 96 |
| No collision hit | 2 | 43, 94 |
| Other component behind isolated target | 1 | 81 |

The last ray is at pixel (323,359), on the image bottom edge: its collision is
6.506 cm behind isolated target depth. The no-hit rays are (321,207) in frame 43
and (307,157) in frame 94. Do not classify these as model misses or missing
buildings. A single unresolved sampled ray makes the current entire target/frame
UNKNOWN, regardless of the many matched rays elsewhere.

An independent intersection calculation against the declared fixture cubes maps
the other-component hits to pole feet and head-bar clamps. This is geometric
inference, not an actor-name identity receipt. The two no-hit rays do not intersect
the declared fixture cubes, consistent with a silhouette/raster discrepancy.
The evidence supports focused instance-visibility work; it does not prove that
every rendered pixel has correct component identity.

The [geometric crosscheck](../../artifacts.local/nearfield/city-field-approach-20260908/uncertainty-audit-v1/fixture-ray-crosscheck.json)
also places 14 of the 15 unresolved isolated-surface samples outside both current
3 m BODY/HEAD query volumes. Only frame 94's no-hit sample lies inside HEAD.
The current checker samples the whole isolated object, and any unresolved sample
invalidates the whole target/frame. Thus these 14 frame exclusions do not imply
14 failures throughout the near-query volume. Analytic fixture first-hit depths
match recorded collision depths within 0.000000965 m; native city surfaces are
not included in that analytic calculation. The distinction between whole-instance
and per-query reliability should be explicit in a future version, with adequate
independent in-query coverage rather than simply ignoring inconvenient rays.

## Delivered quality checks

`tools/audit_city_target_uncertainty.py` validates capture/spec/raycheck identity
against the completed label receipt, inventories active rendered targets, and
creates an overlay for every UNKNOWN target frame. Green points match, cyan
points are occluded, and red points are unresolved. Reports preserve exact pixel,
depth, component path and source hashes. Outputs must use a fresh artifact folder.

Future label verification now records individual uncertainty reasons and the
unresolved collision-ray records beside each target/frame. The reliability gate
and output masks are unchanged. Nine focused tests passed for projection, mask
gates, ray identity/instance distinctions, and reason classification.

Run from the repository root:

```powershell
python tools/audit_city_target_uncertainty.py --capture artifacts.local/nearfield/city-field-approach-20260908/capture-v1 --labels artifacts.local/nearfield/city-field-approach-20260908/labels-v1 --output artifacts.local/nearfield/city-field-approach-20260908/uncertainty-audit-v2
```

The completed [audit v1](../../artifacts.local/nearfield/city-field-approach-20260908/uncertainty-audit-v1/result.json)
contains all 14 overlays, including
[frame 81](../../artifacts.local/nearfield/city-field-approach-20260908/uncertainty-audit-v1/0081-west_intersection__thin_pole.png).

## Remaining label repair

The 14 frames remain UNKNOWN. Raising tolerances or accepting a majority of
matching rays would conceal component ambiguity: the current 3 cm render gate
can include a nearby support surface as though it were the target. A new label
version needs dense visible-instance evidence, with per-pixel exclusion for
occlusion and silhouette disagreement, independently checked against collision.
Preserve v1 and its consumed diagnostic denominators. Validate such a repair on
the existing ambiguous views plus matched clean controls before new acquisition
or model scoring. This audit is completed; identity-label repair remains open.
