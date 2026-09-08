# City complex asset acquisition

Engineering extension of Street200V7; no training or model promotion.

`tools/make_city_obstacle_suite.py --complex-structures` generates 26 frames:
two existing camera poses (west sidewalk and plaza), each with an empty
baseline, five real-asset center/right-placement pairs, and two table/seating
combinations. These are two backgrounds in the same map, not two held-out worlds.

The existing City Sample meshes are bicycle `SM_Bicycle_A_01`, scaffold metal
frame `SM_Scaffolding_metal_N1`, barricade `SM_Barricade_A`, stone picnic-table
assembly `SM_StoneTable_Square_A`, and park bench `SM_park_bench_N01`.
The scaffold is a frame, not a complete construction-site assembly.

Native capture now optionally anchors an explicit mesh using its transformed
world bounds: front face, lateral center (or right-side minimum), and bottom.
This keeps arbitrary mesh pivots from causing floating or misplaced props.
Requested anchors and actual actor origins are recorded. Bounds are placement
helpers only; visible native depth remains label authority. Existing actor-origin
placement and the earlier 24-frame suite remain supported unchanged.

Retained evidence under `artifacts.local/nearfield/city-pcg-20260908/`:
`complex-suite-v1.json` and `complex-capture-v1/`. The saved V7 map is loaded
without saving; task objects are temporary. Source hashes and grouped baselines
are retained, and model inputs exclude object/group metadata.

The batch passes 26 native RGB/depth exports and all-visible-surface geometry
verification (CUDA, 1.253 s). Process lifecycle is 296.765 s including first-use
asset preparation. At both poses, the centered scaffold has BODY and HEAD
support; other centered props and the centered table combination have BODY
support only. All right-placement controls and both empty baselines have no
visible BODY/HEAD support within the 3 m query. These are observed geometry
results, not class-derived labels or model predictions. Source map integrity
and process release pass. New assets were not transferred to the worker in this
batch; the dependency manifest is retained for a subsequent incremental transfer.

Inspected previews show the scaffold's crossbars and open structure, picnic
table with integrated seating, and separate park bench. Bicycle and barricade
are viewed nearly end-on at the selected orientations. This provides thin
structure diagnostics, not complete orientation coverage.

The proposed online RGB augmentation is complementary future training work,
not additional independent geometry samples or an implemented training result.
Preserve clean originals and group-disjoint splits. Photometric degradation
does not move physical geometry but can erase visible thin structures; strong
degradation cannot automatically inherit an assertion of visible evidence.
Do not use two-dimensional scaling to create distance labels or repeatedly tune
against the consumed Willow result. No augmentation training run was launched.
