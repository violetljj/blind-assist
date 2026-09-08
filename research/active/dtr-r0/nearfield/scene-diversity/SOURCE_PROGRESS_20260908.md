# Equal-budget scene allocation: source engineering, 2026-09-08

Coverage1198 remains the retained model. This work implements the candidate's
sampling and geometry preparation; it supplies no new model result and does
not admit the proposed TRAIN/evaluation regions.

## Implemented comparison inputs

- `scene_diversity_plan.py` allocates the same 64 geometry quartets to two or
  four proposed TRAIN regions: 256 frames per arm, four families and four
  BODY/HEAD relations. Every region contains every family and relation.
- A shared seed17 schedule contains 2,000 batches, each with 20 original1198
  draws and three distinct new complete quartets. Each fit therefore receives
  40,000 original and 24,000 new draws, including exactly 12,000 new BODY-positive
  and 12,000 new HEAD-positive draws. Both arms share original/template indices.
- `scene_diversity_templates.py` builds 64 local quartets, 16 per family, from
  collection seed20260920 without model selection. Grounded backboards replace
  unspecified existing-facade support. Rigid placement moves camera, wearer
  and every assembly part together; world-native labels remain unverified until
  actual capture. The input library's geometry digest is
  `bb468fff8ddc3bce48460802c2bfc4823f15e2e1f99bc4fdf85ed48beacd6029`.
- Validation rejects incomplete relation sets, mismatched geometry/context,
  wrong source roles and unknown native labels. It is not a visibility-isolation
  certificate. Five template tests and three allocation tests passed.

Artifacts are under `artifacts.local/work/scene-diversity-20260908/`:
`allocation-v1/`, `templates-v1/`, and `source-snapshot-v2.json`.
The immutable schedule NPZ SHA256 is
`bc2d8ea9af353eb03caecfc13a18f62fd6efd929f216f43f155fa36e560bfa1f`.
The template library predates the explicit three-axis rotation fix; local
geometry is unchanged, while subsequent placements use the corrected code.

## Existing source work reused

The seven-region v4 scout batch completed successfully (33 frames: three in
candidate01 and five in each other region). The source snapshot checks saved
spec and payload hashes, unchanged source receipts and released process trees.
No duplicate scout batch or native-plugin build was launched by this task.

Native HLOD probe `native-source-probe-v1` under
`artifacts.local/nearfield/city-crossregion-v1-20260908/` completed PASS with
unchanged map and released processes. It exported all 925 loaded HLOD1 proxies
and 3,342 valid source identities. All sources are still nested HLOD0 proxies;
none is in that export's loaded proxy/actor set. Thus HLOD1-to-HLOD0 mapping
works, but leaf actor/component/instance closure and rendered visibility remain
UNVERIFIED. Native membership file SHA256:
`6879d75c89c4f899a089c998f3512e35e05ecd48bfe0d954790215eef816ea1e`.

Source review also found real camera issues: declared fixed ground height does
not match every floor hit; some views contain near trees or poles. Floor first
hits and geographic box separation do not prove walkability, unobstructed CLEAR
labels, or independent visible backgrounds. Several reviewed views are wide
waterfront/paved areas; they do not yet establish varied street-content strata.

## Engineering capture and remaining dependency

The first geometry canary failed before frame export because rigid placement
supplied yaw without explicit pitch/roll. The capture API requires all three.
The placement helper now supplies all axes, and the inverse-placement test also
checks this native-input requirement. Failed v1 evidence is retained; its
process tree was released. It is not a valid data point.

Corrected `geometry-canary-v2` completed PASS with 16 frames at candidate01's
paved point (-165, -1589), floor0.7m and yaw180 degrees. Native world-support
verification ran on CUDA (RTX5060 Laptop GPU); all16 BODY/HEAD labels match the
four intended relations across the four families, including four CLEAR views
with no observed support. All16 vertical floor probes hit0.7m (within floating
point tolerance). This is a local placement/label check, not a certificate for
the remaining60 quartets (240 frames) or source isolation. Floor-patch checking remains
explicitly disabled in the inherited template; vertical probes do not replace
walkability acceptance. The world verifier, capture and quartet reports are
`world-verification.json`, `receipt.json` and `quartet-admission.json` in that
capture directory. Process release is verified with no survivors.

The equal-budget two-fit experiment remains unstarted. Final poses and fixture
labels, visible-background closure against old TRAIN and new evaluation regions,
and scene-content diversity still need source acceptance before the exact
comparison is frozen. No loss, checkpoint, threshold or model-selection change
was made. This engineering dependency is not another negative algorithm result.
