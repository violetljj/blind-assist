# MZ62: matched complete profile coverage

2026-09-11 EXPLORE. The definition below is finalized before registration;
registration and primary resource admission precede any fitting or inference.
The candidate schedule was prepared before this proposed run; no result is
claimed here. MZ61 source capture is a separate workstream and not an input.

MZ59's600-step mixed-source schedule presented only34 of100 awning BODY_NEAR
training positives under DROP, in40 presentations. The eligible population
is MZ55 TRAIN1600, not the1231 unique frames previously sampled/cached.
MZ48 fit has1280 eligible frames, not its1250 cached sampled uniques. At four
MZ55 frames per step and a three-profile cycle,600 steps supply only800
DROP slots; uniform full1600-frame coverage is impossible at that budget.

Hypothesis: covering each training frame under every packet profile improves
native near-body recovery and retained false-alert costs compared with
random profile assignment of the exact same material presentations. This
tests an exposure recipe, not a new loss or full architecture ceiling.

## Frozen candidate material and two-arm implementation

Candidate schedule `artifacts.local/work/mz62-exposure-design-20260911/schedule.npz`
SHA256 `9424f5e85ef985ab999541cf75f8c4725cbff5237e0245bf4fa1b5946867d672`.
Both arms fit1200 steps from the exact original MZ56 GLOBAL initialization:
11020-parameter AnchorQuery, full-raster geometry, original frozen encoder
and normalization, Adam0.001 and seed151. Use the **original MZ59 loss**:
balanced local BCE +0.25 original OPEN-max frame-query BCE +0.25 unchanged
selected-old-negative replay. Do not use MZ60's native-positive-pool loss;
MZ60 remains a negative control. No new candidate architecture or GT input.

CONTROL permutes three copies of all1600 MZ55 TRAIN IDs over4800 positions.
COVERAGE independently permutes all1600 IDs once for each of IDEAL, MERGE
and DROP. Both retain step-modulo3 profile order and four MZ55 slots per
step. Thus every frame appears exactly3 times in both arms; total geometry,
context, source and label presentation counts are identical. Construction does not read
labels/family/model outcomes. It changes profile assignment and presentation
order jointly, not independently. Per-profile label marginals are reported.

Both arms repeat the exact original600-step `shared[:, :4]` MZ48 block,
`OLD_NEG` and `query` arrays twice. Each arm presents4800 MZ48 frames,
4800 MZ55 frames and9600 selected-old-negative frames. CONTROL DROP covers
1122 distinct frames and74/100 awning positives in101 presentations;
COVERAGE covers all1600 and100/100 in100 presentations. Original600-step
MZ59/MZ60 outputs remain descriptive comparators, not matched1200 controls.
No MZ55 CALIBRATION or HELDOUT frame may enter either fit or calibration.

One cutoff vector per new arm uses the original zero-added rule on
DEV1000+MZ48cal256 DROP only. No MZ55 cutoff rows, threshold sweep, checkpoint
selection or tuned FP budget. Each candidate preserves MZ37 positives;
final output is fixed OR with unchanged OLD_NEG union. Preserve every MZ60
prediction array and old groups exactly; do not reinfer prior models.

## Predeclared retaining definition

Under DROP, COVERAGE must add more native-winning BODY_NEAR events beyond
OLD_NEG than CONTROL on MZ55 new-family held480. Its final OR must have
at least CONTROL's TP and no greater CONTROL FP on each of held640,
legacy noncal and MZ48 nonfit. All seven clauses are required for this
retaining claim. Report raw per-cohort TP/FP/FN/exact, each candidate and OR,
all3 profiles, all role/family/query/site/context splits, paired losses and
FP types, and costs versus retained OLD_NEG/MZ57 and saved59/60. Do not
hide a matched-arm success that still trails retained older comparators.
Root's pre-fit `primary-definition.json` binds this exact declaration.

Audit DROP TRAIN100 and held40 awning positives: native versus known
nonnative/UNKNOWN winners and native-winner scores below the original-rule
cutoffs. Better exposure without native recovery is a negative result for
this one-pass recipe; training gain with held costs is a transfer tradeoff.
Neither authorizes an automatic extension or proves absent information.

## Implementation and acceptance checks

`mz62_exposure_source.prepare` is CPU-only source assembly. It binds MZ60
predictions/groups and the exact candidate bytes, validates all per-frame
counts and the old replay, and preserves source labels/UNKNOWN as loss or
evaluator inputs. `FeatureStore` inherits unchanged MZ59 encoder arithmetic
under a new task guard. Actual consumed IDs are MZ48 1095 +old3533 +MZ55 1600
=6228 rows, FP32 `[6228,64,45,80]`,5,739,724,928 file bytes including header.
Previously deleted caches remain deleted. Build one temporary shared cache
in batches16, share evaluation features across both arms/profiles, then
release handles and remove only the owned scratch cache after scoring.

CPU preparation must preserve every old array, exact group and schedule
bytes, source role/known identity, and reject corrupted coverage, MZ48,
old-negative, query and profile arrays plus a wrong task owner. Synthetic
CPU tests cover the output decision/OR arithmetic and loss identity.
Before a registered fit, compare at most16 TRAIN feature rows against saved
MZ56 arithmetic at inherited tolerances and verify identical initial
parameters plus exact-zero all-missing anchors for both arms. Retain
support/winner/anchor availability/vector and geometry in new saved traces.

Budget: two1200-step fits, two cutoff vectors, one shared
seven-cohort/three-profile feature traversal for the new two heads and one
independent saved-output score. Measured MZ60 fit19.138s/600 steps suggests
about76.55s total fitting, excluding shared encoding, evaluation and I/O;
this is an extrapolation, not measured MZ62 runtime. Primary GPU must be
free of UE capture before dispatch. Preserve failures and stop at budget.

All sources/sites are previously consumed controlled Development. Local
UNKNOWN remains despite known frame queries; missing returns are not free
space. Native winner agreement does not establish causal feature use.
No calibrated VL53L8CX, natural-scene, Android latency or safety claim.
