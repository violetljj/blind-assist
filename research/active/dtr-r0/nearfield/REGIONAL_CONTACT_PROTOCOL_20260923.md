# Local RGB / ToF correspondence: aligned versus within-frame broken pairing

User-authorized EXPLORE. Test whether correctly located regional RGB features improve
contact-position transfer compared with the SAME regional model given deliberately
mispaired RGB. This is a new matched spatial-input comparison, not a rerun of the
consumed global CDF or spatial-query fitting packages. Regional pooling/attention
already exists in this repository; no novelty claim follows from using it here.

Reuse the consumed1728image public cache:40x8x14frozen visual grid plus64canonical
ToF tokens[range/8,valid,y0,x0,y1,x1]. Pool9bilinear subcell feature locations per
sensor footprint using the existing sample_region_features. This coarse grid and
its receptive fields limit locality. No native depth, target ids, geometry or labels
enter the feature transform. Boxes remain regional sensing footprints, never actual
surface extents or exact per-pixel depth. No global flattened RGB bypass exists.

ALIGNED: concatenate each region's40visual features with its6sensor fields.
MISALIGNED: independently derange the64visual vectors within each frame; sensor
values/flags/boxes remain fixed. One deterministic derangement from SHA256 of public
raw feature bytes plus constant seed-domain, shared across all queries and epochs;
duplicate public features yield duplicate permutations. Save all permutations.
Channel normalization uses aligned TRAIN regions only, same mean/std both arms;
no position-specific normalization or row ids reach the model. Wrong pairing changes
RGB-to-position AND RGB-to-range association, so an advantage does not isolate a
causal RGB-range interaction or establish ownership of a measured return.

Shared RegionContactModel: token46->64->64SiLU, key64->64, width/layer query3->64SiLU,
scaled dot-product attention, pooled64+mean64+condition3 ->64SiLU->3. Outputs same
q,mu,scale normalized logistic CDF as prior MetricContactModel. Query horizon enters
only analytic CDF. Invalid ranges remain explicitly missing, RGB/context retained.
Identical initial weights(seed202609231),864/288/576layout split and100epoch order,
batch32,AdamW1e-3/decay1e-4,2700updates perarm. Two fits total, no additional seed.
Supervision is existing sampled72binary queries plus exact train targets from the
completed exact-contact run: weighted binaryBCE50168/12040 + intervalNLL weight1,
same fixed5cmposition interval and right-censor3m. No new labels/geometry are needed.

Same original72query devBCE every10epochs selects checkpoint; unchanged devFPR<=5%
max-recall rule chooses cutoff. This retains its known position-selection limitation
but does not confound the two arms with different selection objectives. Save all10dev
logits; no posthoc metric selection. Seal all selected public-input predictions and
width.6q/mu/scale components before joining evaluation targets/control results.

Primary matched evidence: ALIGNED improves BOTH held actual grid Z5cm and internal
conditional-median Z5cm by>=10points over MISALIGNED; width5cm loss<=3points,
query recall loss<=3points,FPR rise<=2points,and false Z crossing rate rise<=2points.
Report every term, train/evaluation effects, original full component gate, allfinite
denominators/missing/conditionalMAE/p90, false crossings, families,UNKNOWN and width
reversals. Compare historical exact-global CDF and sampled geometry descriptively,
not as parameter-matched evidence of localization. Retain partial effects separately.
One corruption realization/seed, frozen coarse features, consumed generator only.

One paired run after unit tests and backend selection; no query/loss/head/epoch/cutoff
retry or automatic richer-feature successor. Mechanical failures preserve receipts;
do not restart a begun fit. End after independent pairing/selection/metric audit,
disposition, registry and scoped Git delivery. No capture,paid worker,App,A/LOCAL or
UNKNOWN change; trained estimates are not observed free space or safety evidence.
