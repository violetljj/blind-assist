# Boundary-aware continuous query sampling

User-authorized new EXPLORE after identifying the old72-query integration
operator nullspace. Preserve the completed contact-boundary negative control.
Question: does spending the SAME query budget near contact boundaries improve
metric boundary learning? This tests a sampling package, not unique1200-cell
occupancy reconstruction or the cause of every old error.

Reuse frozen RGB/ToF features, train-only normalization, initial weights and
100epoch image schedule from `ba-contact-boundary-20260923`; source48layouts,
24/8/16groups,864/288/576images and all earlier evidence boundaries unchanged.
The existing saved direct/geometry fits and predictions are fixed controls;
no redundant control fit or old threshold adjustment. Exact inherited source,
feature, weights, schedule and control hashes are recorded before consumption.

For each training image only, draw ONE fixed72-query set, seed202609232+index:
36uniform queries (18perBODY/HEAD) with w in[.36,1.08],h in[.6,3];36queries in
18pairs around exact controlled-object contact boundaries. Per layer9pairs,
alternating critical-width-at-fixed-h and first-horizon-at-fixed-w, offsets
.01/.02/.05m. Accept only pairs entirely within the original training domain
and with opposite labels. Try at most8random slices per pair; unavailable
boundaries use ordinary uniform queries, with explicit counts. No manufactured
boundary or expansion into old extrapolation widths. Same generated set for
both arms and all100epochs: still72unique queries/image, not7200. Training
geometry supplies query locations/labels only; no geometry, pair receipt, identity,
relation, metadata or native depth enters the prediction function.

Subclass the old model solely to batch different query sets per image. Parameter
keys, counts and forward arithmetic are unchanged; shared-query and per-image
logit/gradient parity are tested. Same initial state,batch32,AdamW1e-3,decay1e-4,
2700updates/arm. Preserve old positive-class weight50168/12040, rather than also
changing the loss when query prevalence changes. Dev selection remains old
seen72-query BCE every10epochs and atomic maximum recall at devFPR<=5%.
Old selected states/cutoffs remain frozen. Independently report new models at
the old cutoff as a diagnostic, without selecting it as a replacement.

Reuse every old evaluation query set and exact ground truth. Former interpolation
coordinates remain untrained exact coordinates but lie in the newly continuously
sampled domain; call them fixed off-grid evaluation queries, not an unseen size
distribution. Evaluation layouts remain disjoint from this fit but the entire
source is consumed Development. Width extrapolation [.24,1.20] stays extrapolation.
Save both new checkpoints, selections and all public-input predictions before
joining evaluation targets. No evaluation geometry/query-driven training, retries,
extra seeds, loss/epoch/cutoff sweeps, architecture change or automatic distance
regression successor. Mathematical operator proof precedes fitting; a nonzero
float32 roundoff residual is not hidden as exact arithmetic equality.

Primary diagnostic: width/horizon within5cm on ALL finite true boundaries,
coverage, conditional MAE and right-censored false crossings, on train AND
evaluation layouts. Preserve query TP/FP/FN/TN,FPR,precision,Brier,pair ordering,
joint pair correctness, families,relations,UNKNOWN and original80%component gate.
Meaningful joint boundary improvement: both width and horizon within5cm gain
>=10percentage points versus corresponding old model, with right-censored
false-crossing rate increase<=5points on each axis. Retain narrower partial
gains separately. Query-cost guard: primary FPR increase<=2points and recall
loss<=3points. Bootstrap differences by16whole evaluation layouts1000times.

If train and evaluation boundaries improve, sampling is an evidenced contributor
within this package. Train-only improvement points toward remaining transfer
limitations. Neither improving weakens coarse-query sampling as the main remedy;
it does not prove sensor impossibility. Report mixed-axis outcomes rather than
forcing one of these categories. Stop after two new100epoch fits and complete
audits, result/inheritance/registry and scoped Git delivery. All payloads remain
under the canonical artifacts.local junction; no capture, services or paid worker.
