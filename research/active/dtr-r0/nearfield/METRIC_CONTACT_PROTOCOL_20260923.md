# Direct first-contact CDF pilot

User-authorized EXPLORE, consumed same-generator Development. Hypothesis: replace
the sampled geometry intensity integration with a direct width/layer-conditioned
first-contact CDF to reduce axial boundary errors and false crossings.
The saved boundary-sampled geometry model is the primary frozen control.

Reuse exactly its 864x72 sampled queries and binary labels, frozen4864D features,
train normalization, 864/288/576 layout split, shared trunk initialization,
100epoch batch32 schedule, AdamW1e-3/decay1e-4 and positive weight50168/12040.
No geometry files or extra exact-distance supervision are needed. New head has
131->128->64->3 dimensions, SiLU, seeded202609231; parameter counts are disclosed.
Only width and layer condition the head; horizon enters the analytic CDF.
Outputs q=sigmoid(a), mu=.3+2.7sigmoid(b), scale=.01+softplus(c),
p(h)=q*sigmoid((h-mu)/scale)/sigmoid((3-mu)/scale).
q means modeled contact within3m, not measured occupancy or validated clear space.
Binary current-status weighted BCE is used; this is not exact-distance regression.
The location parameter alone is not a calibrated boundary. Width monotonicity
is not enforced and reversals must be reported. Horizon monotonicity is structural.

One fit,100epochs/2700updates; every10epochs save dev logits and choose lowest
unweighted original72query devBCE, then original max-recall devFPR<=5% cutoff.
Save predictions for all legacy query sets and continuous horizon crossings before
loading evaluation targets/controls. No loss,seed,cutoff,epoch or head rescue.
Mechanical pre-fit repairs are allowed with receipts; do not restart a begun fit.
Preserve source/input/output hashes and original negative controls.

Primary: inherited grid horizon within5cm improves>=10points over sampled geometry,
right-censored horizon false crossings do not increase, query recall loss<=3points,
FPR increase<=2points, and width within5cm loss<=3points. Report every term plus
original full component gate, train/evaluation gap, conditional MAE,p90 error,
missing crossings, false crossings, subgroups, paired16-layout bootstrap and UNKNOWN.
Continuous crossing is a separately labeled diagnostic, never substituted for grid
headline. q/scale/location identifiability is limited by sampled binary observations.
All data are previously consumed Development, not fresh confirmation, real bodies,
event-alert timing, hardware or safety. No App/A/LOCAL/UNKNOWN change.

Use governed research-ue with exact asset inputs, GPU backend probe, no new capture
or paid service. End after one fit and independent audit, disposition and scoped
Git delivery. Preserve partial gains even when joint criteria fail.
