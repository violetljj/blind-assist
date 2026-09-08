# Native-depth routing: no HEAD gain, but not a general depth ceiling

2026-09-09 EXPLORE, consumed same-world320 frames. Zero training and zero model
inference. B is retained. No monocular-depth model was downloaded or fitted.

## Tested intervention

[Protocol](BODY_QUERY_DEPTH_ORACLE_PROTOCOL_20260909.md),
[implementation](body_query_depth_oracle.py). Original native arrays were hash
verified. Nearest-pixel observed axial camera-X depth was compared with lattice
camera-X depth, never Euclidean range or interpolated depth. Signed/absolute
residual arrays are retained. Full-cell ownership agrees with prior Q3 arrays.

This is a cell reliability veto on frozen B count probabilities, NOT depth
features added to the point MLP. A cell gate g scales classes1/2/3+ and transfers
removed probability to0; original count units and six-cell near semantics remain.
There are no sample-count-as-pixel-count substitutions. The seven fixed gates
use interval membership, full XYZ membership, Gaussian residual sigma0.375m,
or full membership after the predeclared depth perturbations. Cell gates take
the maximum of valid sample gates. UNKNOWN samples pass through as1; they are
never treated as known empty. This design conservatively leaves uncertain
cells unchanged and is not an upper bound over depth-conditioned models.

## EVAL results

Each head has16 positives and24 negatives. Each gate uses its own original-policy
DEV cutoff, frozen before EVAL reporting. Oracle TP at FP<=2 is descriptive only.

| Gate | BODY TP/FP | HEAD TP/FP | HEAD oracle TP at FP<=2 |
| --- | ---: | ---: | ---: |
| Original B |7/0|8/2|8|
| Native axial interval |10/0|7/2|7|
| Native full XYZ ownership |9/0|7/2|7|
| Native Gaussian residual |11/0|7/3|7|
| Full ownership, depth x0.9 |6/0|7/3|7|
| Full ownership, depth x1.1 |11/0|7/2|7|
| Full ownership, depth -0.2m |6/0|7/3|7|
| Full ownership, depth +0.2m |11/0|7/2|7|

At original B cutoffs, native full ownership gives BODY3/0 and HEAD7/2.
Reported BODY gains with reselected DEV cutoffs are privileged-information
results; they do not promote an RGB-only model. No gate reaches the proposed
HEAD11-12/16 headroom criterion, even with descriptive EVAL FP<=2 selection.

## Why this cannot reject depth-aware representations

EVAL has1261 unknown samples and217 cells containing at least one unknown
sample. All217 are HEAD cells, out of240 HEAD cells (90.4%). Such a sample makes
the entire cell gate pass through by construction. Full ownership actually vetoes
only13 HEAD cells; the Gaussian affects23. Full ownership changes10 HEAD frame
scores, while Gaussian changes11. Thus most HEAD evidence is untouched.

HEAD results are identical across the four depth perturbations. This is not
evidence of robust depth attribution: the UNKNOWN passthrough often prevents
the perturbation from changing the gate. BODY is visibly sensitive, but that
does not characterize error distributions of any learned depth model.

Consequently the supported negative conclusion is narrow: this conservative
sampled-depth cell veto does not improve HEAD. It neither proves nor disproves
that predicted depth or a different depth-conditioned representation can help.
The test does not provide a meaningful general mechanism ceiling, so do not use
its failure to declare the entire depth hypothesis dead. Conversely, native
depth generated the labels: a future perfect-ownership gate scoring well would
not independently establish that missing depth caused the RGB model failure.

No automatic frozen-monocular-prior integration follows this diagnostic. The
required evidence for that decision is still missing. Preserve original B and
the known limits of the earlier near-only signal; stop this bounded run.

## Literature scope

Primary records support depth-aware methods in their own settings:
[ISO](https://arxiv.org/abs/2407.11730) uses pretrained depth and line-of-sight
projection; [DGOcc](https://arxiv.org/abs/2504.07524) uses depth context with query
interaction; [SGR-OCC](https://arxiv.org/abs/2603.14076) describes Gaussian
soft-gating. These sources motivate a candidate mechanism, not a causal
explanation or performance guarantee for our thin near-field obstacles.

## Validation and artifacts

`artifacts.local/work/body-query-depth-oracle-20260909/run-v2/` is the final result.
It retains all residuals, masks, gates, count/near probabilities, cutoffs,
conditions, matched pairs and receipts. `validation.json` independently matches
exact ownership against Q3, verifies all native/source/result hashes, count
probability conservation, score nonincrease, exact unchanged scores for fully
passing heads, and21 split/gate frame confusions.

`run-v1/` and its source snapshot are retained. A numerical correction explicitly
preserves saved B scores when all six gates pass and prevents roundoff-induced
score increases. Recomputing the same seven comparisons as v2 did not change
the reported EVAL TP/FP outcomes; no tolerance or scientific variant was added.
All work ran as cached/sampled scalar CPU scoring and completed; no worker/GPU
allocation remains. This is not fresh confirmation or device performance.
