# MZ8: thin detections recover, but attribution and false-positive gates fail

2026-09-10 EXPLORE. **Retain MZ5, not this MZ8 recipe.** One angular-return
readout fit recovers all49 thin far opportunities, but22/25 paired pole-absent
frames falsely assert BODY_FAR. Wrong visual correspondence still recovers49/49.
There is useful single-frame response, but no accepted thin-pole attribution
mechanism or upgrade at the frozen per-query false-positive budget. Temporal
remains closed; the failed gate does not warrant fresh confirmation collection.

## One declared change and data exposure

[Protocol](MZ8_ATTRIBUTION_PROTOCOL_20260910.md), [features](mz8_features.py),
[readout](mz8_attribution.py), [fit](mz8_train.py), [audit](mz8_audit.py).
Frozen RGB backbone produces64-channel18x32 maps. Within each8x8 ToF zone,
7x7 angular midpoint hypotheses carry each of two observed radial ranges.
A shared65->32->1 MLP scores local visual/range evidence; geometrically eligible
candidates are max-pooled to four queries with four biases. **2,149 trainable
readout parameters**, not total pipeline parameters. Intersection alone never
asserts occupancy. No-support remains masked even after threshold application.

One1200-step Adam fit, seed108, lr0.001, batch16 (eight old TRAIN, eight MZ6).
Old2500 TRAIN plus all200 MZ6 frames train the model; the200 are explicitly
**targeted training/regression**, not independent validation. Old1000 DEV fits
four thresholds subject to MZ5's per-query FP budgets. No EVAL scoring or new
collection. Native sampled depth supplies auxiliary0.10m echo-compatibility
labels, query truth supplies BCE; neither native data nor target IDs, poses,
paired empty images, or history enter inference. The intervention combines new
representation, geometry and supervision; comparisons cannot isolate these
components' training effects.

CUDA extraction passes original query-pooled feature parity on3700 frames:
max error2.3842e-6. The25 thin paired frozen-map responses are nonzero:
standardized token-difference RMS0.3404–0.5203. This is response, not semantic
separability. All92 MZ6 positive opportunities, including49 thin, have at least
one geometrically eligible sampled candidate also compatible with sampled native
depth. That is a finite-grid descriptive ceiling, not true measured ownership.

## Frozen threshold results

DEV threshold fitting gives exact912/1000 for both MZ5 and MZ8. TP changes
[187,180,183,165] to[170,191,199,174]; FP[6,11,10,7] to[3,11,5,7]. BODY_NEAR loses
17 true positives even in this fitting cohort; aggregate equality hides a cost.
The following200-frame results are training regression, not transfer evidence:

| MZ6 measure | MZ5 | MZ8 correct | MZ8 wrong-zone |
| --- | ---: | ---: | ---: |
| Four outputs exact /200 | 143 | 177 | 156 |
| BODY_NEAR TP /9 | 2 | 9 | 6 |
| BODY_FAR TP /36 | 6 | 35 | 29 |
| HEAD_NEAR TP /9 | 9 | 9 | 9 |
| HEAD_FAR TP /38 | 0 | 38 | 29 |
| FP, BN/BF/HN/HF | 5/3/19/0 | 0/22/0/0 | 0/25/0/0 |
| Thin far TP /49 | 0 | 49 | 49 |
| Pole-absent frames exact /25 | 25 | 3 | 0 |

MZ8 removes all16 approaching-bar wrong HEAD_NEAR activations and detects its
HEAD_FAR14/14. Overall wrong-zone correspondence reduces exact by21 and loses
3 BODY_NEAR,6 BODY_FAR and9 HEAD_FAR true positives. This supports sensitivity
to visual placement elsewhere, but **the thin-pole recovery itself does not
depend on correct correspondence under this control**. It does not pass that
predeclared mechanism check. Total FP27->22 is not a substitute for preserving
the per-query budgets: BODY_FAR3->22 and paired-control collapse are material.

With the same MZ6 artificial foreground-return deletions, correct MZ8 gives
176 exact and thin48/49 (one HEAD_FAR lost); wrong-zone remains156 and49/49.
Do not interpret this one repeated-frame loss as evidence for a temporal upgrade.
There is no deployed retention or new-placement confirmation claim.

## Why the false positives matter

The zero-fit winning-candidate audit examines all49 target far decisions and22
pole-absent false positives. Only19/49 target winning candidates are within0.10m
of their sampled native ray, although compatible alternatives exist for all49.
Only1/49 target winners has a similar echo in the paired absent packet; therefore
the positive decisions cannot simply be described as all using unchanged
background returns. Their sampled spatial attribution is nevertheless unreliable.

All22 absent false-positive winners satisfy the auxiliary0.10m distance
compatibility rule, yet **none of the actual sampled native points belongs to
BODY_FAR**. Actual forward positions are3.200022m, beyond the3.18m endpoint.
Hypothesized positions are3.144235–3.179090m; radial discrepancies only
0.021617–0.058751m. Assigning a zone-averaged return to another within-zone ray
and accepting a coarse distance tolerance crosses a body-query boundary.
Thus distance compatibility is not region-consistent ownership supervision.
This is an observed limitation of this recipe, not a claim that every echo or
every spatial fusion method fails. Do not tighten this completed fit's tolerance
post hoc, or hide these negatives by changing the query endpoint.

The next unresolved single-frame question is region-consistent attribution and
uncertainty near query boundaries while preserving the new detections; it needs
a distinct bounded test. This round stops at the failed gate, with MZ5 retained.

## Verification and durable evidence

`artifacts.local/work/mz8-attribution-20260910/cache-v5/receipt.json` identifies
the admitted3700-frame cache and source hashes. `run-v1/` contains checkpoint,
normalization, thresholds, raw support/logits, predictions, representation
diagnostic, fit history and receipts. CUDA fit plus preprocessing/scoring took
14.34s on RTX5060 Laptop; this is experiment wall time, not per-frame latency.

[Five focused geometry/causal tests](test_mz8_attribution.py) pass. Independent
pre-fit code review found no material input/label alignment or inference leakage
defect. Scalar reconstruction passes45 arm/clip metric groups, and replay from
dense maps preserves200 thin-pair output bits. Recomputed MZ5 matches all800
stored clean MZ6 bits. No native values enter model-visible cached observations.

`engineering-repairs.json` records pre-training cache failures. Source E/F
junction spellings were normalized; original dataset batch grouping was restored
after selected-role truncation changed full/short batches. No tolerance widening,
EVAL-image access or training occurred in those failed preparations;3488 cached
prefix rows were revalidated before reuse. Failed partials remain as evidence.
Task processes have exited; no UE, worker, port or persistent runtime was started.
Unrelated working-tree changes and frozen MZ5 outputs are preserved.
