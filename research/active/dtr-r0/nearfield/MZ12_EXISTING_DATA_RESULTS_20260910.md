# MZ12: existing20000-frame inventory and frozen different-placement replay

The existing data suffice to expose the next failure; no new capture was needed.
All20000 accepted frame records are accounted for. A fixed3000-frame replay
recovers57 extra far TP bits with MZ11 but introduces18 FP bits,17 on hanging
signs. Its zero-added-FP behavior on the earlier calibrated DEV does not transfer
unchanged. Keep MZ5; no refit, cutoff rescue, baseline promotion or collection.

## What the20000 frames contain

| Accepted source | Frames | TRAIN / DEV / EVAL | Sites | Complete units |
|---|---:|---|---:|---:|
| old body-query5000 | 5000 | 2500 /1000 /1500 | 250 | 1000 relation groups |
| relation expansion10000 | 10000 | 5000 /2000 /3000 | 500 | 2000 relation groups |
| distance pairs5000 | 5000 | 2500 /1000 /1500 | 500 shared with expansion | 2500 pairs |

All three final manifests are PASS with index/summary hash bindings verified.
The20000 recorded RGB hashes are unique, with no pairwise source overlap.
All40000 RGB/native paths exist. This inventory validates recorded identities
and source bindings; it does not rehash all20000 image/depth payloads. The selected
3000 RGB and3000 native files were individually rehashed during replay. Canaries,
failed attempts and incomplete retries were excluded from the accepted total.
The union has750 site identifiers, not1250 independent locations: relation and
distance use the same500 sites and share CitySample assets/backgrounds.

Old5000 contains crossbar2000 and cabinet/oblique_rod/hanging_sign1000 each;
relation10000 contains crossbar4000 and the other families2000 each. Their
CLEAR/BODY_ONLY/HEAD_ONLY/BOTH and LOW/ABOVE/LATERAL_OUT/FAR_OUT controls allow
region-specific positive and negative comparisons. Distance5000 has1250 frames
per family and only HEAD_ONLY near/far pairs. It supplies HEAD distance and BODY
false-activation checks, but no BODY-positive or HEAD_ANY-negative denominator.
Some rods are centimeter-scale, but these four-family configurations are not
the MZ6 isolated vertical3.5cm pole. Pair anchor near1.0–1.3m and far2.1–2.6m are
not measured nearest surfaces or an exhaustive sweep at the near/far boundary.

## How the data have actually been used

The current system was not trained end-to-end on all20000 frames. Its frozen
visual backbone and ContextEvidence decoder use relation TRAIN5000; historical
DEV selected parameters/thresholds and EVAL was already reported. MZ5 heads use
old TRAIN2500. MZ9 SOURCE and adapted heads add the consumed200-frame MZ6 source;
MZ11 fits its gate on old TRAIN plus those200 frames. Distance5000 already had
OLD/NEW and LOCAL/JOINT frozen inference, with no extra distance-source fit in
those recorded runs. The replay is a different-placement Development check,
not recovered independent confirmation. Absence of a new MZ head fit on a
partition does not mean the full pipeline has never used that partition.

Usage evidence is recorded in inventory-v1 with source paths/hashes, including
the original B fit, decoder, distance diagnostic, MZ1/MZ9 and MZ11 records.
Collection receipt fields saying zero inference describe collection time only;
they do not supersede later model usage.

## Frozen comparison using existing inputs

[Protocol](MZ12_EXISTING_DATA_PROTOCOL_20260910.md) selects all relation DEV2000
and all distance DEV1000 by source partition, before scoring. Relation covers100
sites/400 relation groups; distance covers the same100 sites/500 rigid pairs.
No EVAL pixels, new collection, training or threshold selection. All methods use
the same frozen visual extraction and generic two-return ToF simulation.

| Cohort / method | Four bits correct | TP BN/BF/HN/HF | FP BN/BF/HN/HF |
|---|---:|---|---|
| relation / MZ5 | 1883/2000 | 382/357/388/375 | 2/5/26/14 |
| relation / adapted MZ5 | 1882/2000 | 369/359/388/385 | 1/9/20/16 |
| relation / SOURCE | 1843/2000 | 343/378/399/376 | 10/8/24/15 |
| relation / MZ11 | **1905/2000** | **392/375/395/387** | **6/7/27/16** |
| distance / MZ5 | 920/1000 | 0/0/489/446 | 0/0/25/9 |
| distance / adapted MZ5 | 929/1000 | 0/0/494/464 | 0/3/25/12 |
| distance / SOURCE | 939/1000 | 0/0/498/458 | 7/10/0/0 |
| distance / MZ11 | **929/1000** | **0/0/498/473** | **6/3/25/9** |

MZ11 adds47 relation TP bits including30 far, plus36 distance TP bits including27
far. It preserves every MZ5-positive bit. But FP increases47->56 on relation and
34->43 on distance. Overall exact rises2803->2834/3000 while FP rises81->99; total
exactness alone would conceal this tradeoff. The predeclared favorable-transfer
criterion fails, without modifying cutoffs to compensate.

Relation complete-group correctness improves325->334/400 and completely correct
sites47->55/100. Distance complete-pair correctness improves424->433/500, but
fully correct sites remain57/100. These group counts retain correlation structure;
the two cohorts' overlapping sites cannot be counted as200 independent places.

## Where the regression occurs

Of18 additional FP bits,17 arise in hanging_sign frames:8 in relation and9 in
distance. The remaining one is relation cabinet HEAD_NEAR. Relation hanging-sign
exactness drops393->386/400; distance hanging-sign exactness drops250->242/252.
In contrast, oblique-rod exactness improves377->385/400 and223->233/250. These
are descriptive pre-existing family slices, not a post-hoc new test selection.

All9 distance additions are erroneous BODY activations on HEAD_ONLY examples
(6 BODY_NEAR,3 BODY_FAR). They expose region attribution rather than a lack of
HEAD evidence. The original wrong-range activations remain: MZ11 preserves25
HEAD_NEAR activations on far-only distance endpoints and9 HEAD_FAR activations
on near-only endpoints. SOURCE alone removes those particular wrong-range
activations but introduces17 BODY FPs and has other coverage limits; it is not
a drop-in replacement either.

This distinguishes useful transferred recall from reliable arbitration. The
data already contain the offending sign configurations and negative denominators.
The next diagnostic should exploit these existing cases before considering more
acquisition. This run does not establish that extra training alone would fix
them, and no successor fit is started.

## Verification, limitations and disposition

The16-frame old TRAIN integration check passed: maximum visual feature error
8.345e-7, packet range error0m, unchanged event/support and MZ5 prediction flags.
All frozen checkpoint/cutoff hashes match their receipts. The independent audit
checks selection, scalar composition, per-query/group/site/family metrics and
the failed transfer criterion. The3000-frame run completed on CUDA in78.912s;
this includes file hashing, decoding, observation construction and all arms,
not online frame latency.

There are184 relation frames without a valid simulated packet and zero in the
distance cohort. Unknown native pixels remain recorded (101169908 relation,
48858524 distance); absent support is not safety CLEAR. Positive truth uses
full-image native support and the unchanged MZ event convention. Generic ToF
simulation, shared assets, and existing background visual limitations remain;
no hardware, deployment or independent natural-scene claim follows.

Artifacts: `artifacts.local/work/mz12-existing-data-20260910/inventory-v1/` and
`run-v1/`. Retain MZ12 as the negative transfer control for the unchanged MZ11
gate. MZ5 and MZ9 dispositions remain unchanged. Processes exited and temporary
delivery resources are released; all durable source and result evidence remains.
