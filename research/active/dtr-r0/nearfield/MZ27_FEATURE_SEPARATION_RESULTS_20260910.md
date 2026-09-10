# MZ27: raw visual neighbors do not explain the remaining support errors

2026-09-10, consumed Development, zero training steps. The fixed descriptor
comparison does not support raw-feature expansion as the next error-separation
mechanism. Under the same angular coordinate and packet-matched reference pool,
packet-only neighbors classify27/29 scored high-score negative locations
correctly, raw64 does10/29, raw3x3 does8/29 and learned40 does4/29. Raw3x3 helps
some positive witnesses, so this is not a rejection of visual information.
Retain the controlled diagnostic and packet/coordinate comparator as a COMPONENT;
MZ5 baseline and MZ20 challenger remain unchanged. No task prediction was changed.

## Controlled reference comparison

The [protocol](MZ27_FEATURE_SEPARATION_PROTOCOL_20260910.md) fixes110 anchor
requests: three residual MZ26 false additions, eight highest negative-availability
bags from each of TRAIN/oldDEV/relationDEV/distanceDEV, and actual contributor
witnesses for all75 MZ20 placement far additions. All75 have a witness. These110
requests select110 distinct frame/zone/cell locations; no duplicates collapsed.
Anchors intentionally target difficult errors and recovered positives, not an
unbiased sample of task opportunities.

References use the exact7562 TRAIN frame IDs. Old index and placement row metadata
establish site/group identity, with source RGB hashes checked. Exclude same site,
same group and self; shared placement site IDs are compared across dataset names.
The200 sequence TRAIN frames have no explicit site/group mapping and are excluded,
leaving7362 identity-known candidates before per-anchor exclusions. At the same
zone/cell, select128 nearest four-vector packets (two cleaned ranges/4, two
validity bits) without labels, then require at least five references of each
native known class. No reference replacement or class balancing is performed.

Compare pooled top-five majority labels using fixed RMS Euclidean distance:
packet4; normalized64-channel raw sample; nine raw samples offset by one feature
pixel in a3x3 neighborhood (576 values); and the40 inputs of the frozen MZ26
availability MLP. The learned40 includes local16, zone mean16, coordinates4 and
packet4. This is neighbor retrieval in those descriptors, not the MLP prediction.
Native known0 means unavailable in-domain native evidence, not no obstacle/CLEAR.

## Results and coverage

92/110 anchors are scored. Eighteen lack both-class support in the fixed128:
one residual false addition, three negative tails and14 positive witnesses.
They remain unscored; no all-anchor accuracy is inferred from the scored subset.

| Scored selection | Count | Packet4 correct | Raw64 correct | Raw3x3 correct | Learned40 correct |
|---|---:|---:|---:|---:|---:|
| Residual false additions | 2 | 2 | 1 | 1 | 1 |
| High-score negative tails | 29 | 27 | 10 | 8 | 4 |
| Actual far-positive witnesses | 61 | 49 | 49 | 53 | 51 |

On negative tails, raw64 versus packet changes0 wrong to correct and17 correct
to wrong; raw3x3 changes0/19. On real witnesses, raw64 changes8/8, raw3x3 changes
8/4. Thus context helps a subset of true witnesses but does not supply the
predeclared advantage over both packet-only and learned descriptors.

For source slices containing both known classes, relationDEV balanced agreement
is0.798 packet,0.489 raw64,0.526 raw3x3,0.489 learned40 (50 scored anchors);
distanceDEV is1.000/0.533/0.642/0.583 (26 anchors). TRAIN and oldDEV selections
contain only negatives and cannot supply balanced agreement. These are correlated
diagnostic locations, not independent accuracy trials or natural-scene metrics.

| Residual event | Zone/cell | MZ26 availability logit | Reference known0/known1 | Retrieval finding |
|---|---:|---:|---:|---|
| relation473 HEAD_FAR, oblique rod | 43/3 | +2.901 | 49/79 | Packet known0; all visual descriptors known1 |
| relation1677 BODY_FAR, hanging sign | 51/43 | +2.478 | 0/128 | Insufficient class support, unscored |
| distance778 BODY_NEAR, hanging sign | 50/37 | +2.064 | 103/25 | All four descriptors known0 |

## Interpretation and next decision

The result weakens the specific hypothesis that the current failures can be
resolved by directly retaining more raw visual features under this local metric.
It does not prove raw information is absent, that a learned metric cannot work,
or that every failure has one cause. Negative anchors are selected for high MZ26
scores; learned-descriptor positive neighborhoods are therefore not independent
proof of compression failure. The reference pool is deliberately packet-matched,
so this compares conditional information within that pool, not unrestricted
retrieval or overall sensor superiority.

Packet-and-coordinate conditional support deserves a fixed full-task comparator.
It is not ready to replace learned availability: even this selected positive
subset has12/61 packet-known errors and14 more positive anchors unscored. The
relation1677 case exposes absent negative coverage in its matched pool. A successor
must report all candidate/witness coverage and actual gains/losses on the fixed
cohorts; no reclassification of missing support as CLEAR or posthoc threshold
rescue. MZ27 itself remains a completed no-fit diagnostic.

## Verification and lifecycle

Frozen descriptor inference completes in15.08s on CUDA; this excludes camera and
backbone extraction and is not device latency. Independent audit verifies31 input
hashes, all110 anchor memberships, site/group exclusions and label-blind reference
selection,47104 descriptor distances/ranks/votes, and every source/kind summary.
Independent NumPy bilinear reconstruction of30 selected descriptor frame instances
agrees within5.48e-6; reconstructed learned-vector-to-head error is at most1.43e-6.
The run receipt and independent audit both PASS.

Outputs, all128 reference identities/labels/distances, descriptors, missing-support
records and audit are retained under
`artifacts.local/work/mz27-feature-separation-20260910/run-v1/`. Processes exited;
no optimizer, capture, device session or persistent GPU allocation was created.
No independent-source, temporal, App or safety claim follows. Broad goal active.
