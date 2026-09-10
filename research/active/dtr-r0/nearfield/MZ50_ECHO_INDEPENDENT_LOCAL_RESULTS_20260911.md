# MZ50: richer local learning helps; unrestricted pooling underperforms

The [fixed experiment](MZ50_ECHO_INDEPENDENT_LOCAL_20260911.md) completes one
600-step fit of a10,676-parameter spatial query head. All1,280 eligible fit
frames occur in the frozen schedule. The same fitted logits feed two readouts:
GATED requires an eligible measured return; OPEN allows every angular cell.
The richer-data/native-supervision combination yields useful recall, while
the unrestricted readout has not earned a general advantage. Both retain
every original MZ37 positive; false-alert costs prevent default promotion.

The new nonfit set contains640 held-out-site frames and384 withheld birch
frames, totaling1,024 frames with992 true event bits. Neither set enters
fitting or the fixed threshold calculation. Calibration uses the original
1,000 DEV plus256 rich frames under DROP_CLOSE. All evidence is controlled
Development on consumed source sites, not fresh blind or hardware validation.

| Input profile, nonfit1,024 frames | MZ37 TP / FP / FN | GATED TP / FP / FN | OPEN TP / FP / FN |
|---|---:|---:|---:|
| IDEAL | 565 / 31 / 427 | 677 / 34 / 315 | 566 / 31 / 426 |
| MERGE_CLOSE | 620 / 18 / 372 | 672 / 19 / 320 | 621 / 20 / 371 |
| DROP_CLOSE | 382 / 20 / 610 | 453 / 21 / 539 | 386 / 20 / 606 |

Under DROP_CLOSE, GATED adds45 true events and1 false event on the640 site
holdout, plus26 true events and0 false events on the384 birch frames. The
combined71 added true events are19 BODY_NEAR,41 BODY_FAR and11 HEAD_NEAR;
HEAD_FAR does not improve. Complete correct frames rise463to511. OPEN adds
4 true events without a new false event here. Two are not found by GATED and
have no gated candidate; both have a native local witness. This is a small
observed exception, not evidence that unrestricted pooling is generally better.

Of GATED's71 added nonfit true events,52 have the winning angular cell supported
by the actual native query label. The remaining19 are correct frame decisions
without matching winning-cell evidence; do not count them as correct local
attribution. All4 OPEN additions have native winning-cell evidence. This
distinction matters when considering learned appearance/context shortcuts.

| DROP_CLOSE addition over MZ37 | GATED TP / FP added | OPEN TP / FP added |
|---|---:|---:|
| MZ48 fit1,280 | 113 / 0 | 12 / 0 |
| MZ48 nonfit1,024 | 71 / 1 | 4 / 0 |
| Old relation2,000 DEV | 5 / 9 | 2 / 2 |
| Old distance1,000 DEV | 3 / 9 | 0 / 0 |
| Older44 rich frames | 5 / 0 | 0 / 0 |
| MZ36 all400 attempts | 0 / 1 | 0 / 0 |
| Total outside calibration, including fit | 197 / 20 | 18 / 2 |

The fixed zero-added calibration condition passes on its own1,256 frames.
It does not transfer without false-alert cost. Ideal and merged profiles also
show costs on old sources; the complete arrays and per-query/group scores
remain in the result. All400 MZ36 attempts and80UNKNOWN bits are retained.
The predeclared zero-added-error component gate fails for both readouts.

The OPEN and GATED cutoff vectors are respectively
`[2.6791,8.6855,6.7877,7.7106]` and
`[-4.1242,-0.3993,1.9052,3.8192]`, in BODY_NEAR/BODY_FAR/HEAD_NEAR/HEAD_FAR
order. OPEN has a larger raw maximum for every supported GATED event, yet
its separately required conservative calibration can suppress useful events.
The same-weight comparison isolates the readout restriction. The gain over
MZ37 combines new data, supervision and head changes; it does not isolate
dataset size as the sole cause.

The sealed CPU diagnosis identifies all four OPEN cutoff providers as old
DEV negatives, global IDs1328/1215/1331/3135. Each has no GATED candidate;
the rich calibration subset supplies no cutoff maximum. Eligible calibration
negative counts expand from GATED's`[8,26,18,54]` to
OPEN's`[1003,983,988,991]`. These are real candidates of OPEN's inference
domain, so discarding them only from calibration would conceal its false-alert
exposure. This is a spatial-separation problem, not a justification to lower
the registered thresholds. On nonfit DROP, OPEN uniquely finds2 true events;
GATED uniquely finds69 true events and1 false event. The0.61-second CPU
diagnosis preserves original output hashes and computes no new threshold.

The source's paired backgrounds also remain informative: across all1,280
pairs under DROP_CLOSE, changed decisions by query fall from MZ37's
`[45,81,69,128]` to GATED's`[39,47,58,129]`. This includes fit/calibration
frames and is a descriptive context check, not independent confirmation.

Actual CUDA fit time was5.578 seconds. New-frame feature extraction and the
three frozen comparator profiles took34.941 seconds; the complete preparation,
fit and inference pipeline took73.701 seconds on RTX5060 Laptop. CPU scoring
took0.479 seconds. Independent scalar scoring checked586,656 known decisions,
both cutoff vectors and the exact schedule; positive-retention checks pass.
The new checkpoint is109,001 bytes and predictions2,811,757 bytes. New visual
features occupy513,802,240 bytes only in RAM; the existing2.32GB detail mmap
is reused. No permanent dense duplicate or second fit was created. This single
CUDA fit retains the inherited non-bitwise grid-sample backward behavior.

The executable readout's missing-return/unknown-mask checks pass; the source
adapter independently matches a312-frame real shard and rejects incomplete or
inconsistent source bindings. A separate CPU code review found no input leakage
or split/index mismatch. Actual full-source loading requires all2,560 frames
and byte-bound source identity before initializing the GPU. The run process
exits after its fixed budget and evidence; capture processes had been released
before model execution.

Retain GATED as CHALLENGER with its recorded old-source false alerts and local
attribution gap. Preserve OPEN as the same-weight falsifier and retain its two
specific missing-candidate recoveries without promoting an unconditional
fallback. Improve spatial true/false separation and observable coverage before
another learning comparison; do not rescue these results with new thresholds.
Evidence is under`artifacts.local/work/mz50-echo-independent-local-20260911/`
in`run-v1/`, `score-v1/` and the separately saved CPU diagnosis.
