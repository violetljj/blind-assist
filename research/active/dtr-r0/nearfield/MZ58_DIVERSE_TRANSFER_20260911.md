# MZ58: frozen-method transfer to the MZ55 diverse source

2026-09-11 EXPLORE preparation. Execution requires root registration, explicit
GPU GO, and the complete admitted MZ55 source index. No run has been performed
by preparing this brief or implementation.

MZ57's post hoc composition improved the already consumed MZ48 nonfit DROP
count from 463 TP / 20 FP to 487 / 20, with old noncalibration FP increasing
from 45 to 48. Those observations motivated this transfer check. They do not
show that adding diverse source data improved a model: MZ55 is not used for
training, selection or calibration in this experiment.

Use every frame in the sealed `mz55-diverse-source-v1` index at
`work/mz55-diverse-mesh-source-20260911/source-index.json`: exactly 2560 unique
frames, 320 at each of eight sites, four families, four relations, two ranges,
five settings and two support contexts per site. Preserve all 1280 pairs.
The roles TRAIN_CANDIDATE 1600, CALIBRATION 320 and HELDOUT_SITE 640 are
descriptive partitions here; all are evaluated, none are fitted or used to
change thresholds. Sites, source admission and visuals are consumed controlled
Development. Intended relation and range are grouping information, not truth;
use the admitted native-derived actual query labels and retain UNKNOWN.

Freeze ten methods before the source is scored:

- MZ37, using the existing FrozenModels observable-input implementation.
- MZ51 OLD_NEG OPEN and GATED, its completed OLD_NEG checkpoint and its two
  existing DROP cutoffs; retain their boolean OR as OLD_NEG/UNION.
- MZ56 LOCAL_ONLY, GLOBAL_ANCHOR and GLOBAL_SUPPRESSED from completed run-v2.
  The suppressed readout uses the same GLOBAL weights and original GLOBAL
  cutoff, with only its global vector set to zero.
- All three existing MZ57 unions: OLD_NEG/UNION OR each corresponding MZ56
  candidate. Preserve every constituent; do not select a winning union.

Evaluate IDEAL, MERGE_CLOSE and DROP_CLOSE observed-packet profiles. No fit,
optimizer, new cutoff, cutoff recomputation on MZ55, threshold search, source
filtering, epoch extension or old-cohort replay is allowed. Original model
weights, cutoff vectors, normalization and source schemas are SHA bound.
Reuse `FrozenModels.visual/predict`, `mz50_train.infer`, `SpatialQuery` and
`AnchorQuery.inspect` rather than reimplementing frozen computations. Strictly
load and compare every checkpoint parameter and buffer. The only new routing
combines those existing functions; no extra TRAIN parity replay is planned.
If an observed implementation defect requires such a parity check, limit it
to 16 original TRAIN frames and declare it before any transfer scoring.

The neural predictor receives only RGB-derived features, observed ranges and
validity, and fixed calibration. Frame identity routes outputs but is not a
neural input. Family/site/role/setting/context/intent, native depth, native cell
counts, known masks and truth remain evaluator-only. Check the compact RGB
archive's observable calibration declaration: original 640x360 RGB, HFOV100,
eye height 1.7m and the unchanged 45-degree sensor field.

Stream fixed batches of 16. Encode each of the required existing views once
per frame: global RGB resized to 256x144 with BOX, the original 224x224 crop,
and native 640x360 full RGB. Share those features across all methods and three
profiles. No permanent dense file or previously deleted dense cache is used.
Close each decoded image and every compact archive handle; record actual RGB
bytes, encoder/view counts, timings, GPU identity and process release.

Primary descriptive utility checks are the three fixed MZ57 unions versus
OLD_NEG/UNION on all 2560 frames under DROP: TP gained > 0 and FP added = 0.
Report all three checks, not a selected winner. Also report heldout-site 640,
every role/family/site/relation/range/setting/context, all queries and profiles,
TP/FP/FN/TN, UNKNOWN and exact frames, and paired gained/lost events. A positive
check retains a fixed challenger for this transfer; a negative check rejects
the corresponding no-extra-FP transfer claim without threshold rescue.

Independently recompose every candidate from saved raw/support and its frozen
cutoff, every union with boolean OR, and scalar-recount truth/known counts.
Preserve all MZ37 positive decisions. Read only MZ55's existing compact angular
and full-frame native count labels for winning-cell attribution; never reopen
raw native depth. For each union's events gained over OLD_NEG/UNION, report
native winner, full-frame sensor coverage, and absence of any GATED candidate.
Compare GLOBAL with GLOBAL_SUPPRESSED to distinguish active global-context use
from changed learned weights. No inherited MZ37 positive is attributed to a
new path unless that path adds it under its original cutoff.

Budget: one streamed fixed-weight inference pass, four existing cutoff vectors
loaded unchanged, one independent CPU score. Save failures and immutable
receipts. Stop after scoped result and delivery; MZ55 source production is
separate. This check cannot establish that source expansion improved training,
physical sensor calibration, natural-scene performance, clearance or App safety.
