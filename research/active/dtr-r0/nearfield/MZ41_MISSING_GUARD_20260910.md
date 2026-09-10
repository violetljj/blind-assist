# MZ41: query observation guard

EXPLORE, 2026-09-10. MZ40 exposed eight true MZ28 alerts removed by MZ35
under missing close returns. Test an input-only guard: when no valid ToF zone
intersects a query's projected angular bounding rectangle, preserve its existing
RGB-positive / ToF-negative MZ28 alert instead of suppressing it. All MZ37
positives and other decisions remain unchanged. A broad guard that protects
every such removal is the simple comparator. The exact rule is in
[mz41_missing_guard.py](mz41_missing_guard.py); no fitted threshold or label input.

Use only the saved MZ40 ideal/merge/drop 380-frame predictions. Preserve all
400 attempted frames / 80 UNKNOWN truth bits. Compare RGB, ToF, MZ5, MZ28 and
MZ37 with both guards. Gain criterion: recover dropout true bits without new
false bits on any arm; otherwise retain only the observed tradeoff or negative
diagnostic. Scalar action/count replay is the focused verification. One replay,
zero inference/training/capture. Bounding coverage is not proof of visibility
or clearance. This consumed Development check does not establish novelty or
physical VL53L8CX performance.

## Result and decision

The replay completed in 0.141 seconds on CPU, without model execution or fitting.
Each cell is FP / FN / exact frames, with 380 admitted frames unchanged.

| Observation | MZ37 | Query missing guard | Broad protection |
|---|---:|---:|---:|
| Ideal | 3 / 0 / 377 | 3 / 0 / 377 | 5 / 0 / 375 |
| Merged close returns | 4 / 2 / 374 | 4 / 2 / 374 | 6 / 2 / 372 |
| Missing close returns | 2 / 18 / 363 | 6 / 14 / 361 | 8 / 10 / 365 |

The query guard restores four true bits and four false bits in the missing arm;
it changes neither ideal nor merged decisions. Broad protection restores all
eight lost true bits but adds six false bits there, and adds two false bits to
each other arm. Thus missing observations explain some vulnerability but the
binary zero-return condition does not provide sufficient selection quality.
The no-added-FP gain criterion fails. Keep this as a NEGATIVE_CONTROL, with no
replacement of MZ37 or claim that query observation quality is generally useless.

Input hashes, 4,560 scalar query actions, independently counted confusion/exact
scores and preservation of all prior positive scores pass. Evidence is in
`artifacts.local/work/mz41-missing-guard-20260910/run-v1/{receipt,result}.json`
and `predictions.npz`. No task-owned process remains. Future improvements need
richer quality cues or matched training, assessed alongside dataset diversity
and compact source storage rather than extending this fixed rule sweep.
