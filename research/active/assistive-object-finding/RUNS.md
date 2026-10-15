# Runs

| Date | Run | Inputs / changes | Results | Decision / evidence |
| --- | --- | --- | --- | --- |
| 2026-10-15 | object-finding-pilot-20261015 / EXPLORE; source version delivered with this row | Official MeViSv2 valid_u, 12 sources/16 queries/528 rows; DINO-base + CLIP; fixed source-grouped 3-fold learning; real KLT with detector cache; official StreamingRVOS-1B mask cache | DINO 82/327 visible; CLIP 70/327; comparative learner 81/327; posthoc full pool 85/327 (saved 69, harmed 66). Strong cached method 147/327 visible and absent output 88/105; 96 rows NOT_EVALUABLE. Adaptive vs uniform at 119 calls: +8 visible, +19 absent outputs. | COMPLETED: pipeline usable, no reliable method gain; keep direction but defer topic choice. RefEgo pixels NOT_RUN; Ego4D application submitted, agreement/approval pending. [Results](README.md); payload artifacts.local/work/assistive-object-finding-pilot-20261015/ |

GPU process wall: 321.516 s including smoke, 528/528 full inference rows, peak
allocated memory approximately 2.13 GB; no paid compute. Real CPU KLT wall 7.88 s;
learned fitting approximately 0.375 s. Candidate-pool probe reused saved features.
Data and model payload stayed below the 6 GB network cap. The pilot finished
within its 90-minute wall budget; authorization waiting does not create a new run.
Archives, source receipts, predictions, features and failures retained. Duplicate
download chunks removed with hash/path checks; task-owned model processes and
writer lock released. No full StreamingRVOS runtime or first-person video run.

Delivery checks: evaluator self-test passed (join, missing rows, censoring,
unknown FPS and native mask state); all route Python files compiled; saved
full-pool replay reproduced 85/327; 26 local documentation links and scoped
whitespace/identity/path checks passed. Whole-repository structure check retained
one unrelated concurrent failure: the existing `schemas/` surface. The new route
is included in its explicit route list and compact-current budget.
