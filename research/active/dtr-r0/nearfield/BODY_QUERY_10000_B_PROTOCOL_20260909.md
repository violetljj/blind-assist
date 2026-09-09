# 10000-frame expanded-source B comparison

2026-09-09 `EXPLORE`. This protocol tests whether the accepted additional 10000-frame structured source improves the fixed Body Query B alert task under the same architecture and 2000-step fit budget. It does not test deployment, natural-scene generalization, safety, or distance attribution.

## Frozen comparison

- Baseline `OLD`: the accepted expanded-B checkpoint `artifacts.local/work/body-query-expanded-b-20260909/run-v1/NEW-step2000.pt` (SHA-256 `c7aef143bcc7ac464097e95219dc5239523df7ce5716fc95926f37592f94e776`).
- New arm `NEW`: `BodyQueryModel` B initialized from the unchanged G13 tensor `artifacts.local/work/body-query-v1-20260908/model-inputs/checkpoint/decoupled_seed17.pt` (SHA-256 `0c14179486102993508dd2385a3b6f9e17d51a6d53e657c0240325e5f78d5c7b`) with seed 17, AdamW, learning rate `1e-5`, weight decay `1e-4`, batch 32, and exactly 2000 steps. Frozen BN buffers and losses remain unchanged.
- The accepted 10000-frame source is adapted to RGB-only QueryRGB cache with roles TRAIN_ONLY=5000, DEV_ONLY=2000, EVAL_ONLY=3000. The old 5000-frame source is not concatenated. Distance-pair frames are excluded from fitting and scored separately.
- DEV thresholds are selected independently for OLD and NEW at the predeclared empirical FPR rule (`<=10%`, minimum count 48) before any EVAL scoring. EVAL is never used for threshold selection.

## Acceptance gate

Run `body_query_10000_finalize.py` only after all ten source regions have PASS labels and the completed visual-review record. Require `summary.json` and `manifest.json` status PASS, 10000 frames, 2000 complete groups, 500 sites, role counts above, zero old-source RGB/XY overlap, and all recorded hashes. Retain `BACKGROUND_VISUAL_LIMITATION_RETAINED` with notes; it is not converted to CLEAR.

Then run the new source adapter `body_query_10000_data.py`. Require a PASS cache manifest and adapter-validation receipt with all RGB/label/hash, native-count, pooled-UNKNOWN, group/site/region split, and QueryRGB readback checks. The adapter is CPU I/O only and performs no inference.

## Decision metrics and stop

Compare OLD and NEW on the same accepted source: BODY/HEAD TP, FP, FN, TN and AUC on TRAIN/DEV/EVAL; complete counterfactual-group correctness; HEAD-near and HEAD-far query strata; LOW, ABOVE, LATERAL_OUT and FAR_OUT controls; per-region and per-family slices; and processing cost. Preserve all denominators and UNKNOWN cases. A final alert gain does not establish correct distance-bin attribution.

The distance-5000 source is a separate diagnostic with native accepted HEAD-positive near/far pairs. It reports paired far-score direction, reverse/tie counts, both-endpoint alerts, absolute near/far event confusion, and BODY false alerts for OLD and NEW without fitting thresholds or using native truth as model input. It has no HEAD-negative false-positive denominator.

Stop after this one fixed NEW fit, one independent analysis, and the distance diagnostic. Retain the new arm only for a measured controlled-Development gain that survives the declared errors and coverage limits. Do not add losses, pooling, depth, threshold rescue, extra seeds, or more capture automatically.
