# Known-Camera-Height Ground Scale R1 Posthoc Optimization Receipt

Date: 2026-08-04

Status: `POSTHOC_CONSUMED_DEVELOPMENT_OPTIMIZATION_FROZEN`

This is deliberately not a fresh or held-out protocol. The user authorized reuse of consumed data, so R0 failure analysis compared only six bounded causal median windows: 1, 3, 5, 9, 15, and 33 prior-valid scales. Window 9 was selected because it improved coverage, clearance MAE, false-clear rate, and temporal delta MAE relative to the single-frame operator without using future frames.

The runtime operator uses only the current and previous valid scale estimates. It does not read truth at runtime. The selection itself used the same consumed synthetic Development outcomes, so its ceiling is `SAME_CONSUMED_SYNTHETIC_DEVELOPMENT_DIAGNOSTIC_ONLY`.

The R0 absolute gates are unchanged. If any remains failed, this optimization branch stops rather than searching more windows, thresholds, models, or outcome-conditioned selectors.
