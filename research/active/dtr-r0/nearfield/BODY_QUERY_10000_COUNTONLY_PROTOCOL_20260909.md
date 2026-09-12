# Count-authoritative 10k B readout

EXPLORE on consumed shared-site Development. User-requested single 2000-step contrast: test whether removing aggregate alert gradients releases native spatial count attribution. Baseline is original trained 10k B db39ccfcb9f3d1c8f3722711bab314ff0cf2da708e00d91f7fb263e6fec0ece0, not the readout-continuation checkpoint.

Reuse readout protocol initialization, frozen tensors/buffers, exact seed17 schedule, AdamW lr1e-5 wd1e-4, data, DEV selection and all acceptance gates. Only optimization loss changes to unweighted count CE. Alert and support remain diagnostic computations and alert inference remains derived from counts. No alert BCE is included in backward. Compare against original B and historical combined-loss run-v2. Gate: HEAD-near >=294/1788, wrong-far <564/600, HEAD_ONLY BODY FP<38, groups>=508/600, BODY TP>=1187 FP<=69, HEAD TP>=1150 FP<=35. Report the historical combined-loss wrong-far568/600 separately.

One valid fit only. No decoder successor, weight sweep, data expansion or additional seed. Preserve mechanical failures. A negative result rejects this exact count-only shared-linear recipe, not all optimization explanations. A gain is controlled Development evidence, not unique causal proof or deployment evidence.

Execution error: run-v1 was inadvertently started from combined-loss continuation SHA34a690ed...de326 after an incorrect weakening of the expected-SHA assertion. It was interrupted before evaluation; INVALID_FOR_REQUESTED_COMPARISON. Preserve partial output. Restore exact original-B SHA validation for run-v2.
