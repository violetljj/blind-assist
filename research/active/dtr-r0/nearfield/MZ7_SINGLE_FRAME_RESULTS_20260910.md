# MZ7: paired single-frame readout diagnosis and one calibration comparator

2026-09-10, EXPLORE on consumed Development. **Keep MZ5. Positive branch
rescaling cannot recover the 49 thin-pole far opportunities because both branch
logits are negative in every case.** One DEV-fitted scale/threshold comparator
recovers two approaching HEAD_FAR bits but still misses all thin-pole positives
and increases BODY_NEAR false positives. Do not retain this calibration or reopen
temporal work. This does not prove the frozen visual representation lacks usable
local information, nor establish return attribution as the proven cause.

## Paired evidence

[Protocol](MZ7_SINGLE_FRAME_PROTOCOL_20260910.md) and
[diagnostic](mz7_paired_readout.py) reuse admitted MZ6 capture-v3 and its hashed
observations/evaluator cache. All 100 target/empty pairs have identical specified
camera, floor, sample time and non-target objects. Full four-logit target, empty
and delta rows are exported separately for RGB, ToF and fixed fusion, clean and
missing-return conditions. Frozen compact checkpoint SHA256 starts `ccbfd681`.
Recomputed CURRENT matches all 1600 stored output bits across both conditions;
logits agree within atol=1e-4, rtol=1e-5. No RGB re-extraction or retraining.

Only actual positive target bits with negative paired controls enter the table.
Each cell is a repeated-frame opportunity from one configuration, not an
independent obstacle event. Selective response means delta>0 and delta exceeds
every false query's delta in that same target frame; a second truly positive
region is not treated as a false competitor. No small-delta cutoff is fitted.

| Clean scenario/query | Opportunities | RGB hits | ToF hits | Fixed hits | RGB selective delta | ToF selective delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Approaching bar HEAD_NEAR | 9 | 9 | 9 | 9 | 9 | 8 |
| Approaching bar HEAD_FAR | 14 | 0 | 14 | 0 | 0 | 12 |
| Lateral entry BODY_FAR | 11 | 1 | 8 | 6 | 1 | 8 |
| Thin pole BODY_FAR | 25 | 0 | 0 | 0 | 0 | 0 |
| Thin pole HEAD_FAR | 24 | 0 | 0 | 0 | 8 | 2 |
| Lateral exit BODY_NEAR | 9 | 0 | 8 | 2 | 5 | 9 |

The 75 missed positive bits here comprise 22 correct-ToF overrides and 53
both-negative bits. The latter include all 49 thin-pole opportunities, three
entry bits and one exit bit. This is a new selection and denominator, independent
of the old 26/32 lost-exact error attribution.

Thin BODY_FAR RGB target-minus-empty delta has min/median/max
**-2.688/-1.200/+0.405**: 24/25 decrease. ToF increases on 21/25 but is never
selectively stronger than the false regions; median delta +0.387. Thin HEAD_FAR
increases on all 24 positives in both branches, with median deltas +2.724 RGB and
+0.671 ToF, but target logits remain strongly negative (medians -9.461/-3.310).
Thus the frame does affect the output; it does not yet produce reliable correct
body-region decisions. HEAD positive response cannot be generalized to BODY.

## One fixed calibration attempt, then stop

[Calibration code](mz7_calibration.py) fits eight positive inverse temperatures
by binary cross entropy on old MZ1 **DEV_ONLY 1000** rows. Four thresholds then
maximize DEV true positives subject to each query's unchanged MZ5 DEV false
positive budget. Parameters are written before loading MZ6 scoring labels.
No EVAL_ONLY scoring, MZ6 fitting, sequence smoothing, OR or oracle branch choice.
CPU double-precision optimization follows the declared bounded CPU protocol;
cached tiny readout inference also runs on CPU. No backbone work is performed.

DEV is the fitting set, not validation: exact912->943, TP[187,180,183,165] to
[193,189,193,189], FP[6,11,10,7] to[6,11,10,6]. That apparent gain does not transfer
under the required budget:

| Clean MZ6, 200 frames | Fixed MZ5 | Calibrated |
| --- | ---: | ---: |
| Four bits exact | 143 | 141 |
| BODY_NEAR TP /9 | 2 | 3 |
| BODY_FAR TP /36 | 6 | 6 |
| HEAD_NEAR TP /9 | 9 | 9 |
| HEAD_FAR TP /38 | 0 | 2 |
| False positives, BN/BF/HN/HF | 5/3/19/0 | 7/3/19/0 |
| Thin-pole far TP /49 | 0 | 0 |
| Target-absent controls exact /100 | 100 | 100 |

The two recovered HEAD_FAR bits still coexist with erroneous HEAD_NEAR. The
approach clip keeps all 16 wrong near activations. Exit introduces three late
BODY_NEAR false activations, partly offset by one fewer entry false activation.
The missing-return condition has the same decisions and counts. Returning
complete current packets therefore does not make this readout detect thin poles.

No upgrade, no new-placement confirmation collection, no additional fitted
candidate this round. The justified next layer to inspect is position-preserving
evidence and echo-to-query attribution, with wrong-zone correspondence as the
mechanism falsifier. Angular intersection supplies possible support only. The
present diagnostic does not identify whether pooling, learned boundary, or return
ownership is the decisive cause; that distinction requires a new bounded test.
Temporal reopening still requires complete-evidence detection, deletion loss,
then recovery from actual observable history. All 200 samples remain Development;
invalid/no-return observations are UNKNOWN, never safety clearance. No hardware,
fresh-placement, generalization or deployment claim is made.

## Durable outputs and verification

Under `artifacts.local/work/mz7-single-frame-20260910/`:

- `paired-v2/branch-logits.npz`, `paired-rows.json`, `result.json`, `receipt.json`.
- `calibration-v1/parameters.json`, `predictions.npz`, `result.json`, `receipt.json`.
- `audit.json`: scalar metric reconstruction, paired-delta and sign checks.

`paired-v1` stopped at a clip-name assertion before inference: suffix-only pairing
fixes the internal word `target` in `lateral_target_exit_target`; no data or
parameter change. No UE, worker, persistent process, port or temporary runtime
was started. Durable receipts and outputs remain; MZ5 exports are untouched.
