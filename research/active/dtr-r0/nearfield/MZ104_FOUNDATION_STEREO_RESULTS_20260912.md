# MZ104: learned stereo runs, but far-surface errors dominate

2026-09-12. EXPLORE, consumed rendered Development. **Fixed candidate rejected;
retain the original SGBM/ToF baselines.** No model/threshold successor was run.
The GPU replay and evaluation completed; global registry metadata remains pending
the unrelated historical ASE input fingerprint failure. No default-App promotion.

## Intervention and result

Original NVlabs11-33-40 public downloads remained quota-blocked. The explicitly
declared fallback is the official NVIDIA TAO FoundationStereo small dynamic v2.0
FP32 ONNX model, not that original checkpoint. One 384x640 engineering canary
failed before output at the6GiB arena cap. The declared memory recovery uniformly
resizes RGB640x360 to512x288, restores disparity to original pixel units, and
keeps the model, geometry, ToF and two-on/two-off alert state fixed.

All576 RGB pairs completed with no training, GT input, point-span filtering or
score-based configuration changes. The full run used producer/evaluator revision
`d22eb4a1ee57f49a73cbccb47025dca913b1ab59`; the protocol and engineering amendment
precede the run. Original RGB and scene-spec identities are checked against their
capture hashes. The MZ103 gate and negative outcome remain unchanged.

| Panel / stage | SGBM + ToF TP/FP/FN | TAO + ToF TP/FP/FN |
| --- | ---: | ---: |
| MZ101 current spatial support | 219 / 43 / 5 | 221 / 132 / 3 |
| MZ102 current spatial support | 216 / 19 / 6 | 216 / 141 / 6 |
| MZ101 final alerts | 202 / 25 / 22 | 202 / 117 / 22 |
| MZ102 final alerts | 200 / 15 / 22 | 197 / 125 / 25 |

Descriptive pooling only: raw **435/62/11 becomes437/273/9**; final
**402/40/44 becomes399/242/47**, F1 .9054 to .7341. Stereo-only TAO produces
raw433/273/13 and final394/242/52; unchanged ToF rescues some misses but does
not remove stereo false support under this union rule.

Both panels fail the primary fewer-raw-FP gate. Critical raw-TP retention is
33/34 and34/34 for MZ101 thin-left/right; MZ102 thin-left/right are36/36 each,
occluded-thin40/40, but **small_head13/15 (86.7%)** fails the95% condition.
All56 baseline-detected query events remain detected, but entirely false
sessions increase **2 to39**; FP-only segments23 to62. Maximum extra paired
first-correct delay is1.00s in MZ101 and1.25s in MZ102, failing the0.25s check.
Final false duration is10.0 to60.5 query-seconds (BODY/HEAD intervals may overlap).
These delays use captured frame times and **exclude model processing latency**.

Raw changes remove40 oldFP but add251 differentFP; lose6 oldTP and add8.
Final changes remove17 oldFP but add219; lose17 oldTP and add14. Thus the small
net raw-TP gain does not conceal the substantial new errors.

## Error localization, not a rescue experiment

Independent array/hash recomputation confirms the regression. Posthoc native
comparison was performed only after the learned predictions and scores were
sealed; it is diagnostic and does not alter any prediction or admission rule.

- Final false alerts have196 current-support and46 held-only query-frames;
  the old union has21 current-support and19 held-only FP. The main new error
  therefore already exists in current spatial evidence.
- Of273 raw false query-frames, **249 have only native->4m pixels behind their
  learned in-query support** (115/132 in MZ101,134/141 in MZ102).265/273 have
  at least one such far pixel. The model pulls far surfaces into eligible
  near geometry; this is not explained solely by small-object boundary loss.
- In MZ101, overhead-clear alone contributes32 finalFP, head-bar22 and
  lateral-pole17; corresponding MZ102 counts are28,10 and19. Both textured and
  flat conditions regress: finalFP15 to54 /10 to63 on MZ101 and9 to48 /6 to77
  on MZ102. It is not limited to textureless walls.
- On pixels where both predicted and native depth are eligible, sampled
  predicted/native median ratios are .9939 and .9924. This and the verified
  focal/resize arithmetic do not suggest a uniform metric-scale factor error.
  The diagnostic is biased to mutually valid pixels: it is **not** full depth
  accuracy or proof that every preprocessing assumption is correct. The paired
  mean frame MAE is .460m/.654m, with a pronounced underestimation tail.

The [two-frame qualitative inspection](../../../../artifacts.local/work/mz104-foundation-stereo-20260912/posthoc-depth-comparison.png)
shows far background estimated as broad near surfaces while some actual thin
surfaces retain reasonable depth. Repeated procedural backgrounds are a plausible
matching ambiguity, not an independently established causal explanation.

The measured conclusion is limited to **this official adapted checkpoint,
0.8-scale configuration and unconditional point readout**. It does not reject
all learned stereo or establish performance of original11-33-40/full-resolution
inference. A later changed hypothesis could test observable correspondence
reliability before granting dense predictions spatial authority; this run did
not test such a filter, and MZ102 still rejects its previous size-based recipe.

Interpretation clarified by [MZ105](MZ105_RESIDUAL_MATCHING_RESULTS_20260912.md):
SGBM already applies uniqueness10, explicit <=1px left/right consistency and
>=8px connected-valid support; this TAO path lacks equivalent external matching
validation. The comparison replaces complete frontends, so all regression cannot
be attributed to pretrained weights alone. Deletion-only validation cannot
recover the two missing small_head raw TP (95% of15 requires15). Per-panel strict
FP gates require at least90+123=213 removed FP; MZ101 can lose at most2 TP and
MZ102 none. The existing inference cost also remains. No rescue run is implied.
The metric named false_sessions counts fully false contiguous alert segments
separately per episode and BODY/HEAD, not distinct capture scenes/sessions.

## Runtime and evidence

ONNX Runtime GPU1.30.0, RTX5060Laptop, official FP32 graph. The first-frame
profile records976 actual CUDA convolution nodes; CPU shape/control nodes are
also present. Complete replay wall time751.31s includes109.55s initialization.
Mean inference1.074s/pair; mean full pair1.112s, median1.126s, P951.182s.
Measured cost exceeds the0.25s frame budget; this is offline evaluation, not
real-time warning evidence. Sampled device-total peak is7,144,054,784bytes
(6.65GiB), not process-exclusive VRAM. Task-owned inference processes exited.

The base Python environment is unchanged. Official weights, isolated runtime,
failed/passed canaries, all576 outputs and diagnostic evidence remain under
the canonical artifacts junction for reproducibility, with no automatic rerun.

- [Protocol](MZ104_FOUNDATION_STEREO_PROTOCOL_20260912.md), [producer](mz104_foundation_infer.py), [evaluator](run_mz104_foundation_stereo.py)
- [Official model card](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationstereo)
- [Official stereo normalization contract](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/depth_estimation/stereo_depth_estimation.html)
- [Producer receipt](../../../../artifacts.local/work/mz104-foundation-stereo-20260912/frontend-v1/receipt.json), [runtime manifest](../../../../artifacts.local/work/mz104-foundation-stereo-20260912/frontend-v1/manifest.json)
- [Evaluation summary](../../../../artifacts.local/work/mz104-foundation-stereo-20260912/evaluation-v1/summary.json)
- [Independent audit](../../../../artifacts.local/work/mz104-foundation-stereo-20260912/independent-final-audit.json)
- [Far-support attribution](../../../../artifacts.local/work/mz104-foundation-stereo-20260912/posthoc-false-support-origin.json)
- [Paired-depth diagnostic](../../../../artifacts.local/work/mz104-foundation-stereo-20260912/posthoc-depth-diagnostic.json)
- [Pending registration](../../../../artifacts.local/work/mz104-foundation-stereo-20260912/pending-registration.json)

Independent verification covers578 producer output hashes,7 evaluation hashes,
1,152 source RGB hashes, frame/model/producer identities, baseline parity,
raw/final masks, event counts, critical-slice gates and runtime scope. Narrow
source compilation and diff checks passed. Registry registration returned the
existing row303 fingerprint error; no historical ledger row or validator was
changed to bypass it. Intended terminal role is NEGATIVE_CONTROL for this fixed
TAO frontend as unconditional geometric support, with a pending record preserved.
