# MZ46: ancillary supports causally affect these fixed obstacle decisions

2026-09-11 EXPLORE. [Frozen comparison](MZ46_SCAFFOLD_CONTEXT_20260911.md).
Removing the rod's12 ancillary supports, with its target unchanged, makes
MZ5 and MZ43's ensemble miss all four positive events and introduces false
HEAD_NEAR predictions. This verifies a context dependency on these paired
states. The earlier MZ45 failure cannot be attributed solely to unseen
object shape or size: even the familiar unchanged rod now loses detections.

## Target and native evidence remain fixed

One secondary-worker capture produced all four intended states, with no
recapture or exclusions. The original four MZ44 frames supply the baseline.
Both sides have four positive event bits across three positive frames plus
one visible nonintruding frame. All32 event bits are known.

The complete retained target dictionaries and actual controlled-object
receipts match exactly: mesh, world origin, scale, rotation and bounds.
The renderer does not use the now-absent support_parent reference. Material
identity follows the unchanged dictionary, frozen assignment code and
asset hashes; the actor receipt has no independent material measurement.

An independent NumPy reconstruction matches all32 native counts/labels.
All four paired event masks are pixel-identical, and native depth differs
by0m at every event pixel. The surrounding scene depth changes where
supports were removed, as expected. Every paired case is eligible for the
predeclared context attribution. Both source subagent and root viewed all
four new RGB/native panels; root also viewed all eight paired RGB panels.

## Actual matched observations

Entries below are true positives / false positives / false negatives,
supported source to unsupported source. Each side has four positive bits
and12 negative bits; these are event counts, not frame counts.

| Input condition | MZ5 | MZ43 ensemble | MZ35 and MZ37, each |
|---|---:|---:|---:|
| IDEAL comparator | 3/0/1 -> 0/3/4 | 3/0/1 -> 0/3/4 | 3/0/1 -> 3/1/1 |
| Close returns merged | 3/0/1 -> 0/3/4 | 4/0/0 -> 0/3/4 | 3/0/1 -> 3/1/1 |
| Close returns missing | 3/0/1 -> 0/2/4 | 4/0/0 -> 0/2/4 | 3/0/1 -> 3/0/1 |

The retained local-support pipeline remains informative. MZ28 recovers all
three previously correct positives but inherits3/3/2 false bits; MZ35/MZ37
reduce those to1/1/0 without losing these positives. They still miss the
HEAD_FAR-only event. This is a retained comparator's behavior, not a newly
trained or universally promoted method. Joint FUSION also degrades; full
results include all ten fixed methods and every state, not only this table.

## Both modalities contribute to the failure

The prespecified crossed-input diagnostic keeps the target/camera fixed
and exchanges only the old/new RGB or packet input. MZ43 ensemble gives:

| Input condition | Old RGB + old ToF | Old RGB + new ToF | New RGB + old ToF | New RGB + new ToF |
|---|---:|---:|---:|---:|
| IDEAL comparator | 3/0/1 | 3/1/1 | 1/1/3 | 0/3/4 |
| Close returns merged | 4/0/0 | 3/1/1 | 3/1/1 | 0/3/4 |
| Close returns missing | 4/0/0 | 3/0/1 | 3/1/1 | 0/2/4 |

RGB alone changes from3 correct positives to0 under every packet condition.
Restricted-trained ToF alone changes from3 to2 on IDEAL,4 to1 under merging,
and4 to0 under missing returns. In the restricted conditions, restoring
either old modality restores three ensemble positives. The full failure
therefore cannot be assigned only to the RGB averaging scale; the coarse
packet/readout also depends on surrounding geometry. Candidate returns
are not verified target-surface identities. Hybrid observations are
synthetic diagnostics, not deployable simultaneous measurements.
The joint threshold crossing is consistent with the ensemble's fixed
additive mean; it does not prove a nonlinear interaction inside the model.

This intervention identifies the effect of removing these supports,
including resulting appearance, shadow and packet changes. It does not
locate a particular learned feature, prove all training data has the same
shortcut, or explain the entire shape/size transfer gap. The inputs remain
controlled proxies for restricted sensing, not a calibrated VL53L8CX model
or natural-scene/device evidence.

## Decision, verification and cost

Retain MZ46 as a NEGATIVE_CONTROL for global/context-dependent decisions,
alongside MZ45's unsupported new forms. Keep MZ35/MZ37 as strong local
comparators and retain MZ43's earlier measured restricted-training scope.
The next improvement should test local spatial attribution with paired
support/appearance variation in the data; adjusting only the final global
fusion weight would leave the observed representation dependency untested.
No follow-on fitting or capture was performed in MZ46.

All saved arrays for the four original frames replay exactly under all
three conditions. Scalar scoring recounts960 actual-observation event
decisions. An independent branch diagnostic recounts1920 decisions across
all four combinations, matches60 actual metric groups and120 paired-logit
records, and verifies exact fixed-mean arithmetic and branch-exchange
invariance from saved arrays. No model fit, threshold
selection, source extraction or permanent dense-feature cache was used.
Initialization, replay and inference took5.54s on CUDA; visual computation
took0.561s and all actual/hybrid readouts0.412s. These internal timings are
not sensor or end-to-end device latency. Saved predictions/compact features
occupy176,242bytes.

The44-file new source is12,408,301bytes raw and4,437,733bytes packaged,
64.24% smaller for transfer. All44 files passed worker byte comparison and
primary streamed hash verification. This is package size reduction, not
disk space reclaimed from retained raw/package copies. Raw evidence stays
on the worker; owned UE exited and its Zen port has no listener.

Evidence under `artifacts.local/work/mz46-scaffold-context-20260911/`:
`source-receipt.json`, `target-invariance.json`, `return-verification.json`,
`resource-release.json`, `prepared-v1/{receipt.json,native-pairs.json}`,
`inference-v1/{receipt.json,parity.json,predictions.npz}`,
`score-v1/{receipt.json,result.json,paired-logits.json,audit.json}`,
`branch-diagnostic.json` and `paired-context.png`. Frozen spec SHA256:
`89a2313daca556abac923ca7b6b484bc6f217000421e284227bad25912f4dffe`.
