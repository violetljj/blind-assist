# MZ77: fixed-model approach response in a bounded simulated sequence

EXPLORE, simulation only, zero fitting or cutoff selection. The user authorized
the next small continuous approach check after MZ76 replay diagnosis. The
hypothesis is that original MZ70 DIVERSE's static simulated gain over OLD_NEG
also supplies earlier and more persistent BODY/HEAD warnings while approaching
obstacles. This changes the source trajectory, not model weights or alert logic.
No smoothing or history-based repair is introduced.

## Source and fixed limits

Four40-frame clips,160 total: a suspended horizontal head bar, thin vertical
pole, wall, and shared no-added-target control. Static cube geometry, frozen
Willow scene. Camera640x360, HFoV100degrees, level, eye1.7m above floor0.12m;
camera x=26+0.1*i, y=0, z=1.82. Nominal10Hz corresponds to1m/s; sampled positions
are rendered and settled individually, not live walking or measured timing.
The camera stops before contact. Target transforms and control path are fixed
in the source spec before rendering or new model outputs. Native geometry,
readiness and representative visual checks must pass before model use. Preserve
failed source attempts; no geometry rescue based on model outcomes or expansion.

## Observations and comparators

The sensor simulator alone consumes native axial depth and emits the unchanged
generic8x8 two-return geometric packet. Predictor sees RGB, ranges/validity and
fixed calibration only; no native depth, target masks/IDs, world pose, distance
to target, or evaluator labels. Raw pixel zero/nonfinite remains UNKNOWN.

Two paired inputs use the same RGB: CLEAN and CENTER_GAP. CENTER_GAP invalidates
both returns in zone columns2..5 (all8 rows) at clip indices20..22 and30..32,
identically across every clip. These six missing samples are an artificial
partial-field outage stress, not calibrated physical sensor behavior. All other
packets are byte-identical. Report actual deleted valid slots and naturally
zero-valid packets; if no valid returns are removed, that pressure is not
evaluable. No past/future imputation or ALL_INVALID training is introduced.

Freeze MZ70 DIVERSE checkpoint and original cuts, OLD_NEG/MZ37, and the existing
learned RGB, ToF and fixed0.5-logit MZ5 branches. All consume identical observations.
Use explicit contiguous NCHW for the full encoder as verified in the MZ76
diagnosis; keep the baseline crop/low-resolution encoder paths unchanged. Bind
all model hashes and check16 prior HELD rows within the original2e-5/1e-6
tolerance before evaluating the new source. All candidate and union arithmetic
remains original. No threshold, source or winning-model selection after outputs.

## Evaluation and decision

After predictions are sealed, derive full-frame native BODY_NEAR/BODY_FAR/
HEAD_NEAR/HEAD_FAR pixel-support labels (>=3 valid pixels), and separately derive
target-only visible support by native depth agreement with the declared target.
No-positive-support is not certified clear space. Report query label knownness,
invalid native pixels and measurement availability independently.

For each clip and BODY_ANY/HEAD_ANY as well as four range queries, report first
eligible sample, first true detection, nominal delay, target-front distance at
first true detection, positive retention, missed episodes, alert on/off changes,
and longest silent run during positive support. Count target-absent control
activations separately. A wrong near alert during a far episode is a range error;
BODY/HEAD union metrics avoid mislabelling it as total absence of warning.
An alert already active before onset is recorded as pre-existing, not a new
correct onset. Distance divided by1m/s is descriptive time to the target plane
if motion continues, not measured time-to-contact or end-to-end latency.

During each gap window report lost/gained true and false bits versus CLEAN,
head near misses, whether a previously active warning disappears, and samples
to recovery after observed packets return. Do not count persistent silence as
successful recovery. Clips end before contact; right-censored missing recovery
and episode boundaries remain explicit. No release claim follows from clips
without positive-to-negative transitions.

Retain descriptive gains only if first true union warning or positive retention
improves without extra control errors or hidden near/head losses; otherwise
record tradeoffs or failure, retain existing models and stop. This160-frame
pilot cannot establish general dynamic reliability. No automatic successor fit,
capture or application change follows. Capture, inference, scalar scoring and
task-owned process/resource release receipts complete the delivery.

Artifact root: `artifacts.local/work/mz77-approach-sequence-20260911`.
