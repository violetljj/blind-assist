# Original fixture retains partial spatial evidence after background translation

2026-09-09 EXPLORE. On 53 identity-matched admitted pairs, unchanged JOINT strict
pair correctness falls from 41/53 (77.36%) in original backgrounds to 29/53
(54.72%) in new backgrounds, a 22.64 percentage-point decrease. Original B alerts
remain exactly unchanged on all120 captured frames. This supports a narrower
observation of background-dependent degradation, not a unique or complete cause
of the previous combined fixture/background failure.

The predeclared strong attribution decision remains
`BACKGROUND_ATTRIBUTION_INCONCLUSIVE`: new retention is below80%, but the matched
old control is also below its80% prerequisite. Preserve the observed difference;
do not present it as satisfying the stronger criterion or as no effect. No
promotion, ordinal training, fitting or threshold rescue follows this result.

## Matched four-metric comparison

| JOINT, same53 original pair identities | Original background | New background |
| --- | ---: | ---: |
| Near query hit |150/159 (94.34%)|119/159 (74.84%)|
| Near-to-far confusion |2/53 (3.77%)|3/53 (5.66%)|
| Strict both-endpoint pair-correct |41/53 (77.36%)|29/53 (54.72%)|
| Original alert parity |Retained B interface; old forward not rerun|106/106 admitted frames;120/120 captured|

The first three historical values are recomputed from frozen saved probabilities
on EXACT partners of the admitted new pairs, not the old750-pair multi-family
average. BASE near hit is0/159 and strict pairs0/53 in both backgrounds; its
wrong-far count changes52 to42/53. Partial JOINT evidence remains useful within
scope, but robust background retention is not established.

## What changed and what remained fixed

[Generator](body_query_background_spec.py) selects one unused historical EVAL
crossbar BASE pair per new site, in saved metadata order with identical absolute
camera yaw. All60 previous sites are used once, with15pairs per cardinal yaw.
Both endpoint cameras, wearer and all13 fixture components undergo the same
translation. Dimensions, materials, rotations, sun/skylight scales, real target
distance, lateral offset and relative camera geometry remain fixed. No source
selection uses predictions. Independent checks cover1560 object instances with
maximum relative-coordinate error4.44e-16m.

The original C8 capture core is reused (SHA256
`0b7696a7349d8d8500108c3549391162bd28e3a4eefea3f08a58cdd7d55cab0b`),
including original rigid reuse and settling. A pre-capture v1 spec omission of
the native target instance ID was repaired in v2; both versions remain. No
physical geometry changed during that repair. Both60-frame captures and native
world verification pass, with matched map and frozen runtime inputs.

This is a controlled translation to consumed backgrounds in the same City Sample
map. Moving location also changes occlusion, illumination and reflections. It
isolates this background intervention from redesigning the fixture, not those
individual photometric pathways or natural-source generalization.

## Source failure and separate visible-evidence diagnostic

The [original protocol](BODY_QUERY_BACKGROUND_ONLY_20260909.md) required complete
target-extent visibility and an aggregate target-ray PASS. It TERMINATED
NOT_EVALUABLE with0/60pairs; that source failure is preserved and is not a model
result. The inherited full-extent rule was inappropriate for a bar partly covered
by its original clamps. It rejected95frames on aggregate identity and52frames on
extent;14frames also have real background BODY support. Reasons overlap.

Before any model access, a [separate consumed-source diagnostic](BODY_QUERY_BACKGROUND_VISIBLE_HEAD_20260909.md)
was frozen on the SAME120frames. Its [native source operator](body_query_background_source_repair.py)
requires at least3 distinct exact target-component collision MATCH pixels, visible
in scene native depth and inside the intended HEAD range. Native scene BODY/HEAD,
exclusive-range, floor, capture-pose and source-integrity checks remain unchanged.
All original counts are byte-identical; failed reasons and off-corridor UNKNOWN
rays remain in each row. It does not assert full target extent or size invariance.

Independent ray inspection finds5or6 HEAD-corridor target MATCH rays in every
frame. All367 non-MATCH rays are outside that corridor, geometrically consistent
with original clamps. Only13 have direct non-target component inventory binding;
the other354 are NOT promoted from geometric consistency to native identity.
The visible-evidence diagnostic admits26/30 and27/30pairs, exceeding its fixed
24/30 per-region floor. Seven pairs with background BODY contamination remain
excluded. This is a separate diagnostic, never a retroactive pass of the primary.

## Validation and evidence

[Frozen inference](body_query_background_eval.py) ran once, on RGB only, with the
same checkpoint/normalization/threshold hashes as the previous JOINT result.
CUDA RTX5060 Laptop inference took3.49seconds excluding startup; native label
generation ran on the RTX3060 worker. The [saved-probability analysis](body_query_background_analysis.py)
reconstructs range probabilities in float64 (maximum difference1.83e-7) with
identical threshold decisions and reproduces all four new metrics. An independent audit reproduces the same result:25pairs stay correct,16become
wrong,4become correct and8stay wrong; both regions decline. Historical
predictions and native truth are receipt/hash bound. One displayed old/new near
example was visually inspected and shows the same13-part fixture projection.

Durable root: `artifacts.local/work/body-query-background-only-20260909/`.
Key records: `cohort-freeze-v2.json`, `translation-independent-audit.json`,
`visible-head-before-inference.json`, `returned-full-capture-v1/admission-v1/`,
`returned-full-capture-v1/admission-visible-head-v2/`, `inference-v1/`,
`matched-comparison.json`, `independent-final-audit.json`, and
`worker-final-release.json`.
All176 returned capture files were hash checked. Full native/isolated arrays
remain in the task-owned worker capture directory; RGB, native counts, failures,
source manifests and receipts are retained locally. Owned UE processes, scheduled
tasks and Zen listeners are gone; capture temporary directories were removed.
