# MZ80 observability-conditioned visual rescue score ceiling

Decision: `MZ80_CURRENT_RGB_SCORE_NO_SELECTIVE_RESCUE`.

The observability gate fixes the architecture boundary but cannot rescue the
current frozen RGB risk score. It preserves MZ78 CLEAN exactly, yet on every
5-klux MZ79 profile the best posthoc threshold with at most five added false
query bits rescues zero true bits. This rules out threshold rescue of the current
score representation. It does not rule out a newly trained residual visual
expert with appropriate hard negatives.

## Capability question and fixed gate

MZ80 tests the proposed rule:

```text
R = R_ToF OR (U_ToF AND RGB_margin >= threshold)
```

`U_ToF` is not the absence of a ToF alert. For each BODY/HEAD near/far query it
is true only when the conservative MZ79 profile cap (minimum of inner and corner)
is shorter than the full fixed query interval: 1.68, 3.18, 1.63 and 3.13 m.
Consequently all four queries are observable under the ideal 4-m profile, only
FAR is structurally unobservable for the dark 17% profile, and all four are
unobservable under the tested 5-klux profiles.

The primary fixed comparison uses the existing RGB cutoff (`margin >= 0`). A
separate exhaustive threshold scan opens the consumed evaluator and reports the
best possible rescued-TP ceiling at added-FP budgets 0, 5, 10, 20, 40 and 64.
Those thresholds are explicitly posthoc diagnostics and are forbidden as
deployment cuts. There is no training, new cutoff, retry or result promotion.

## Result

| Profile | ToF TP/FP/FN | Fixed conditioned increment | Required rescue (25% FN) | Posthoc best at added FP <=5 |
| --- | --- | --- | ---: | --- |
| Ideal 4 m | 124/2/6 | +0 TP / +0 FP | 2 | 0 / 0 |
| Dark, 54% typical | 124/1/6 | +0 / +5 | 2 | 0 / 0 |
| Dark, 17% typical | 87/1/43 | +1 / +5 | 11 | **1 / 1** |
| 5 klux, 88% typical | 42/0/88 | +8 / +64 | 22 | **0 / 0** |
| 5 klux, 54% typical | 32/0/98 | +9 / +64 | 25 | **0 / 0** |
| 5 klux, 17% typical | 20/0/110 | +13 / +64 | 28 | **0 / 0** |
| 5 klux, 17% minimum | 10/0/120 | +17 / +64 | 30 | **0 / 0** |

The CLEAN invariant is exact: all 640 four-query decisions equal MZ78 because
`U_ToF` is false. Conditioning also reduces the dark-gray fixed OR cost from the
unrestricted MZ79 result to one rescued bit for five added false bits by limiting
RGB to FAR queries.

The 5-klux failure is representational, not a cutoff-choice problem. With 40
added-FP budget, the posthoc ceiling rescues 0 white, 0 light-gray, 2 gray-typical
and 3 gray-minimum bits. Reaching 8--18 rescued bits needs 54--64 added false bits.
None approaches the required 22--30 rescued bits at five added FP.

## Decision and next admissible work

Retain the observability gate as `COMPONENT_OR_CHALLENGER / COMPONENT` because it
protects ToF-observable conditions by construction. Carry the frozen RGB score
as a negative control for selective rescue; do not lower or retune its cutoff on
MZ79. The current final RGB risk margin has not learned the distinction between
an unseen future collision and an UNKNOWN but harmless/pre-onset scene.

A successor must change the learned responsibility rather than the threshold:
train a residual expert only on `U_ToF` examples, with explicit hard negatives
covering UNKNOWN controls, visible but out-of-corridor/pre-onset obstacles,
wrong BODY/HEAD band, and unrelated structure. Selection and evaluation must be
source-separated; the MZ77/MZ79 cohort is consumed and can only remain Development
diagnostic evidence. The acceptance target remains at least 25% FN rescue, at
most five added FP per profile, CLEAN parity, and earlier correct warnings.

The two-dimensional RGB/ToF degradation matrix is not run here. Clean RGB already
fails the selective-rescue ceiling, so adding blur, darkness or glare cannot make
this frozen score pass. A future newly trained rescue expert should be checked in
four quadrants (both good, ToF bad/RGB good, ToF good/RGB bad, both bad) before
any claim of physical failure complementarity.

## Evidence and verification

Twenty-two focused tests pass: four MZ80 gate/ceiling tests plus the eighteen
MZ77--MZ79 tests. CPU execution is scalar and marked `TASK_NOT_GPU_SUITABLE`.
Canonical evidence is under
`artifacts.local/work/mz80-observability-rescue-20260912/run-v1/`, containing
the posthoc result, source snapshots and hash receipt.
