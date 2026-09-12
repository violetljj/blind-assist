# Frozen matched_hold final-error source audit

Read-only, posthoc consumed MZ90 diagnosis. No new policy, training, threshold
search, counterfactual fusion or source expansion. MZ93 remains a negative result;
matched_hold remains the comparison baseline. Attribution shows where errors
occur, not that overriding the selected branch improves the system.

## Paired input degradation

| Regime | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Ideal | 457 | 41 | 34 | .9242 |
| Sensor proxy | 333 | 151 | 158 | .6831 |

Episode IDs, timestamps and truth arrays are identical across regimes. Framewise
transitions:135 TP become FN,11 FN become TP,128 TN become FP,18 FP become TN.
322 TP,23 FN,23 FP and1260 TN remain in their original class. Thus124 lost TP
and110 extra FP are net effects, not counts of one-way transitions.

Of135 lost TP,80 reach valid Radar returns rejected by the raw gate,19 have raw
Radar support not activated by hysteresis,17 have Radar alert overridden by a
nonalerting ToF branch,15 have nonalerting ToF with no Radar alert, and4 have no
valid current return. Of128 new FP,90 use current Radar support,14 use Radar
hysteresis without a current qualified return,7 inherit Radar through outer hold,
5 use current ToF geometry,10 ToF temporal geometry and2 inherit ToF through hold.
These changes apply all proxy effects together; they do not isolate one physical
noise or missing-return mechanism.

## Separate sensor origin from temporal mechanism

Proxy final-alert paths, counted disjointly:

| Origin sensor | Evidence / state mechanism | TP | FP |
|---|---|---:|---:|
| ToF | Current direct geometry sufficient | 90 | 6 |
| ToF | Current return plus temporal geometry required | 17 | 10 |
| Radar | Current qualified return with hysteresis | 168 | 108 |
| Radar | Hysteresis without current qualified return | 34 | 18 |
| ToF | Extra one-frame hold of preceding hard output | 17 | 2 |
| Radar | Extra one-frame hold of preceding hard output | 7 | 7 |
| Total | | 333 | 151 |

Both current and temporal ToF flags are stored independently. For a disjoint
table, direct geometry takes precedence when both suffice; temporal-required
means direct is false. Radar current support can still depend on the earlier
two-of-three activation. It is not an independent single-frame decision.

Outer hold is counted only when it changes hard=false to final=true. It reads
the previous hard result, not the previous held result, so it cannot recursively
extend itself. On a held frame the selected sensor is Radar (no current ToF),
while the inherited source may be ToF. This is why source and persistence are
separate fields. Across these actual held-only frames,24 TP and9 FP are sustained;
these counts do not evaluate an alternative state policy across time.

Every frame row records both branch outputs, raw Radar qualification, current
selected sensor, inherited sensor, evidence mechanism, hard-source frame,
Radar activation frame and supporting seed frames, latest raw support,
current/temporal ToF witness frames, outer hold and final alert-segment start.
Witness-frame truth and initial-segment truth locate whether support originated
before or during a nonhazard interval. These are frame-truth annotations, not
proof that every contributing return represents the hazardous object. Radar
activation can combine different objects across frames; no object track is implied.

## The158 FN: stage where existing support stops

| Frozen decision path | FN |
|---|---:|
| No ToF; valid Radar returns all fail raw gate | 94 |
| ToF selected and nonalerting; Radar branch also nonalerting | 17 |
| No valid current ToF or Radar return | 6 |
| No ToF; raw Radar qualifies but hysteresis has not activated | 21 |
| ToF selected and nonalerting despite Radar branch alert | 20 |

The94 raw-gate FN include82 frames with a current return from a true hazardous
source object and12 without such a Radar return. Within the82, all hazardous
Radar returns fail range in69 frames, closing speed in11 and bearing in10;
criteria overlap and must not be added.66of69 range-blocked cases have future-only
truth. The frozen Radar gate uses current range<3.18m, whereas benchmark truth
also permits a one-second future corridor intersection at current range<3.6m.
This is an explicit task/readout mismatch worth investigating, not a measured
benefit from loosening the gate. Real-time availability, false returns and other
branches must still be handled under any future proposal.

Selection conflicts are bounded and mixed:

| Valid nonalerting ToF selected over alerting Radar | FN | TN |
|---|---:|---:|
| Ideal | 9 | 82 |
| Sensor proxy | 20 | 8 |

15of20 proxy conflicts also have a currently qualified Radar return from a true
hazardous object. The remaining5 need not have such current support; a positive
Radar output alone is not a same-object witness. These are simultaneous branch
states under the original policy, not an OR-policy experiment or a guaranteed
recall/precision tradeoff. The large ideal TN count warns against removing ToF
authority unconditionally.

Other diagnostic cohorts include the correct outputs they preserve:

| Proxy current condition | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| No ToF, qualified Radar but no hysteresis alert | 5 | 1 | 21 | 25 |
| No ToF, valid Radar but none qualifies | 43 | 23 | 94 | 976 |
| No valid current return from either sensor | 10 | 3 | 6 | 208 |

TP/FP can coexist with absent current qualification because history still acts.
No current return is not clearance. Removing a cohort is not the same as changing
one upstream gate; no such policy was evaluated here.

## Physical source provenance is evaluator-only

Exact label-only replay identifies whether a received return came from an object
that is hazardous at the current timestamp, independently of policy acceptance:

| Current hazardous source return availability | TP | FN |
|---|---:|---:|
| ToF and Radar | 88 | 24 |
| ToF only | 19 | 5 |
| Radar only | 191 | 110 |
| Neither | 35 | 19 |

139of158 FN have at least one current hazardous-object return;19 do not. Only6
FN have no valid return of any kind; the other13 can contain nonhazard returns.
This separates absent target returns from available data rejected downstream.
It does not prove that the algorithm can distinguish a hazardous return from
clutter using its observables. Likewise35 TP with no current hazardous return may
use history or coincident support and are not automatically correct localization.

## Research implication, not a new fusion rule

The evidence prioritizes (1) the current-range versus future-risk contract and
(2) whether a valid ToF observation covers the same spatial hazard implicated by
Radar. Most FP still have current Radar support:108 FP versus168 TP. Pure state
exit work cannot explain or remove that majority by itself. Any next mechanism
must distinguish useful from erroneous support on its actual final decision path;
these counts alone establish neither separability nor a ready replacement.

## Verification and artifacts

Canonical detailed output: artifacts.local/work/mz90-final-decision-audit-20260912/run-v2.
`ideal-rows.json` and `sensor_proxy-rows.json` retain all1920 rows each, including
every FP/FN; `paired-transitions.json` retains every changed frame; `result.json`
holds aggregate tables. `validation.json` adds read-only transition/gate cross-tabs.
run-v1 is the retained initial audit before adding independent ToF temporal flags
and detailed Radar-gate cohorts. Both reproduce the same frozen hard/hold outputs;
no policy rerun or parameter selection occurred between the diagnostic versions.

All22 input receipt hashes verified. Label-only ToF and Radar provenance replay
preserves every RNG draw and numeric stable ordering; every raw and evaluator
array matches exactly. Reconstructed matched_hard and matched_hold equal the
sealed predictions for both regimes. Three focused tests cover Radar activation,
decay and nonrecursive hold, inherited sensor versus current branch, causal
prefixes and episode resets. All four detailed output payload hashes verify.
Independent read-only review checked attribution boundaries. No workers, paid
allocations, modified predictions, learned parameters or counterfactual policies.
