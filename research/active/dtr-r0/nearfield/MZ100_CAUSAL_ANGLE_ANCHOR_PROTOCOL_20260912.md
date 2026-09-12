# MZ100: causal angle anchors for ToF-missing Radar fallback

EXPLORE, one fixed observable-only candidate on consumed MZ99 128 episodes/5120
frames. This is a development feasibility test, not independent confirmation.
No training, threshold search, collection, full decision model, ghost classifier,
fixed-width geometry change or automatic successor. Original packets and all
MZ99 evidence remain immutable. Artifacts: artifacts.local/work/
mz100-causal-angle-anchor-20260912/run-v1. CPU only.

Hypothesis: unique co-observed ToF/Radar groups constrain a shared azimuth offset
that can remain useful during subsequent ToF absence. MZ99 exact-angle benefit
does not establish this offset's observability or achievable correction.

Use the existing15-key raw schema and all8 ToF horizontal slots. Earlier narrative
references to16 keys are counting errors; RAW_KEYS itself is unchanged. Status5 positive
finite received ToF zones form contiguous clusters if adjacent range difference
is <=0.30m. Each cluster uses median range and min/max zone-center angle expanded
by the half-zone2.8125deg. Candidate Radar/cluster edges require range difference
<=0.35m and wrapped center-angle difference<=20deg. Only mutually unique edges
contribute; no nearest-candidate tie breaking. Distinct unambiguous pairs can
contribute even if other clusters have ambiguous edges. Missing is never clearance.

Each pair supplies a nominal offset interval: measured Radar angle minus cluster
angular interval, expanded by9deg (5deg quantization half-step + two2deg noise
standard deviations), clipped to[-20,20]deg. This is a tolerance hypothesis, NOT
a calibrated confidence interval or guaranteed object-center enclosure. ToF
slant-range aggregation and unknown object extent can invalidate the association.
Intersect same-frame intervals; conflicting pairs clear history and calibration.
Over a1s sliding window require >=3 contributing frames spanning>=0.2s and an
intersection width<=10deg. Empty temporal intersection clears history/calibration.
Cache the midpoint only on actual qualifying anchor frames. Ambiguous/missing
frames do not renew the cache. Expire1s after its last contributing anchor; reset
at each episode. An existing calibration may persist through ambiguous frames
until expiration. Do not shrink uncertainty by counting repeated data as independent.

Use only the preceding frame's cached estimate at each prediction time. Apply
it only when current ToF has no valid observation and Radar has valid returns.
Correct all valid Radar returns equally; identity is unavailable. Other fields,
R angular-wedge admission, ToF priority, hysteresis and outer hold are frozen.
The estimator never reads evaluator geometry, provenance, scene bias/IDs as
features, or future packets. Episode ID is used only for reset.

Stage1: seal estimates, anchor trace and corrected input before evaluator access.
On actual application frames only, require all three feasibility conditions:
1. coverage>=10% of Radar-available/ToF-missing frames;
2. >=90% of applied frame estimates are within5deg of simulator scene offset;
3. paired absolute angular measurement error against pre-noise bearing decreases
   on valid real-actor returns in those same application frames (nonempty set).
Truth is only a posthoc diagnostic; it never chooses individual applications or
parameter values. Report paired errors, coverage, age, ambiguity/conflicts,
offset strata and cold-start gaps. Estimation cannot exclude ghosts by identity:
they may form accidental unique correspondences. Audit pair provenance only after
sealing; a paired real-return tag alone does not prove the same ToF/Radar object.

If any stage1 condition fails, stop the algorithm test and report the failure;
do not relax tolerances or run/score downstream candidate alerts. If all pass,
run frozen R and corrected-input R, seal both predictions before task scoring,
and verify baseline parity to MZ99. Compare CURRENT_ROUTE primary, BODY_1S and
legacy separately. Retention conditions: >=98% paired route TP, no lost reference
route interval, maximum extra onset delay<=0.2s, fewer FP and no increase in
false segments or fragments. Include causal warning-session and contact timing
from frozen MZ99 evaluator. Passing only establishes a consumed component,
not promotion or independent calibration validation. No contact100% claims.

Unit checks cover unique/ambiguous matching, disjoint conflict, prior-only
application, expiry without cache renewal, episode reset, prefix causality,
invalid packets and angle-only mutation. Independent implementation review
before outcomes. Freeze code/protocol before executing. Preserve mechanical
failures if any; fixes do not authorize outcome-driven policy changes. Finish
scoped report, experiment/inheritance recording, commit and normal master push.
