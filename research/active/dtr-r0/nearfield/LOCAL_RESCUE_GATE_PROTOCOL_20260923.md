# One ambiguity-fraction rescue gate

The preceding consumed diagnostic found that all LOCAL-only true frames have
less winning-query possible-only pixel fraction than false frames, separately
on transfer and stability. This feature was selected after both cohorts were
inspected; neither cohort is an independent validation for feature selection.
The absolute separating intervals differ. Do not claim a universal cutoff.

One scalar rule: retain A OR (LOCAL and possible-only fraction <= cutoff).
The fraction is existing public RGB/ToF interval representation slot 939 for
the winning centre query, not actual obstacle area or measured ownership.
Choose cutoff once using only the original transfer's 36 positive and 2 negative
LOCAL-only frames: midpoint between maximum positive and minimum negative.
If they overlap, stop rather than search another rule. Do not use stability to
adjust this cutoff. Save gate before evaluating either cohort's event metrics.

Use the original diagnostic joint criteria: >=80% incremental TP retained,
>=50% incremental FP removed, no event loss or onset delay versus LOCAL.
If both consumed cohorts pass, run one new 576-frame/48-clip controlled source
with new dimensions, lateral placements and two new sampled trajectories.
Freeze geometry and gate before capture. Compare A, LOCAL, two-frame confirmation
and the scalar gate. On new source apply the same gain/cost gate, requiring at
least 8 incremental TP and 2 incremental FP for evaluability. Preserve full
counts even if opportunity is inadequate. No refit after fresh outcomes.

Fresh refers only to new controlled instances absent from gate selection;
same renderer/materials/sensor law and selected feature limit generalization.
Sampled timing is not real motion or device latency. Neither result promotes
the App or validates UNKNOWN abstention. CPU scalar selection, zero model fits.
