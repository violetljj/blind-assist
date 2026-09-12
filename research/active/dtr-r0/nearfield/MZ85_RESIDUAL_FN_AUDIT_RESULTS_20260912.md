# MZ85 residual generic-FN audit results

Decision: `RESIDUAL_FN_SPLIT_NO_SINGLE_DOMINANT_MECHANISM`.

The 11 remaining MZ85 generic false-negative frames do not reduce to one
dominant 6--8-frame mechanism. A read-only audit of hash-verified sealed MZ84
observations and MZ85 predictions divides them into `4 + 4 + 3`. No prediction,
threshold, state or source was changed.

## Primary mutually exclusive decomposition

| Mechanism | FN frames | Meaning |
| --- | ---: | --- |
| Lateral history cold start | 4 | All four lateral-crossing episodes are positive at frame 0, before the fixed ToF angular projection has a previous ray; radar radial evidence is also inactive. |
| Radar evidence present but not activated | 4 | ToF is UNKNOWN and a current raw radar packet passes range/Doppler/angle cuts, but two-of-three activation has not fired. |
| Dual instantaneous observation absence | 3 | ToF is UNKNOWN and no current radar return passes the fixed cuts. |
| Other | 0 | No residual was forced into an unexplained bucket. |

The largest individual slice is therefore an episode-boundary/history
initialization artifact, not a newly identified sensor failure. Future temporal
sources should provide pre-roll or report cold-start scoring separately.

## Gap and authority-hold scope

Five FN frames occur inside the multi-target ToF gap; four are at gap onset.
Only two of those onset frames have an already-active collision state in the
immediately preceding frame. Those are exactly the two events that fragment in
MZ84/MZ85.

Consequently, a narrowly defined authority-aware hold has a direct target of
two FN frames and two fragmentations, not all five gap FN and not all 11 residual
FN. The other two gap-onset misses begin their truth-positive interval at the
same frame as the gap and have no previous positive state to preserve. The final
gap FN has neither ToF nor qualifying instantaneous radar evidence.

This supports keeping a later MZ86 tiny: preserve only an already-established
hazard across a brief loss of sensor authority when no valid-clear contradiction
exists. It should not be described as the main residual-recall solution.

## HEIGHT_UNKNOWN is a separate axis

Thirty-nine frames are correct generic true positives supplied only by radar and
remain `HEIGHT_UNKNOWN`. They are attribution debt, not part of the 11 generic
FN and not a cause of generic misses. Query-level BODY/HEAD evaluation must keep
that distinction explicit.

## Implication

There is no evidence here for one large learned-state-estimator opportunity.
The residual requires at least three separately testable questions:

- temporal cold-start/pre-roll semantics for lateral projection;
- activation/continuity authority when ToF becomes UNKNOWN;
- truly missing dual-sensor evidence, which policy logic alone cannot recover
  without importing stronger dynamics or persistence assumptions.

The next most informative robustness step remains leaving ideal analytic
orientation: inject bounded gyro bias, noise, timing and extrinsic error on a
freshly declared falsifier. MZ86 can still remove the two known continuity
defects, but its maximum direct scope is now quantitatively bounded.

## Evidence boundary and verification

[Audit protocol](MZ85_RESIDUAL_FN_AUDIT_20260912.md). Canonical output is
`artifacts.local/work/mz85-residual-fn-audit-20260912/run-v2/`. The audit verifies
all MZ84 and MZ85 receipt-listed artifact hashes before reading them. It is a
post-result diagnosis of consumed constructed evidence, not new-source or
hardware confirmation. Three classification tests, Python compilation,
knowledge validation and `git diff --check` pass.
