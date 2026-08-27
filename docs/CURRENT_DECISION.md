# Current decisions: L10-R0 active; Dynamic Travel Risk R2 established

Status: `L10_R0_ACTIVE / CONTROLLED_MECHANISM_POSITIVE` and
`DTR_R2_PUBLIC_REAL_PRIVILEGED_CEILINGS_ESTABLISHED /
NO_LOCAL_RECORDING / DETECTOR_RUNTIME_DOWNSTREAM`

## Parallel product lines

Ten-metre goal completion and obstacle-risk guidance are separate lines. L10
does not wait for DTR, DTR does not change L10, and evidence from one line does
not count for the other.

L10-R0 remains the goal-conditioned controller for readable destinations. Its
controlled Development benchmark reached 87.5% task completion, 81.2%
post-occlusion reacquisition, 93.1% direction accuracy, 1.7% wrong-lock frames,
and `0/50` target-absent false completions. A second seed reproduced 88.5%
completion, 82.4% reacquisition, 91.9% direction accuracy, 1.8% wrong-lock
frames, and another `0/50` false completions. These remain controller/mechanics
results, not real-camera or product evidence.

## DTR-R2 decision

Accept R2 as the current dynamic-track algorithm. It combines robust
finite-horizon route-occupancy consensus with a fixed 1.5-second imminent
route-intersection guard and produces stable
`ONSET / HOLD / ESCALATE / CLEAR` events.

- On route-authoritative THÖR-MAGNI, R2 recalls `10/10` events with 42 false
  alert segments versus R0's `9/10` and 55.
- On 27 JRDB test sequences, R2 recalls `164/175` with 256 false alerts versus
  R0's `161/175` and 260.
- On CODa development plus rainy holdout, R2 recalls `119/122` pedestrian
  events with 285 false alerts versus R0's `122/122` and 286. The three-event
  cost is retained rather than threshold-tuned away.

Accept S3 as the current static-obstacle ceiling. Its causal curved route plus
bounded vertical occupancy recalls all `12/12` CODa barrier, fixed-structure,
and temporary-obstacle path contacts while reducing 3 m proximity false alerts
from 104 to 10 (`90.4%`) and clearing `4/4` eligible events.

The decision is algorithm success at a privileged public-data ceiling, not
detector or product completion. CODa supplies positive dynamic events only for
pedestrians and positive static events only for the observed
barrier/fixed/temporary classes. Bicycle, vehicle, vegetation, thin-branch,
drop-off, and head-clearance positive recall remain unproved. GOOSE does not
supply synchronized route and ground-clearance truth for a trustworthy hanging
branch event, so head-clearance is `NOT_EVALUABLE`, not a negative result.

## What stops here

- Do not record the superseded 24 canary or 120 staged local RGB clips.
- Do not widen the public cohorts or tune tracker, support, tube, horizon,
  urgency, lifecycle, or guard thresholds against the opened outcomes.
- Do not create separate per-class test matrices to make the numbers look more
  complete.
- Do not treat `UNKNOWN` or `NOT_EVALUABLE` as safe.

The next increment, only when deployment evidence is wanted, is to replace
privileged boxes/tracks with a real RGB/LiDAR detector and tracker behind the
already frozen metric-frame adapters. Phone recording and live-device testing
are not the current blocker or current priority. Full methods, receipts, and
claim limits are in [the DTR route README](../research/active/dtr-r0/README.md).
