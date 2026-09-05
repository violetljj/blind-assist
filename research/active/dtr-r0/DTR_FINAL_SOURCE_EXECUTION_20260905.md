# R1 avoidance-only source execution

Status: `NOT_EVALUABLE_SOURCE_CAPTURE_INTERRUPTED`; no algorithm comparison.

All three groups pass all ten instance/witness source gates. FIT_ONLY completed
all four modalities and the R1 join. FINAL_A completed RGB but its depth server
exited before the first depth frame: UE reported `Shader compilation failures
are Fatal` after 59 seconds (crash process 12492). FINAL_B RGB/depth were not
started. Nine of twelve sensor shards are complete. No detector inference,
fitting, final prediction, or final method score was opened. Task-owned CARLA
processes, ports, and capacity leases are released; durable evidence remains.

`execution-terminal.json` and `engine-crash-final-a-depth/` under the source seal
retain the failure and crash payloads. This is missing source support, not
negative evidence against any avoidance algorithm. The frozen execution is not
retried or reclassified as fresh confirmation. Algorithm inheritance is unchanged.

The user's accepted next step is the existing avoidance R1 comparison, excluding
L10 and any new learner. The eleven frozen arms remain byte-identical. No
algorithm contribution result is claimed by this source preparation.

## Source findings and correction

The consumed visual-shell probe was checked against all ten strata using
hash-verified instance and independent witness shards. Seven source checks pass;
three fail. S02/S03 pass only prospective raw-source removal eligibility, not a
detector collision credential. The historical audit is retained at
`artifacts.local/evidence/dtr-final-reckoning-roster-r1/source-audit-20260905-b.json`.

- S06 has independently visible targets but no shared bounding-box overlap.
  The corrected alias crosses the target bearing before contact and clears the
  route laterally; it does not add another intended collision actor.
- S09's only zero within the declared window is startup sample 0. Its partial
  interval has disconnected non-singleton pixel components. Every stratum now
  uniformly excludes sample 0 from visibility and method metrics, retaining the
  raw payload. An internal zero still fails the gate.
- S10's old negative interval and disappearance do not overlap sufficiently.
  A new analytic design has two future-contact intervals, a planned 1.8–3.8 s
  disappearance, and 1.8–3.1 s known-negative support inside it. These are analytic
  predictions were independently verified in all three groups: zero pixels at
  samples 18–39, with known-negative support inside that interval. The negative
  interval starts before disappearance, so this does not establish the stronger
  temporal ordering of an ACTIVE event entering loss and then clearing inside it.

All cells retain 9 s source capture for the 3 s horizon. Scored windows are
0.1–3.5 s for S01/S09 and 0.1–6.0 s otherwise; post-contact out-of-view tails are
not clean/partial-visibility test windows. The two unobscured reference captures
remain evaluator-only auxiliaries outside the ten-episode denominator.

## Sealed execution and limits

`artifacts.local/evidence/dtr-final-reckoning-roster-r1/source-execution-20260905-c`
contains three source protocols/annexes, a 308-file source snapshot, and
`source-freeze.json`. Seeds remain FIT_ONLY 517031, FINAL_A 517131, FINAL_B 517231.
Its `raw` junction points into the governed CARLA experiments tree on the same
artifact volume. No prior probe pixels enter these groups.

The source-only capture path verifies the exact protocol binding before launch,
captures instance and witness, then runs the full source audit. A durable-frame
source failure is NOT_EVALUABLE with no scene repair or retry. This seal grants
no detector inference, model fitting, or final algorithm outcome access.

The nine nonlearned arms, two learned stages, and twenty-episode shared scorer
now have integration adapters. They have only synthetic/mock validation so far.
The raw detector ledger is retained; fixed interventions remove all candidates
on the declared frame indices. Adjacent raw X24 frames must have a currently
measured, confirmed collision-risk track. This admission proxy does not read
native target IDs or truth, and failing it never relocates the indices.

Not executed because source capture terminated: shared detector ledger admission,
nonlearned prediction seals, FIT-only fitting once, learned prediction seals,
then one joint final scoring. The adapters are implementation deliverables, not
observed algorithm results. No X97 is authorized.

## Generic join compatibility

Before any RGB/depth join, code inspection identified an inapplicable legacy
check: C2 demands a nonempty complete-disappearance run even for S09's maximum
complete-disappearance duration of zero. R1 instead requires partial visibility
and disconnected raster regions. `finalize_dtr_final_roster_join.py` preserves
the generic C2 result and binds its eighteen other integrity checks plus its
independent replay check to the already sealed ten-stratum R1 source evaluator.
It does not change pixels, source geometry, evaluator rows, or source thresholds.
A false replay/integrity check or any failed R1 stratum still rejects the join.
Its distinct `r1-joined-result.json` is required before detector preparation.

The first real join exposed a second manifestation of that legacy mismatch:
the complete-occlusion routine indexes a non-collision visual shell in the
collision-polygon map and raises `KeyError` before producing a result. Original
failure logs remain retained. A versioned join bridge marks this legacy routine
NOT_APPLICABLE and false, while retaining every other check and the sealed R1
source gates. `join-bridge-annex-v2.json` freezes that bridge against all three
existing source gates before rejoining. Only empty failed-join output directories
are removed; no pixels, actor rows, collision truth, or source gates are edited.

Validation: 58 distinct focused tests passed across source/materialization,
retained event metrics, prediction stages, detector intervention, and joining.
PowerShell parsing and repository artifact hygiene passed. The CARLA client
environment lacks Pillow, so the source
audit explicitly uses the existing research Python environment; capture still
uses the CARLA client environment. No packages were installed.

## Capture efficiency finding

The user's runtime concern is supported by the execution logs. FIT_ONLY's four
shards spent approximately 181.5 s between server and client starts and 1400.9 s
in client capture, for 1092 frames per modality. The successful join bridge took
about 91 s. These are filesystem/log timestamp estimates, not profiler timings.
The dominant cost is four separate frame loops, approximately 3 frames/s each.
The runner also repeats capacity checks and reads hardlinked payloads for hashes
multiple times; current timestamps cannot isolate their exact costs.

The operational profile attributes the single-camera path to a historical
UE4.26.2 D3D12 asynchronous PSO failure, explicitly not OOM. An existing 640x360,
four-camera, 12-frame smoke passed, so the limitation is not fundamental, but
that short result does not establish 1280x720 multi-camera stability. A bounded
future engineering check should compare serial versus same-tick RGB/depth on
two short 720p scenes, measuring wall time and alignment before adopting it.
No throughput or speedup claim has been measured for that proposed change.

During FINAL_A capture, a separate Frostpunk process also used the GPU; total
utilization reached 100% and memory roughly 7.4/8 GiB. This is concurrent host
load, not an algorithm outcome or a controlled throughput benchmark.

The later native crash message confirms a shader compilation fatal, not an
observed OOM. GPU contention explains a throughput concern but does not establish
the cause of that crash. Repeated fresh-server, single-camera execution did not
eliminate the historical failure mode.

## Retention and registration

Both the source-seal directory and the governed raw directory are registered in
the asset catalog with the protocol-required `sealed_final` retention token.
That token protects storage; it does not mean final algorithm scoring succeeded.
The source seal's raw junction is excluded from its metadata inventory, and the
physical raw tree is registered separately.

Experiment registration was attempted through `tools/knowledge.py` and remains
blocked by the existing `experiments/index.jsonl:252` input fingerprint mismatch.
No manual registry entry or unrelated repair was made. Accordingly this remains
an explicit working source record, not a newly admitted high-authority terminal.
The proposed engineering inheritance is a source-stability negative control:
single-camera fresh-server D3D12 did not prevent shader compilation failure.
No algorithm inheritance role is revoked or promoted by this failure.
