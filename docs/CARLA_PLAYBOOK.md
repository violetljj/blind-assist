# CARLA integration playbook

CARLA is an isolated local causal-lab and evidence backend for BlindAssist DTR.
It is **not** copied into this repository, imported into the BlindAssist Python
environment, or added to Android/Gradle. BlindAssist owns the research question,
consumer contract, evaluator, and claim; the external CARLA library owns its
runtime and frozen raw evidence.

## First use

The bridge resolves CARLA in this order:

1. `-CarlaRoot` / `-CarlaPython` command arguments;
2. `carla_root` / `carla_python` in ignored `config/local.toml`;
3. `BLINDASSIST_CARLA_ROOT` / `BLINDASSIST_CARLA_PYTHON` environment variables;
4. the workspace sibling `../CARLA` and its `client-env/Scripts/python.exe`.

Run the read-only admission check before consuming an asset:

```powershell
pwsh -NoProfile -File tools/carla_assist.ps1 check
```

Useful read-only operations are:

```powershell
pwsh -NoProfile -File tools/carla_assist.ps1 status
pwsh -NoProfile -File tools/carla_assist.ps1 list
pwsh -NoProfile -File tools/carla_assist.ps1 verify -Asset c2-v16-payload-authority
pwsh -NoProfile -File tools/carla_assist.ps1 explain -Asset c2-v17-footprint-guarded
pwsh -NoProfile -File tools/carla_assist.ps1 next
```

`check` verifies the project-owned consumer manifest against the external
catalog, runtime profile, status, authority, result paths, result hashes, and
the external verifier. Add `-Deep` only when every indexed source payload must
be rehashed. None of these commands starts or stops CARLA.

To create the normalized context consumed by project-side experiment tooling:

```powershell
pwsh -NoProfile -File tools/carla_assist.ps1 materialize
```

The output is
`artifacts.local/evidence/dtr-r0/carla/asset-context.json`. It is local,
ignored evidence containing the selected external root, frozen identities and
hashes, exact claim ceilings, and next-route metadata. It copies no sensor
payloads and creates no Android runtime dependency.

## Current admitted assets

The project contract is
[`consumer-manifest.json`](../research/active/dtr-r0/carla/consumer-manifest.json):

- `c2-v16-payload-authority` is a synthetic privileged-source canary with
  capture-time authority over 1,608 depth and instance payloads.
- `c2-v17-footprint-guarded` is a fully evaluable valid negative: the corrected
  guard-before-refresh arm covered `0/11` physical-loss frames while the Oracle
  covered `11/11`.

These are mechanism and source Development assets. They are not a frozen X21
source-disjoint confirmation, natural tracking evidence, real-device evidence,
or a safety result.

## Runnable DTR-CARLA-C0 benchmark

Run a fresh benchmark with a unique `RunId`:

```powershell
pwsh -NoProfile -File tools/run_dtr_carla_c0.ps1 -RunId c0-canary-YYYYMMDD-HHMMSS
```

The runner uses one fresh packaged CARLA server and one long-lived camera per
modality in the fixed order `instance -> rgb -> depth -> flow`. The instance
gate must admit all six twin families before observable payload capture
continues. Partial run directories are never overwritten; use a new `RunId`
after a failed capture. Raw evidence stays under the external CARLA library,
while joined observation, teacher, truth, sealed prediction, and result files
are written under ignored `artifacts.local/evidence/dtr-carla-c0/<RunId>`.

The first complete run is
[`DTR_CARLA_C0_RESULT_2026-08-30.md`](../research/active/dtr-r0/carla/DTR_CARLA_C0_RESULT_2026-08-30.md).
All four frozen-R2 arms recalled `7/7` events, but O0 RGB was best with three
false segments and 82.35% event F1. O1 depth produced six false segments, O2T
CARLA flow produced eight, and O3 privileged current state produced seven. The
route-turn pair failed in every arm; teacher flow also broke the static/dynamic
background pair.

## Current algorithm and source line

The planned-route question has now run. X24 combines the immutable issued plan,
online adherence, truth-blind RGB-D tracking, and bounded occlusion memory; it
met its same-source C2 Development gate. X26 support consensus and the X27--X30
occupancy-lineage successor improved some fresh-cohort metrics but did not meet
their frozen gates. X31 is the frozen ambiguity-preserving transport successor.

C8, C9, and C10 are terminal source-level `NOT_EVALUABLE` results: respectively
scripted-pose drift, occluder/wearer route intersection, and a raster-permeable
occluder stopped admission before any X31 metric result. See
[`DTR_CARLA_C8_X31_SOURCE_NOT_EVALUABLE_2026-08-30.md`](../research/active/dtr-r0/carla/DTR_CARLA_C8_X31_SOURCE_NOT_EVALUABLE_2026-08-30.md),
[`DTR_CARLA_C9_X31_SOURCE_NOT_EVALUABLE_2026-08-30.md`](../research/active/dtr-r0/carla/DTR_CARLA_C9_X31_SOURCE_NOT_EVALUABLE_2026-08-30.md), and
[`DTR_CARLA_C10_X31_SOURCE_NOT_EVALUABLE_2026-08-30.md`](../research/active/dtr-r0/carla/DTR_CARLA_C10_X31_SOURCE_NOT_EVALUABLE_2026-08-30.md).

C11 changes only the occluder to the observed solid-body firetruck and freezes
one fresh capture in
[`dtr_carla_c11_x31_solid_body_protocol.json`](../research/active/dtr-r0/carla/dtr_carla_c11_x31_solid_body_protocol.json).
It is pending and `NOT_RUN`; no source result or X24/X31 metric is claimed.
The CARLA line remains synthetic Development evidence and is separate from the
required real source-disjoint X21 confirmation.
