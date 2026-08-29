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

## Next integration layer

The next useful CARLA surface is a fresh DTR causal benchmark, not bulk random
image generation. It should cover six paired interventions: motion, route,
time-to-contact, visibility, visible-to-occluded continuity, and
static-to-dynamic background. Evaluate the same episodes with an Oracle ladder
from RGB-only through depth and optical-flow observations to privileged truth.

CARLA optical flow may first serve as teacher/evaluator evidence. Making flow a
deployment input or a new threshold gate is a separate source decision. A
frozen X21 replay also needs an explicit observable-source adapter and remains
separate from required real source-disjoint confirmation.
