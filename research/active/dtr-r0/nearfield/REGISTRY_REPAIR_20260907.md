# Near-field registration repair and backfill

The pre-existing line252 input-fingerprint failure is repaired without changing
that historical row or its original inputs. The stored digest
`def2fb408fe93292a6e0c503f55bd40cf79140599685df5f7e474d657a491673`
exactly matches raw Git blobs at its recorded revision
`0a2224093ca12f4a920967b75789021bc052b492`. Successor `6ff906e1` changed the
shared freeze Python implementation; the original protocol remained identical.
The current checkout digest is
`82ff2ef00f28e584c1fde7b57f9cf1f7a58d791407b0fa3577c5f0fdd9c56604`.

The registry now verifies current input bytes first, then exact historical blobs
at the full recorded code revision. Neither hash matching means rejection.
Historical verification establishes original evidence integrity only: it does
not make changed current inputs fresh or permit consumed confirmation reuse.
Missing historical objects, invalid revisions and altered digests still fail.
No threshold, research outcome, historical hash or terminal was rewritten.

## Actual registrations

All six were successfully registered with `tools/knowledge.py register-experiment`,
using `active` status for Development association without inventing terminals:

- `nf-g0-representation-2026-09-07`
- `nf-g1-ground-anchor-2026-09-07`
- `nf-g2-surface-support-2026-09-07`
- `nf-g3-distinct-views-2026-09-07`
- `nf-g4-evidence-fusion-2026-09-07`
- `nf-g5-frontend-domain-2026-09-07`

Retrospective input manifests under `experiments/nearfield/` bind the immutable
result receipts, recorded input identities and executed-code hashes. They are
registration evidence manifests, not newly collected experiment inputs. NF-G0
through NF-G4 code hashes matched their respective delivered Git revisions:
5/5,11/11,13/13,19/19 and23/23 files. NF-G5 explicitly records its Git revision
as a base only; executed WIP hashes remain in the receipts. Its saved root
pre-score protocol controls attribution, while the mistakenly copied old briefs
in individual run directories remain preserved and disclosed.

Earlier reports and failed registration logs describe the historical publication
gap. This repair closes central registration; it does not change their measured
results, create terminal inheritance assignments or promote any algorithm.

## Verification and scoped delivery

The full knowledge unit suite passed11/11, including historical checkout drift,
deleted current inputs, invalid revisions and tampered-hash rejection. The real
shared library validates and the regenerated shared decision index passes its
freshness check with256 run associations. Original ledger bytes are preserved;
only the six CLI-produced rows were added.

Delivery stages only task-owned tool hunks, six registration rows and manifests,
this note and the owning near-field changes. The generated delivery index is
rebuilt from the exact staged source snapshot and validated separately. It must
not absorb unrelated unstaged registry/terminal/inheritance work. The existing
hook reads the shared checkout and rejects such WIP; equivalent snapshot checks
are required for a one-command hook skip, without changing persistent hook config.

Exact staged-source verification passed:10/10 knowledge tests,library validation,
all34 decision-engine cases,and generated-index freshness. Its238 experiments,
44 current terminals and236 run associations exclude the20 unrelated WIP rows
and their additional terminals. The shared checkout retains its separately
generated258-experiment/256-association index; it is not staged wholesale.

Initial snapshots, forensic receipt, actual registration logs and task-only
patches are retained under
`artifacts.local/nearfield/registry-repair-20260907-v1/`.
