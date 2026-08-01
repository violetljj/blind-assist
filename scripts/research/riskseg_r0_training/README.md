# RISKSEG-R0 training data preparation

## Status

Development-only infrastructure for the frozen `RISKSEG_R0` task and data
contract. Full sequential execution authority is already recorded in that
contract; this utility does not bypass its event-data, technical-preflight, or
promotion gates.

## Stable interface

`materialize_reencoded_view.py` reads the frozen data-role ledger and only the
three source manifests declared by that ledger. It selects exactly the declared
train/dev source sessions and frame counts, verifies source hashes, and writes a
new four-class mask view. `dev_200` and the old blind 120 use
`canonical_passthrough`; the fresh 200 first use the SHA-bound frozen R2-P0
`source_native` canonicalizer. Only then is the old canonical order remapped:

```text
legacy 0 -> RISKSEG-R0 0
legacy 1 -> RISKSEG-R0 2
legacy 2 -> RISKSEG-R0 1
legacy 3 -> RISKSEG-R0 3
```

Example:

```powershell
python -m scripts.research.riskseg_r0_training.materialize_reencoded_view `
  --repo-root E:\linnan\linnan `
  --output-root artifacts.local\evidence\riskseg-r0\training-reencoded-view
```

The output directory is atomically published below `artifacts.local/` and
contains:

- `manifest.jsonl`: one row per selected frame with role, explicit dataset root
  and decoder, native source hash, in-memory old-canonical PNG hash, new mask
  path/hash, both mapping hashes, dimensions, and four-class pixel counts.
- `receipt.json`: total, role, session, and class coverage; source/ledger
  identity; class/session coverage gates.
- `masks/{train,dev}/`: re-encoded single-channel PNG masks.

Source RGB images are hash-verified and referenced, not copied.

## Safety boundary and stop conditions

The materializer never modifies the ledger or source data. It fails closed and
publishes no output for manifest/source hash drift, path traversal, unknown
class IDs, duplicate selected IDs, exact session/frame-count drift, image/mask
dimension mismatch, or insufficient obstacle/boundary coverage. In particular,
both `blocking_obstacle` and `boundary_level_change` must have nonzero pixels in
at least two train sessions and two dev sessions; failure is
`DATA_SPLIT_NOT_READY`.

This module does not inspect event-eval review records, candidate outputs, or
model outputs. The 90-frame fixed-regression role is not an input.

## Research question and evidence ceiling

This utility answers only whether the already-consumed 520-frame train/dev view
can be reproducibly remapped to the frozen four-class contract with closed
source ancestry. Pixel coverage is a data-readiness check, not event-level
model evidence, deployment evidence, or safety evidence. Previously consumed
assets remain Development-only.
