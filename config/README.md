# Configuration

Repository configuration lives here. Start with the file matching the task:

| File | Purpose |
| --- | --- |
| [local.example.toml](local.example.toml) | Copy to ignored `local.toml` for local tool paths and output locations used by [`tools/ba.ps1`](../tools/ba.ps1) |
| [remote-worker.example.json](remote-worker.example.json) | Template for the remote worker connection; follow the [worker handoff](../docs/operations/WORKER_HANDOFF.md) |
| [worker-research.requirements.txt](worker-research.requirements.txt) | Research worker Python dependencies |
| [worker-export.constraints.txt](worker-export.constraints.txt) | Export dependency constraints; applied with `pip -c`, not a standalone install list |
| [public_release_assets.json](public_release_assets.json) | Public app asset paths, provenance, sizes and SHA-256; validated by [the release readiness check](../scripts/check_open_source_readiness.ps1) |

The public asset manifest moved here from `configs/`; its asset and notice paths
remain relative to the repository root. Model ownership and limitations are in
the [model card](../docs/MODEL_CARD.md).

Experiment parameters and frozen protocols stay with their owning
`research/active/<route>/` implementation. Downloads, local runtimes, data and
outputs use [`artifacts.local/`](../docs/LOCAL_ARTIFACTS.md). Machine-specific
configuration and credentials do not belong in tracked templates.
