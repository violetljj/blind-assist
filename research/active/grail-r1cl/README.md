# GRAIL R1C-L

Status: `ACTIVE / EXPLORE / TRAIN_VALIDATION_ONLY / FINAL_UNOPENED`

R1C-L learns a pairwise owner-local slot coordinate from two RGB observations
and their owner/sibling masks. It is the only active BlindAssist algorithm
experiment. Historical GRAIL, SAGE, dual-loop, RCLE, TARO, USTRF, and other
routes are preserved by tag `archive/pre-agent-surface-2026-08-26` and indexed
in `experiments/index.jsonl`; they are not current source-tree entrypoints.

## One entrypoint

```powershell
pwsh -NoProfile -File tools/ba.ps1 setup  research-r1cl
pwsh -NoProfile -File tools/ba.ps1 doctor research-r1cl
pwsh -NoProfile -File tools/ba.ps1 smoke  research-r1cl
pwsh -NoProfile -File tools/ba.ps1 run    research-r1cl -- <training arguments>
```

Configuration resolves in this order: explicit CLI option, ignored
`config/local.toml`, `BLINDASSIST_*` environment variable, repository-relative
default, then limited command discovery. The command prints the selected repo,
Python, CUDA, backbone, dataset, and output paths before execution.

## Active files

- `grail_pairwise_owner_coordinate_r1cl.py`: dataset, model, and loss.
- `train_grail_pairwise_owner_coordinate_r1cl.py`: two-seed train/validation.
- `collect_grail_pairwise_owner_coordinate_r1cl.py`: ProcTHOR pair collection.
- `run_grail_procthor_native_m0.py`: shared ProcTHOR position utilities.
- `run_grail_r1cl_sharded_collection.py` and
  `merge_grail_r1cl_collection_shards.py`: resumable collection.
- `grail_r1c_l_manifest_v3.json`: frozen roster and single architecture.
- `smoke_r1cl.py`: two-sample real DINO forward/loss smoke.

## Decision

Train and validate the one frozen architecture. If validation slot uplift over
the frozen OA-V2 baseline is below `+8`, stop without opening final. If it meets
the gate, update `docs/CURRENT_DECISION.md` before the one final access. This is
controlled synthetic mechanism evidence only; it does not change the Android
default app or establish natural, product, user, or safety performance.
