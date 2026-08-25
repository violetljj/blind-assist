# GRAIL R1C-L

Status: `STOPPED / DEVELOPMENT_GATE_NOT_MET / FINAL_UNOPENED`

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
- `evaluate_grail_r1cl_oa_v2_baseline.py`: frozen paired-relative OA-V2
  validation slot baseline.
- `grail_r1c_l_manifest_v3.json`: frozen roster and single architecture.
- `smoke_r1cl.py`: two-sample real DINO forward/loss smoke.

## Decision

The one frozen architecture completed both seeds. The selected learned arm was
`1497/1806 = 82.89%`; the paired-relative OA-V2 baseline on the same validation
pairs was `1456/1806 = 80.62%`. The `+2.27` point uplift is below the frozen
`+8` gate, so R1C-L stops without final access. See
`grail_r1c_l_development_result_v1.json` for hashes and exact scope. This is
controlled synthetic mechanism evidence only; it does not change the Android
default app or establish natural, product, user, or safety performance.
