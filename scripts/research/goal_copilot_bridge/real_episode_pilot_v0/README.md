# Public-real episode mining + selective-guidance pilot V0

状态：`IMPLEMENTED / FRESH_PUBLIC_METADATA_COHORT_READY_8X89 / CONSUMED_DEVELOPMENT_BASELINE_SMOKE / MANUAL_CAPTURE_NOT_BLOCKING / NO_P1 / DEFAULT_APP_UNCHANGED`

This package automatically converts public real-world sequence metadata into goal-driven approach episodes, reuses
frozen current-frame provider output, applies Selective Guidance V0, and evaluates only truth-supported denominators.
The prospective entrypoint rejects a roster unless its public Goal Contract and entrance candidate set were frozen
before Mapillary metadata, pixels, model output, and evaluator truth.

Truth authority is ordered as native GT, map/trajectory-derived truth, independent-teacher consensus,
`AMBIGUOUS/UNKNOWN`, then manual annotation as a last resort. Missing exact frame-region visibility truth never becomes
a fabricated negative. `UNIQUE`, `SET_VALUED`, and `AMBIGUOUS` remain distinct.

```powershell
python -m scripts.research.goal_copilot_bridge.real_episode_pilot_v0.public_real_mining prospective `
  --goal-roster <frozen-public-goals.json> `
  --mapillary-metadata <sequence-metadata.json> `
  --output-dir <new-output-dir>

python -m scripts.research.goal_copilot_bridge.real_episode_pilot_v0.baseline `
  --public-manifest <public.json> --provider-observations <provider.json> `
  --config scripts/research/goal_copilot_bridge/real_episode_pilot_v0/baseline_config.json `
  --output <prediction.json>

python -m scripts.research.goal_copilot_bridge.real_episode_pilot_v0.evaluate `
  --annotation <annotation.json> --prediction <prediction.json> --output <evaluation.json>
```

`adapt-consumed-replay` is a smoke-only adapter for the sealed Last-10m Mapillary sequence. It labels the source
`PROJECT_CONSUMED_DEVELOPMENT_PIPELINE_SMOKE_ONLY`; it cannot establish freshness or performance. The current fresh
successor is automatic Mapillary + OSM/Overture mining. ADT is calibration/mechanism support, Ego4D is domain-realism
support, and Habitat is explicit-goal mechanics support. Physical capture is not a current blocker and is considered
only if public data cannot answer a separately stated, high-value question.

The executed prospective V1 froze four unused, venue-taxonomy-gated goals before Mapillary access, expanded 14 full
sequences from bbox-nearby metadata, and mined 8 episodes / 89 observations. It downloaded no pixels and made no model
calls. The next bounded stage may materialize only these frozen observations; it may not replace goals after outcome.

No component adds tracking, re-ID, persistence, world memory, VIO/SLAM, model/threshold search, completion authority,
or default-App integration. Public media cannot support a blind-user effectiveness claim.

```powershell
python -m unittest `
  scripts.research.goal_copilot_bridge.selective_guidance_v0.test_contract `
  scripts.research.goal_copilot_bridge.real_episode_pilot_v0.test_pilot
```
