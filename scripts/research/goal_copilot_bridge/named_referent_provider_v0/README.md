# NamedReferentProviderV0

Status: `REVERSIBLE_EXPLORATION / CANARY_LITE / PLATFORM_ENGINEERING_CANARY / ENGINEERING_MECHANICS_ONLY`.

This directory is the minimal provider seam for a current RGB frame plus a Goal Reference Pack containing a name,
aliases, reference images, an optional logo asset, and an optional map bearing. It emits exactly four independent
channels:

- `text_evidence`: PP-OCRv5 text, confidence, polygon/bbox, raw string and per-name exact/substring/fuzzy diagnostics;
- `visual_reference_evidence`: DINOv2-S reference/logo cosine and rank with both source hashes;
- `proposal_evidence`: existing pinned Grounding DINO proposal-only mechanical output;
- `bearing_evidence`: optional map bearing and, when a current heading exists, signed angular delta.

Every channel includes provider/model/runtime identity, latency, source image/crop/polygon/bbox or bearing provenance,
raw and normalized match fields, and explicit error semantics. An unavailable or failed provider produces an empty
`NOT_EVALUABLE` channel. It never fabricates evidence.

There is no cross-channel fusion, referent selection, truth/evaluator field, temporal identity, tracker, navigation
action, arrival decision, safety authority, Android path, default-App change, or model promotion. Grounding DINO's
inherited prompt and thresholds are disclosed only as the configuration of a mechanical smoke; this package does not
select, tune, or promote them. OCR match class and DINOv2 similarity are diagnostics, not physical-referent authority.

## Focused tests

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts/research/goal_copilot_bridge/named_referent_provider_v0/test_schema.py `
  scripts/research/goal_copilot_bridge/named_referent_provider_v0/test_provider.py
```

## Public repeatable canary

The acquisition command resolves a fixed small roster by exact Wikimedia Commons titles and writes source page,
download URL, creator, license, role, dimensions, and SHA-256 under ignored `artifacts.local/`. It is explicitly
public, consumed, non-fresh, non-formal data and is never committed.

```powershell
$artifactRoot = 'artifacts.local/named_referent_provider_v0'
$python = "$artifactRoot/v/Scripts/python.exe"
& $python -m scripts.research.goal_copilot_bridge.named_referent_provider_v0.cli acquire-canary `
  --artifact-root $artifactRoot
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.goal_copilot_bridge.named_referent_provider_v0.cli run-vision-canary `
  --artifact-root $artifactRoot `
  --grounding-dino-model-dir F:/ba-data/blindassist-artifacts-20260805/models/grounding-dino-tiny-a2bb814 `
  --reference-model-dir F:/ba-data/blindassist-artifacts-20260805/models/p1_a2_dinov2_small_ed25f3a
& $python -m scripts.research.goal_copilot_bridge.named_referent_provider_v0.cli run-canary `
  --artifact-root $artifactRoot `
  --grounding-dino-model-dir F:/ba-data/blindassist-artifacts-20260805/models/grounding-dino-tiny-a2bb814 `
  --reference-model-dir F:/ba-data/blindassist-artifacts-20260805/models/p1_a2_dinov2_small_ed25f3a `
  --ocr-devices cpu,gpu:0
```

The torch-backed vision shard runs in the project Python separately from the Paddle shard because the official Windows
50-series Paddle wheel and PyTorch load different CUDA DLL envelopes. The final artifact only packages the unchanged
four independent channel results; it does not combine scores or make a decision.

The canary runs real PP-OCRv5 over English/Chinese storefronts and a derived Chinese distance/scale ladder, preserves
polygons and exact/substring/fuzzy diagnostics, records CPU/GPU latency, compares same-POI views against same-brand and
unrelated distractors with DINOv2-S, and runs one Grounding DINO mechanical smoke. Machine output and the closeout stay
under `artifacts.local/named_referent_provider_v0/evidence/` with hard claim ceiling
`ENGINEERING_MECHANICS_ONLY`.

For a one-frame integration call, use `run-provider` with `--frame-json`, `--goal-pack-json`, and only the adapter model
paths available on the host. Omitted adapters remain `NOT_EVALUABLE`; they are never silently substituted.
