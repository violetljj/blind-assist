# R1C-L active route

This directory is the sole active algorithm workset. Read `README.md` and
`../../../docs/CURRENT_DECISION.md`; do not search the archive tag unless the
task explicitly requests reproduction.

- Inference inputs are paired RGB plus owner-union and sibling-centroid masks.
- The single model is DINOv2-S plus two bidirectional cross-attention blocks and
  a 36-bin symmetry-marginalized owner-coordinate head.
- Development uses train/validation only. Do not open the final-test roster or
  data unless `docs/CURRENT_DECISION.md` explicitly advances to `FINAL`.
- The primary Development decision is validation slot uplift over the frozen
  OA-V2 baseline. Uplift below `+8` stops the route without final access.
- Keep outputs below `artifacts.local/evidence/grail-r1cl/`.
- Use `tools/ba.ps1`; do not hand-compose Python, CUDA, model, or data paths.

Run one focused mechanics test or the two-sample smoke after code changes. Do
not add another architecture, seed, sweep, protocol, or governance layer unless
the current decision explicitly changes.
