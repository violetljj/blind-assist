# Q3 native query coverage and attribution diagnostic

2026-09-09 EXPLORE. Frozen320-frame source and Q2 point dumps; zero new RGB
inference or fitting in this diagnostic. Retrieve original native arrays, verify
all against the existing world-verification hashes and reproduce query counts.
Use primary GPU for native geometry; no worker training or capture.

Question: does existing query sampling overlap cell-specific visible surfaces,
and are frozen strong point readouts selective for those surfaces?

Compare nearest native-pixel ownership at each valid projected ray with a coarse
local proxy: max-pool each cell's native membership into20x20 pixel blocks and
apply the model's exact18x32 bilinear projection weights. This latter measure is
local footprint coverage, not the true learned feature receptive field. Never
interpolate depth across surfaces. Preserve invalid depth as UNKNOWN. Native
ownership uses exactly the old cell partition and visible-surface semantics.

Report both coverage definitions on nonempty cells by split, head and range;
point strong>=.5 conditional on owned/known-other/UNKNOWN, and BODY_ONLY HEAD
strong-ray height/distance categories. Cached count nonempty>=.5 defines misses.
Do not interpret these as calibrated point detectors or a network recall ceiling.
No automatic mapping from one low metric to a particular neural architecture.

Choose the next implementation from the combined coverage, response and negative
control evidence. Do not retune frozen scores, extend the completed V1 budget,
or claim fresh confirmation. Record a separate successor before any new fit.
