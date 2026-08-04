# Scale-Free Traversability R0 Protocol

Date: 2026-08-04

Status: `FROZEN_BEFORE_SCALE_FREE_OUTPUT_EXECUTION`

This Development-only diagnostic reuses the three consumed fixed-phone RGB
sessions. It does not read Samsung Quick Measure distance, camera height, or any
metric truth. The question is limited to whether a deterministic scale-free
three-band operator can execute stably enough to justify a later independent
evaluation. It cannot establish accuracy, clearance, safety, or a usable route.

## Frozen operator

The source is the already locked Depth Anything V2 Metric Hypersim ViT-S
checkpoint with SHA-256
`B782898D8A3E8BE1F639DE33837ED85E9B4B73E40F8F5E5CD99067588D722545`.
Its positive depth output is converted to log inverse depth `q = -log(depth)`.
Multiplying every depth value by an arbitrary positive constant adds one common
offset to `q`; the following row-relative subtraction removes that offset.

- analysis ROI: normalized `x=[0.05,0.95)`, `y=[0.30,0.90)`;
- bands: left `x=[0.05,0.35)`, center `x=[0.35,0.65)`, right
  `x=[0.65,0.95)`;
- each ROI row requires at least 90% valid positive depth;
- row baseline: 25th percentile of valid log inverse depth across the full ROI
  width;
- per-pixel relative intrusion: `max(0, q - row_baseline)`;
- band score: 85th percentile of valid per-pixel intrusion;
- a frame is valid only when at least 90% of every band is valid;
- causal smoothing: median of the current and previous four valid scores from
  the same session;
- the first four frames are `UNKNOWN_WARMUP`;
- after warm-up, the lowest-score band is reported as relatively most open only
  when its margin to the second-lowest band is at least `0.08` log units and it
  won at least four of the current five raw frames; otherwise output
  `AMBIGUOUS`.

The only permitted labels are `RELATIVELY_OPEN_LEFT`,
`RELATIVELY_OPEN_CENTER`, `RELATIVELY_OPEN_RIGHT`, `AMBIGUOUS`, and `UNKNOWN`.
The words clear, safe, blocked, metres, distance, collision probability, and
future prediction are not authorized output semantics.

## Frozen diagnostic summaries

The session, not a frame, is the independent unit. Report execution coverage,
non-ambiguous recommendation coverage, per-session label counts, modal-label
fraction, and per-band temporal median absolute deviation. Do not compare with
the consumed Quick Measure values. No threshold, percentile, ROI, smoothing
window, checkpoint, or band definition may be changed after reading outputs.

Positive mechanics terminal:
`SCALE_FREE_TRAVERSABILITY_R0_EXECUTES_STABLY_DEVELOPMENT_ONLY`, requiring all
three sessions to have execution coverage at least 0.95 and modal-label fraction
at least 0.80. This terminal only justifies designing an independent evaluation.
Any other outcome is
`SCALE_FREE_TRAVERSABILITY_R0_UNSTABLE_OR_AMBIGUOUS_DO_NOT_INTEGRATE`.
