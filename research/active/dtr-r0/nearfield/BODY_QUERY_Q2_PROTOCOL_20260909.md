# Q2 point-evidence decomposition: frozen B, zero training

2026-09-09 EXPLORE. Inspect BODY-QUERY-V1 B final2000, SHA
dd6fab0bddf436477e9076bb47a0df8914b4ddd21dcf0ea9897143643460726b.
Run one inference pass through the existing320 TRAIN/DEV/EVAL RGB frames on the
primary GPU with the previous batch32 policy. No capture, optimization, V2 fit,
threshold selection or expansion of the previous training budget is authorized
by this diagnostic. Previously consumed same-world Development remains consumed.

Question: are missed nonempty query cells associated with rare strong projected
point readouts that disappear under the existing masked mean, or with weak
responses at every sampled point under the existing decoder?

Capture inputs/outputs of the existing query_point MLP and apply the unchanged
linear count readout to each point. Verify that masked mean point logits reproduce
original count logits, including bias. The identity holds for logits, not averaged
softmax probabilities. Mask invalid projections; report effective point counts.
No point-level classifier has been trained or calibrated independently.

Predeclare mean, best-point and top3-point mean-logit diagnostics. Rank valid
points by logsumexp(nonempty class logits)-empty logit. Use nonempty probability
>=.5 and report the stricter .9 point threshold too. Break down true nonempty
misses and empty cells by head/range and condition on every split. Frozen B near
aggregation and DEV cutoffs are retained for descriptive top3/max readout effects;
they are not newly selected or deployed thresholds.

Grounding proxy: bilinearly sample the cached18x32 native HEAD/BODY support at
the projected points. Preserve known/UNKNOWN mass and report positive-support
overlap. This is a coarse2D head-support proxy, not per-cell3D point truth or a
receptive-field bound. Strong logits on a point are not proof of obstacle evidence.

A pooling dilution hypothesis needs available sparse responses plus negative-control
selectivity; widespread new empty-cell/LOW/ABOVE alerts weakens that interpretation.
All-weak point readouts only localize failure to the combined sampling/feature/MLP/
decoder path, not the backbone alone. No automatic choice of attention or
cross-attention follows either category. Complete the zero-fit report first.
