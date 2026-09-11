# MZ65: reuse controlled obstacle actors during capture

EXPLORE engineering. MZ61 first-primary-shard timing put492.096 of537.561
pipeline seconds in capture. The source collector already writes native pairs
asynchronously without observed queue pressure. Earlier viewport and reduced
settling probes did not establish a quality-preserving improvement. The new
question is whether retaining reusable obstacle actors lowers capture time while
preserving the same settled images, native evidence and weak ToF packets.

Use a versioned copy of the completed MZ48 capture runtime, not the concurrent
checkout collector. All arms have identical phase instrumentation and cleanup.
Only B enables actor pooling. A and R execute the original create/destroy policy;
R is the original repeat after B. The candidate key contains the complete
immutable object descriptor and excludes only center, scale, size and rotation.
Only actor-origin placement is admitted. Reapply the requested transforms;
retain material/mesh identity and active object order. Hide inactive slots in
game and editor, disable component visibility, shadows and collision; restore
the exact saved flags on reactivation. Destroy every pooled actor at exit,
including inactive slots. Do not save or modify any source map or asset.

Freeze16 cases selected from the completed MZ61 worker-owned dense06 site001
source: four object families, HEAD_ONLY near and BODY_ONLY far, both support
contexts. Preserve source world coordinates and metadata. The16-case order puts
the two support contexts next to one another. A/R specs are byte-identical;
B differs only by the pooling flag. Keep original readiness, settling intervals,
render-call sites, native depth export, packet formation and asynchronous writer.
No frame skipping, RGB reuse, reduced resolution, compression change or model
change is part of this contrast.

Budget is three16-frame passes on the same worker, A then B then R,48 captures
total. Do not run another worker model or capture job concurrently. Primary
model work may continue. Record the existing desktop GPU use; do not stop
unrelated processes. Bind code, specs, source identities and exact quality
definitions before launch. A mechanical failure retains its bytes and may be
repaired with the same scientific inputs, without a quality-driven retry.

The fixed comparison uses exact query/event labels, native validity and UNKNOWN,
world-support arrays, full-frame counts, weak45-degree packets and paired target
depth. A/R must satisfy the corresponding exact baseline identity checks.
For RGB and depth values, compare whole image, the union of fixed A native
query-positive pixels and its3-by3 boundary. Per-case MAE, p95, p99, maximum error
and changed-value counts for B versus A must not exceed the corresponding
A-versus-R repeat difference. An exact A/R component therefore requires exact
B parity. Freeze masks, arithmetic, empty-mask handling and floating-point
comparison in the scorer before any capture; do not widen tolerances after B.
Report complete residuals, including failures. Require identical readiness
configuration and actual settling/render counts, with every readiness check READY
and no incomplete-resource indication. Report asynchronous readiness poll counts,
elapsed time and resets descriptively; those may vary with completion latency.
Inspect all16 A/B/R panels for actual placement,
occlusion, support context and absence of inactive actors. Flag assertions alone
are not image-quality evidence.

The speed check requires B's map-excluded capture wall time to be below both
A and R, with its saving exceeding the absolute A/R wall-time drift, and its
total actor-preparation time below both baselines. Report total job and map-load
costs, per-phase timings, created/reused slots, readiness, render calls and writer
drain time. Removed actor operations are not a measured speedup. No sweep,
outcome-driven case selection or additional pass is allowed.

Quality plus speed supports keeping this implementation for this controlled
capture workload. Quality PASS without speed gain does not justify deployment
for throughput. Any quality failure keeps the original collector; baseline
instability is NOT_EVALUABLE for preservation, not a model or sensor result.
This48-frame engineering check is neither a new training dataset nor proof of
natural-scene, model, hardware or safety performance. Preserve every original
source and failed receipt. Return thin evidence and reviewed panels; stop and
verify release of task-owned UE, capture, evaluator and export processes and the
worker scheduled task. Durable captured evidence remains on its owner host.
