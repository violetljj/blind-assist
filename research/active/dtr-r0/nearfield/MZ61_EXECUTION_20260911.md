# MZ61: execution and evidence preservation

The [source protocol](MZ61_GEOMETRY_SOURCE_20260911.md) used ten original shards
and exactly 4096 captures. Primary owned dense05 (28 canary + 2020 main), worker
dense06 (36 canary + 2012 main). Both initial canaries passed before the remaining
4032 were admitted. Main allocation stayed explicit; only host-local `map_file`
was remapped with the same map SHA and all other case/spec fields unchanged.
Primary main launch followed MZ62 process exit; no primary UE/model overlap occurred.

Primary main session 14320 exited 0 after 2100.009 seconds for four shards.
Worker main shard work summed to 2284.487 seconds. These scopes omit their
separate canaries and differ from end-to-end two-host elapsed time; summing them
does not measure a speedup. Both hosts used existing warm DDC. Capture-stage
time includes map loading, readiness, rendering/export and shutdown; it is not
a pure frame-loop or cold-start benchmark.

The [first primary 504-frame timing diagnostic](../../../../artifacts.local/work/mz61-geometry-source-20260911/collection-throughput-diagnostic-v1/result.json)
measured 537.561 seconds total, of which capture was 492.096 seconds (91.54%).
Within its recorded evidence, map load was 15.360 seconds, readiness polls
13.347 seconds and UE callbacks 419.657 seconds. These are not established
nonoverlapping components and must not be summed as a new timing total.
Settling remained 32 ticks, 64 on first use; its isolated elapsed share was
not measured. The 91.54% finding applies to this one shard, not all source or
a controlled machine comparison. Further packing compression cannot remove
that measured dominant capture cost.

Owner-local independent block-loop audits checked 32 primary and 48 worker
predeclared native frames: 1,152,000 event cells, 288,000 valid cells and 40
raw mask/depth pairs, all exact. Audit CPU times were 3.076 and 4.622 seconds.
Root separately viewed every one of the 80 reserved cases, recorded in the
[final visual review](../../../../artifacts.local/work/mz61-geometry-source-20260911/visual-review-final-v1.json).
Owner full-frame derivation summed to 20.200/20.609 seconds;
this excludes unrelated failed preflights and is not a model-inference cost.

Preserved failures were repaired without recapture or scientific input changes:

- Initial full-frame consumers expected slash-separated receipt member keys;
  Windows backslashes caused `KeyError evaluator/metadata.json`. Versioned
  consumers normalized separators and derived labels from the same captured
  bytes. Original failure logs/helpers/configs remain in
  [windows-path-fix-v1](../../../../artifacts.local/work/mz61-geometry-source-20260911/windows-path-fix-v1/receipt.json).
- The first primary main invocation had a PowerShell argument-binding error
  before the driver body. Explicit named arguments launched the unchanged
  approved driver; no source capture was duplicated.
- `combined-v1` compared a world-space target hash with a frozen camera-relative
  geometry ID. The [coordinate repair](../../../../artifacts.local/work/mz61-geometry-source-20260911/merge-frame-fix-v1/result.json)
  verified all 4096 original forward transforms and 1024 canonical hashes;
  maximum inverse position/yaw errors were 2.274e-13 m / 1.421e-14 degrees.
  Raw source, labels, roles, geometry IDs and frozen specifications were unchanged.
  Its original failed output remains; fresh `combined-v2` passed in 10.051
  CPU seconds, with root-observed session 37690 exit 0.

Transparent NTFS compression was a separate storage action, not a scientific
variant. Exact owner-native paths were inspected for ownership, schema,
no reparse points and writers, then compressed using `compact /c /q` without
recursive scope, forced recompression, folder inheritance or deletion.
All before/after SHA, array bits, file IDs, mtimes and owned aliases passed.
Native savings were 497,922,048 primary + 906,506,240 worker = 1,404,428,288 bytes.
Their compact-only times were 14.073/17.223 seconds; full apply/verification
26.177/34.640 seconds. These are per-file allocated data savings, not whole-volume
free-space deltas. Primary world-support saved 873,488,384 bytes in 14.429
apply/verification seconds (4.445 compact-only); its storage-before SHA also
matched existing world-verification mask hashes. Worker world-support saved 911,081,472 bytes in 17.365 apply/verification
seconds (4.423 compact-only). Its [final release](../../../../artifacts.local/work/mz61-geometry-source-20260911/returned-worker-world-support-v1/worker-world-support-v1/release.json)
confirms released owner processes/port; the compression command exited 0. Combined native/support
savings are 3,188,998,144 allocated bytes across 8192 files; no raw was deleted.

[Primary release](../../../../artifacts.local/work/mz61-geometry-source-20260911/primary-release-final.json)
and [worker release](../../../../artifacts.local/work/mz61-geometry-source-20260911/returned-worker-main-v2/worker-main-release-final.json)
confirm capture resources were released. Primary storage session 6701 exited 0
and its separate release confirmed no remaining owned process. All durable raw,
failure evidence, native compression records and source receipts are preserved.
The [parts manifest](../../../../artifacts.local/work/mz61-geometry-source-20260911/parts.json)
binds the source; subsequent model transfer has its own budget and conclusions.
