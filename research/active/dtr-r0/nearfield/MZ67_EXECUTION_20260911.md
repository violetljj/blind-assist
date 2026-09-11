# MZ67: execution, storage and evidence

The fixed 4096-frame source uses ten original shards with explicit owner
allocation; its 64 canary frames are part of the budget.
Original runtime, native kernel, source specs and geometry were bound. Only
host-local map paths may differ when their bytes match. Existing failures and
owner logs remain; this record does not authorize recapture or remove evidence.

| Owner / shard | Frames | Recorded full shard seconds |
|---|---|---|
| primary/canary-dense_candidate_05 | 32 | 148.156 |
| worker/canary-dense_candidate_06 | 32 | 138.077 |
| primary/main-mz36_dense_candidate_05_site_001 | 496 | 526.814 |
| primary/main-mz36_dense_candidate_05_site_002 | 512 | 525.692 |
| primary/main-mz36_dense_candidate_05_site_004 | 496 | 513.607 |
| primary/main-mz36_dense_candidate_05_site_007 | 512 | 529.280 |
| worker/main-mz36_dense_candidate_06_site_001 | 496 | 569.946 |
| worker/main-mz36_dense_candidate_06_site_002 | 512 | 603.542 |
| worker/main-mz36_dense_candidate_06_site_003 | 496 | 559.848 |
| worker/main-mz36_dense_candidate_06_site_004 | 512 | 575.343 |

Times are actual full-shard capture/world/dataset/auxiliary/fullframe/pack work,
not pure render-loop time. Primary and worker overlapped, so summed shard times
are not two-host wall time or a speedup measurement. Transfer duration is not
inferred from archive bytes. No unmeasured end-to-end time is reported.

| Owner | Audited frames | Native pairs | Event cells | Known cells | Audit seconds |
|---|---|---|---|---|---|
| primary | 40 | 20 | 576000 | 144000 | 3.209 |
| worker | 40 | 20 | 576000 | 144000 | 4.194 |

Recorded owner full-frame derivation seconds:
`{"primary": 20.151087999984156, "worker": 20.633613999933004}`. Final metadata/index
merge took 10.721 CPU seconds, with actual exit
0. It decoded no raw native on the controller.
The final index binds all source/render/floor health, roles/pairs, RGB/member
hashes, native derivation and actual 80-view review. There are
4216 byte-verified compact members totaling
1774283868 logical member bytes; original RGB file hashes
match for 4096 frames. These totals do not count
archive compression as deletion of original evidence.

## Exact owner-file NTFS compression

| Owner / representation | Files | Logical B | Allocated before B | Allocated after B | Saved allocated B | Compact s | Apply + verify s |
|---|---|---|---|---|---|---|---|
| primary/native | 2048 | 1887698944 | 1895825408 | 1391943680 | 503881728 | 13.987 | 26.156 |
| primary/world_support | 2048 | 943980544 | 947912704 | 74665984 | 873246720 | 4.294 | 14.663 |
| worker/native | 2048 | 1887698944 | 1895825408 | 977534976 | 918290432 | 16.993 | 32.694 |
| worker/world_support | 2048 | 943980544 | 947912704 | 36700160 | 911212544 | 3.930 | 17.288 |

Total actual allocated reduction: 3206631424 bytes
(5687476224 to 2480844800); logical
file bytes remain 5663358976. These are per-file allocation
measurements, not whole-volume free-space changes. They exclude filesystem
metadata and other representations. Reported original raw and compact archive
sizes (7547614895 / 1703344513 bytes) have
different coverage and are not added to this saving.

Only exact owner-local native float32 NPY and world_support int8 files were
compressed via `compact /c /q`, without recursion, force, directory inheritance,
deletion, relocation or source-index mutation. File SHA, decoded array bits,
logical size, file IDs, mtimes and accepted owner hardlink names remained exact;
UNKNOWN/NaN representations were retained. Native audit arrays stay on their
owner. Compact-only and full apply/verification times above are distinct;
warm-cache verification reads are not production latency claims.

Both owner release records and all four compression execution records confirm
actual exit 0 and no surviving owned process. No unrelated processes were stopped
by this record generator. Exact paths/hashes and source release evidence are in
[the sealed delivery bindings](../../../../artifacts.local/work/mz67-topology-source-20260911/records-inputs-sealed.json) and
[storage completion](../../../../artifacts.local/work/mz67-topology-source-20260911/storage-completion-v1.json); the generated
[execution completion](../../../../artifacts.local/work/mz67-topology-source-20260911/execution-completion.json) binds the two public
documents, terminal, COMPONENT inheritance and the sole archived experiment row.
All source, checkpoints, raw, compact packs, prior outputs and failure evidence
are retained. Source completion is not completion of the overall capability goal.
