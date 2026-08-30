# CARLA storage compaction — 2026-08-31

## Outcome

CARLA experiment storage was compacted without deleting a cohort, path, or
payload. `E:\linnan\CARLA` is a junction to
`F:\ba-data\blindassist-artifacts-20260805\runtime\carla-asset-library`; the
physical experiment root is that target's `experiments` directory. All byte
limits and free-space measurements below therefore refer to F:, not E:.

| Measure | Before | After |
| --- | ---: | ---: |
| Named files | 144,641 | 144,641 |
| Logical bytes | 98,810,166,491 (92.024139 GiB) | 98,810,166,491 (92.024139 GiB) |
| Unique NTFS file identities | 144,641 | 87,069 |
| Unique physical bytes | 98,810,166,491 (92.024139 GiB) | 42,926,979,146 (39.978865 GiB) |
| Hard-link savings | 0 | 55,883,187,345 (52.045274 GiB) |
| F: free bytes observed by apply | 278,817,595,392 (259.669121 GiB) | 334,586,789,888 (311.608231 GiB) |

The exact identity-based saving is 55,883,187,345 bytes. The free-space delta
observed during the operation was 55,769,194,496 bytes (51.939110 GiB); the
small difference is volume activity outside this maintenance operation and is
not used as the deduplication authority.

## Frozen scope and receipts

The reviewed plan admitted 95,255 old, manifest-listed, sealed PNG paths. It
excluded 42,434 unsealed PNG paths, found no unlisted sealed candidate, and
found no alternate data stream. The plan contained 29,848 duplicate groups and
57,572 action paths. Every affected duplicate identity was fully observed.

- Frozen plan identity:
  `43D698E18E9A1CA11B27558821AA8D0FDCA68758654A158959CA00ED56B08D1F`
- Dry-run plan file:
  `artifacts.local/maintenance/carla-storage-20260831/dry-run-v2/dedupe-plan.json`
- Dry-run plan file SHA-256:
  `0B7541CE66A8A5ED31D54C7EAFB9B4643A9BCD378FFDE80FB688B1741D9A5EBE`
- Apply receipt directory:
  `artifacts.local/maintenance/carla-storage-20260831/apply-v2/`
- Apply result SHA-256:
  `88E80E7E29F7FDB661725C635F9352D4DD6F9AFB74B96C7ADB86D443D6D24AD7`
- Apply event journal SHA-256:
  `A6BABC334E7206EF009F1F26E8BFD8E9F9409A8CE9FD68540232E8BE758751DA`

Apply completed 57,572 `STARTED` and 57,572 `COMPLETE` path events. Its result
sets `content_or_path_deleted` to `false`.

## Verification

- A post-apply audit kept the named-file count and logical bytes unchanged and
  measured the exact 55,883,187,345-byte hard-link saving.
- A representative C11 canonical/model pair retained identical 604,400-byte
  content and SHA-256
  `003AD26F746AB6A505AC4D74272204FFD01DD9BB673AD32335F1E5F5C7B6C95E`,
  and both paths now resolve to the same NTFS identity with link count 2.
- All 36 sealed-manifest/result artifact hash pairs were rechecked with zero
  failures.
- The post-verification dry run found zero duplicate groups, zero action paths,
  and zero reclaimable bytes. Its plan identity is
  `B2179AA6B7A3EB86D3A5B4DE174AC15842B77F1247C0B8265BA58FD669CDEFF4`.
- No maintenance lock, coordination lock, active lease, or temporary dedupe
  file remained.

## Bounded structure going forward

`carla_storage_policy.json` freezes an 80 GiB unique-experiment cap, a 100 GiB
backing-volume free-space floor, an 8 GiB normal run reservation, and a 16 GiB
C4 reservation. Official C0/C1/C2/C4/N1/N2 runners acquire an atomic persistent
lease before output creation and recheck the live boundary after each material
stage. C4 package/reuse trees use same-volume hard links with copy fallback.
Automatic evidence deletion remains disabled: overflow means
`REFUSE_NEW_RUN`, not eviction.

Hard links are not backups. All linked names address the same bytes, so sealed
payloads must remain immutable. The implementation follows Microsoft's
[hard-link and junction model](https://learn.microsoft.com/en-us/windows/win32/fileio/hard-links-and-junctions),
uses NTFS file identity/link count for physical accounting as exposed by
[`BY_HANDLE_FILE_INFORMATION`](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/ns-fileapi-by_handle_file_information),
and retains path plus SHA-256 as the durable evidence identity. The operational
boundary also follows the useful distinction in
[DVC's large-dataset guidance](https://docs.dvc.org/user-guide/large-dataset-optimization):
hard-link-backed content must be treated as read-only.

## Claim boundary

This is storage-maintenance evidence only. It changes no CARLA observation,
prediction, metric, Development result, source-disjoint status, or safety claim.
