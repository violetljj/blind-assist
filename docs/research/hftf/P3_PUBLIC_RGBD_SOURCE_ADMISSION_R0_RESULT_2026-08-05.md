# P3 public RGB-D source admission R0 result

## Conclusion

The label-blind public-source audit reached:

```text
P3_PUBLIC_RGBD_SOURCE_ADMISSION_R0_CAPACITY_READY_FOR_ROLE_FREEZE
```

The official [Bonn RGB-D Dynamic Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) full archive contains all 26 published recording-sequence identities. After excluding 15 exact sequences already inspected, selected or consumed by repository-recorded development programs, 11 independent sequence parents remain. Every actual RGB/depth file and both timestamp indexes were SHA256-processed, and every remaining parent contains at least 332 unique, timestamp-synchronized RGB-D identities. The frozen minimum is 8 parents and the target is 12, so the minimum capacity gate passes while the target is not reached.

This means controlled self-collection is not currently required. It does not mean that a P3 holdout is sealed, that transition coverage is sufficient, or that P3 R0.2 training is authorized.

## Integrity finding and treatment

The 15.26-GiB full archive has SHA256 `D2AFDC28...A84B`; all 59,243 ZIP entries pass CRC. Nevertheless, all 26 `depth.txt` indexes contain at least one reference whose PNG is absent. An independently downloaded official `rgbd_bonn_kidnapping_box.zip` reproduces the same nine absent references, so this is not a partial local download or extraction error.

Attempt 02 deliberately failed before writing output because its producer required every index reference to exist. The correction was committed before Attempt 03. Attempt 03:

- hashes every actual RGB/depth file and the two source timestamp indexes;
- discloses every absent depth reference in the local receipt;
- never admits an absent reference as a metric-valid frame;
- admits only unique RGB-depth pairs within 50 ms;
- requires a four-frame RGB run whose adjacent gaps are at most 500 ms.

This is evidence-validity handling, not a lowered parent gate. A missing sensor sample becomes invalid evidence; it does not silently become depth truth, and it does not automatically invalidate the thousands of real synchronized samples in the same recording.

## Bound evidence

| Evidence | SHA256 |
|---|---|
| Protocol | `DD494869...E0C03ED` |
| Frozen auditor | `2B8DD4FD...6F7052` |
| Bonn identity producer | `848B5546...2D348` |
| Bonn ancestry exclusions | `31B4E806...79B33D` |
| Full Bonn archive | `D2AFDC28...A84B` |
| Attempt 03 identity catalog | `0F6307FB...4E20E` |
| Attempt 03 identity receipt | `6E4089B3...F6D56` |
| Attempt 03 result and exact replay | `3F9BCC4F...F4A28` |

The result replay is byte-identical. Auditor tests and materializer tests each pass `3/3`.

## Authority boundary

The result authorizes only a new, pre-frozen P3 R0.2 data-role and sealing contract. That successor must still prove parent disjointness and sealed transition coverage before activation. No holdout outcome was opened, and no checkpoint, P3 model, optimizer or training runtime was created.

TUM, JRDB, EgoBody, HOI4D and additional ARKitScenes identities remain metadata-eligible within the frozen public-source universe, but their payloads were not materialized after Bonn crossed the minimum gate. They may be opened only by a separately frozen cross-domain or 12-parent target-expansion audit; they are not post-hoc replacements for a later failed sealed cohort.
