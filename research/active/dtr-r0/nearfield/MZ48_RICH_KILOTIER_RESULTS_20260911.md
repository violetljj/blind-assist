# MZ48: 2,560 paired rich-object frames delivered

The [fixed source acquisition](MZ48_RICH_KILOTIER_20260911.md) completes all
2,560 intended frames on two machines. All pass native/source checks; all
1,280 target/context pairs retain identical event masks and event-surface
depth. The fixed64-frame canary passed before the2,496-frame expansion.
The original relation intent matches2,432 frames. All128 mismatches remain
with their actual native labels, including small pipe/ladder placements whose
visible surface does not meet the intended body query.

| Source property | Delivered value |
|---|---:|
| Sites / frames per site | 8 / 320 |
| Training-candidate / held-out-site frames | 1,920 / 640 |
| Families | pipe, ladder, pouch, woody birch, oblique rod |
| Relations / support states / ranges / settings | 4 / 2 / 2 / 4 |
| Actual BODY_NEAR / BODY_FAR / HEAD_NEAR / HEAD_FAR positives | 512 / 640 / 640 / 640 |
| Source-valid frames / frame-contract UNKNOWN bits | 2,560 / 0 |
| Predeclared native samples independently checked and viewed | 80 |
| Primary / secondary captured frames, including canary | 936 / 1,624 |

The80 visual/native checks comprise64 canary frames and two prescribed frames
from each main site shard. Automated source, packet, label, pair, RGB-byte and
archive checks cover every frame. These checks do not imply that all2,560
images were manually viewed or that every physical volume is observable.

The compact training ZIPs total1,095,307,842 bytes, with260,314 bytes of
additional native cell supervision. Complete raw captures total4,740,414,257
bytes and remain on their owning machines. The training representation is
23.1% of the raw byte count: original PNG bytes, compact range/validity packets,
labels/metadata/provenance and80 native audit samples are retained in the ZIPs.
The remaining full native evidence stays with raw captures. This reduces the
working/transfer representation; the raw evidence has not been deleted.

An additional training/evaluator-only label stores whether any native point
in each45-degree angular cell belongs to each query, independent of which
echo bins were selected. It is separate from the immutable original ZIPs.
Every old selected-echo query label is a subset, and local-known masks match.
There are7,169,167 unknown angular cells and352 positive frame-event bits with
no native witness inside this crop. Preserve these gaps; zero frame-contract
UNKNOWN bits does not establish complete local visibility. Merge proxies do
not inherit selected-bin contributor labels. None of these labels, native
depths or family/site/instance IDs enter the predictor.

Actual scheduling improvements include dual-host nonoverlapping shards,
reuse of primary's existing CitySample derived-data cache after an evidenced
zero-frame cold-cache compile stall, and exclusive transfer of an unstarted
312-frame shard to the faster worker. Primary's launch guard records the
intentional handoff before creating any capture directory. Only primary map
paths were adapted; reversing that mapping restores each original spec byte
hash. Case geometry, assets, runtime and scientific sample identities remain
unchanged. The interrupted zero-frame attempt remains as mechanical evidence.

Primary's three completed shards took21.9 minutes including capture and
checks. All owned UE/Zen/dispatch processes and checked ports are released
on both machines. No model fit, inference or persistent dense-feature cache
belongs to this source experiment. Full counts, ten-shard bindings, hashes,
mechanical mapping and release receipts are under
`artifacts.local/work/mz48-rich-kilotier-20260911/`:
`source-index.json`, `source-total-receipt.json`, `primary-source-index.json`,
`primary-release-final.json`, `secondary-resource-release.json`, and
`owned-source-check-final.json`.

Retain this source as COMPONENT for controlled Development. The objects remain
synthetic opaque fixtures, including leafless woody birch; scales/support
arrangements and consumed sites limit transfer claims. The original640-site
holdout remains a future-fit split, not fresh blind evidence. The next
decision is whether native local supervision and candidates that survive
missing echoes improve useful detection with acceptable false alerts.
