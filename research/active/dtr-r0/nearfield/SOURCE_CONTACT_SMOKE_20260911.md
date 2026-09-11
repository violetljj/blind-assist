# Rendered support and terrain check

Four new diagnostic frames completed on the secondary worker in83.837s:
corrected wooden slats and compound duct, each with its original context partner.
Original target transforms,1x scale and assets remain unchanged. These frames
were not admitted to a training dataset and are not inputs to MZ76.

All six unique foot centres, checked in both contexts (12 native traces), hit
sidewalk at0.7299999618530273m: exactly the planned foot-bottom height. Each of
the four224-point ground grids also hit only sidewalk. Task-owned collector and
floor-probe copies pass the actual controlled actors to ActorsToIgnore, including
camera-floor probes. Twelve CPU forwarding checks passed before capture;
original runtime files were rehashed unchanged. Settling/render/export settings
and original-component target-ray code were not modified.

| Assembly, each partner | Scene positive query pixels | Isolated target contribution |
| --- | ---: | ---: |
| Wooden slats | HEAD_NEAR696; other queries0 | 25+671=696 within original3cm scene-depth tolerance |
| Compound duct | HEAD_FAR2242; other queries0 | 2175 within original3cm scene-depth tolerance |

The remaining67 duct scene positives are retained without target attribution.
Each context pair has identical event masks, scene depth on the event union,
and isolated target arrays. Component-ray checks remain0MATCH/32UNKNOWN for
each wooden target and1MATCH/31UNKNOWN per duct frame. These UNKNOWN results
are not ToF invalidity. Terrain traces do not establish whole-foot contact or
mechanical stability; the wood point/line contact and duct attachment limitations
in the [geometry report](SOURCE_CONTACT_REPAIR_20260911.md) still apply.

Root and worker actually viewed the four-frame preview. UE and owned Python/Zen
processes were released; the final worker inventory is empty. No large collection
or raw-data deletion followed this diagnostic. User requested a pause after MZ76.

[Report](../../../../artifacts.local/work/source-contact-smoke-20260911/REPORT.md),
[result](../../../../artifacts.local/work/source-contact-smoke-20260911/result.json),
[receipt](../../../../artifacts.local/work/source-contact-smoke-20260911/receipt.json),
[final receipt](../../../../artifacts.local/work/source-contact-smoke-20260911/final-receipt.json),
[preview](../../../../artifacts.local/work/source-contact-smoke-20260911/preview-all4.jpg).

Result SHA256:134590010ea8392a78dcf7e250aad329d586e51209fd4b1eebc9202ffe54b13a.
Receipt SHA256:42062832b85c730db400dac13afba4a99085b09d12927c7143f13ee8a9fb8d2e.
Final receipt SHA256:3bf58cdf537c91ba56c98d3bbf172025f6581b617f79298ed5e6816bab19526b.
