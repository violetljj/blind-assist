# Executable support contact correction

The CPU repair produced complete name-matched cube transforms for the wood and
duct supports. Target mesh scale1x, rotation and actor origin remain unchanged.
No new meshes, import step or asset changes are required. The calculation uses
the previously exported actual LOD0 triangles, not bounding-box contact.

| Support | Corrected top height above site floor (m) | Increase (mm) | Horizontal target contact area (mm2) |
| --- | ---: | ---: | ---: |
| Wood left bracket | 1.662247914292 | 3.578613 | 0: point/line contact only |
| Wood right bracket | 1.659831156019 | 1.161854 | 0: point/line contact only |
| Duct rail0 | 2.002197341919 | 502.197342 | 641.324 |
| Duct rail1 | 1.502874832153 | 2.874832 | 523.526 |

All four calculated residual gaps are zero. The two wood posts and four duct
legs were resized/repositioned consistently with the changed brackets and rails.
The geometric checks cover six floor-touching parts and eight positive-area
cube-to-cube contact interfaces. The wood mesh has no exactly horizontal source
triangle; closing its gap against a horizontal bracket yields point/line contact,
not positive-area contact. Wood stability, friction and attachment remain unknown.

These are source-mesh/primitive calculations against the declared floor plane.
They do not prove actual terrain contact, stable support or native visibility.
The new transforms have not yet been rendered. Their next check should ignore
controlled actors when probing terrain and preserve actual extra query bits;
the [four-frame surface smoke](SOURCE_SURFACE_SMOKE_20260911.md) already showed
why collision-depth agreement and rendered target contribution must be distinct.

CPU execution passed in0.313s, exit0; no UE, capture, native-depth decoding or
model work occurred. No failed MZ72 source was rewritten or admitted.

[Complete replacement payload](../../../../artifacts.local/work/source-contact-repair-20260911/payload.json),
[contact polygons and triangle witnesses](../../../../artifacts.local/work/source-contact-repair-20260911/geometry-evidence.json),
[CPU script](../../../../artifacts.local/work/source-contact-repair-20260911/repair_supports.py),
[report](../../../../artifacts.local/work/source-contact-repair-20260911/REPORT.md),
[receipt](../../../../artifacts.local/work/source-contact-repair-20260911/receipt.json).
Payload SHA-256: fdaaf80acc502cc4ad2135e98f9776593c76127a80b00fb6a9620e8175a23431.
Receipt SHA-256: 44979b4f3658ecaea98ba67a6c4b9c34311c7ab926b7e33e21cf9a8bab4199f5.
