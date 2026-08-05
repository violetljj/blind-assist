# P3 R0.2 local asset inventory

The route-specific inventory is now current for everything downloaded during the public-source audit.

- Full Bonn official archive: 16,395,422,367 bytes.
- Independently downloaded Bonn `kidnapping_box` verification archive: 618,957,920 bytes.
- Full extracted Bonn tree: 59,164 files and 16,821,630,031 bytes.
- Bonn identity catalog, receipt and capacity evidence are SHA-bound and retained.
- The eight-video ARKit validation roster and 40-asset HEAD receipt are present, but media body bytes remain zero.

The archive and extracted Bonn tree intentionally coexist, so current route payload is about 33.84 GB before the planned 5.39-GB ARKit transfer. The individual Bonn archive is retained only as the independent reproduction proving that missing depth references also exist in the official per-sequence package.

The repository-wide `DATASET_MASTER_LEDGER.json/.csv` was last generated on 2026-08-02 and does not include this route. It will be regenerated once after the locked ARKit transfer and continuity audit, avoiding two expensive full-workspace scans. Until then, this scoped inventory is the authoritative P3 storage record.
