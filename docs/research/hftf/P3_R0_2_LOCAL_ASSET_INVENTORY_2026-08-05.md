# P3 R0.2 local asset inventory

The route-specific inventory is now current for everything downloaded during the public-source audit.

- Full Bonn official archive: 16,395,422,367 bytes.
- Independently downloaded Bonn `kidnapping_box` verification archive: 618,957,920 bytes.
- Full extracted Bonn tree: 59,164 files and 16,821,630,031 bytes.
- Bonn identity catalog, receipt and capacity evidence are SHA-bound and retained.
- The eight-video ARKit validation extension retains 9,609 files and 267,157,183 bytes after temporary source ZIP deletion; its 2,400 selected identities passed independent SHA and continuity verification.

The archive and extracted Bonn tree intentionally coexist, so current retained route payload is about 34.10 GB. The ARKit transfer read about 5.39 GB from the source but retained only the scoped continuous frames, intrinsics and trajectories. The individual Bonn archive is retained only as the independent reproduction proving that missing depth references also exist in the official per-sequence package.

The repository-wide `DATASET_MASTER_LEDGER.json/.csv` now includes all eight ARKit media sessions. The new prefix was fully decoded and hashed, while unchanged historical sessions retained their prior hashes; global duplicate groups, role conflicts, summaries and reports were recomputed. The resulting JSON SHA-256 is `5A0BBFEA3467F2D160D70C6E838D28147BEF7B8190948ECD63DACAE8227BBE42`. This file remains the concise authoritative P3 storage and retention record.
