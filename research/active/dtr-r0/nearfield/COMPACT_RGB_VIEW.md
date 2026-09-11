# Opt-in lossless RGB view

`compact_rgb_view.py` writes a fresh, separately named WebP view using fixed
`lossless=True, method=0, quality=100, exact=True`. It never rewrites PNGs,
archives, existing views or frozen predictor manifests. Only single-frame
8-bit RGB/RGBA PNG inputs are supported; every output must decode to exactly
the original RGBA bytes, including RGB values behind transparent alpha.

```json
{
  "schema": "compact-rgb-view-input-v1",
  "images": [
    {
      "frame_id": "example",
      "source": {"path": "F:/explicit/original.png", "sha256": "SHA256"},
      "output": "model/rgb/example.webp"
    }
  ]
}
```

A compact archive member is also accepted without extraction: replace `source`
with `{"archive":"F:/explicit/training.source.zip","archive_sha256":"SHA256",
"member":"model/rgb/0000.png","sha256":"ORIGINAL_PNG_SHA256"}`. Container
paths must be absolute; output/member paths must be clean relative paths.
Duplicate frame IDs or case-folded output filenames are refused.

```text
python compact_rgb_view.py --manifest explicit-images.json --output NEW_VIEW
python test_compact_rgb_view.py --output NEW_TEST_EVIDENCE
```

`mapping.json` retains ordered frame IDs, original source/member identity,
original/output SHA, decoded RGBA SHA and output filenames. A consumer must
explicitly adopt its `rgb_files` mapping; an old `.png` predictor path is not
silently redirected. This is an RGB-only view, not a replacement complete
dataset. Its ordinary directory can be packed with `data_lightweight.py` or
used alongside unchanged packet/evaluator inputs. Unlike
`mz72_transfer_source.py`, this changes encoded image bytes and therefore must
have its own source identity; the existing byte-preserving transfer view remains
unchanged.

`result.json` records actual bytes and timings. `receipt.json` binds the input
manifest, original containers and every output. Partial failures retain
`progress.jsonl` and `failure.json`; rerun into a new directory after diagnosis.
Existing output roots are rejected even if their contents appear identical.
Pillow with WebP support is required; the builder installs nothing and creates
no background processes or feature cache. Pixel preservation does not imply
PNG metadata preservation, hardware validation or model performance equivalence.

The first bounded acceptance used the same eight existing MZ67 TRAIN images as
the earlier codec probe. The complete builder took 0.783 s, including 0.199 s
encoding, and reduced encoded RGB bytes from 3,546,777 to 2,412,046 (31.99%).
Every decoded RGBA value was exact. Five focused checks passed; packing the new
view and reading it through `CompactSource` also passed. This is an eight-image
measurement, not a full-dataset compression ratio or actual freed disk space.
No original archive was converted or deleted. [Acceptance report](../../../../artifacts.local/work/vl53l8cx-packet-contract-20260911/compact-rgb-v1/REPORT.md).
