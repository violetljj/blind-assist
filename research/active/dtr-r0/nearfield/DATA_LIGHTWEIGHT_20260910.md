# Lossless compact sources for nearfield data

2026-09-10 engineering result. MZ36's 400-frame source (1,247 files) is now
available as a directly readable lossless package: 751,784,227 bytes become
307,320,958 bytes, a 59.12% reduction. Pack time was 6.96s; full decode, SHA-256
and original-byte comparison took 2.30s. These are measurements of this batch,
not a projection of savings for every dataset. Original source files remain;
the smaller package does not itself reclaim their disk space.

Ordinary transparent NTFS compression was then applied only to the completed
MZ36 `returned-full-v1` source. Its measured file allocation fell from
755,059,248 to 424,073,776 bytes: 330,985,472 bytes (315.65 MiB, 43.84%) were
released in 11.73s. Paths, file identities, logical sizes and modification times
are unchanged. All 1,247 files again match the verified package byte for byte.
The package remains a separate recoverable transfer copy; these allocation
savings describe the original source tree, not net savings across both copies
or the whole drive. No source was deleted or moved.

[data_lightweight.py](data_lightweight.py) stores existing PNG bytes and deflates
NPY/metadata bytes, with no dtype conversion or quantization. The package records
relative paths, byte sizes and SHA-256. Pack and verify use only the Python3.10+
standard library. Array/image readers load NumPy/Pillow on demand.

```python
from data_lightweight import CompactSource

with CompactSource(source_directory_or_zip) as source:
    depth = source.load_array(relative_native_npy_path)
    image = source.load_image(relative_rgb_png_path)
    metadata = source.read_json(relative_json_path)
```

```powershell
python data_lightweight.py pack --source CAPTURE --output PACK.source.zip --receipt pack.json
python data_lightweight.py verify --archive PACK.source.zip --source CAPTURE --receipt verify.json
```

Put the output outside CAPTURE. Both directory and ZIP use the same relative
paths; no full extraction is needed. Existing frozen absolute-path runners are
preserved, and future readers can adopt this interface without regenerating
source arrays or permanent dense-feature caches.

All 1,247 packed entries passed byte comparison. The focused
[loader check](data_lightweight_smoke.py) covers six actual frames across two
scenes: float32 native depth, int8 support, PNG pixels, and the exact existing
RGB BOX resize/crop preprocessing match direct-file reads. NaN payload, Inf and
signed-zero bytes also survive. Traversal, duplicate members and replacement of
an existing package are rejected. Task-owned temporary fixtures were removed.

Canonical evidence and package:
`artifacts.local/work/data-lightweight-20260910/` contains
`mz36-full-v1.source.zip`, `pack.json`, `verify.json`, `loader-smoke.json`,
`code-check.json` and detailed `README.md`. No model execution, source deletion
or claims about lossy precision changes are part of this result.
The bounded filesystem change is recorded in `ntfs-summary.json`,
`ntfs-before.json`, `ntfs-after.json`, `ntfs-command.json` and
`ntfs-byte-verification.json`. No hardlinked files, reparse points or conflicting
writers were admitted; no forced or executable/WOF compression was used.
