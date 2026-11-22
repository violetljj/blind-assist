import json, time
from pathlib import Path
from google.colab import drive

ddir = Path("/content/drive/MyDrive/BlindAssist/datasets/hypersim_full_v1")
m0 = json.loads((ddir / "manifest_w0.json").read_text(encoding="utf-8"))
m1 = json.loads((ddir / "manifest_w1.json").read_text(encoding="utf-8"))
merged = {
    "dataset": "hypersim_full_v1",
    "license": m0["license"],
    "split_policy": "official train+val ONLY; official test strictly excluded",
    "total_scenes": len(m0["completed_scenes"]) + len(m1["completed_scenes"]),
    "total_frames": m0["total_frames"] + m1["total_frames"],
    "total_bytes": m0["total_bytes"] + m1["total_bytes"],
    "failed_scenes": m0["failed_scenes"] + m1["failed_scenes"],
    "shards": sorted(m0["shards"] + m1["shards"], key=lambda s: s["filename"]),
    "merged_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(ddir / "manifest_sha256.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(f"Merged manifest_sha256.json: {merged['total_scenes']} scenes, {merged['total_frames']} frames, {merged['total_bytes']/1e9:.2f} GB, {len(merged['shards'])} shards", flush=True)
drive.flush_and_unmount()
print("Drive cleanly flushed and unmounted!", flush=True)
