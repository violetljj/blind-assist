"""Phase 6 48-vCPU Parallel Treasure Data Packer (G4):
1. Packs ALL 444 remaining unused SANPO-Real outdoor walking sessions (70% head / 30% chest,
   strictly disjoint from eval40, phase4 train76, and official test split) with drift-free +/-0.8s
   local-window GT -> Drive /BlindAssist/datasets/sanpo_real_extra444_v0/ (incremental 30-scene .tar shards).
2. Packs 210 additional strictly visit-disjoint ARKitScenes real indoor captures (expanding indoor
   training from 90 -> 300 captures) -> Drive /BlindAssist/datasets/arkit_indoor_extra210_v0/.
3. Keeps all packed .npz locally in /content/p6_data/ so the SAME G4 session can immediately train
   EgoClear-v5 without re-downloading or re-copying from Drive!
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/content")

from pack_arkit_colab import pack_arkit_split
from pack_phase2_colab import pack_real_worker
from phase4_simreal_aria_onnx_g4 import build_local_bev_and_rows_for_real_train

DRIVE_ROOT = Path("/content/drive/MyDrive/BlindAssist/datasets")
DRIVE_REAL444 = DRIVE_ROOT / "sanpo_real_extra444_v0"
DRIVE_ARKIT210 = DRIVE_ROOT / "arkit_indoor_extra210_v0"

LOCAL_REAL444 = Path("/content/p6_data/real_extra444")
LOCAL_ARKIT210 = Path("/content/p6_data/arkit_extra210")


def pack_one_real_local_gt(args: tuple[str, str, str]) -> dict | None:
    """Download, pack 5Hz SANPO-Real session, and immediately rebuild drift-free +/-0.8s local GT."""
    scene, cam, out_dir = args
    inf = pack_real_worker((scene, cam, out_dir))
    if inf is None:
        return None
    npz_p = Path(out_dir) / f"{scene}__{cam}.npz"
    if not npz_p.exists():
        # pack_real_worker may save as <scene>.npz
        alt = Path(out_dir) / f"{scene}.npz"
        if alt.exists():
            npz_p = alt
        else:
            return None
    build_local_bev_and_rows_for_real_train(str(npz_p))
    inf["stem"] = npz_p.stem
    return inf


def write_incremental_shards(local_dir: Path, stems: list[str], drive_dir: Path, prefix: str, shard_size: int = 30) -> None:
    drive_dir.mkdir(parents=True, exist_ok=True)
    shards_meta = []
    for si in range(0, len(stems), shard_size):
        chunk = stems[si : si + shard_size]
        tname = f"{prefix}_shard{si // shard_size:02d}.tar"
        tpath = drive_dir / tname
        if tpath.exists():
            tpath.unlink()
        with tarfile.open(tpath, "w") as tf:
            for st in chunk:
                p = local_dir / f"{st}.npz"
                if p.exists():
                    tf.add(p, arcname=p.name)
        h = hashlib.sha256()
        with open(tpath, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        shards_meta.append({"file": tname, "bytes": tpath.stat().st_size, "sha256": h.hexdigest(), "scenes": chunk})
    man = {"prefix": prefix, "n_scenes": len(stems), "shards": shards_meta}
    mpath = drive_dir / "manifest_sha256.json"
    if mpath.exists():
        mpath.unlink()
    mpath.write_text(json.dumps(man, indent=2), encoding="utf-8")


def main():
    t0 = time.time()
    ncpu = os.cpu_count() or 8
    n_workers = min(24, max(6, ncpu - 2))
    man = json.loads(Path("/content/phase6_treasure_manifest.json").read_text())
    print(f"[p6-pack] Starting 48-vCPU treasure pack (ncpu={ncpu}, workers={n_workers})", flush=True)

    # 1. Pack 444 unused SANPO-Real outdoor sessions with drift-free local GT
    LOCAL_REAL444.mkdir(parents=True, exist_ok=True)
    existing_real = {p.stem for p in LOCAL_REAL444.glob("*.npz")}
    # Restore from Drive if partially/fully packed previously
    if len(existing_real) < 350 and (DRIVE_REAL444 / "manifest_sha256.json").exists():
        print("[p6-pack] restoring sanpo_real_extra444_v0 from Drive...", flush=True)
        dm = json.loads((DRIVE_REAL444 / "manifest_sha256.json").read_text())
        for sh in dm["shards"]:
            with tarfile.open(DRIVE_REAL444 / sh["file"], "r") as tf:
                tf.extractall(LOCAL_REAL444)
        existing_real = {p.stem for p in LOCAL_REAL444.glob("*.npz")}

    todo_real = [r for r in man["sanpo_real_extra444"] if r["scene"] not in existing_real and f"{r['scene']}__{r['cam']}" not in existing_real]
    if todo_real:
        t1 = time.time()
        print(f"[p6-pack] packing {len(todo_real)} unused SANPO-Real outdoor sessions (existing={len(existing_real)})...", flush=True)
        ok_cnt = 0
        fail_cnt = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(pack_one_real_local_gt, (r["scene"], r["cam"], str(LOCAL_REAL444))) for r in todo_real]
            for fut in as_completed(futs):
                try:
                    inf = fut.result()
                    if inf is not None:
                        ok_cnt += 1
                    else:
                        fail_cnt += 1
                except Exception as e:
                    fail_cnt += 1
                    if fail_cnt <= 8:
                        print(f"  [warn] real session failed: {e}", flush=True)
                if (ok_cnt + fail_cnt) % 50 == 0:
                    print(f"  [p6-real] progress {ok_cnt+fail_cnt}/{len(todo_real)} (ok={ok_cnt}, skip/fail={fail_cnt}) elapsed={time.time()-t1:.1f}s", flush=True)
        all_real_stems = sorted(p.stem for p in LOCAL_REAL444.glob("*.npz"))
        print(f"[p6-pack] SANPO-Real extra DONE: {len(all_real_stems)} total valid sessions in {time.time()-t1:.1f}s; writing shards to Drive...", flush=True)
        write_incremental_shards(LOCAL_REAL444, all_real_stems, DRIVE_REAL444, "sanpo_real_extra444_v0", shard_size=30)
        print(f"[p6-pack] Drive saved sanpo_real_extra444_v0 ({len(all_real_stems)} scenes)", flush=True)
    else:
        print(f"[p6-pack] SANPO-Real extra already complete ({len(existing_real)} scenes)", flush=True)

    # 2. Pack 210 additional strictly visit-disjoint ARKitScenes indoor sessions
    t2 = time.time()
    ark_stems, _ = pack_arkit_split(
        man["arkit_extra_pool"],
        man["arkit_extra_target_n"],
        LOCAL_ARKIT210,
        DRIVE_ARKIT210,
        "arkit_indoor_extra210_v0",
        min_valid_frames=20,
        max_workers=n_workers,
    )
    print(f"[p6-pack] ARKitScenes extra210 DONE: {len(ark_stems)} scenes in {time.time()-t2:.1f}s", flush=True)
    print(f"[p6-pack] ALL TREASURE DATA PACKED & CACHED IN DRIVE in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
