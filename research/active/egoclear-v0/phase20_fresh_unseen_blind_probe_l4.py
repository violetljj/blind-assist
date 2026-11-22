"""Phase 20: Strictly-Disjoint 4-Domain Fresh Unseen Blind Probe (Colab L4, < 1 CU, ZERO Official Test Touch!).

Packs 4 completely brand-new, strictly zero-overlap unseen datasets (verified by hard set assertions against
ALL prior train/dev manifests AND official SANPO-Real/Synthetic `test_session_ids.txt`):
  1. `fresh_coda_unseen`        : 6 brand-new UT Austin CODa outdoor campus sequences (excluding {0,3,5,6,10,13})
                                  -> ~24 brand-new 128-line LiDAR + FLIR zero-shot clips!
  2. `fresh_arkit_val_unseen`   : 40 brand-new visit-disjoint real indoor LiDAR captures from ARKitScenes `Validation`
                                  (excluding ALL 830 previously used train+val visit_ids!)
  3. `fresh_sanpo_real_unseen`  : 35 brand-new real outdoor pedestrian `camera_head` sessions from SANPO-Real
                                  (strictly excluding `test_session_ids.txt` AND ALL 308 previously used sessions!)
                                  + DeepLabV3-ResNet50 dynamic pedestrian cleaning.
  4. `fresh_sanpo_synth_unseen` : 50 brand-new SANPO-Synthetic sessions
                                  (strictly excluding `test_session_ids.txt` AND ALL 1,187 previously used sessions!).

Then runs a STRICT FROZEN BLIND AUDIT (zero weight/threshold tuning!) across:
  - EgoClear-v6.6 (DAV2-Small baseline)
  - EgoClear-v11  (MoGe-2 Full old flagship)
  - EgoClear-v12  (MoGe-Turbo Round 1)
  - EgoClear-v14  (Hyper-Omni new 26.5 FPS FP16 / 35.9 FPS Mixed champion)
  - Dual-NPU ENS(v6.6 + v14)

Measures: frozen global thr=0.50 late/FA, @80/@120/@200/@350/@500 curves, dyn@0.5, paired bootstrap (v14 vs v6.6/v11),
Dark 0.18x stress (Fresh Real + Fresh ARKit), CODa iso_crop, and full 6-channel Omni + 5-mode optical UNKNOWN audits!
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torchvision

sys.path.insert(0, "/content")

import pack_coda_zeroshot_colab as coda_mod
from egoclear_model import RGB_H, RGB_W
from egoclear_v4_model import EgoClearV4Net
from egoclear_v11_model import EgoClearV11Net
from egoclear_v12_model import EgoClearV12Net
from egoclear_v14_model import EgoClearV14Net
from eval_closed_loop import eval_score_curve, summarize_at_fa_caps
from pack_arkit_colab import pack_arkit_worker
from pack_phase6_treasure_colab import write_incremental_shards
from pack_phase7_real_dualcam_colab import pack_head_noseg_worker
from pack_sanpo_colab import H as H_EVAL, process_session
from phase7_train_v6_g4 import eval_v6_all
from phase11_stage2_diagnostics_g4 import paired_cluster_bootstrap, per_route_at_thr
from phase11_train_v7_g4 import eval_dark_domain, print_block
from phase11_train_v75_g4 import IDX_4M_EXACT, prepare_scene_slice_tensors
from phase12_v9_probe_g4 import logit, sigmoid, summarize_scores
from phase13_train_v10_g4 import eval_frontend_mode
from phase14_train_v105_g4 import paired_route_audit_from_scores
from phase15_train_v106_g4 import audit_prehit_leakage
from phase17_train_v12_turbo_omni_g4 import (
    DRIVE_ROOT,
    audit_depth_scale,
    audit_omni_v12,
    enrich_pool_v12,
)
from train_and_eval_g4 import decode_rgb_batch, load_scene

DRIVE_DATA = DRIVE_ROOT / "datasets"
DRIVE_PROBE_DATA = DRIVE_DATA / "unseen_fresh_blind_probe_v0"
DRIVE_RUNS = DRIVE_ROOT / "runs/egoclear_v0_phase20_fresh_blind"
LOCAL_PROBE = Path("/content/fresh_blind_probe_v0")
LOCAL_RUNS = Path("/content/egoclear_runs_p20_fresh_blind")

OLD_CODA_SEQS = {0, 3, 5, 6, 10, 13}
CANDIDATE_NEW_CODA_SEQS = [1, 2, 4, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22]


def _fetch_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "BlindAssist-Research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def _list_stems_in_drive_dataset(ds_dir: Path) -> set[str]:
    """Read every .npz stem from manifest or .tar headers without extracting payloads."""
    stems: set[str] = set()
    if not ds_dir.exists():
        return stems
    man_p = ds_dir / "manifest_sha256.json"
    if man_p.exists():
        try:
            mj = json.loads(man_p.read_text(encoding="utf-8"))
            for sh in mj.get("shards", []):
                for s in sh.get("scenes", []):
                    stems.add(Path(str(s)).stem)
        except Exception:
            pass
    if not stems:
        for tp in sorted(ds_dir.glob("*.tar")):
            try:
                with tarfile.open(tp, "r") as tf:
                    for m in tf.getmembers():
                        if m.name.endswith(".npz"):
                            stems.add(Path(m.name).stem)
            except Exception:
                pass
    for npz_p in ds_dir.glob("*.npz"):
        stems.add(npz_p.stem)
    return stems


def _restore_if_cached(sub_name: str, local_dir: Path, min_n: int) -> bool:
    d_sub = DRIVE_PROBE_DATA / sub_name
    if (d_sub / "manifest_sha256.json").exists():
        local_dir.mkdir(parents=True, exist_ok=True)
        if len(list(local_dir.glob("*.npz"))) < min_n:
            man = json.loads((d_sub / "manifest_sha256.json").read_text(encoding="utf-8"))
            for sh in man.get("shards", []):
                with tarfile.open(d_sub / sh["file"], "r") as tf:
                    tf.extractall(local_dir, filter="data")
        if len(list(local_dir.glob("*.npz"))) >= min_n:
            print(f"[cache-hit] {sub_name}: restored {len(list(local_dir.glob('*.npz')))} scenes from Drive", flush=True)
            return True
    return False


def _archive_sub(local_dir: Path, sub_name: str, extra_meta: dict) -> None:
    stems = sorted(p.stem for p in local_dir.glob("*.npz"))
    if not stems:
        return
    d_sub = DRIVE_PROBE_DATA / sub_name
    d_sub.mkdir(parents=True, exist_ok=True)
    write_incremental_shards(local_dir, stems, d_sub, sub_name, shard_size=25)
    (d_sub / "disjoint_audit.json").write_text(json.dumps(extra_meta, indent=2), encoding="utf-8")


def pack_fresh_coda(target_seqs: int = 6, clips_per_seq: int = 4) -> tuple[Path, dict]:
    local_dir = LOCAL_PROBE / "fresh_coda"
    local_dir.mkdir(parents=True, exist_ok=True)
    if _restore_if_cached("fresh_coda_unseen", local_dir, min_n=12):
        meta = json.loads((DRIVE_PROBE_DATA / "fresh_coda_unseen/disjoint_audit.json").read_text(encoding="utf-8"))
        return local_dir, meta

    t0 = time.time()
    picked_seqs: list[int] = []
    stats = []
    for seq in CANDIDATE_NEW_CODA_SEQS:
        if len(picked_seqs) >= target_seqs:
            break
        assert seq not in OLD_CODA_SEQS, f"Leakage! CODa seq {seq} was in OLD_CODA_SEQS!"
        url = f"{coda_mod.TACC_BASE}/{seq}.zip"
        try:
            rz = coda_mod.RemoteZip64(url)
        except Exception as e:
            print(f"  [fresh-coda] skip seq {seq}: {e}", flush=True)
            continue
        pose_key = f"poses/dense_global/{seq}.txt" if f"poses/dense_global/{seq}.txt" in rz.entries else f"poses/dense/{seq}.txt"
        cal_k = f"calibrations/{seq}/calib_cam0_intrinsics.yaml"
        cal_t = f"calibrations/{seq}/calib_os1_to_cam0.yaml"
        if pose_key not in rz.entries or cal_k not in rz.entries or cal_t not in rz.entries:
            print(f"  [fresh-coda] skip seq {seq}: missing pose/calib", flush=True)
            continue

        _ts, R_w, t_w = coda_mod.parse_pose_file(rz.read(pose_key).decode("utf-8"))
        K_rect, orig_hw, T_os1_cam0rect = coda_mod.parse_cam0_calib(
            rz.read(cal_k).decode("utf-8"), rz.read(cal_t).decode("utf-8")
        )
        avail_cam, avail_pc, bbox_frs = set(), set(), set()
        pref_c = f"2d_rect/cam0/{seq}/2d_rect_cam0_{seq}_"
        pref_p = f"3d_comp/os1/{seq}/3d_comp_os1_{seq}_"
        pref_b = f"3d_bbox/os1/{seq}/3d_bbox_os1_{seq}_"
        for name in rz.entries:
            if name.startswith(pref_c) and name.endswith(".png"):
                avail_cam.add(int(name[len(pref_c):-4]))
            elif name.startswith(pref_p) and name.endswith(".bin"):
                avail_pc.add(int(name[len(pref_p):-4]))
            elif name.startswith(pref_b) and name.endswith(".json"):
                bbox_frs.add(int(name[len(pref_b):-5]))
        avail = avail_cam & avail_pc
        clips = coda_mod.select_walking_clips(t_w, R_w, avail, bbox_frs, n_clips=clips_per_seq, clip_frames_5hz=120, stride=2)
        if not clips:
            print(f"  [fresh-coda] skip seq {seq}: 0 valid walking clips", flush=True)
            continue
        coda_mod.SEQ_META[seq] = f"FreshSeq{seq:02d}"
        seq_ok = 0
        for ci, (_st, frs, _pm, _nbb) in enumerate(clips):
            info = coda_mod.pack_one_clip(rz, seq, ci, frs, R_w, t_w, K_rect, orig_hw, T_os1_cam0rect, local_dir)
            if info:
                stats.append(info)
                seq_ok += 1
                print(
                    f"  [fresh-coda] seq {seq} c{ci}: ok={info['n_ok']}/{info['n_frames']} haz={info['n_haz']} "
                    f"dyn={info['n_dyn_only']} path={info['path_m']}m ({info['sec']}s)",
                    flush=True,
                )
        if seq_ok > 0:
            picked_seqs.append(seq)

    assert len(set(picked_seqs) & OLD_CODA_SEQS) == 0
    meta = {
        "pool": "fresh_coda_unseen",
        "old_excluded_seqs": sorted(OLD_CODA_SEQS),
        "new_seqs": picked_seqs,
        "overlap_count": 0,
        "n_clips": len(stats),
        "wall_sec": round(time.time() - t0, 1),
    }
    _archive_sub(local_dir, "fresh_coda_unseen", meta)
    print(f"[Fresh CODa DONE] {meta}", flush=True)
    return local_dir, meta


def pack_fresh_arkit_val(n_workers: int, target_n: int = 40) -> tuple[Path, dict]:
    local_dir = LOCAL_PROBE / "fresh_arkit_val"
    local_dir.mkdir(parents=True, exist_ok=True)
    if _restore_if_cached("fresh_arkit_val_unseen", local_dir, min_n=20):
        meta = json.loads((DRIVE_PROBE_DATA / "fresh_arkit_val_unseen/disjoint_audit.json").read_text(encoding="utf-8"))
        return local_dir, meta

    t0 = time.time()
    csv_txt = _fetch_text("https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/metadata.csv")
    rows = list(csv.DictReader(io.StringIO(csv_txt)))

    used_vids: set[str] = set()
    for dsub in ("arkit_indoor_p5_v0", "arkit_indoor_extra210_v0", "arkit_indoor_p13_500_v0"):
        for st in _list_stems_in_drive_dataset(DRIVE_DATA / dsub):
            used_vids.add(st.replace("arkit_", ""))

    vid_to_visit = {r["video_id"]: r["visit_id"] for r in rows}
    used_visits = {vid_to_visit[v] for v in used_vids if v in vid_to_visit and vid_to_visit[v] != "NA"}

    # Strictly Validation fold, strictly visit-disjoint from ALL 830 prior ARKit captures!
    cand_by_visit: dict[str, str] = {}
    for r in rows:
        if r["fold"] == "Validation" and r["visit_id"] != "NA":
            if r["video_id"] not in used_vids and r["visit_id"] not in used_visits and r["visit_id"] not in cand_by_visit:
                cand_by_visit[r["visit_id"]] = r["video_id"]

    cand_items = sorted(cand_by_visit.items())
    rng = np.random.default_rng(20261122)
    perm = rng.permutation(len(cand_items))
    todo = [cand_items[i] for i in perm[: min(len(cand_items), target_n + 20)]]
    print(f"[Fresh ARKit Val] Packing {target_n} strictly visit-disjoint Validation captures from {len(cand_items)} unused visits (excluded {len(used_vids)} prior vids / {len(used_visits)} visits)...", flush=True)

    packed_vids: list[str] = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(pack_arkit_worker, (vid, "Validation", str(local_dir), 35)): (vis, vid) for vis, vid in todo}
        for fut in as_completed(futs):
            vis, vid = futs[fut]
            try:
                res = fut.result()
            except Exception:
                res = None
            if res is not None:
                packed_vids.append(vid)
                if len(packed_vids) % 10 == 0 or len(packed_vids) >= target_n:
                    print(f"  [Fresh ARKit Val] {len(packed_vids)}/{target_n} packed ({time.time()-t0:.1f}s)", flush=True)
                if len(packed_vids) >= target_n:
                    break

    # Trim to exact target_n if extra completed concurrently
    all_npzs = sorted(local_dir.glob("*.npz"))
    for p in all_npzs[target_n:]:
        p.unlink(missing_ok=True)
    final_vids = [p.stem.replace("arkit_", "") for p in sorted(local_dir.glob("*.npz"))]
    final_visits = {vid_to_visit[v] for v in final_vids if v in vid_to_visit}
    assert len(set(final_vids) & used_vids) == 0
    assert len(final_visits & used_visits) == 0

    meta = {
        "pool": "fresh_arkit_val_unseen",
        "fold": "Validation",
        "excluded_prior_vids": len(used_vids),
        "excluded_prior_visits": len(used_visits),
        "n_scenes": len(final_vids),
        "overlap_vids": 0,
        "overlap_visits": 0,
        "video_ids": final_vids,
        "wall_sec": round(time.time() - t0, 1),
    }
    _archive_sub(local_dir, "fresh_arkit_val_unseen", meta)
    print(f"[Fresh ARKit Val DONE] n={len(final_vids)} (0 overlap) in {meta['wall_sec']}s", flush=True)
    return local_dir, meta


def _strip_sanpo_real_id(stem: str) -> str:
    s = stem
    for tok in ("_camera_chest", "_camera_head_noseg", "_camera_head", "__camera_chest", "__camera_head"):
        s = s.replace(tok, "")
    return s


def pack_fresh_sanpo_real(n_workers: int, device: torch.device, target_n: int = 35) -> tuple[Path, dict]:
    local_dir = LOCAL_PROBE / "fresh_sanpo_real"
    local_dir.mkdir(parents=True, exist_ok=True)
    if _restore_if_cached("fresh_sanpo_real_unseen", local_dir, min_n=25):
        meta = json.loads((DRIVE_PROBE_DATA / "fresh_sanpo_real_unseen/disjoint_audit.json").read_text(encoding="utf-8"))
        return local_dir, meta

    t0 = time.time()
    train_ids = _fetch_text("https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real/splits/train_session_ids.txt").split()
    test_ids = set(_fetch_text("https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real/splits/test_session_ids.txt").split())

    used_real_ids: set[str] = set()
    for dsub in (
        "sanpo_real_40_v0",
        "sanpo_real_train75_v0",
        "sanpo_real_extra22_v0",
        "sanpo_real_chest240_v0",
        "sanpo_real_head160_noseg_v0",
        "sanpo_real_head100_autoseg_v0",
        "sanpo_real_p13_extra_v0",
    ):
        for st in _list_stems_in_drive_dataset(DRIVE_DATA / dsub):
            used_real_ids.add(_strip_sanpo_real_id(st))

    avail = [sc for sc in train_ids if sc not in test_ids and sc not in used_real_ids]
    rng = np.random.default_rng(20261122)
    perm = rng.permutation(len(avail))
    todo = [avail[i] for i in perm[: min(len(avail), target_n + 25)]]
    print(
        f"[Fresh SANPO-Real] Packing {target_n} unused non-test camera_head sessions from {len(avail)} available "
        f"(hard-excluded {len(test_ids)} official test + {len(used_real_ids)} prior train/eval sessions)...",
        flush=True,
    )

    ok = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(pack_head_noseg_worker, (sc, str(local_dir))) for sc in todo]
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception:
                res = None
            if res is not None:
                ok += 1
                if ok % 10 == 0 or ok >= target_n:
                    print(f"  [Fresh SANPO-Real raw] {ok}/{target_n} packed ({time.time()-t0:.1f}s)", flush=True)
                if ok >= target_n:
                    break

    all_npzs = sorted(local_dir.glob("*.npz"))
    for p in all_npzs[target_n:]:
        p.unlink(missing_ok=True)

    # Clean dynamic pedestrians/vehicles with DeepLabV3-ResNet50 (exact Stage A2 protocol!)
    print(f"  [Fresh SANPO-Real] Running DeepLabV3-ResNet50 dynamic pedestrian cleaning on {len(list(local_dir.glob('*.npz')))} scenes...", flush=True)
    seg_m = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT").to(device).eval()
    dyn_cls = torch.tensor([2, 6, 7, 8, 12, 14, 15], device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    n_dyn_fr = 0
    for f in sorted(local_dir.glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        n = len(d["eval_rows"])
        rgb = decode_rgb_batch(d, np.arange(n))
        dyn_frac = np.zeros(n, dtype=np.float32)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, n, 32):
                x = (torch.from_numpy(rgb[i : i + 32]).to(device) - mean) / std
                pred = seg_m(x)["out"].argmax(dim=1)
                is_dyn = torch.isin(pred, dyn_cls)
                c_dyn = is_dyn[:, int(0.25 * RGB_H) :, int(0.25 * RGB_W) : int(0.75 * RGB_W)].float().mean(dim=(1, 2))
                dyn_frac[i : i + 32] = c_dyn.cpu().numpy()
        er = d["eval_rows"].copy()
        gt_st = d["gt_bev_static"].copy()
        gt_dy = d["gt_bev_dynamic"].copy()
        is_d_fr = dyn_frac >= 0.025
        er[is_d_fr, 2] = 1.0
        ambig = is_d_fr & np.isfinite(er[:, 0]) & (er[:, 0] <= H_EVAL)
        er[ambig, 1] = np.nan
        gt_dy[is_d_fr] = np.maximum(gt_dy[is_d_fr], gt_st[is_d_fr])
        gt_st[ambig] = 0
        n_dyn_fr += int(is_d_fr.sum())
        d["eval_rows"] = er
        d["gt_bev_static"] = gt_st
        d["gt_bev_dynamic"] = gt_dy
        d["has_dyn_gt"] = np.array(1, dtype=np.uint8)
        np.savez(f, **d)
    del seg_m
    torch.cuda.empty_cache()

    final_ids = [_strip_sanpo_real_id(p.stem) for p in sorted(local_dir.glob("*.npz"))]
    assert len(set(final_ids) & test_ids) == 0, "FATAL: touched official SANPO-Real test split!"
    assert len(set(final_ids) & used_real_ids) == 0, "FATAL: overlapped prior SANPO-Real train/eval!"

    meta = {
        "pool": "fresh_sanpo_real_unseen",
        "excluded_official_test": len(test_ids),
        "excluded_prior_train_eval": len(used_real_ids),
        "n_scenes": len(final_ids),
        "n_dyn_frames_autoseg": n_dyn_fr,
        "overlap_test": 0,
        "overlap_prior": 0,
        "session_ids": final_ids,
        "wall_sec": round(time.time() - t0, 1),
    }
    _archive_sub(local_dir, "fresh_sanpo_real_unseen", meta)
    print(f"[Fresh SANPO-Real DONE] n={len(final_ids)} (0 overlap with test & prior) in {meta['wall_sec']}s", flush=True)
    return local_dir, meta


def _pack_synth_one(args: tuple[str, str]) -> str | None:
    sc, out_dir = args
    dst = Path(out_dir) / f"{sc}.npz"
    if dst.exists() and dst.stat().st_size > 5000:
        return sc
    try:
        with ThreadPoolExecutor(max_workers=8) as dl_pool:
            raw_bytes, _meta = process_session(sc, "train", dl_pool)
        dst.write_bytes(raw_bytes)
        return sc if dst.stat().st_size > 5000 else None
    except Exception:
        return None


def pack_fresh_sanpo_synth(n_workers: int, target_n: int = 50) -> tuple[Path, dict]:
    local_dir = LOCAL_PROBE / "fresh_sanpo_synth"
    local_dir.mkdir(parents=True, exist_ok=True)
    if _restore_if_cached("fresh_sanpo_synth_unseen", local_dir, min_n=40):
        meta = json.loads((DRIVE_PROBE_DATA / "fresh_sanpo_synth_unseen/disjoint_audit.json").read_text(encoding="utf-8"))
        return local_dir, meta

    t0 = time.time()
    train_ids = _fetch_text("https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-synthetic/splits/train_session_ids.txt").split()
    test_ids = set(_fetch_text("https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-synthetic/splits/test_session_ids.txt").split())

    used_synth_ids: set[str] = set()
    for dsub in ("sanpo_synth_287_v0", "sanpo_synth_extra450_v0", "sanpo_synth_p13_450_v0"):
        used_synth_ids.update(_list_stems_in_drive_dataset(DRIVE_DATA / dsub))
    m287_p = Path("/content/sanpo_287_manifest.json")
    if m287_p.exists():
        used_synth_ids.update(json.loads(m287_p.read_text(encoding="utf-8"))["splits"].keys())

    avail = [sc for sc in train_ids if sc not in test_ids and sc not in used_synth_ids]
    rng = np.random.default_rng(20261122)
    perm = rng.permutation(len(avail))
    todo = [avail[i] for i in perm[: min(len(avail), target_n + 20)]]
    print(
        f"[Fresh SANPO-Synth] Packing {target_n} unused non-test sessions from {len(avail)} available "
        f"(hard-excluded {len(test_ids)} official test + {len(used_synth_ids)} prior train/eval)...",
        flush=True,
    )

    ok = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(_pack_synth_one, (sc, str(local_dir))) for sc in todo]
        for fut in as_completed(futs):
            if fut.result() is not None:
                ok += 1
                if ok % 15 == 0 or ok >= target_n:
                    print(f"  [Fresh SANPO-Synth] {ok}/{target_n} packed ({time.time()-t0:.1f}s)", flush=True)
                if ok >= target_n:
                    break

    all_npzs = sorted(local_dir.glob("*.npz"))
    for p in all_npzs[target_n:]:
        p.unlink(missing_ok=True)
    final_ids = [p.stem for p in sorted(local_dir.glob("*.npz"))]
    assert len(set(final_ids) & test_ids) == 0, "FATAL: touched official SANPO-Synth test split!"
    assert len(set(final_ids) & used_synth_ids) == 0, "FATAL: overlapped prior SANPO-Synth train/eval!"

    meta = {
        "pool": "fresh_sanpo_synth_unseen",
        "excluded_official_test": len(test_ids),
        "excluded_prior_train_eval": len(used_synth_ids),
        "n_scenes": len(final_ids),
        "overlap_test": 0,
        "overlap_prior": 0,
        "session_ids": final_ids,
        "wall_sec": round(time.time() - t0, 1),
    }
    _archive_sub(local_dir, "fresh_sanpo_synth_unseen", meta)
    print(f"[Fresh SANPO-Synth DONE] n={len(final_ids)} (0 overlap) in {meta['wall_sec']}s", flush=True)
    return local_dir, meta


def main():
    t_all = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    LOCAL_PROBE.mkdir(parents=True, exist_ok=True)
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_PROBE_DATA.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

    ncpu = os.cpu_count() or 8
    n_workers = min(10, max(4, ncpu - 2))
    print(f"=== [Phase 20 Fresh Unseen Blind Probe] GPU={torch.cuda.get_device_name(0)} | ncpu={ncpu} | workers={n_workers} ===", flush=True)

    # Restore sanpo_287_manifest.json from Drive (for bw_speeds & synth exclusion)
    m287_local = Path("/content/sanpo_287_manifest.json")
    if not m287_local.exists():
        shutil.copy2(DRIVE_DATA / "sanpo_synth_287_v0/sanpo_287_manifest.json", m287_local)
    bw_speeds = json.loads(m287_local.read_text(encoding="utf-8"))["bw_speeds"]

    # 1. Pack all 4 strictly-disjoint brand-new unseen pools (ZERO official test touch!)
    p_coda, meta_coda = pack_fresh_coda(target_seqs=6, clips_per_seq=4)
    p_arkit, meta_arkit = pack_fresh_arkit_val(n_workers=n_workers, target_n=40)
    p_real, meta_real = pack_fresh_sanpo_real(n_workers=n_workers, device=device, target_n=35)
    p_synth, meta_synth = pack_fresh_sanpo_synth(n_workers=n_workers, target_n=50)

    pools = {
        "fresh_sanpo_real_unseen": {p.stem: load_scene(p) for p in sorted(p_real.glob("*.npz"))},
        "fresh_arkit_val_unseen": {p.stem: load_scene(p) for p in sorted(p_arkit.glob("*.npz"))},
        "fresh_coda_unseen": {p.stem: load_scene(p) for p in sorted(p_coda.glob("*.npz"))},
        "fresh_sanpo_synth_unseen": {p.stem: load_scene(p) for p in sorted(p_synth.glob("*.npz"))},
    }
    omni_gt_stats = {
        "fresh_sanpo_real_unseen": enrich_pool_v12(pools["fresh_sanpo_real_unseen"], "fresh_real", has_dyn_gt=True, allow_flat_pose_neg=False),
        "fresh_arkit_val_unseen": enrich_pool_v12(pools["fresh_arkit_val_unseen"], "fresh_arkit", has_dyn_gt=False, allow_flat_pose_neg=True),
        "fresh_coda_unseen": enrich_pool_v12(pools["fresh_coda_unseen"], "fresh_coda", has_dyn_gt=True, default_h_cam=0.704, allow_flat_pose_neg=True),
        "fresh_sanpo_synth_unseen": enrich_pool_v12(pools["fresh_sanpo_synth_unseen"], "fresh_synth", has_dyn_gt=True),
    }

    # 2. Load ALL 4 frozen models on CPU and move ONE AT A TIME to L4 GPU (peak VRAM < 5 GB!)
    print("\n=== Loading Frozen Checkpoints (v6.6, v11, v12, v14) on CPU ===", flush=True)
    m_v66 = EgoClearV4Net(mode="full", c_bev=128, freeze_dav2=True).eval()
    m_v66.load_state_dict(
        torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase9_v66_coda/egoclear_v66_cleankd_small.pt", map_location="cpu", weights_only=False)["head_state_dict"],
        strict=False,
    )

    m_v11 = EgoClearV11Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), unlock_moge_neck_and_last2=False, rgb_feat_drop_p=0.0).eval()
    m_v11.load_state_dict(
        torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase16_v11/egoclear_v11_omni_moge2_small.pt", map_location="cpu", weights_only=False)["state_dict"],
        strict=True,
    )

    m_v12 = EgoClearV12Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), unlock_vit_last_n=0, rgb_feat_drop_p=0.0, use_moge_pretrained=True).eval()
    m_v12.load_state_dict(
        torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase17_v12/egoclear_v12_turbo_omni.pt", map_location="cpu", weights_only=False)["state_dict"],
        strict=True,
    )

    m_v14 = EgoClearV14Net(moge_id="Ruicheng/moge-2-vits-normal", c_feat=64, c_bev=128, lags=(1, 2), unlock_vit_last_n=0, rgb_feat_drop_p=0.0, use_moge_pretrained=True).eval()
    m_v14.load_state_dict(
        torch.load(DRIVE_ROOT / "runs/egoclear_v0_phase19_v14/egoclear_v14_hyper_omni.pt", map_location="cpu", weights_only=False)["state_dict"],
        strict=True,
    )

    models = {"v6.6": m_v66, "v11": m_v11, "v12": m_v12, "v14": m_v14}
    thrs = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.50, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 0.98], dtype=np.float64)

    all_evals: dict[str, dict] = {k: {} for k in models}
    fixed50_audits: dict[str, dict] = {k: {} for k in models}
    omni_audits: dict[str, dict] = {k: {} for k in ("v11", "v12", "v14")}
    prehit_audits: dict[str, dict] = {k: {} for k in ("v11", "v12", "v14")}
    depth_audits: dict[str, dict] = {k: {} for k in ("v11", "v12", "v14")}
    boot_v14_vs_v66: dict[str, dict] = {}
    boot_v14_vs_v11: dict[str, dict] = {}
    ens_v66_v14: dict[str, dict] = {}
    iso_coda: dict[str, dict] = {}
    scores_all: dict[str, dict[str, dict]] = {dname: {k: {} for k in models} for dname in pools}
    ev_all: dict[str, dict] = {dname: {} for dname in pools}

    # Evaluate ONE model at a time across all pools + dark stress (zero VRAM contention!)
    for mname, net in models.items():
        net.to(device).eval()
        torch.cuda.empty_cache()
        print(f"\n================ EVALUATING FROZEN MODEL: {mname} ================", flush=True)
        for dname, dmap in pools.items():
            all_evals[mname][dname] = eval_v6_all(net, dmap, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=True)
            print_block(mname, dname, all_evals[mname][dname])
            for sc, d in dmap.items():
                er = d["eval_rows"]
                if len(er) < 5:
                    continue
                inp = prepare_scene_slice_tensors(d, 0, len(er), device)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    scores_all[dname][mname][sc] = net.forward_sequence(**inp)["corr_probs"][0, :, IDX_4M_EXACT].float().cpu().numpy()
                ev_all[dname][sc] = er
            fixed50_audits[mname][dname] = paired_route_audit_from_scores(dmap, None, scores_all[dname][mname], bw_speeds, thrs, fixed_thr=0.50)
            f50 = fixed50_audits[mname][dname]["fixed_thr_0.50"]
            print(f"    [{mname:4s} FROZEN thr=0.50] {dname:24s} late={f50['late']}/{f50['avoidable']}({f50['late_pct']}%) | FA={f50['fa_fr_km']} fr/km", flush=True)

            if mname in ("v11", "v12", "v14"):
                omni_audits[mname][dname] = audit_omni_v12(net, dmap, None, device)
                if dname in ("fresh_sanpo_real_unseen", "fresh_arkit_val_unseen"):
                    prehit_audits[mname][dname] = audit_prehit_leakage(net, dmap, None, device)
                if any("depth_s8" in d for d in dmap.values()):
                    depth_audits[mname][dname] = audit_depth_scale(net, dmap, device)

        for dname, dark_key in [
            ("fresh_sanpo_real_unseen", "fresh_sanpo_real_dark018"),
            ("fresh_arkit_val_unseen", "fresh_arkit_val_dark018"),
        ]:
            all_evals[mname][dark_key] = eval_dark_domain(net, pools[dname], None, bw_speeds, device, gain=0.18)
            print_block(mname, dark_key, all_evals[mname][dark_key])

        net.cpu()
        torch.cuda.empty_cache()

    # Paired bootstraps + Dual-NPU ENS(v6.6 + v14)
    print("\n================ PAIRED BOOTSTRAPS & DUAL-NPU ENS ================", flush=True)
    for dname, dmap in pools.items():
        ev_m = ev_all[dname]
        scores_by_m = scores_all[dname]
        sens_m = {sc: sigmoid(0.5 * logit(scores_by_m["v6.6"][sc]) + 0.5 * logit(scores_by_m["v14"][sc])) for sc in ev_m}
        ens_v66_v14[dname] = summarize_scores(ev_m, sens_m, bw_speeds)
        c66 = summarize_at_fa_caps(eval_score_curve(ev_m, scores_by_m["v6.6"], bw_speeds, thrs, higher_is_alert=True))
        c11 = summarize_at_fa_caps(eval_score_curve(ev_m, scores_by_m["v11"], bw_speeds, thrs, higher_is_alert=True))
        c14 = summarize_at_fa_caps(eval_score_curve(ev_m, scores_by_m["v14"], bw_speeds, thrs, higher_is_alert=True))
        t66 = float(c66["late_at_fa_fr<=120.3"]["thr"]) if c66.get("late_at_fa_fr<=120.3") else 0.50
        t11 = float(c11["late_at_fa_fr<=120.3"]["thr"]) if c11.get("late_at_fa_fr<=120.3") else 0.50
        t14 = float(c14["late_at_fa_fr<=120.3"]["thr"]) if c14.get("late_at_fa_fr<=120.3") else 0.50

        boot_v14_vs_v66[dname] = {
            "at_120": paired_cluster_bootstrap(per_route_at_thr(dmap, None, scores_by_m["v6.6"], bw_speeds, t66), per_route_at_thr(dmap, None, scores_by_m["v14"], bw_speeds, t14)),
            "at_fixed_0.50": paired_cluster_bootstrap(per_route_at_thr(dmap, None, scores_by_m["v6.6"], bw_speeds, 0.50), per_route_at_thr(dmap, None, scores_by_m["v14"], bw_speeds, 0.50)),
        }
        boot_v14_vs_v11[dname] = {
            "at_120": paired_cluster_bootstrap(per_route_at_thr(dmap, None, scores_by_m["v11"], bw_speeds, t11), per_route_at_thr(dmap, None, scores_by_m["v14"], bw_speeds, t14)),
            "at_fixed_0.50": paired_cluster_bootstrap(per_route_at_thr(dmap, None, scores_by_m["v11"], bw_speeds, 0.50), per_route_at_thr(dmap, None, scores_by_m["v14"], bw_speeds, 0.50)),
        }
        print(f"  [{dname:24s}] ENS(v6.6+v14)={ens_v66_v14[dname]}", flush=True)
        print(f"    Boot v14 vs v6.6 @120={boot_v14_vs_v66[dname]['at_120']}", flush=True)
        print(f"    Boot v14 vs v11  @120={boot_v14_vs_v11[dname]['at_120']}", flush=True)
        print(f"    Boot v14 vs v11  0.50={boot_v14_vs_v11[dname]['at_fixed_0.50']}", flush=True)

    # CODa isotropic crop frontend check (one pair at a time on GPU)
    m_v66.to(device).eval()
    for mname in ("v11", "v12", "v14"):
        models[mname].to(device).eval()
        iso_coda[f"{mname}__iso_crop"] = eval_frontend_mode(models[mname], m_v66, pools["fresh_coda_unseen"], None, bw_speeds, device, "iso_crop")
        print(f"  [Fresh CODa iso_crop {mname}] {iso_coda[f'{mname}__iso_crop']}", flush=True)
        models[mname].cpu()
        torch.cuda.empty_cache()
    m_v66.cpu()

    results = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "protocol": "STRICTLY-DISJOINT FRESH UNSEEN BLIND PROBE (0 overlap with train/dev & 0 reads of official SANPO-Real/Synth test splits!)",
        "disjoint_proofs": {
            "fresh_coda_unseen": meta_coda,
            "fresh_arkit_val_unseen": meta_arkit,
            "fresh_sanpo_real_unseen": meta_real,
            "fresh_sanpo_synth_unseen": meta_synth,
        },
        "omni_gt_stats": omni_gt_stats,
        "evals": all_evals,
        "frozen_thr050_audits": fixed50_audits,
        "ens_v66_v14": ens_v66_v14,
        "iso_coda": iso_coda,
        "paired_bootstrap_v14_vs_v66": boot_v14_vs_v66,
        "paired_bootstrap_v14_vs_v11": boot_v14_vs_v11,
        "omni_audits": omni_audits,
        "prehit_audits": prehit_audits,
        "depth_audits": depth_audits,
        "total_wall_sec": round(time.time() - t_all, 1),
    }
    res_p = LOCAL_RUNS / "phase20_fresh_blind_results.json"
    res_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    shutil.copy2(res_p, DRIVE_RUNS / res_p.name)
    os.sync()
    print(f"\n[Phase 20 Fresh Blind Probe] ALL DONE in {results['total_wall_sec']}s! Saved to {DRIVE_RUNS}", flush=True)


if __name__ == "__main__":
    main()
