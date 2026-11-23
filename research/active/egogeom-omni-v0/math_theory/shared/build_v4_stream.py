"""Materialize label-free, causally usable v4 observations and verify the archive bridge."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

ROUTE = Path(__file__).resolve().parents[2]
REPO = ROUTE.parents[2]
sys.path[:0] = [str(ROUTE), str(ROUTE.parent / "egoclear-v0")]
from eval_egogeom_v3_zeroshot import (
    H_IN,
    IM_MEAN,
    IM_STD,
    W_IN,
    EgoGeomOmniV4Net,
    EgoGeomV4Config,
    TemporalBEVConfig,
    compute_gravity_aligned_ray_map,
    corridor_dist_from_depth,
    decode_isotropic_frame,
    run_bev_on_clip,
    score_dataset,
)

WORK = REPO / "artifacts.local/work/egogeom-omni-v0"
ROOT = WORK / "math_theory/v4_stream"
SUITES = {
    "arkit_val30_day": ("arkit_val30_shard00", 1.0, 2., 30),
    "coda_campus": ("coda_zeroshot_shard000", 1.0, 4., 24),
    "aria_gen2": ("aria_g2_20clips", 1.0, 4., 19),
    "arkit_val30_dark018": ("arkit_val30_shard00", 0.18, 2., 30),
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def strict_headings(suite, packed, poses_r, poses_t):
    fwd = packed.copy()
    if suite == "aria_gen2":
        for i in range(min(2, len(fwd))):
            xy = poses_t[i, :2] - poses_t[0, :2]
            if np.linalg.norm(xy) < .15:
                xy = poses_r[i, :2, 2]
            norm = float(np.linalg.norm(xy))
            fwd[i, :2] = xy / norm if norm > 1e-6 else [1., 0.]
            fwd[i, 2] = 0
    return fwd


@torch.inference_mode()
def observe(model, source, suite, shade, batch):
    with np.load(source, allow_pickle=False) as data:
        blob, offs = data["rgb_blob"], data["rgb_offsets"]
        hw, intr = data["orig_hw"], data["K_orig"]
        rot, trans, fwd = data["poses_R"], data["poses_t"], data["causal_fwd"]
    n = len(trans)
    true_fwd = strict_headings(suite, fwd, rot, trans)
    keys = ("depth_s4", "dh_s4", "nrm_s4", "ls_s4")
    outputs = ("depth_m", "delta_h_m", "normal_grav", "log_sigma")
    obs = {key: np.empty((n,) + ((3,) if key == "nrm_s4" else ()) +
                          (H_IN//4, W_IN//4), np.float32) for key in keys}
    obs.update(K=np.empty((n, 3, 3), np.float32), gray_s4=np.empty((n, H_IN//4, W_IN//4), np.uint8),
               poses_R=rot, poses_t=trans, causal_fwd=fwd, strict_fwd=true_fwd)
    metric_keys = ("d_1f_wz", "d_1f_strict")
    for key in metric_keys:
        obs[key] = np.empty(n, np.float32)
    # The frozen projection samples only [::4, ::4] of its two full-sized inputs.
    dep = np.zeros((H_IN, W_IN), np.float32)
    unc = np.zeros_like(dep)
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        frames, gravity = [], []
        for i in range(start, stop):
            rgb, obs["K"][i] = decode_isotropic_frame(blob[offs[i]:offs[i+1]].tobytes(), hw, intr, shade)
            obs["gray_s4"][i] = np.rint(255 * cv2.cvtColor(rgb[::4, ::4], cv2.COLOR_RGB2GRAY)).astype(np.uint8)
            frames.append(((rgb - IM_MEAN) / IM_STD).transpose(2, 0, 1))
            g = rot[i].T @ np.array([0., 0., -1.], dtype=np.float32)
            gravity.append(g / (np.linalg.norm(g) + 1e-6))
        x = torch.from_numpy(np.stack(frames)).cuda()
        rays = compute_gravity_aligned_ray_map(torch.from_numpy(obs["K"][start:stop]).cuda(),
                                               torch.from_numpy(np.stack(gravity)).cuda(), H_IN, W_IN)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, rays, None, None, return_features=False)
        for key, field in zip(keys, outputs):
            pred = out[field][:, :, ::4, ::4]
            obs[key][start:stop] = (pred if key == "nrm_s4" else pred[:, 0]).float().cpu().numpy()
        del out, pred, x, rays
        for i in range(start, stop):
            dep[::4, ::4] = obs["depth_s4"][i]
            unc[::4, ::4] = obs["ls_s4"][i]
            obs["d_1f_wz"][i] = corridor_dist_from_depth(dep, None, unc, obs["K"][i], rot[i], fwd[i], False)[0]
            obs["d_1f_strict"][i] = obs["d_1f_wz"][i]
            if not np.array_equal(fwd[i], true_fwd[i]):
                obs["d_1f_strict"][i] = corridor_dist_from_depth(dep, None, unc, obs["K"][i], rot[i], true_fwd[i], False)[0]
    bev_args = [obs[key] for key in (*keys, "K", "poses_R", "poses_t")]
    def bev(headings):
        return run_bev_on_clip(*bev_args, headings, TemporalBEVConfig())
    obs["d_bev_v23"], _, obs["r_hyb_v23"] = bev(fwd)
    if np.array_equal(true_fwd, fwd):
        obs["d_bev_strict"], obs["r_hyb_strict"] = obs["d_bev_v23"], obs["r_hyb_v23"]
    else:
        obs["d_bev_strict"], _, obs["r_hyb_strict"] = bev(true_fwd)
    return obs


def host_headroom():
    if psutil.virtual_memory().available >= 2 * 2**30:
        return
    if os.name == "nt":
        # This worker maps a checkpoint and imports large native DLLs. Release
        # its unused resident pages before waiting for other local work.
        import ctypes
        from ctypes import wintypes
        trim = ctypes.windll.psapi.EmptyWorkingSet
        trim.argtypes = [wintypes.HANDLE]
        trim(wintypes.HANDLE(-1))
    for _ in range(60):
        if psutil.virtual_memory().available >= 2 * 2**30:
            return
        time.sleep(2)
    raise MemoryError("Local host remains below 2 GiB available; existing clips can be resumed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    ckpt = WORK / "egogeom_omni_v4_unified_best.pt"
    model_digest = digest(ckpt)
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    with torch.device("cuda"):
        model = EgoGeomOmniV4Net(EgoGeomV4Config(**checkpoint["cfg"]), pretrained_backbone=False)
    model.load_state_dict(checkpoint["ema"], strict=True)
    del checkpoint
    model.cuda().eval()
    protocol_files = [ROUTE / "eval_cross_glasses_zeroshot.py",
                      ROUTE / "egogeom_temporal_bev.py", ROUTE / "egogeom_omni_v4_model.py",
                      ROUTE / "egogeom_omni_model.py", ROUTE / "egogeom_omni_v3_model.py",
                      ROUTE / "egogeom_losses_and_physics.py"]
    observation_code = inspect.getsource(observe) + inspect.getsource(strict_headings)
    identity = hashlib.sha256((observation_code + "".join(digest(p) for p in protocol_files)).encode()).hexdigest()
    archive = json.loads((WORK / "cross_glasses_zeroshot_v4_final.json").read_text())
    manifest = {"checkpoint_sha256": model_digest, "protocol_sha256": identity, "batch": args.batch,
                    "device": torch.cuda.get_device_name(), "torch_version": str(torch.__version__),
                    "observation_precision": "original BF16 CUDA autocast, heads copied to FP32", "suites": {}}
    for suite, (folder, shade, cutoff, expected) in SUITES.items():
        sources = sorted((WORK / "zeroshot_eval" / folder).rglob("*.npz"))
        if len(sources) != expected or len({p.stem for p in sources}) != expected:
            raise ValueError(f"{suite}: expected {expected} unique clips, found {len(sources)}")
        dest = ROOT / suite
        dest.mkdir(exist_ok=True, parents=True)
        clips, entries = [], []
        for i, source in enumerate(sources, 1):
            host_headroom()
            target = dest / source.name
            tag = f"{model_digest}:{identity}:{args.batch}:{digest(source)}:{shade}"
            if target.exists():
                with np.load(target, allow_pickle=False) as pack:
                    if str(pack["identity"]) != tag:
                        raise ValueError(f"Incompatible cache {target}")
            else:
                obs = observe(model, source, suite, shade, args.batch)
                temp = target.with_suffix(".partial")
                try:
                    with temp.open("wb") as stream:
                        np.savez(stream, identity=tag, **obs)
                    os.replace(temp, target)
                finally:
                    temp.unlink(missing_ok=True)
                del obs
            # Evaluator truth is opened only after the observable stream is finalized.
            with np.load(target) as pack, np.load(source) as truth:
                clip = {k: pack[k] for k in ("d_1f_wz", "d_bev_v23", "r_hyb_v23")}
                clip.update(name=source.stem, eval_rows=truth["eval_rows"])
            clips.append(clip)
            entries.append({"name": source.stem, "frames": len(clip["d_1f_wz"]), "input_sha256": tag.split(":")[3]})
            if i == 1 or i % 5 == 0 or i == expected:
                print(f"{suite} {i}/{expected}, peak VRAM {torch.cuda.max_memory_reserved()/2**30:.2f} GiB, "
                      f"RAM available {psutil.virtual_memory().available/2**30:.2f} GiB", flush=True)
        metrics = {"1f_world_z": score_dataset(clips, "d_1f_wz", None, cutoff),
                   "stage2_bev_v23_hyb": score_dataset(clips, "d_bev_v23", "r_hyb_v23", cutoff)}
        delta = {arm: {k: (v, archive["suites"][suite][arm][k]) for k, v in results.items()
                       if v != archive["suites"][suite][arm][k]} for arm, results in metrics.items()}
        manifest["suites"][suite] = {"clips": entries, "metrics": metrics, "mismatches_new_archived": delta}
        print(f"{suite} archive differences {delta}", flush=True)
    manifest["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    tmp = ROOT / "manifest.partial"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, ROOT / "manifest.json")


if __name__ == "__main__":
    main()
