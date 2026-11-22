"""90-second Phase 4 finisher on G4:
Loads saved egoclear_v3_full.pt & egoclear_v3_full_simreal.pt, saves complete phase4_results.json
(with Aria baselines + Sim-Only + Sim+Real across all 5 regimes), exports & verifies ONNX-17
2-stage streaming encoder+step models, copies to Drive, and flushes Drive cleanly.
"""
from __future__ import annotations

import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/content")
ud_root = Path("/content/unidepth_src")
if ud_root.exists():
    sub = next(ud_root.glob("UniDepth-*"), None)
    if sub is not None:
        sys.path.insert(0, str(sub))

from egoclear_v3_model import EgoClearV3Net
from phase4_simreal_aria_onnx_g4 import DRIVE_RUNS, LOCAL_RUNS, eval_aria_baselines, export_and_verify_onnx
from train_and_eval_g4 import load_scene
from train_and_eval_phase3_g4 import eval_v3_on_regime


def main():
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    LOCAL_RUNS.mkdir(parents=True, exist_ok=True)
    DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

    m287 = json.loads(Path("/content/sanpo_287_manifest.json").read_text())
    bw_speeds = m287["bw_speeds"]

    eval_synth = {}
    for sc, sp in m287["splits"].items():
        if sp == "eval":
            p = Path("/content/sanpo_287/scenes") / f"{sc}.npz"
            if p.exists():
                eval_synth[sc] = load_scene(p)
    eval_real = {p.stem: load_scene(p) for p in sorted(Path("/content/sanpo_p2/real_scenes").glob("*.npz"))}
    aria_dir = Path("/content/aria_g2_scenes")
    aria_dicts = {p.stem: load_scene(p) for p in sorted(aria_dir.glob("*.npz"))}
    z_rloc = np.load("/content/sanpo_p2/real_local_gt_cache.npz", allow_pickle=False)
    real_local_rows = {sc: z_rloc[f"r_{sc}"] for sc in eval_real}

    print(f"[p4-fin] eval_synth={len(eval_synth)}, eval_real={len(eval_real)}, aria_g2={len(aria_dicts)}", flush=True)
    aria_baselines = eval_aria_baselines(aria_dicts, bw_speeds, device)

    # 1. Sim-Only Zero-Shot on Aria Gen 2
    p3_ckpt = torch.load("/content/egoclear_runs_p3/egoclear_v3_full.pt", map_location="cpu", weights_only=False)
    model = EgoClearV3Net(mode="full", c_bev=128, freeze_dav2=True).to(device)
    model.load_state_dict(p3_ckpt["head_state_dict"], strict=False)
    ev_aria_simonly = eval_v3_on_regime(model, aria_dicts, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=True)

    # 2. Sim+Real (75 disjoint real) across all 5 benchmarks
    p4_ckpt = torch.load(LOCAL_RUNS / "egoclear_v3_full_simreal.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(p4_ckpt["head_state_dict"], strict=False)
    ev_simreal = {
        "synth_clean": eval_v3_on_regime(model, eval_synth, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=True),
        "synth_vio_1x": eval_v3_on_regime(model, eval_synth, None, bw_speeds, device, vio_scale=1.0),
        "real_40_local": eval_v3_on_regime(model, eval_real, real_local_rows, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=True),
        "real_40_global": eval_v3_on_regime(model, eval_real, None, bw_speeds, device, vio_scale=0.0),
        "aria_g2_zeroshot": eval_v3_on_regime(model, aria_dicts, None, bw_speeds, device, vio_scale=0.0, compute_uncertainty_audit=True),
    }

    # 3. Export & verify 2-stage mobile streaming ONNX (now using ONNX-17 native torch.sort!)
    onnx_info = export_and_verify_onnx(model, LOCAL_RUNS)

    out_all = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "finisher_wall_sec": round(time.time() - t0, 1),
        "counts": {"synth_train": 627, "real_train": 75, "eval_synth": len(eval_synth), "eval_real": len(eval_real), "aria_g2_clips": len(aria_dicts)},
        "aria_baselines": aria_baselines,
        "aria_egoclear_v3_simonly_zeroshot": ev_aria_simonly,
        "egoclear_v3_simreal": {"history": p4_ckpt.get("history", []), "evals": ev_simreal},
        "onnx_export": onnx_info,
    }
    res_local = LOCAL_RUNS / "phase4_results.json"
    res_local.write_text(json.dumps(out_all, indent=2))
    res_drive = DRIVE_RUNS / "phase4_results.json"
    if res_drive.exists():
        res_drive.unlink()
    shutil.copy2(res_local, res_drive)
    print(f"[p4-fin] DONE saved phase4_results.json ({res_local.stat().st_size} B) in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
