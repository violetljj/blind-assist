"""Prepare label-free inputs and compare published/new graphs on identical host and device frames."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from eval_on_v4 import read_inputs, session
from quant_head import ASSETS, OUT
from silicon_run import IO, WORK


def prepare():
    import torch
    route = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(route))
    sys.path.insert(0, str(route.parent / "egoclear-v0"))
    from egogeom_losses_and_physics import compute_gravity_aligned_ray_map
    from eval_cross_glasses_zeroshot import (
        H_IN,
        IM_MEAN,
        IM_STD,
        W_IN,
        decode_isotropic_frame,
    )
    torch.set_num_threads(1)
    panel = WORK / "panel_inputs"
    panel.mkdir(exist_ok=True, parents=True)
    rows = []
    intrinsics = {}
    for name, folder, shade in [("aria","aria_g2_20clips",1.),("coda","coda_zeroshot_shard000",1.),
                                ("day","arkit_val30_shard00",1.),("dark","arkit_val30_shard00",.18)]:
        clips = sorted((ASSETS/"zeroshot_eval"/folder).rglob("*.npz"))
        # Four evenly distributed clips; use each middle frame without opening labels.
        for j in range(4):
            source = clips[(2*j+1)*len(clips)//8]
            with np.load(source, allow_pickle=False) as pack:
                offs = pack["rgb_offsets"]
                t = (len(offs)-1)//2
                data = pack["rgb_blob"][offs[t]:offs[t+1]].tobytes()
                rgb, K = decode_isotropic_frame(data,pack["orig_hw"],pack["K_orig"],shade)
                g = pack["poses_R"][t].T @ np.array([0.,0.,-1.], np.float32)
            g /= np.linalg.norm(g) + 1e-6
            rays = compute_gravity_aligned_ray_map(torch.tensor(K[None]), torch.tensor(g[None]),H_IN,W_IN)
            frame_id = f"{name}_{j}"
            dest = panel/frame_id
            dest.mkdir(exist_ok=True)
            for filename, data in (("rgb.raw", ((rgb-IM_MEAN)/IM_STD).transpose(2,0,1)),
                                   ("grav_rays.raw", rays.numpy())):
                path = dest / filename
                blob = data.astype("<f4").tobytes()
                if path.exists():
                    if path.read_bytes() != blob:
                        raise ValueError(f"Prepared input differs from existing evidence at {path}")
                else:
                    path.write_bytes(blob)
            rows.append({"id":frame_id,"domain":name,"clip":source.stem,"frame":t})
            intrinsics[frame_id] = K.tolist()
    (panel/"frames.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    (panel/"intrinsics.json").write_text(json.dumps(intrinsics,indent=2),encoding="utf-8")
    return rows


def metric(value, ref):
    error = 100*(value.astype(float)/ref-1)
    return {"mean_abs_pct":float(np.abs(error).mean()),"p95_abs_pct":float(np.quantile(np.abs(error),.95)),
            "signed_mean_pct":float(error.mean())}


def analyze(folder):
    receipt = json.loads((folder/"receipt.json").read_text())
    panel = WORK/"panel_inputs"
    # Use the first measured block of a given arm for fidelity; all blocks are retained for timing.
    blocks = {(a["arm"],a["mode"]):folder/a["block"] for a in reversed(receipt["blocks"])}
    result = {}
    for mode in dict.fromkeys(k[1] for k in blocks):
        kind = "pure_rgb" if mode in ("rgb","panel") else "omni"
        prefix = "rgb" if kind == "pure_rgb" else "omni"
        reference = session(ASSETS/f"egogeom_omni_v4_best_{kind}_fp32.onnx")
        codec = session(OUT/f"m5_v4_{kind}.onnx")
        rows = json.loads((panel/"frames.json").read_text()) if mode == "panel" else [{"id":"archived"}]
        measurements = []
        for i,row in enumerate(rows):
            inputs = read_inputs(reference,mode,panel/row["id"] if mode == "panel" else IO)
            base_host = reference.run(["depth_m"], inputs)[0].ravel()
            new_host = codec.run(["depth_m"], inputs)[0].ravel()
            base_device, new_device = [np.fromfile(blocks[(prefix+"_"+key, mode)]/f"Result_{i}/depth_m.raw",
                                                  dtype="<f4") for key in ("base","codec")]
            if not all(a.shape == (288*384,) and np.isfinite(a).all() and a.min()>0
                       for a in (base_host,new_host,base_device,new_device)):
                raise ValueError("Unexpected depth output")
            item = {**row,"device_base_vs_host_base":metric(base_device,base_host),
                    "device_codec_vs_host_base":metric(new_device,base_host),
                    "device_codec_vs_host_codec":metric(new_device,new_host),
                    "host_codec_vs_host_base":metric(new_host,base_host),
                    "device_codec_vs_device_base":metric(new_device,base_device),
                    "levels_host_codec":len(np.unique(new_host)),"levels_device_codec":len(np.unique(new_device))}
            if mode != "panel":
                archived = IO / ("out_pure_rgb" if mode == "rgb" else "out_omni_"+mode) / "depth_m.raw"
                item["base_matches_archive_exactly"] = np.array_equal(base_device,np.fromfile(archived,"<f4"))
            measurements.append(item)
        result[mode] = measurements
        del reference,codec
    (folder/"fidelity.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    for mode, rows in result.items():
        print(mode,{k:round(np.mean([r[k]["mean_abs_pct"] for r in rows]),4)
                    for k in ("device_base_vs_host_base","device_codec_vs_host_base","host_codec_vs_host_base")})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","analyze"))
    parser.add_argument("--run",type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        print(f"Prepared {len(prepare())} matched device inputs.")
    else:
        if not args.run:
            parser.error("Specify --run")
        analyze(WORK/args.run)
