"""Calibrate on distinct unlabelled ARKit visits, compile QNN contexts and evaluate paired outputs."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parents[1]
ROOT = ROUTE.parents[2]
ASSETS = ROOT / "artifacts.local/work/egogeom-omni-v0"
OUT = ASSETS / "math_theory" / HERE.name
PANEL = ASSETS / "math_theory/m5_info_geom_quant_reparam/silicon_followup/panel_inputs"
IO = ASSETS / "qnn_v75_v41_final"
_default_sdk=next((p/"qairt"/"2.47.0.260601" for p in Path(sys.executable).parents
                   if (p/"qairt"/"2.47.0.260601").is_dir()),Path("missing-qairt-sdk"))
SDK = Path(os.environ.get("QAIRT_SDK_ROOT", _default_sdk))
QPY = Path(os.environ.get("QAIRT_PYTHON", Path(sys.executable).parent.parent.parent/"qairt-2.47-py310/Scripts/python.exe"))
BINDIR = SDK / "bin/x86_64-windows-msvc"
LIBDIR = SDK / "lib/x86_64-windows-msvc"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def qenv():
    return dict(os.environ, PYTHONPATH=str(SDK / "lib/python"), QAIRT_SDK_ROOT=str(SDK),
                QNN_SDK_ROOT=str(SDK), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                PATH=os.pathsep.join([str(LIBDIR), str(BINDIR), os.environ["PATH"]]))


def model_path(kind):
    return ASSETS / f"egogeom_omni_v41_best_{kind}_fp32.onnx"


def calibration():
    import torch
    sys.path[:0] = [str(ROUTE), str(ROUTE.parent / "egoclear-v0")]
    from egogeom_losses_and_physics import compute_gravity_aligned_ray_map
    from egogeom_omni_v41_model import analytical_host_focal_from_K
    from eval_cross_glasses_zeroshot import decode_isotropic_frame, IM_MEAN, IM_STD
    torch.set_num_threads(1)
    held = {r["clip"] for r in json.loads((PANEL / "frames.json").read_text())}
    available = sorted(p for p in (ASSETS / "zeroshot_eval/arkit_val30_shard00").glob("*.npz")
                       if p.stem not in held)
    selected = [available[(2*i+1)*len(available)//16] for i in range(8)]
    rows = []
    rng = np.random.default_rng(20261124)
    for i,p in enumerate(selected):
        dest = OUT / "cal" / p.stem
        dest.mkdir(parents=True, exist_ok=True)
        with np.load(p, allow_pickle=False) as data:
            offs = data["rgb_offsets"]
            t = (len(offs)-1)//2
            raw = data["rgb_blob"][offs[t]:offs[t+1]].tobytes()
            rgb,K = decode_isotropic_frame(raw,data["orig_hw"],data["K_orig"],1. if i%2==0 else .18)
            gravity = data["poses_R"][t].T @ np.array([0,0,-1],np.float32)
            gravity /= np.linalg.norm(gravity) + 1e-6
        rays = compute_gravity_aligned_ray_map(torch.tensor(K[None]),torch.tensor(gravity[None]),288,384)
        inputs = {"rgb":((rgb-IM_MEAN)/IM_STD).transpose(2,0,1), "grav_rays":rays.numpy(),
                  "host_focal":analytical_host_focal_from_K(K[0,0],K[1,1],K[0,2],K[1,2])}
        # Independent synthetic mode patterns; the signed H3 evaluation input is never used for calibration.
        has = i%4 != 0
        tof = np.zeros((20,72,96),np.float32)
        if has:
            active = rng.uniform(size=(8,8)) > .35
            z = rng.uniform(.3,4.8,(8,8))
            h = rng.normal(0,.65,(16,8,8))
            h[0] += 2.
            for y,x in np.ndindex(8,8):
                h[int(z[y,x]/.301)%16,y,x] += 1.8
            coarse = np.concatenate([(np.log1p(z)*active)[None],active[None],
                                     np.sign(h)*np.log1p(10*np.abs(h)),np.ones((2,8,8))])
            tof = np.repeat(np.repeat(coarse,9,axis=1),12,axis=2)
        inputs.update(tof_grid_s4=tof,has_tof=float(has))
        for name,arr in inputs.items():
            np.asarray(arr,dtype="<f4").tofile(dest / (name+".raw"))
        rows.append({"clip":p.stem,"frame":t,"dark":bool(i%2),"pack_sha256":sha(p),
                     "raw_sha256":{k:sha(dest/(k+".raw")) for k in inputs}})
    for kind in ("pure_rgb","omni"):
        keys = ["rgb","grav_rays","host_focal"]
        if kind=="omni":
            keys.extend(["tof_grid_s4","has_tof"])
        (OUT/("cal_"+kind+".txt")).write_text("\n".join(
            " ".join(k+":="+str(OUT/"cal"/r["clip"]/(k+".raw")) for k in keys) for r in rows)+"\n")
    assert not held.intersection(r["clip"] for r in rows)
    dump(OUT/"calibration.json",rows)


def execute(cmd,folder,stage):
    with (folder/(stage+".log")).open("wb") as f:
        cp = subprocess.run([str(x) for x in cmd],cwd=folder,env=qenv(),stdout=f,stderr=subprocess.STDOUT)
    print(folder.name,stage,cp.returncode,flush=True)
    return {"argv":list(map(str,cmd)), "returncode":cp.returncode,"log":stage+".log"}


def convert(kind, variety="native"):
    import onnx
    source = model_path(kind) if variety=="native" else OUT/("equal_"+kind+".onnx")
    g = onnx.load(source)
    onnx.checker.check_model(g)
    folder = OUT/(variety+"_"+kind)
    folder.mkdir(parents=True,exist_ok=True)
    candidate = folder/"base.dlc"
    layouts=[]
    for side,vs in [("input",g.graph.input),("output",g.graph.output)]:
        for v in vs:
            for layout in ["source_model","desired"]:
                layouts.extend([f"--{layout}_{side}_layout",v.name,"NCHW"])
    del g
    command = [QPY,BINDIR/"qairt-converter","-i",source,"-o",candidate,
               "--preserve_io_datatype",*layouts]
    if variety=="mixed":
        command.extend(["--quantization_overrides",folder/"overrides.json"])
    record = {"onnx_sha256":sha(source),"onnx_checker":"passed","conversion":execute(command,folder,"convert")}
    if record["conversion"]["returncode"]==0:
        record["dlc_sha256"] = sha(candidate)
    dump(folder/"conversion.json",record)
    if not record.get("dlc_sha256"):
        raise RuntimeError(f"Converter failed; inspect {folder/'convert.log'}")


def build(kind,bits,variety="native", override=None):
    src = OUT/(variety+"_"+kind)
    folder = OUT/f"{variety}_{kind}_a{bits}"
    folder.mkdir(parents=True,exist_ok=True)
    record = {"kind":kind,"weight_bits":8,"activation_bits":bits,"variety":variety}
    dlc = folder/"graph.dlc"
    listing=OUT/f"cal_{kind}.txt"
    if variety=="focal":
        # Post-result sensitivity test: the public model contract clamps the analytical factor to [.2,4].
        # Reuse exactly the same RGB/ToF samples. Only the synthetic host_focal changes in two extra rows.
        src=OUT/("native_"+kind)
        lines=listing.read_text().splitlines()
        for index,endpoint in enumerate((.2,4.)):
            point=folder/f"endpoint_{index}.raw"
            np.asarray([endpoint],dtype="<f4").tofile(point)
            lines.append(" ".join("host_focal:="+str(point) if token.startswith("host_focal:=") else token
                                  for token in lines[index].split()))
        listing=folder/"cal_focal_contract.txt"
        listing.write_text("\n".join(lines)+"\n")
    cmd = [QPY,BINDIR/"qairt-quantizer","-i",override or src/"base.dlc","-o",dlc,
           "--input_list",listing,"--weights_bitwidth","8","--act_bitwidth",str(bits),
           "--float_bitwidth","16","--bias_bitwidth","32","--preserve_io_datatype","--dump_encoding_json"]
    if variety=="matched":
        original=OUT/f"native_{kind}_a{bits}"
        dlc=original/"graph.dlc"
        record["quantization"]=json.loads((original/"build.json").read_text())["quantization"]
        record["quantized_dlc_sha256"]=sha(dlc)
    else:
        record["quantization"] = execute(cmd,folder,"quantize")
    if record["quantization"]["returncode"]==0:
        template = IO/("htp_rgb.json" if kind=="pure_rgb" else "htp_omni.json")
        config = json.loads(template.read_text())
        # QAIRT imports the output DLC stem ("base"), as verified by context metadata.
        config["graphs"][0]["graph_names"]=["base"]
        dump(folder/"htp.json",config)
        dump(folder/"ext.json",{"backend_extensions":{"shared_library_path":str(LIBDIR/"QnnHtpNetRunExtensions.dll"),
                                                      "config_file_path":str(folder/"htp.json")}})
        record["compile"] = execute([BINDIR/"qnn-context-binary-generator.exe","--backend",LIBDIR/"QnnHtp.dll",
            "--model",LIBDIR/"QnnModelDlc.dll","--dlc_path",dlc,"--config_file",folder/"ext.json",
            "--binary_file","graph","--output_dir",folder],folder,"compile")
        binary=folder/"graph.bin"
        if record["compile"]["returncode"]==0 and binary.is_file():
            record["info"] = execute([BINDIR/"qnn-context-binary-utility.exe","--context_binary",binary,
                "--json_file",folder/"info.json","--unified_qairt_format"],folder,"info")
            info=json.loads((folder/"info.json").read_text())
            record.update(context_bytes=binary.stat().st_size,context_sha256=sha(binary),arch=info["contextMetadata"]["dspArch"],
                          soc=info["socModel"],graph_name=info["graphs"][0]["graphName"],
                          graph_blob=info["graphs"][0]["graphBlobInfo"])
            assert (record["arch"],record["soc"])==(75,57)
    dump(folder/"build.json",record)
    if not record.get("context_sha256"):
        raise RuntimeError(f"Quantization or context generation failed; inspect {folder}")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("task",choices=["calibrate","convert","build","device","report","equalize","mixed"])
    ap.add_argument("--kind",choices=["pure_rgb","omni"],default="pure_rgb")
    ap.add_argument("--bits",type=int,choices=[8,16],default=8)
    ap.add_argument("--variety",default="native")
    args=ap.parse_args()
    if args.task=="calibrate":
        calibration()
    elif args.task=="convert":
        convert(args.kind,args.variety)
    elif args.task=="build":
        build(args.kind,args.bits,args.variety)
    elif args.task=="equalize":
        from equalize import export
        export(args.kind)
    elif args.task=="mixed":
        from equalize import mixed
        mixed(args.kind)
    else:
        from device_panel import run,report
        if args.task=="device":
            run(args.kind,args.variety,args.bits)
        else:
            report()
