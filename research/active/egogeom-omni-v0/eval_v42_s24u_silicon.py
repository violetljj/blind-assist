"""Live S24 Ultra (SM-S9280 / Hexagon v75 / QAIRT 2.47) benchmark for EgoGeom-Omni v4.2.

Compiles & runs BOTH:
  - fp16:          Full FP16 reference (~108-109 MB)
  - pc_fp16dec_a16: Track A winning 2x-compressed zero-interference profile (~57-59 MB, W8A16 encoder + FP16 decoder/ToF)
against v4.2's OWN FP32 ONNXRuntime host reference on the identical 16-frame 4-domain panel (day, dark, aria, coda).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROUTE = Path(__file__).resolve().parent
M13 = ROUTE / "math_theory/m13_fisher_so_c_w8a8_silicon"
sys.path[:0] = [str(M13)]

from eval_empirical import BINDIR, IO, LIBDIR, OUT as M13_OUT, PANEL, QPY, dump, execute, sha
from device_panel import inputs as panel_inputs
from silicon_perchannel_probe import ensure_focal_cal_list

ASSETS = ROUTE.parents[2] / "artifacts.local/work/egogeom-omni-v0"
V42_OUT = ASSETS / "s24u_v42_silicon"
RUNTIME = "/data/local/tmp/ba_qairt_htp_r0"


def v42_onnx_path(kind: str) -> Path:
    name = "egogeom_omni_v42_best_pure_rgb_fp32.onnx" if kind == "pure_rgb" else "egogeom_omni_v42_best_omni_fp32.onnx"
    p = ASSETS / name
    assert p.is_file(), f"Missing {p}"
    return p


def make_fp16_dec_overrides_for_onnx(onnx_path: Path, out_json: Path) -> Path:
    """Place exact INT->FP16 boundary at (f1,f2,f3,f4): ConvNeXt encoder W8A16, entire FPN+ToF+head FP16."""
    import onnx

    m = onnx.load(str(onnx_path))
    act_ov: dict[str, list[dict]] = {}
    param_ov: dict[str, list[dict]] = {}

    def is_encoder_tensor(name: str) -> bool:
        return name.startswith("/core/enc/") or name.startswith("core.enc.") or name == "rgb"

    for node in m.graph.node:
        if any(not is_encoder_tensor(o) for o in node.output):
            for o in node.output:
                if o:
                    act_ov[o] = [{"bitwidth": 16, "dtype": "FLOAT"}]
            for inp in node.input:
                if inp and not is_encoder_tensor(inp):
                    param_ov[inp] = [{"bitwidth": 16, "dtype": "FLOAT"}]
    for inp in m.graph.input:
        if inp.name != "rgb":
            act_ov[inp.name] = [{"bitwidth": 16, "dtype": "FLOAT"}]
    for out in m.graph.output:
        act_ov[out.name] = [{"bitwidth": 16, "dtype": "FLOAT"}]

    payload = {
        "activation_encodings": act_ov,
        "param_encodings": param_ov,
    }
    dump(out_json, payload)
    return out_json


def compute_v42_host_ref(kind: str, mode: str, rows: list[dict]) -> Path:
    """Compute v4.2 FP32 ONNXRuntime reference outputs for the 16-frame panel."""
    ref_dir = V42_OUT / "host_ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    sentinel = ref_dir / f"{kind}_{mode}_done.json"
    if sentinel.exists():
        return ref_dir
    import onnxruntime as ort

    sess = ort.InferenceSession(str(v42_onnx_path(kind)), providers=["CPUExecutionProvider"])
    in_shapes = {i.name: [int(d) if isinstance(d, int) else 1 for d in i.shape] for i in sess.get_inputs()}
    for r in rows:
        feed = {}
        for k, p in panel_inputs(kind, r["id"], mode).items():
            arr = np.fromfile(p, dtype="<f4").reshape(in_shapes[k])
            feed[k] = arr
        depth = sess.run(["depth_m"], feed)[0].astype(np.float32).reshape(-1)
        np.save(ref_dir / f"host_{kind}_{mode}_{r['id']}.npy", depth)
    dump(sentinel, {"onnx": str(v42_onnx_path(kind)), "sha256": sha(v42_onnx_path(kind)), "frames": len(rows)})
    return ref_dir


def build_v42_arm(kind: str, arm: str) -> dict:
    folder = V42_OUT / f"{arm}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    onnx_p = v42_onnx_path(kind)

    import onnx

    g = onnx.load(str(onnx_p))
    layouts = []
    for side, vs in [("input", g.graph.input), ("output", g.graph.output)]:
        for v in vs:
            for layout in ["source_model", "desired"]:
                layouts.extend([f"--{layout}_{side}_layout", v.name, "NCHW"])
    del g

    base_dlc = folder / "base.dlc"
    conv_cmd = [QPY, BINDIR / "qairt-converter", "-i", onnx_p, "-o", base_dlc, "--preserve_io_datatype", *layouts]
    if arm == "pc_fp16dec_a16":
        ov_path = make_fp16_dec_overrides_for_onnx(onnx_p, folder / "fp16_dec_overrides.json")
        conv_cmd += ["--quantization_overrides", ov_path]
    c0 = execute(conv_cmd, folder, "convert")
    if c0["returncode"] != 0:
        raise RuntimeError(f"Converter failed for v42 {arm}_{kind}; check {folder/'convert.log'}")

    if arm == "fp16":
        dlc = base_dlc
    elif arm == "pc_fp16dec_a16":
        cal_list = ensure_focal_cal_list(kind)
        dlc = folder / "graph.dlc"
        q_cmd = [
            QPY, BINDIR / "qairt-quantizer",
            "-i", base_dlc, "-o", dlc,
            "--input_list", cal_list,
            "--weights_bitwidth", "8",
            "--act_bitwidth", "16",
            "--float_bitwidth", "16",
            "--bias_bitwidth", "32",
            "--preserve_io_datatype",
            "--use_per_channel_quantization",
            "--use_per_row_quantization",
            "--act_quantizer_calibration", "sqnr",
            "--param_quantizer_calibration", "sqnr",
        ]
        q_res = execute(q_cmd, folder, "quantize")
        if q_res["returncode"] != 0:
            raise RuntimeError(f"Quantizer failed for v42 {arm}_{kind}; check {folder/'quantize.log'}")
    else:
        raise ValueError(f"Unknown arm {arm}")

    template = IO / ("htp_rgb.json" if kind == "pure_rgb" else "htp_omni.json")
    cfg = json.loads(template.read_text(encoding="utf-8"))
    cfg["graphs"][0]["graph_names"] = ["base"]
    if arm == "fp16":
        cfg["graphs"][0]["fp16_relaxed_precision"] = 1
    dump(folder / "htp.json", cfg)
    dump(
        folder / "ext.json",
        {"backend_extensions": {"shared_library_path": str(LIBDIR / "QnnHtpNetRunExtensions.dll"),
                                "config_file_path": str(folder / "htp.json")}},
    )
    c_res = execute(
        [BINDIR / "qnn-context-binary-generator.exe",
         "--backend", LIBDIR / "QnnHtp.dll",
         "--model", LIBDIR / "QnnModelDlc.dll",
         "--dlc_path", dlc,
         "--config_file", folder / "ext.json",
         "--binary_file", "graph",
         "--output_dir", folder],
        folder, "compile",
    )
    binary = folder / "graph.bin"
    if c_res["returncode"] != 0 or not binary.is_file():
        raise RuntimeError(f"Compile failed for v42 {arm}_{kind}; check {folder/'compile.log'}")

    execute(
        [BINDIR / "qnn-context-binary-utility.exe",
         "--context_binary", binary,
         "--json_file", folder / "info.json",
         "--unified_qairt_format"],
        folder, "info",
    )
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    rec = {
        "arm": arm,
        "kind": kind,
        "context_bytes": binary.stat().st_size,
        "context_sha256": sha(binary),
        "arch": info["contextMetadata"]["dspArch"],
        "soc": info["socModel"],
        "graph_name": info["graphs"][0]["graphName"],
    }
    dump(folder / "build.json", rec)
    return rec


def run_v42_on_s24u(kind: str, arm: str) -> dict:
    exe = shutil.which("adb.exe") or shutil.which("adb")
    devs = subprocess.check_output([exe, "devices"], text=True, timeout=30)
    online = [s.split()[0] for s in devs.splitlines() if s.endswith("\tdevice")]
    assert len(online) == 1, f"Expected 1 ADB device, found {online}"
    serial = online[0]

    def adb(*argv):
        for attempt in range(4):
            p = subprocess.run([exe, "-s", serial, *map(str, argv)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
            diag = p.stdout + p.stderr
            if p.returncode == 0 or not any(x in diag for x in ("cannot connect to daemon", "failed to start daemon")):
                break
            time.sleep(1.5 + attempt)
        if p.returncode:
            raise RuntimeError((p.stderr + p.stdout)[-2000:])
        return p.stdout.strip()

    def shell(*argv):
        return adb("shell", shlex.join(list(map(str, argv))))

    folder = V42_OUT / f"{arm}_{kind}"
    b = json.loads((folder / "build.json").read_text(encoding="utf-8"))
    context = folder / "graph.bin"
    graph = b["graph_name"]
    assert shell("getprop", "ro.product.model") == "SM-S9280"

    local = folder / "device"
    local.mkdir(parents=True, exist_ok=True)
    temp = shell("mktemp", "-d", "/data/local/tmp/v42_s24u_XXXXXXXX")
    env = ["env", f"LD_LIBRARY_PATH={RUNTIME}", f"ADSP_LIBRARY_PATH={RUNTIME};/vendor/lib/rfsa/adsp;/dsp"]
    cache: dict[Path, str] = {}

    def ship(path: Path) -> str:
        if path not in cache:
            h = sha(path)
            remote = f"{temp}/{h[:18]}{path.suffix}"
            adb("push", path, remote)
            cache[path] = remote
        return cache[path]

    rows = json.loads((PANEL / "frames.json").read_text(encoding="utf-8"))
    result = {"arm": arm, "kind": kind, "context_bytes": b["context_bytes"], "runs": {}, "fidelity": {}}
    try:
        cfg = json.loads((IO / ("htp_rgb.json" if kind == "pure_rgb" else "htp_omni.json")).read_text(encoding="utf-8"))
        cfg["graphs"][0]["graph_names"] = [graph]
        if arm == "fp16":
            cfg["graphs"][0]["fp16_relaxed_precision"] = 1
        dump(local / "htp.json", cfg)
        dump(
            local / "ext.json",
            {"backend_extensions": {"shared_library_path": f"{RUNTIME}/libQnnHtpNetRunExtensions.so",
                                    "config_file_path": ship(local / "htp.json")}},
        )
        modes = ["rgb"] if kind == "pure_rgb" else ["signed", "zero"]
        for mode in modes:
            ref_dir = compute_v42_host_ref(kind, mode, rows)
            for phase in (["speed", "panel"] if mode != "zero" else ["panel"]):
                tag = f"{mode}_{phase}"
                chosen = rows[:1] if phase == "speed" else rows
                n = 50 if phase == "speed" else len(chosen)
                listing = local / f"{tag}.txt"
                listing.write_text(
                    "\n".join(" ".join(f"{k}:={ship(p)}" for k, p in panel_inputs(kind, r["id"], mode).items()) for r in chosen) + "\n",
                    encoding="utf-8",
                )
                dest = f"{temp}/{tag}"
                shell(
                    *env, f"{RUNTIME}/qnn-net-run",
                    "--backend", f"{RUNTIME}/libQnnHtp.so",
                    "--config_file", ship(local / "ext.json"),
                    "--retrieve_context", ship(context),
                    "--input_list", ship(listing),
                    "--output_dir", dest,
                    "--num_inferences", str(n),
                    "--keep_num_outputs", str(len(chosen)),
                    "--perf_profile", "burst",
                    "--profiling_level", "basic",
                )
                prof = shell(*env, f"{RUNTIME}/qnn-profile-viewer", "--input_log", f"{dest}/qnn-profiling-data.log")
                rec = {"n": n}
                for key, pat in [("htp_us", r"Backend \(Accelerator \(execute\) time\):\s*([\d.]+) us"),
                                 ("netrun_us", r"NetRun:\s*([\d.]+) us")]:
                    m = re.search(pat, prof.split("Execute Stats (Average):")[-1])
                    if m:
                        rec[key] = float(m[1])
                result["runs"][tag] = rec
                if phase == "panel":
                    errs = []
                    for i, r in enumerate(rows):
                        out_raw = local / f"{mode}_{r['id']}.raw"
                        adb("pull", f"{dest}/Result_{i}/depth_m.raw", out_raw)
                        ref = np.load(ref_dir / f"host_{kind}_{mode}_{r['id']}.npy")
                        z = np.fromfile(out_raw, "<f4")
                        errs.append(float(100.0 * np.mean(np.abs(z.astype(float) / ref - 1.0))))
                    by_dom = {
                        d: round(float(np.mean([v for r, v in zip(rows, errs) if r["domain"] == d])), 4)
                        for d in sorted({r["domain"] for r in rows})
                    }
                    result["fidelity"][mode] = {
                        "mean_pct": round(float(np.mean(errs)), 4),
                        "by_domain": by_dom,
                    }
        if kind == "omni":
            rgb_dir = V42_OUT / f"{arm}_pure_rgb" / "device"
            if (rgb_dir / "rgb_aria_0.raw").exists():
                zi = max(
                    float(np.max(np.abs(np.fromfile(rgb_dir / f"rgb_{r['id']}.raw", "<f4")
                                        - np.fromfile(local / f"zero_{r['id']}.raw", "<f4"))))
                    for r in rows
                )
                result["fidelity"]["no_tof_max_abs_vs_pure_rgb_m"] = round(zi, 6)
    finally:
        shell("rm", "-rf", temp)
    dump(local / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["fp16", "pc_fp16dec_a16"])
    ap.add_argument("--kinds", nargs="+", default=["pure_rgb", "omni"])
    args = ap.parse_args()
    out = {}
    for arm in args.arms:
        for kind in args.kinds:
            print(f"\n================ [v4.2 | {arm} | {kind}] ================", flush=True)
            build_v42_arm(kind, arm)
            out[f"{arm}_{kind}"] = run_v42_on_s24u(kind, arm)
    dump(V42_OUT / "v42_s24u_silicon_summary.json", out)
    print(f"Saved -> {V42_OUT / 'v42_s24u_silicon_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
