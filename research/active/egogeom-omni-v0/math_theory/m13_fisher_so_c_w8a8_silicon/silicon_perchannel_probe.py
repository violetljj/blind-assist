"""Track A: Per-Channel + SQNR/Percentile + CLE + Focal-Endpoint W8A8 / W8A16 Live S24 Ultra Silicon Probe.

Directly tests the exact root causes identified in M13 THEORY.md on live Samsung Galaxy S24 Ultra
(SM-S9280 / Hexagon v75 / QAIRT 2.47) against the identical frozen 16-frame 4-domain panel and FP32 host ref:
  - Baseline (M13 focal-endpoint per-tensor W8A8): Pure-RGB 4.508 ms / 11.837%, Omni 4.939 ms / 8.999%
  - Baseline (v4.1 FP16):                          Pure-RGB 16.493 ms / 0.587%, Omni 19.237 ms / 0.377%
  - Candidate arms (ZERO retraining, strictly unlabelled non-panel ARKit calibration + [0.2, 4.0] focal contract):
      1. pc_sqnr_a8:  Per-channel/per-row W8 + SQNR act/param + focal endpoints
      2. pc_pct_a8:   Per-channel/per-row W8 + Percentile(99.99) act + SQNR param + focal endpoints
      3. pc_cle_a8:   Per-channel/per-row W8 + CLE + SQNR + focal endpoints
      4. pc_sqnr_a16: Per-channel/per-row W8A16 + SQNR + focal endpoints
      5. pc_fp16tail_a8: Per-channel W8A8 backbone + FP16 terminal disparity/softplus/focal/reciprocal head
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_empirical import BINDIR, IO, LIBDIR, OUT, PANEL, QPY, dump, execute, model_path, qenv, sha
from device_panel import inputs as panel_inputs, reference as host_reference

SUB = OUT / "perchannel_probe"
RUNTIME = "/data/local/tmp/ba_qairt_htp_r0"


def ensure_focal_cal_list(kind: str) -> Path:
    """Build strictly unlabelled ARKit calibration list + [0.2, 4.0] physical focal contract endpoints."""
    SUB.mkdir(parents=True, exist_ok=True)
    src_list = OUT / f"cal_{kind}.txt"
    assert src_list.exists(), f"Missing {src_list}; run eval_empirical.py calibrate first"
    lines = src_list.read_text(encoding="utf-8").splitlines()
    for idx, endpoint in enumerate((0.2, 4.0)):
        ep_raw = SUB / f"focal_ep_{idx}.raw"
        np.asarray([endpoint], dtype="<f4").tofile(ep_raw)
        lines.append(
            " ".join(
                f"host_focal:={ep_raw}" if tok.startswith("host_focal:=") else tok
                for tok in lines[idx].split()
            )
        )
    out_list = SUB / f"cal_{kind}_focal.txt"
    out_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_list


def make_fp16_tail_overrides(kind: str, out_json: Path) -> Path:
    """Mark ONLY the tiny terminal disparity/softplus/focal/reciprocal (+ global_ctx) tensors as FP16."""
    import onnx

    m = onnx.load(model_path(kind))
    tail_keys = (
        "disp_head",
        "global_ctx/Slice_2",
        "host_focal",
        "/core/Add_3",
        "/core/Abs",
        "/core/Neg",
        "/core/Exp",
        "/core/Add_4",
        "/core/Div",
        "/core/Mul",
        "/core/Add_5",
        "/core/Add_6",
        "/core/Add_7",
        "/core/Add_8",
        "/core/Relu",
        "/core/Clip",
        "/core/Resize_4",
        "/core/Reciprocal",
        "depth_m",
        "tof_branch/disp_res_head",
        "tof_branch/Tanh",
        "tof_branch/Mul",
    )
    act_enc = {}
    for n in m.graph.node:
        for o in n.output:
            if any(k in o or k in n.name for k in tail_keys):
                if o and o not in ("normal_grav", "delta_h_m", "log_sigma"):
                    act_enc[o] = [{"bitwidth": 16, "dtype": "FLOAT"}]
    act_enc["host_focal"] = [{"bitwidth": 16, "dtype": "FLOAT"}]
    act_enc["depth_m"] = [{"bitwidth": 16, "dtype": "FLOAT"}]
    payload = {"activation_encodings": act_enc, "param_encodings": {}}
    dump(out_json, payload)
    return out_json


def make_stem16_tailfp16_overrides(kind: str, out_json: Path, stem_only_int16: bool = True) -> Path:
    """Surgical mixed-precision overrides:
      - If stem_only_int16=True (base act=8): stem+stages_0+edge_s2 in INT16 (kills dark 0.18x 11% error!),
        stages_1..3 + FPN in fast W8A8, tiny global_ctx+disp_head+softplus+focal+1/disp+tof_branch in FP16!
      - If stem_only_int16=False (base act=16): full W8A16 enc+FPN, tiny global_ctx+disp_head+softplus+focal+1/disp+tof_branch in FP16!
    """
    import onnx

    m = onnx.load(model_path(kind))
    fp16_prefixes = (
        "/core/global_ctx/",
        "/core/disp_head/",
        "/core/tof_branch/",
        "/core/Add_3", "/core/Add_4", "/core/Add_5", "/core/Add_6", "/core/Add_7", "/core/Add_8",
        "/core/Abs", "/core/Neg", "/core/Exp", "/core/Div", "/core/Relu",
        "/core/Mul", "/core/Clip", "/core/Resize_4", "/core/Reciprocal",
    )
    int16_prefixes = ("/core/enc/stem_", "/core/enc/stages_0/", "/core/edge_s2/")
    act_enc = {}
    if stem_only_int16:
        act_enc["rgb"] = [{"bitwidth": 16, "dtype": "INT"}]
    for inp in m.graph.input:
        if inp.name in ("host_focal", "tof_grid_s4", "has_tof"):
            act_enc[inp.name] = [{"bitwidth": 16, "dtype": "FLOAT"}]
    for n in m.graph.node:
        nm = n.name
        is_fp16 = any(nm.startswith(p) for p in fp16_prefixes)
        # Avoid catching Div_1 (normal_grav) or Clip_3/Clip_4
        if nm in ("/core/Div_1", "/core/Clip_3", "/core/Clip_4"):
            is_fp16 = False
        if is_fp16:
            for o in n.output:
                if o:
                    act_enc[o] = [{"bitwidth": 16, "dtype": "FLOAT"}]
        elif stem_only_int16 and any(nm.startswith(p) for p in int16_prefixes):
            for o in n.output:
                if o:
                    act_enc[o] = [{"bitwidth": 16, "dtype": "INT"}]
    act_enc["depth_m"] = [{"bitwidth": 16, "dtype": "FLOAT"}]
    dump(out_json, {"activation_encodings": act_enc, "param_encodings": {}})
    return out_json


def make_fp16_dec_overrides(kind: str, out_json: Path, stem_int16: bool = False) -> Path:
    """Keep 100% of the 50M ConvNeXt encoder (`enc.*`) in INT8/W8A16; put ONLY the tiny FPN+ToF+Head decoder in FP16.

    Single clean INT->FP16 boundary at (f1, f2, f3, f4) -> minimal cast overhead AND preserves exact 0.000000m
    zero-interference between Pure-RGB and Omni! If stem_int16=True, also upgrades rgb + enc.stem + enc.stages_0 to INT16.
    """
    import onnx

    m = onnx.load(model_path(kind))
    int16_prefixes = ("/core/enc/stem_", "/core/enc/stages_0/")
    act_enc = {}
    if stem_int16:
        act_enc["rgb"] = [{"bitwidth": 16, "dtype": "INT"}]
    for inp in m.graph.input:
        if inp.name != "rgb":
            act_enc[inp.name] = [{"bitwidth": 16, "dtype": "FLOAT"}]
    for n in m.graph.node:
        if "/core/enc/" in n.name:
            if stem_int16 and any(n.name.startswith(p) for p in int16_prefixes):
                for o in n.output:
                    if o:
                        act_enc[o] = [{"bitwidth": 16, "dtype": "INT"}]
            continue
        for o in n.output:
            if o:
                act_enc[o] = [{"bitwidth": 16, "dtype": "FLOAT"}]
    payload = {"activation_encodings": act_enc, "param_encodings": {}}
    dump(out_json, payload)
    return out_json


def build_arm(kind: str, arm: str) -> dict:
    folder = SUB / f"{arm}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    base_dlc = OUT / f"native_{kind}" / "base.dlc"
    assert base_dlc.exists(), f"Missing {base_dlc}"
    cal_list = ensure_focal_cal_list(kind)

    use_dlc = base_dlc
    bits = 16 if arm.endswith("_a16") else 8
    extra_q: list[str] = [
        "--use_per_channel_quantization",
        "--use_per_row_quantization",
    ]
    if arm == "pc_sqnr_a8":
        extra_q += ["--act_quantizer_calibration", "sqnr", "--param_quantizer_calibration", "sqnr"]
    elif arm == "pc_pct_a8":
        extra_q += [
            "--act_quantizer_calibration", "percentile",
            "--percentile_calibration_value", "99.99",
            "--param_quantizer_calibration", "sqnr",
        ]
    elif arm == "pc_cle_a8":
        extra_q += ["--apply_algorithms", "cle", "--act_quantizer_calibration", "sqnr", "--param_quantizer_calibration", "sqnr"]
    elif arm == "pc_sqnr_a16":
        extra_q += ["--act_quantizer_calibration", "sqnr", "--param_quantizer_calibration", "sqnr"]
    elif arm in (
        "pc_fp16tail_a8", "pc_fp16dec_a8", "pc_fp16dec_a16", "pc_fp16dec_mm_a8", "pc_fp16dec_mm_a16",
        "pc_stem16_tailfp16_a8", "pc_a16_fp16tail_a16", "pc_stem16_fp16dec_a8",
    ):
        if arm == "pc_fp16tail_a8":
            ov_path = make_fp16_tail_overrides(kind, folder / "fp16_tail_overrides.json")
        elif arm == "pc_stem16_tailfp16_a8":
            ov_path = make_stem16_tailfp16_overrides(kind, folder / "stem16_tailfp16_overrides.json", stem_only_int16=True)
        elif arm == "pc_a16_fp16tail_a16":
            ov_path = make_stem16_tailfp16_overrides(kind, folder / "a16_tailfp16_overrides.json", stem_only_int16=False)
        elif arm == "pc_stem16_fp16dec_a8":
            ov_path = make_fp16_dec_overrides(kind, folder / "stem16_fp16dec_overrides.json", stem_int16=True)
        else:
            ov_path = make_fp16_dec_overrides(kind, folder / "fp16_dec_overrides.json")
        import onnx
        g = onnx.load(model_path(kind))
        layouts = []
        for side, vs in [("input", g.graph.input), ("output", g.graph.output)]:
            for v in vs:
                for layout in ["source_model", "desired"]:
                    layouts.extend([f"--{layout}_{side}_layout", v.name, "NCHW"])
        del g
        use_dlc = folder / "base.dlc"
        conv_res = execute(
            [QPY, BINDIR / "qairt-converter", "-i", model_path(kind), "-o", use_dlc,
             "--preserve_io_datatype", "--quantization_overrides", ov_path, *layouts],
            folder, "convert",
        )
        if conv_res["returncode"] != 0:
            raise RuntimeError(f"Converter failed for {arm}_{kind}; check {folder/'convert.log'}")
        if "_mm_" in arm:
            extra_q += ["--act_quantizer_calibration", "min-max", "--param_quantizer_calibration", "sqnr"]
        else:
            extra_q += ["--act_quantizer_calibration", "sqnr", "--param_quantizer_calibration", "sqnr"]
    else:
        raise ValueError(f"Unknown arm {arm}")

    dlc = folder / "graph.dlc"
    q_cmd = [
        QPY, BINDIR / "qairt-quantizer",
        "-i", use_dlc, "-o", dlc,
        "--input_list", cal_list,
        "--weights_bitwidth", "8",
        "--act_bitwidth", str(bits),
        "--float_bitwidth", "16",
        "--bias_bitwidth", "32",
        "--preserve_io_datatype",
        "--dump_encoding_json",
        *extra_q,
    ]
    q_res = execute(q_cmd, folder, "quantize")
    if q_res["returncode"] != 0:
        raise RuntimeError(f"Quantizer failed for {arm}_{kind}; inspect {folder/'quantize.log'}")

    template = IO / ("htp_rgb.json" if kind == "pure_rgb" else "htp_omni.json")
    cfg = json.loads(template.read_text(encoding="utf-8"))
    cfg["graphs"][0]["graph_names"] = ["base"]
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
        raise RuntimeError(f"Compile failed for {arm}_{kind}; inspect {folder/'compile.log'}")

    i_res = execute(
        [BINDIR / "qnn-context-binary-utility.exe",
         "--context_binary", binary,
         "--json_file", folder / "info.json",
         "--unified_qairt_format"],
        folder, "info",
    )
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    clog = (folder / "compile.log").read_text(errors="replace")
    ddr = {k: int(v) for k, v in re.findall(r"\b(spill_bytes|fill_bytes|write_total_bytes|read_total_bytes)=(\d+)", clog)}
    rec = {
        "arm": arm,
        "kind": kind,
        "bits": bits,
        "context_bytes": binary.stat().st_size,
        "context_sha256": sha(binary),
        "arch": info["contextMetadata"]["dspArch"],
        "soc": info["socModel"],
        "graph_name": info["graphs"][0]["graphName"],
        "ddr": ddr,
    }
    dump(folder / "build.json", rec)
    return rec


def run_on_s24u(kind: str, arm: str) -> dict:
    exe = shutil.which("adb.exe") or shutil.which("adb")
    serial = os.environ.get("M13_DEVICE_SERIAL", "")
    if not serial:
        devs = subprocess.check_output([exe, "devices"], text=True, timeout=30)
        online = [s.split()[0] for s in devs.splitlines() if s.endswith("\tdevice")]
        assert len(online) == 1, f"Expected 1 online ADB device, found {online}"
        serial = online[0]

    def adb(*argv):
        for attempt in range(4):
            p = subprocess.run([exe, "-s", serial, *map(str, argv)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
            diag = p.stdout + p.stderr
            if p.returncode == 0 or not any(x in diag for x in ("cannot connect to daemon", "failed to start daemon", "cannot connect to 127.0.0.1")):
                break
            time.sleep(1.5 + attempt)
        if p.returncode:
            raise RuntimeError((p.stderr + p.stdout)[-2000:])
        return p.stdout.strip()

    def shell(*argv):
        return adb("shell", shlex.join(list(map(str, argv))))

    folder = SUB / f"{arm}_{kind}"
    b = json.loads((folder / "build.json").read_text(encoding="utf-8"))
    context = folder / "graph.bin"
    graph = b["graph_name"]
    assert shell("getprop", "ro.product.model") == "SM-S9280"

    local = folder / "device"
    local.mkdir(parents=True, exist_ok=True)
    temp = shell("mktemp", "-d", "/data/local/tmp/m13pc_XXXXXXXX")
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
    host_reference(kind, rows)
    result = {"arm": arm, "kind": kind, "context_bytes": b["context_bytes"], "ddr": b["ddr"], "runs": {}, "fidelity": {}}
    try:
        cfg = json.loads((IO / ("htp_rgb.json" if kind == "pure_rgb" else "htp_omni.json")).read_text(encoding="utf-8"))
        cfg["graphs"][0]["graph_names"] = [graph]
        dump(local / "htp.json", cfg)
        dump(
            local / "ext.json",
            {"backend_extensions": {"shared_library_path": f"{RUNTIME}/libQnnHtpNetRunExtensions.so",
                                    "config_file_path": ship(local / "htp.json")}},
        )
        modes = ["rgb"] if kind == "pure_rgb" else ["signed", "zero"]
        for mode in modes:
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
                        ref = np.load(OUT / "device" / f"host_{kind}_{mode}_{r['id']}.npy")
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
            rgb_dir = SUB / f"{arm}_pure_rgb" / "device"
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
    ap.add_argument("--arms", nargs="+", default=["pc_sqnr_a8", "pc_pct_a8", "pc_sqnr_a16", "pc_fp16tail_a8"])
    ap.add_argument("--kinds", nargs="+", default=["pure_rgb", "omni"])
    args = ap.parse_args()
    all_res = {}
    for arm in args.arms:
        for kind in args.kinds:
            print(f"\n================ [{arm} | {kind}] ================", flush=True)
            try:
                build_arm(kind, arm)
                all_res[f"{arm}_{kind}"] = run_on_s24u(kind, arm)
            except Exception as e:
                print(f"  [FAILED {arm}_{kind}]: {e}", flush=True)
                all_res[f"{arm}_{kind}"] = {"error": str(e)}
            dump(SUB / "perchannel_probe_summary.json", all_res)


if __name__ == "__main__":
    main()
