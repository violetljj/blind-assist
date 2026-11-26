"""Live S24 Ultra (SM-S9280 / Hexagon v75) Compilation & Real-Silicon Verification of Fused `v4.1 + G2` (`pc_fp16dec_a16`).

Zero edits to frozen `egogeom_omni_v41_model.py` or `:app`.
Builds two fused graphs (`pure_rgb_g2` and `omni_g2`) by attaching frozen `v41_frozen_glass_head_g2.pt` (0.411M)
onto frozen `v4.1` `feat_s16` + `feat_s32`:
  - Exports static ONNX-17 outputting `(depth_m, delta_h_m, normal_grav, log_sigma, glass_prob_s16)`.
  - Quantizes with the exact `pc_fp16dec_a16` profile:
      * 50M ConvNeXt `enc.*` -> Per-Channel/Per-Row W8A16 SQNR
      * FPN + ToF + Depth/Normal/Uncert Heads + Glass Head G2 -> FP16 (zero interference!)
  - Compiles Hexagon v75 context binaries via QAIRT 2.47 (`qnn-context-binary-generator`).
  - Runs live on Samsung Galaxy S24 Ultra (`R5CX10M8Y8X`) via `qnn-net-run`:
      * N=50 burst HTP latency (us/ms)
      * 16-frame 4-domain fidelity panel (depth error % + `glass_prob_s16` max/mean diff vs PyTorch FP32 + 0 NaN/Inf)
      * Real-silicon zero-interference: `max|Omni+G2(has_tof=0) - PureRGB+G2|` on BOTH `depth_m` and `glass_prob_s16`!
  - Cleans up `/data/local/tmp/v41g2_*` in `finally:`.
"""
from __future__ import annotations

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
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parent
REPO = ROUTE.parents[2]
M13_DIR = ROUTE / "math_theory/m13_fisher_so_c_w8a8_silicon"
ART = REPO / "artifacts.local/work/egogeom-omni-v0"
PROBE_DIR = ART / "cane_blind_band_probe"
WORK = PROBE_DIR / "s24u_v41_g2"

sys.path[:0] = [str(HERE), str(ROUTE), str(M13_DIR)]

from egogeom_omni_v41_model import EgoGeomOmniV41Net, EgoGeomV41Config  # noqa: E402
from eval_empirical import BINDIR, IO, LIBDIR, OUT, PANEL, QPY, dump, execute, sha  # noqa: E402
from device_panel import inputs as panel_inputs  # noqa: E402
from silicon_perchannel_probe import RUNTIME, ensure_focal_cal_list  # noqa: E402


class FrozenDINOv3GlassHeadFused(nn.Module):
    def __init__(self, c3: int = 384, c4: int = 768, hidden: int = 128):
        super().__init__()
        self.lat3 = nn.Sequential(nn.Conv2d(c3, hidden, 1, bias=False), nn.GroupNorm(8, hidden), nn.SiLU(inplace=True))
        self.lat4 = nn.Sequential(nn.Conv2d(c4, hidden, 1, bias=False), nn.GroupNorm(8, hidden), nn.SiLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, 96, 3, 1, 1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(96, 48, 3, 1, 1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 1, 1),
        )

    def forward(self, f3: torch.Tensor, f4: torch.Tensor) -> torch.Tensor:
        p3 = self.lat3(f3)
        p4 = F.interpolate(self.lat4(f4), size=(18, 24), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.fuse(torch.cat([p3, p4], dim=1)))


class PureRgbWithGlassWrapper(nn.Module):
    def __init__(self, core: EgoGeomOmniV41Net, glass_head: FrozenDINOv3GlassHeadFused):
        super().__init__()
        self.core = core
        self.core.tof_branch = None
        self.glass_head = glass_head

    def forward(
        self, rgb: torch.Tensor, grav_rays: torch.Tensor, host_focal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.core(rgb, grav_rays, None, None, host_focal=host_focal, return_features=True)
        glass_prob = self.glass_head(out["feat_s16"], out["feat_s32"])
        return out["depth_m"], out["delta_h_m"], out["normal_grav"], out["log_sigma"], glass_prob


class OmniWithGlassWrapper(nn.Module):
    def __init__(self, core: EgoGeomOmniV41Net, glass_head: FrozenDINOv3GlassHeadFused):
        super().__init__()
        self.core = core
        self.glass_head = glass_head

    def forward(
        self,
        rgb: torch.Tensor,
        grav_rays: torch.Tensor,
        tof_grid_s4: torch.Tensor,
        has_tof: torch.Tensor,
        host_focal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.core(rgb, grav_rays, tof_grid_s4, has_tof, host_focal=host_focal, return_features=True)
        glass_prob = self.glass_head(out["feat_s16"], out["feat_s32"])
        return out["depth_m"], out["delta_h_m"], out["normal_grav"], out["log_sigma"], glass_prob


def load_fused_model(kind: str) -> nn.Module:
    ckpt_p = ART / ("egogeom_omni_v41_pure_rgb_best.pt" if kind == "pure_rgb" else "egogeom_omni_v41_unified_best.pt")
    st = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    core = EgoGeomOmniV41Net(EgoGeomV41Config(**st["cfg"]), pretrained_backbone=False)
    if kind == "pure_rgb":
        core.tof_branch = None
        core.load_state_dict({k: v for k, v in st["ema"].items() if not k.startswith("tof_branch.")}, strict=True)
    else:
        core.load_state_dict(st["ema"], strict=True)
    core.eval()

    g2_st = torch.load(PROBE_DIR / "v41_frozen_glass_head_g2.pt", map_location="cpu", weights_only=False)
    gh = FrozenDINOv3GlassHeadFused().eval()
    gh.load_state_dict(g2_st["state_dict"], strict=True)
    return PureRgbWithGlassWrapper(core, gh).eval() if kind == "pure_rgb" else OmniWithGlassWrapper(core, gh).eval()


def export_onnx_and_host_refs(kind: str, rows: list[dict]) -> tuple[Path, dict]:
    folder = WORK / kind
    folder.mkdir(parents=True, exist_ok=True)
    onnx_path = folder / f"v41_{kind}_g2.onnx"
    ref_dir = folder / "host_ref"
    ref_dir.mkdir(parents=True, exist_ok=True)

    model = load_fused_model(kind)
    rgb_d = torch.zeros(1, 3, 288, 384, dtype=torch.float32)
    rays_d = torch.zeros(1, 3, 288, 384, dtype=torch.float32)
    hf_d = torch.ones(1, 1, 1, 1, dtype=torch.float32)
    outputs = ["depth_m", "delta_h_m", "normal_grav", "log_sigma", "glass_prob_s16"]

    if not onnx_path.exists():
        print(f"[export] Exporting {onnx_path.name}...", flush=True)
        if kind == "pure_rgb":
            inputs_d = (rgb_d, rays_d, hf_d)
            in_names = ["rgb", "grav_rays", "host_focal"]
        else:
            tof_d = torch.zeros(1, 20, 72, 96, dtype=torch.float32)
            ht_d = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
            inputs_d = (rgb_d, rays_d, tof_d, ht_d, hf_d)
            in_names = ["rgb", "grav_rays", "tof_grid_s4", "has_tof", "host_focal"]
        kw = dict(input_names=in_names, output_names=outputs, opset_version=17, do_constant_folding=True)
        try:
            torch.onnx.export(model, inputs_d, str(onnx_path), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(model, inputs_d, str(onnx_path), **kw)
        onnx.checker.check_model(str(onnx_path))

    # Compute PyTorch FP32 host references on the 16-frame panel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    modes = ["rgb"] if kind == "pure_rgb" else ["signed", "zero"]
    with torch.inference_mode():
        for mode in modes:
            for r in rows:
                d_npy = ref_dir / f"{mode}_{r['id']}_depth.npy"
                g_npy = ref_dir / f"{mode}_{r['id']}_glass.npy"
                if d_npy.exists() and g_npy.exists():
                    continue
                inp_map = panel_inputs(kind, r["id"], mode)
                rgb_t = torch.from_numpy(np.fromfile(inp_map["rgb"], "<f4").reshape(1, 3, 288, 384)).to(device)
                rays_t = torch.from_numpy(np.fromfile(inp_map["grav_rays"], "<f4").reshape(1, 3, 288, 384)).to(device)
                hf_t = torch.from_numpy(np.fromfile(inp_map["host_focal"], "<f4").reshape(1, 1, 1, 1)).to(device)
                if kind == "pure_rgb":
                    dep, _, _, _, gls = model(rgb_t, rays_t, hf_t)
                else:
                    tof_t = torch.from_numpy(np.fromfile(inp_map["tof_grid_s4"], "<f4").reshape(1, 20, 72, 96)).to(device)
                    ht_t = torch.from_numpy(np.fromfile(inp_map["has_tof"], "<f4").reshape(1, 1, 1, 1)).to(device)
                    dep, _, _, _, gls = model(rgb_t, rays_t, tof_t, ht_t, hf_t)
                np.save(d_npy, dep.cpu().numpy().reshape(-1))
                np.save(g_npy, gls.cpu().numpy().reshape(-1))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return onnx_path, {"ref_dir": str(ref_dir)}


def build_pc_fp16dec_a16(kind: str, onnx_path: Path) -> dict:
    folder = WORK / kind
    ov_path = folder / "fp16_dec_g2_overrides.json"
    m = onnx.load(str(onnx_path))

    # Keep 100% of ConvNeXt `enc.*` in W8A16; put ALL non-encoder nodes (FPN + ToF + heads + glass_head) in FP16!
    act_enc = {}
    for inp in m.graph.input:
        if inp.name != "rgb":
            act_enc[inp.name] = [{"bitwidth": 16, "dtype": "FLOAT"}]
    for n in m.graph.node:
        if "/core/enc/" in n.name:
            continue
        for o in n.output:
            if o:
                act_enc[o] = [{"bitwidth": 16, "dtype": "FLOAT"}]
    dump(ov_path, {"activation_encodings": act_enc, "param_encodings": {}})

    layouts = []
    for side, vs in [("input", m.graph.input), ("output", m.graph.output)]:
        for v in vs:
            for layout in ["source_model", "desired"]:
                layouts.extend([f"--{layout}_{side}_layout", v.name, "NCHW"])
    del m

    base_dlc = folder / "base.dlc"
    print(f"[qairt] Converting {kind}...", flush=True)
    conv_res = execute(
        [QPY, BINDIR / "qairt-converter", "-i", onnx_path, "-o", base_dlc,
         "--preserve_io_datatype", "--quantization_overrides", ov_path, *layouts],
        folder, "convert",
    )
    if conv_res["returncode"] != 0:
        raise RuntimeError(f"Converter failed for {kind}; inspect {folder/'convert.log'}")

    cal_list = ensure_focal_cal_list(kind)
    dlc = folder / "graph.dlc"
    print(f"[qairt] Quantizing {kind} (pc_fp16dec_a16)...", flush=True)
    q_res = execute(
        [QPY, BINDIR / "qairt-quantizer", "-i", base_dlc, "-o", dlc,
         "--input_list", cal_list,
         "--weights_bitwidth", "8",
         "--act_bitwidth", "16",
         "--float_bitwidth", "16",
         "--bias_bitwidth", "32",
         "--preserve_io_datatype",
         "--dump_encoding_json",
         "--use_per_channel_quantization",
         "--use_per_row_quantization",
         "--act_quantizer_calibration", "sqnr",
         "--param_quantizer_calibration", "sqnr"],
        folder, "quantize",
    )
    if q_res["returncode"] != 0:
        raise RuntimeError(f"Quantizer failed for {kind}; inspect {folder/'quantize.log'}")

    template = IO / ("htp_rgb.json" if kind == "pure_rgb" else "htp_omni.json")
    cfg = json.loads(template.read_text(encoding="utf-8"))
    cfg["graphs"][0]["graph_names"] = ["base"]
    dump(folder / "htp.json", cfg)
    dump(
        folder / "ext.json",
        {"backend_extensions": {"shared_library_path": str(LIBDIR / "QnnHtpNetRunExtensions.dll"),
                                "config_file_path": str(folder / "htp.json")}},
    )

    print(f"[qairt] Compiling Hexagon v75 context binary for {kind}...", flush=True)
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
        raise RuntimeError(f"Compile failed for {kind}; inspect {folder/'compile.log'}")

    execute(
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
        "kind": kind,
        "context_bytes": binary.stat().st_size,
        "context_mb": round(binary.stat().st_size / 1e6, 3),
        "context_sha256": sha(binary),
        "arch": info["contextMetadata"]["dspArch"],
        "soc": info["socModel"],
        "graph_name": info["graphs"][0]["graphName"],
        "ddr": ddr,
    }
    dump(folder / "build.json", rec)
    return rec


def run_on_s24u(kind: str, rows: list[dict]) -> dict:
    exe = shutil.which("adb.exe") or shutil.which("adb")
    devs = subprocess.check_output([exe, "devices"], text=True, timeout=30)
    online = [s.split()[0] for s in devs.splitlines() if s.endswith("\tdevice")]
    assert len(online) == 1, f"Expected 1 online ADB device, found {online}"
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

    folder = WORK / kind
    b = json.loads((folder / "build.json").read_text(encoding="utf-8"))
    context = folder / "graph.bin"
    graph = b["graph_name"]
    assert shell("getprop", "ro.product.model") == "SM-S9280"

    local = folder / "device"
    local.mkdir(parents=True, exist_ok=True)
    ref_dir = folder / "host_ref"
    temp = shell("mktemp", "-d", "/data/local/tmp/v41g2_XXXXXXXX")
    env = ["env", f"LD_LIBRARY_PATH={RUNTIME}", f"ADSP_LIBRARY_PATH={RUNTIME};/vendor/lib/rfsa/adsp;/dsp"]
    cache: dict[Path, str] = {}

    def ship(path: Path) -> str:
        if path not in cache:
            h = sha(path)
            remote = f"{temp}/{h[:18]}{path.suffix}"
            adb("push", path, remote)
            cache[path] = remote
        return cache[path]

    result = {
        "kind": kind,
        "context_bytes": b["context_bytes"],
        "context_mb": b["context_mb"],
        "ddr": b["ddr"],
        "runs": {},
        "fidelity": {},
    }
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
                print(f"[s24u] Running {kind} {tag} (N={n})...", flush=True)
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
                        rec[key.replace("_us", "_ms")] = round(float(m[1]) / 1e3, 3)
                result["runs"][tag] = rec
                if phase == "panel":
                    d_errs, g_max_diffs, g_mean_diffs = [], [], []
                    n_nan = 0
                    for i, r in enumerate(rows):
                        d_raw = local / f"{mode}_{r['id']}_depth.raw"
                        g_raw = local / f"{mode}_{r['id']}_glass.raw"
                        adb("pull", f"{dest}/Result_{i}/depth_m.raw", d_raw)
                        adb("pull", f"{dest}/Result_{i}/glass_prob_s16.raw", g_raw)
                        z = np.fromfile(d_raw, "<f4")
                        gp = np.fromfile(g_raw, "<f4")
                        if not (np.isfinite(z).all() and np.isfinite(gp).all()):
                            n_nan += 1
                        d_ref = np.load(ref_dir / f"{mode}_{r['id']}_depth.npy")
                        g_ref = np.load(ref_dir / f"{mode}_{r['id']}_glass.npy")
                        d_errs.append(float(100.0 * np.mean(np.abs(z.astype(float) / d_ref - 1.0))))
                        gd = np.abs(gp.astype(float) - g_ref.astype(float))
                        g_max_diffs.append(float(np.max(gd)))
                        g_mean_diffs.append(float(np.mean(gd)))
                    by_dom = {
                        d: round(float(np.mean([v for r, v in zip(rows, d_errs) if r["domain"] == d])), 4)
                        for d in sorted({r["domain"] for r in rows})
                    }
                    result["fidelity"][mode] = {
                        "nan_or_inf_frames": n_nan,
                        "depth_mean_err_pct": round(float(np.mean(d_errs)), 4),
                        "depth_by_domain_pct": by_dom,
                        "glass_prob_mean_abs_diff": float(f"{float(np.mean(g_mean_diffs)):.4e}"),
                        "glass_prob_max_abs_diff": float(f"{float(np.max(g_max_diffs)):.4e}"),
                    }
        if kind == "omni":
            rgb_dir = WORK / "pure_rgb/device"
            if (rgb_dir / "rgb_aria_0_depth.raw").exists():
                zi_d = max(
                    float(np.max(np.abs(np.fromfile(rgb_dir / f"rgb_{r['id']}_depth.raw", "<f4")
                                        - np.fromfile(local / f"zero_{r['id']}_depth.raw", "<f4"))))
                    for r in rows
                )
                zi_g = max(
                    float(np.max(np.abs(np.fromfile(rgb_dir / f"rgb_{r['id']}_glass.raw", "<f4")
                                        - np.fromfile(local / f"zero_{r['id']}_glass.raw", "<f4"))))
                    for r in rows
                )
                result["fidelity"]["no_tof_max_abs_depth_vs_pure_rgb_m"] = round(zi_d, 6)
                result["fidelity"]["no_tof_max_abs_glass_vs_pure_rgb"] = round(zi_g, 6)
    finally:
        shell("rm", "-rf", temp)
    dump(local / "summary.json", result)
    return result


def main() -> None:
    t0 = time.time()
    rows = json.loads((PANEL / "frames.json").read_text(encoding="utf-8"))
    summary = {
        "meta": {
            "target_soc": "SM-S9280 / Snapdragon 8 Gen 3 / Hexagon v75 (QAIRT 2.47)",
            "profile": "pc_fp16dec_a16 + frozen G2 glass_head (0.411M)",
            "baseline_v41_no_g2": {
                "pure_rgb_mb": 57.496,
                "pure_rgb_htp_ms": 14.28,
                "omni_mb": 59.105,
                "omni_htp_ms": 17.04,
            },
        },
        "results": {},
    }
    for kind in ("pure_rgb", "omni"):
        onnx_p, _ = export_onnx_and_host_refs(kind, rows)
        build_pc_fp16dec_a16(kind, onnx_p)
        res = run_on_s24u(kind, rows)
        summary["results"][kind] = res
    summary["meta"]["total_elapsed_s"] = round(time.time() - t0, 2)
    out_json = PROBE_DIR / "s24u_v41_g2_silicon_report.json"
    dump(out_json, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\n[done] Saved {out_json} in {summary['meta']['total_elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
