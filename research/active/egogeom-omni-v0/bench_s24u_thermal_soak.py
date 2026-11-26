"""Sustained live S24 Ultra (SM-S9280 / Hexagon v75) thermal, current & latency soak test.

Directly compares:
  - v4.1 FP16 Omni+SignedH3          (~109.4 MB)
  - v4.1 pc_fp16dec_a16 Omni+SignedH3 (~59.1 MB, Track A 2x-compressed zero-interference champion)
over matched continuous NPU load (6 waves x 300 inferences = 1,800 continuous inferences per arm),
logging Hexagon HMX/HVX die temps, DDR temp, CPU temp, battery temp/current, and wave-by-wave HTP latency drift.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROUTE = Path(__file__).resolve().parent
M13 = ROUTE / "math_theory/m13_fisher_so_c_w8a8_silicon"
sys.path[:0] = [str(M13)]

from eval_empirical import IO, OUT as M13_OUT, PANEL, dump, sha
from device_panel import inputs as panel_inputs

ASSETS = ROUTE.parents[2] / "artifacts.local/work/egogeom-omni-v0"
RUNTIME = "/data/local/tmp/ba_qairt_htp_r0"

ARMS = {
    "v41_fp16_omni": {
        "binary": ASSETS / "qnn_v75_v41_final/egogeom_omni_v41_omni_v75_fp16.bin",
        "graph_name": "egogeom_omni_v41_best_omni_fp32",
        "fp16_relaxed": True,
    },
    "v41_pc_fp16dec_a16_omni": {
        "binary": M13_OUT / "perchannel_probe/pc_fp16dec_a16_omni/graph.bin",
        "graph_name": "base",
        "fp16_relaxed": False,
    },
}


def get_adb():
    exe = shutil.which("adb.exe") or shutil.which("adb")
    devs = subprocess.check_output([exe, "devices"], text=True, timeout=30)
    online = [s.split()[0] for s in devs.splitlines() if s.endswith("\tdevice")]
    assert len(online) == 1, f"Expected 1 online ADB device, found {online}"
    serial = online[0]

    def adb(*argv):
        for attempt in range(4):
            p = subprocess.run([exe, "-s", serial, *map(str, argv)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
            diag = p.stdout + p.stderr
            if p.returncode == 0 or not any(x in diag for x in ("cannot connect to daemon", "failed to start daemon")):
                break
            time.sleep(1.5 + attempt)
        if p.returncode:
            raise RuntimeError((p.stderr + p.stdout)[-2000:])
        return p.stdout.strip()

    def shell(cmd: str) -> str:
        return adb("shell", cmd)

    return adb, shell, serial


def sample_telemetry(shell) -> dict:
    types_raw = shell("head -n 1 /sys/class/thermal/thermal_zone*/type /sys/class/thermal/thermal_zone*/temp")
    # Simpler & rock-solid: single python-parsed shell loop without pipes or case
    raw_tz = shell("for z in /sys/class/thermal/thermal_zone*; do echo TZ $z $(cat $z/type) $(cat $z/temp); done")
    raw_bat = shell("dumpsys battery")
    hmx, hvx, cpu = [], [], []
    ddr = None
    batt = {}
    for line in raw_tz.splitlines():
        parts = line.strip().split()
        if len(parts) >= 4 and parts[0] == "TZ":
            k, v = parts[2], parts[3]
            try:
                c = float(v) / 1000.0
            except ValueError:
                continue
            if k.startswith("nsphmx-"):
                hmx.append(c)
            elif k.startswith("nsphvx-"):
                hvx.append(c)
            elif k.startswith("cpuss-"):
                cpu.append(c)
            elif k == "ddr":
                ddr = c
    for line in raw_bat.splitlines():
        s = line.strip()
        if ":" in s and not s.startswith("11-") and not s.startswith("12-"):
            k, v = s.split(":", 1)
            kl = k.strip().lower().replace(" ", "_")
            if kl in ("temperature", "voltage", "current_now", "level"):
                try:
                    val = float(v.strip())
                    if kl == "temperature":
                        batt["battery_c"] = round(val / 10.0, 2)
                    elif kl == "voltage":
                        batt["voltage_mv"] = int(val)
                    elif kl == "current_now":
                        batt["current_ma"] = int(val)
                    elif kl == "level":
                        batt["level_pct"] = int(val)
                except ValueError:
                    pass
    return {
        "nsphmx_max_c": round(max(hmx), 2) if hmx else None,
        "nsphvx_max_c": round(max(hvx), 2) if hvx else None,
        "ddr_c": round(ddr, 2) if ddr is not None else None,
        "cpuss_max_c": round(max(cpu), 2) if cpu else None,
        **batt,
    }


def wait_cooldown(shell, target_hmx_c: float = 36.0, max_wait_s: int = 45) -> dict:
    t0 = time.time()
    tel = sample_telemetry(shell)
    while (tel["nsphmx_max_c"] or 0.0) > target_hmx_c and (time.time() - t0) < max_wait_s:
        time.sleep(5.0)
        tel = sample_telemetry(shell)
    return tel


def run_arm_soak(adb, shell, arm_name: str, spec: dict, waves: int, iters_per_wave: int) -> dict:
    binary: Path = spec["binary"]
    assert binary.is_file(), f"Missing {binary}"
    local_dir = ASSETS / "s24u_thermal_soak" / arm_name
    local_dir.mkdir(parents=True, exist_ok=True)
    temp = shell("mktemp -d /data/local/tmp/s24u_soak_XXXXXXXX")
    env_prefix = f"LD_LIBRARY_PATH={RUNTIME} ADSP_LIBRARY_PATH='{RUNTIME};/vendor/lib/rfsa/adsp;/dsp'"

    try:
        # Resolve exact graph name from context binary if needed
        cfg = json.loads((IO / "htp_omni.json").read_text(encoding="utf-8"))
        cfg["graphs"][0]["graph_names"] = [spec["graph_name"]]
        if spec["fp16_relaxed"]:
            cfg["graphs"][0]["fp16_relaxed_precision"] = 1
        dump(local_dir / "htp.json", cfg)
        adb("push", local_dir / "htp.json", f"{temp}/htp.json")
        dump(
            local_dir / "ext.json",
            {"backend_extensions": {
                "shared_library_path": f"{RUNTIME}/libQnnHtpNetRunExtensions.so",
                "config_file_path": f"{temp}/htp.json",
            }},
        )
        adb("push", local_dir / "ext.json", f"{temp}/ext.json")
        adb("push", binary, f"{temp}/graph.bin")

        rows = json.loads((PANEL / "frames.json").read_text(encoding="utf-8"))
        r0 = rows[0]
        parts = []
        for k, p in panel_inputs("omni", r0["id"], "signed").items():
            remote = f"{temp}/in_{k}.raw"
            adb("push", p, remote)
            parts.append(f"{k}:={remote}")
        listing = local_dir / "in.txt"
        listing.write_text(" ".join(parts) + "\n", encoding="utf-8")
        adb("push", listing, f"{temp}/in.txt")

        # Cooldown before starting wave 0
        t_base = wait_cooldown(shell, target_hmx_c=36.0, max_wait_s=45)
        print(f"[{arm_name}] baseline telemetry: {t_base}", flush=True)

        wave_logs = []
        t_soak_start = time.time()
        for w in range(waves):
            dest = f"{temp}/w{w}"
            t_w0 = time.time()
            shell(
                f"{env_prefix} {RUNTIME}/qnn-net-run "
                f"--backend {RUNTIME}/libQnnHtp.so "
                f"--config_file {temp}/ext.json "
                f"--retrieve_context {temp}/graph.bin "
                f"--input_list {temp}/in.txt "
                f"--output_dir {dest} "
                f"--num_inferences {iters_per_wave} "
                f"--keep_num_outputs 1 "
                f"--perf_profile burst "
                f"--profiling_level basic >/dev/null 2>&1"
            )
            wall_s = time.time() - t_w0
            tel = sample_telemetry(shell)
            prof = shell(f"{env_prefix} {RUNTIME}/qnn-profile-viewer --input_log {dest}/qnn-profiling-data.log")
            tail = prof.split("Execute Stats (Average):")[-1]
            htp_m = re.search(r"Backend \(Accelerator \(execute\) time\):\s*([\d.]+) us", tail)
            nr_m = re.search(r"NetRun:\s*([\d.]+) us", tail)
            htp_ms = round(float(htp_m[1]) / 1000.0, 3) if htp_m else None
            nr_ms = round(float(nr_m[1]) / 1000.0, 3) if nr_m else None
            rec = {
                "wave": w + 1,
                "elapsed_s": round(time.time() - t_soak_start, 2),
                "wall_s": round(wall_s, 2),
                "iters": iters_per_wave,
                "htp_ms": htp_ms,
                "netrun_ms": nr_ms,
                **tel,
            }
            wave_logs.append(rec)
            print(
                f"  [{arm_name} w{w+1}/{waves}] HTP={htp_ms}ms NetRun={nr_ms}ms "
                f"HMX={tel['nsphmx_max_c']}C HVX={tel['nsphvx_max_c']}C DDR={tel['ddr_c']}C Batt={tel['battery_c']}C",
                flush=True,
            )

        htps = [x["htp_ms"] for x in wave_logs if x["htp_ms"] is not None]
        nrs = [x["netrun_ms"] for x in wave_logs if x["netrun_ms"] is not None]
        summary = {
            "arm": arm_name,
            "binary_mb": round(binary.stat().st_size / (1024 * 1024), 2),
            "total_inferences": waves * iters_per_wave,
            "total_soak_s": round(time.time() - t_soak_start, 2),
            "baseline": t_base,
            "peak_nsphmx_c": max(x["nsphmx_max_c"] for x in wave_logs if x["nsphmx_max_c"] is not None),
            "peak_nsphvx_c": max(x["nsphvx_max_c"] for x in wave_logs if x["nsphvx_max_c"] is not None),
            "peak_ddr_c": max(x["ddr_c"] for x in wave_logs if x["ddr_c"] is not None),
            "peak_battery_c": max(x["battery_c"] for x in wave_logs if x.get("battery_c") is not None),
            "delta_nsphmx_c": round(
                max(x["nsphmx_max_c"] for x in wave_logs if x["nsphmx_max_c"] is not None) - (t_base["nsphmx_max_c"] or 0.0), 2
            ),
            "delta_ddr_c": round(
                max(x["ddr_c"] for x in wave_logs if x["ddr_c"] is not None) - (t_base["ddr_c"] or 0.0), 2
            ),
            "delta_battery_c": round(
                max(x["battery_c"] for x in wave_logs if x.get("battery_c") is not None) - (t_base.get("battery_c") or 0.0), 2
            ),
            "htp_ms_wave1": htps[0],
            "htp_ms_last": htps[-1],
            "htp_ms_mean": round(sum(htps) / len(htps), 3),
            "htp_ms_drift_pct": round(100.0 * (htps[-1] - htps[0]) / max(htps[0], 1e-6), 2),
            "netrun_ms_mean": round(sum(nrs) / len(nrs), 3),
            "sustained_htp_fps": round(1000.0 / (sum(htps) / len(htps)), 2),
            "waves": wave_logs,
        }
        return summary
    finally:
        shell(f"rm -rf {shlex.quote(temp)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", type=int, default=6)
    ap.add_argument("--iters-per-wave", type=int, default=300)
    args = ap.parse_args()

    adb, shell, serial = get_adb()
    model = shell("getprop ro.product.model")
    assert model == "SM-S9280", f"Unexpected model {model}"
    print(f"Live S24 Ultra ({serial} / {model}) sustained thermal soak: {args.waves} waves x {args.iters_per_wave} iters/arm", flush=True)

    # Check exact graph name for v41_fp16_omni from info or default
    fp16_info_p = ASSETS / "qnn_v75_v41_final/s24u_v41_final_npu_report.json"
    if fp16_info_p.exists():
        pass  # use qnn-context-binary-utility to read exact graph name safely
    from eval_empirical import BINDIR, execute
    tmp_info = ASSETS / "s24u_thermal_soak/fp16_info"
    tmp_info.mkdir(parents=True, exist_ok=True)
    execute(
        [BINDIR / "qnn-context-binary-utility.exe",
         "--context_binary", ARMS["v41_fp16_omni"]["binary"],
         "--json_file", tmp_info / "info.json",
         "--unified_qairt_format"],
        tmp_info, "info",
    )
    info = json.loads((tmp_info / "info.json").read_text(encoding="utf-8"))
    ARMS["v41_fp16_omni"]["graph_name"] = info["graphs"][0]["graphName"]

    out = {"device_model": model, "serial": serial, "arms": {}}
    for arm_name, spec in ARMS.items():
        print(f"\n================ SOAK: {arm_name} ================", flush=True)
        out["arms"][arm_name] = run_arm_soak(adb, shell, arm_name, spec, args.waves, args.iters_per_wave)

    dst = ASSETS / "s24u_thermal_soak/thermal_soak_fp16_vs_pc_fp16dec_a16.json"
    dump(dst, out)
    print(f"\nSaved -> {dst}", flush=True)


if __name__ == "__main__":
    main()
