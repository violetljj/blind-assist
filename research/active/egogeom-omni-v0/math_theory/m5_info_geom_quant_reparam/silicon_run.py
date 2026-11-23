"""Run paired frozen/candidate HTP contexts on a development handset with the project runtime."""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time

from quant_head import ASSETS, OUT

IO = ASSETS / "qnn_v75_v4_final"
WORK = OUT / "silicon_followup"
RUNTIME = "/data/local/tmp/ba_qairt_htp_r0"
CONTEXTS = {
    "rgb_base": IO / "egogeom_omni_v4_pure_rgb_v75_fp16.bin",
    "rgb_codec": OUT / "m5_v4_pure_rgb_v75.bin",
    "omni_base": IO / "egogeom_omni_v4_omni_v75_fp16.bin",
    "omni_codec": OUT / "m5_v4_omni_v75.bin",
    "matched_taps": WORK / "probe/matched_taps/v4_pure_rgb_final_v75.bin",
    "matched_control": WORK / "probe/matched_control/v4_pure_rgb_final_v75.bin",
    "byte": WORK / "byte/head_byte_v75.bin",
    "byte_both": WORK / "byte_both/head_byte_both_v75.bin",
    "omni_both": WORK / "factorial/omni_both/v4_omni_final_v75.bin",
    **{arm: WORK / "probe" / arm / "v4_pure_rgb_final_v75.bin"
       for arm in ("matched_codec", "matched_codec_r4", "matched_softplus")},
    **{arm: WORK / "factorial" / arm / "v4_pure_rgb_final_v75.bin" for arm in ("focal", "both")},
}


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def case_rows(mode, panel):
    if mode == "panel":
        return [{k: panel / row["id"] / f"{k}.raw" for k in ("rgb","grav_rays")}
                for row in json.loads((panel / "frames.json").read_text())]
    row = {"rgb": IO / "rgb.raw", "grav_rays": IO / "grav_rays.raw"}
    if mode != "rgb":
        row["tof_grid_s4"] = IO / f"tof_{mode}.raw"
        row["has_tof"] = IO / ("ht_zero.raw" if mode == "zero" else "ht_one.raw")
    return [row]


def design(task):
    if task == "transfer":
        return [(arm, mode, 50) for mode in ("zero", "cnh", "dep")
                for arm in ("omni_base", "omni_both", "omni_both", "omni_base")]
    if task == "matched_codecs":
        return [(arm, "rgb", 50) for arm in ("rgb_base", "matched_codec", "matched_codec_r4")]
    if task == "factor":
        return [(arm, "rgb", 50) for arm in ("rgb_base", "matched_softplus", "focal", "both",
                                             "both", "focal", "matched_softplus", "rgb_base")]
    if task == "factor_panel":
        return [(arm, "panel", 0) for arm in ("matched_softplus", "focal", "both")]
    if task == "probe":
        return [(arm,"rgb",1) for arm in ("rgb_base","matched_control","matched_taps")]
    if task == "byte":
        return [(arm,"rgb",50) for arm in ("rgb_base","byte","byte","rgb_base")]
    if task == "byte_panel":
        return [("byte","panel",0)]
    if task == "corrected_byte":
        return [(arm,"rgb",50) for arm in ("both","byte_both","byte_both","both")]
    if task == "corrected_byte_panel":
        return [("byte_both","panel",0)]
    if task == "panel":
        return [("rgb_base", "panel", 0), ("rgb_codec", "panel", 0)]
    blocks = []
    for prefix, mode in (("rgb", "rgb"), ("omni", "cnh")):
        blocks.extend((prefix + "_" + arm, mode, 50) for arm in ("base", "codec", "codec", "base"))
    for mode in ("zero", "dep"):
        blocks.extend(("omni_" + arm, mode, 1) for arm in ("base", "codec"))
    return blocks


def run(args):
    exe = args.adb or shutil.which("adb.exe") or shutil.which("adb")
    if exe is None:
        raise FileNotFoundError("Set --adb to Android platform-tools/adb")
    def adb(*command, limit=180):
        for retry in range(3):
            cp = subprocess.run([exe, "-s", args.serial, *map(str, command)], capture_output=True,
                                encoding="utf-8", errors="replace", timeout=limit, check=False)
            connection_failed = any(s in cp.stderr + cp.stdout for s in
                                    ("failed to start daemon", "cannot connect to daemon", "cannot connect to 127.0.0.1"))
            if cp.returncode == 0 or not connection_failed:
                break
            time.sleep(retry+1)
        if cp.returncode:
            raise RuntimeError(f"adb {command[0]} exited {cp.returncode}: {(cp.stdout+cp.stderr)[-1600:]}")
        return (cp.stdout + cp.stderr).strip()
    def shell(*command):
        return adb("shell", shlex.join(command))
    out = WORK / args.run
    out.mkdir(exist_ok=False, parents=True)
    signature = shell("getprop", "ro.product.model")
    root = shell("mktemp", "-d", "/data/local/tmp/egogeom_m5_XXXXXXXX")
    if not re.fullmatch(r"/data/local/tmp/egogeom_m5_[a-zA-Z0-9]{8}", root):
        raise RuntimeError("Unexpected task-owned phone directory")
    env = ["env", f"LD_LIBRARY_PATH={RUNTIME}", f"ADSP_LIBRARY_PATH={RUNTIME};/vendor/lib/rfsa/adsp;/dsp"]
    memo = {}
    def ship(p):
        if p not in memo:
            h = digest(p)
            target = root + "/" + h[:12] + "_" + p.name
            adb("push", p, target)
            if shell("sha256sum", target).split()[0] != h:
                raise OSError(f"Device input hash mismatch: {p.name}")
            memo[p] = target, h
        return memo[p][0]
    results = {"model": signature, "serial": args.serial, "task": args.task,
               "runtime": RUNTIME, "temp_dir": root, "backend_extensions":not args.without_extensions, "blocks": []}
    try:
        results["version"] = shell(*env, RUNTIME + "/qnn-net-run", "--version").splitlines()[-1]
        for i, (arm, mode, repeats) in enumerate(design(args.task)):
            if i < args.start:
                continue
            binary = CONTEXTS[arm]
            rows = case_rows(mode, WORK / "panel_inputs")
            if arm in ("focal", "both", "omni_both", "byte_both"):
                for row in rows:
                    tag = row["rgb"].parent.name if mode == "panel" else "archived"
                    row["host_focal"] = WORK / "focal_inputs" / (tag + ".raw")
            block = out / f"{i:02}_{arm}_{mode}"
            block.mkdir()
            lines = [" ".join(f"{k}:={ship(p)}" for k,p in row.items()) for row in rows]
            input_list = block / "inputs.txt"
            input_list.write_text("\n".join(lines)+"\n", encoding="ascii")
            remote_out = f"{root}/out_{i}"
            options = []
            if not args.without_extensions:
                graph_name = {"omni_base":"v4_omni_final", "omni_both":"v4_omni_final",
                              "rgb_codec":"m5_v4_pure_rgb","omni_codec":"m5_v4_omni",
                              "byte":"head_byte", "byte_both":"head_byte_both"}.get(arm,"v4_pure_rgb_final")
                config = json.loads((IO/"htp_rgb.json").read_text())
                config["graphs"][0]["graph_names"] = [graph_name]
                htp = block/"htp.json"
                htp.write_text(json.dumps(config))
                ext = block/"ext.json"
                ext.write_text(json.dumps({"backend_extensions":{
                    "shared_library_path":RUNTIME+"/libQnnHtpNetRunExtensions.so","config_file_path":ship(htp)}}))
                options.extend(["--config_file",ship(ext)])
            if arm in ("byte","byte_both"):
                options.append("--use_native_output_files")
            log = shell(*env, RUNTIME+"/qnn-net-run", "--backend", RUNTIME+"/libQnnHtp.so",
                        "--retrieve_context", ship(binary), "--input_list", ship(input_list),
                        "--output_dir", remote_out, "--num_inferences", str(repeats or len(rows)),
                        "--keep_num_outputs", str(len(rows)), "--profiling_level", "basic",
                        "--perf_profile", "burst", *options)
            (block/"netrun.log").write_text(log,encoding="utf-8")
            profile = shell(*env, RUNTIME+"/qnn-profile-viewer", "--input_log", remote_out+"/qnn-profiling-data.log")
            (block/"profile.txt").write_text(profile,encoding="utf-8")
            for j in range(len(rows)):
                adb("pull", remote_out + f"/Result_{j}", block / f"Result_{j}")
            match = re.search(r"Execute Stats \(Average\):(.*?)(?:Execute Stats|$)", profile,re.DOTALL)
            if match is None:
                raise ValueError(f"Missing device profile in {block.name}")
            stats = {}
            for key, pattern in [("htp_us",r"Backend \(Accelerator \(execute\) time\):\s*([\d.]+) us"),
                                 ("netrun_us",r"NetRun:\s*([\d.]+) us")]:
                hit = re.search(pattern, match[1])
                if hit:
                    stats[key] = float(hit[1])
            results["blocks"].append({"block":block.name,"arm":arm,"mode":mode,"inferences":repeats or len(rows),
                                      "saved":len(rows),"context_sha256":memo[binary][1],**stats})
            print(block.name,stats,flush=True)
    finally:
        shell("rm","-r",root)
        results["device_temp_released"] = adb("shell", f"test ! -e {root} && echo released") == "released"
        results["file_hashes"] = {target.split("/")[-1]: h for target,h in memo.values()}
        (out/"receipt.json").write_text(json.dumps(results,indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", required=True)
    ap.add_argument("--task", choices=("archive","probe","panel","byte","byte_panel","matched_codecs",
                                       "factor","factor_panel","transfer","corrected_byte","corrected_byte_panel"), required=True)
    ap.add_argument("--without-extensions",action="store_true")
    ap.add_argument("--start",type=int,default=0,help="Resume a failed design at this block in a new receipt")
    ap.add_argument("--run", required=True, help="New evidence subdirectory")
    ap.add_argument("--adb", default=os.environ.get("ADB"))
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        ap.error("Use a simple run name")
    run(args)
