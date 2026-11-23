"""Exhaust the representable disparity input codes of the frozen RGB reciprocal tail on HTP."""

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
from eval_on_v4 import qairt_convert, session
from onnx import TensorProto, helper, numpy_helper
from quant_head import HI, LO
from silicon_probe import FULL, NATIVE, ORIGINAL_IO, TAPS
from silicon_run import RUNTIME, WORK, digest

DIR = WORK / "terminal_sweep"
MODEL = DIR / "terminal.onnx"
BINARY = DIR / "terminal_v75.bin"
FIRST, LAST = [int(np.float32(v).view(np.uint32)) for v in (1 / HI, 1 / LO)]
SIZE = int(np.prod(FULL))
TOTAL = LAST - FIRST + 1
NMAP = (TOTAL + SIZE - 1) // SIZE
TAP_DIR = WORK / "matched_probes/02_matched_taps_rgb/Result_0"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def export_tail():
    original = onnx.load(NATIVE)
    by = {v: n for n in original.graph.node for v in n.output}
    clip = by["depth_m"]
    mul = by[clip.input[0]]
    inv = by[mul.input[0]]
    source = inv.input[0]
    assert source == TAPS["tap_preinv"][0] and [
        n.op_type for n in (inv, mul, clip)
    ] == ["Reciprocal", "Mul", "Clip"]
    const = lambda name: float(numpy_helper.to_array(by[name].attribute[0].t).item())
    assert const(by[source].input[1]) == np.float32(0.04) and not by[source].input[2]
    assert const(by[TAPS["tap_scaled"][0]].input[2]) == np.float32(1 / 0.15)
    assert [const(mul.input[1]), const(clip.input[1]), const(clip.input[2])] == [
        1,
        np.float32(0.15),
        25,
    ]
    nodes = [by[mul.input[1]], by[clip.input[1]], by[clip.input[2]], inv, mul, clip]
    # Only rename the graph input to a safe QNN CLI name. Its producer (Clip_2)
    # belongs to the removed prefix; legal inputs are exactly its bounded domain.
    inv.input[0] = "preinv"
    graph = helper.make_graph(
        nodes,
        MODEL.stem,
        [helper.make_tensor_value_info("preinv", TensorProto.FLOAT, FULL)],
        [helper.make_tensor_value_info("depth_m", TensorProto.FLOAT, FULL)],
    )
    model = helper.make_model(
        graph, opset_imports=original.opset_import, ir_version=original.ir_version
    )
    onnx.checker.check_model(model)
    onnx.save(model, MODEL)
    d = np.linspace(
        np.float32(0.04), np.float32(1 / 0.15), SIZE, dtype=np.float32
    ).reshape(FULL)
    cpu = session(MODEL).run(None, {"preinv": d})[0]
    assert np.array_equal(cpu, np.clip(1 / d, np.float32(0.15), np.float32(25)))
    return {
        "original_onnx_sha256": digest(NATIVE),
        "terminal_input_source": source,
        "nodes": [n.name for n in nodes if n.op_type != "Constant"],
        "cut_sha256": digest(MODEL),
    }


def layer_table(sdk, python, source, tag):
    lib = sdk / "lib/x86_64-windows-msvc"
    bindir = sdk / "bin/x86_64-windows-msvc"
    env = dict(
        os.environ,
        QAIRT_SDK_ROOT=str(sdk),
        QNN_SDK_ROOT=str(sdk),
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(sdk / "lib/python"),
        PATH=os.pathsep.join(map(str, (lib, bindir, os.environ["PATH"]))),
    )
    path = DIR / f"{tag}_layers.csv"
    with (DIR / f"{tag}_layers.log").open("wb") as f:
        subprocess.run(
            [
                str(python),
                str(bindir / "qairt-dlc-info"),
                "-i",
                str(source),
                "-s",
                str(path),
            ],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f) if len(r) > 6 and r[0].isnumeric()]
    if tag == "full":
        rows = [
            r
            for r in rows
            if any(
                word in r[1]
                for word in ("Reciprocal", "/core/Mul_3", "/core/Clip_3", "depth_m")
            )
        ]
    if not rows:
        raise ValueError(f"No tail DLC layer records: {tag}")
    return [{"name": r[1], "type": r[2], "inputs": r[3], "outputs": r[4]} for r in rows]


def disp_codes(i):
    begin = FIRST + i * SIZE
    return np.arange(begin, min(begin + SIZE, LAST + 1), dtype="<u4").view("<f4")


def prepare(sdk, python):
    if TOTAL != 61_963_852 or SIZE != 110_592:
        raise ValueError("Unexpected domain")
    DIR.mkdir(parents=True, exist_ok=True)
    evidence = export_tail()
    args = ["--float_bitwidth", "16", "--preserve_io_datatype"]
    for kind, key in (("input", "preinv"), ("output", "depth_m")):
        for prefix in ("source_model", "desired"):
            args.extend([f"--{prefix}_{kind}_layout", key, "NCHW"])
    evidence["build"] = qairt_convert(sdk, python, MODEL, extra_args=args)
    if not BINARY.is_file() or evidence["build"]["htp_v75_build"]["exit_code"] != 0:
        raise ValueError(evidence["build"])
    info = json.loads(BINARY.with_suffix(".info.json").read_text())["graphs"][0]
    for key in ("graphInputs", "graphOutputs"):
        (t,) = info[key]
        assert (
            t["dataType"] == "QNN_DATATYPE_FLOAT_32" and tuple(t["dimensions"]) == FULL
        )
    evidence.update(
        {
            "context_sha256": digest(BINARY),
            "cut_layers": layer_table(sdk, python, MODEL.with_suffix(".dlc"), "cut"),
            "full_layers": layer_table(
                sdk, python, ORIGINAL_IO / "v4_pure_rgb_final.dlc", "full"
            ),
        }
    )
    payload = DIR / "inputs"
    payload.mkdir(exist_ok=True)
    index = []
    for i in range(NMAP):
        d = disp_codes(i)
        if len(d) < SIZE:
            d = np.pad(d, (0, SIZE - len(d)), mode="edge")
        path = payload / f"codes_{i:03d}.raw"
        d.tofile(path)
        index.append(digest(path))
    shutil.copyfile(TAP_DIR / "tap_preinv.raw", payload / "bridge.raw")
    evidence.update(
        {
            "first_bits": hex(FIRST),
            "last_bits": hex(LAST),
            "n_positive_fp32_codes": TOTAL,
            "maps": NMAP,
            "nchwdims": FULL,
            "padded_last_slots": NMAP * SIZE - TOTAL,
            "ordered_input_sha256": index,
            "bridge_sha256": digest(payload / "bridge.raw"),
        }
    )
    write_json(DIR / "prepare.json", evidence)
    return evidence


def run_device(adb, serial):
    prep = json.loads((DIR / "prepare.json").read_text())
    assert digest(BINARY) == prep["context_sha256"]

    def call(*args):
        p = subprocess.run(
            [str(adb), "-s", serial, *map(str, args)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=500,
            check=False,
        )
        if p.returncode:
            raise RuntimeError((p.stderr + p.stdout)[-1500:])
        return (p.stdout + p.stderr).strip()

    def shell(*args):
        return call("shell", shlex.join(args))

    result = {
        "model": shell("getprop", "ro.product.model"),
        "serial": serial,
        "context_sha256": digest(BINARY),
    }
    env = [
        "env",
        f"LD_LIBRARY_PATH={RUNTIME}",
        f"ADSP_LIBRARY_PATH={RUNTIME};/vendor/lib/rfsa/adsp;/dsp",
    ]
    root = shell("mktemp", "-d", "/data/local/tmp/egogeom_recip_XXXXXXXX")
    if not re.fullmatch(r"/data/local/tmp/egogeom_recip_[a-zA-Z0-9]{8}", root):
        raise ValueError(root)
    try:
        folders = DIR / "device_outputs"
        if folders.exists():
            raise FileExistsError(folders)
        for i in range(NMAP):
            if (
                digest(DIR / "inputs" / f"codes_{i:03d}.raw")
                != prep["ordered_input_sha256"][i]
            ):
                raise ValueError("Input chunk changed")
        assert digest(DIR / "inputs/bridge.raw") == prep["bridge_sha256"]
        files = [f"codes_{i:03d}.raw" for i in range(NMAP)] + ["bridge.raw"]
        listing = DIR / "inputs.txt"
        listing.write_text(
            "".join("preinv:=" + root + "/inputs/" + name + "\n" for name in files),
            encoding="ascii",
        )
        ext = DIR / "device.ext.json"
        write_json(
            ext,
            {
                "backend_extensions": {
                    "shared_library_path": RUNTIME + "/libQnnHtpNetRunExtensions.so",
                    "config_file_path": root + "/terminal.htp.json",
                }
            },
        )
        call("push", DIR / "inputs", root + "/inputs")
        for path in (BINARY, listing, ext, DIR / "terminal.htp.json"):
            call("push", path, root + "/" + path.name)
        result["version"] = shell(
            *env, RUNTIME + "/qnn-net-run", "--version"
        ).splitlines()[-1]
        log = shell(
            *env,
            RUNTIME + "/qnn-net-run",
            "--backend",
            RUNTIME + "/libQnnHtp.so",
            "--retrieve_context",
            root + "/" + BINARY.name,
            "--input_list",
            root + "/" + listing.name,
            "--output_dir",
            root + "/outputs",
            "--config_file",
            root + "/" + ext.name,
            "--num_inferences",
            str(len(files)),
            "--keep_num_outputs",
            str(len(files)),
            "--profiling_level",
            "basic",
            "--perf_profile",
            "burst",
        )
        (DIR / "netrun.log").write_text(log, encoding="utf-8")
        call("pull", root + "/outputs", folders)
        result["saved_maps"] = len(list(folders.glob("Result_*/depth_m.raw")))
        assert result["saved_maps"] == len(files)
        profile = shell(
            *env,
            RUNTIME + "/qnn-profile-viewer",
            "--input_log",
            root + "/outputs/qnn-profiling-data.log",
        )
        (DIR / "profile.txt").write_text(profile, encoding="utf-8")
        result["complete"] = True
    finally:
        shell("rm", "-r", root)
        result.update(
            {
                "temp_released": call("shell", f"test ! -e {root} && echo released")
                == "released"
            }
        )
        write_json(DIR / "device.json", result)
        if not result["temp_released"]:
            raise RuntimeError("Device temp cleanup failed")


def stats(z, d):
    ref = np.clip(1 / d.astype("f8"), LO, HI)
    loss = np.abs(z.astype("f8") / ref - 1)
    i = int(np.argmax(loss))
    return {
        "pixels": len(d),
        "mean_pct": float(100 * loss.mean()),
        "max_pct": float(100 * loss[i]),
        "at_or_over_0_15pct": int((loss >= 0.0015).sum()),
        "index": i,
        "d_at_max": float(d[i]),
        "device_z_at_max": float(z[i]),
        "reference_z_at_max": float(ref[i]),
    }


def analyze():
    base = np.fromfile(
        WORK / "matched_probes/00_rgb_base_rgb/Result_0/depth_m.raw", "<f4"
    )
    fitted = np.fromfile(TAP_DIR / "depth_m.raw", "<f4")
    tap = np.fromfile(TAP_DIR / "tap_preinv.raw", "<f4")
    bridge_roi = json.loads((WORK / "matched_probes/receipt.json").read_text())
    assert len(tap) == len(fitted) == SIZE and np.array_equal(base, fitted)
    assert all(b["mode"] == "rgb" and b["saved"] == 1 for b in bridge_roi["blocks"])
    bridge = {
        "tap_sha256": digest(TAP_DIR / "tap_preinv.raw"),
        "taps_depth_sha256": digest(TAP_DIR / "depth_m.raw"),
        "unmodified_baseline_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
        "full_network_tail_error": stats(fitted, tap),
    }
    dirs = DIR / "device_outputs"
    independent = np.fromfile(dirs / f"Result_{NMAP}/depth_m.raw", "<f4")
    if len(independent) != SIZE or not np.isfinite(independent).all():
        raise ValueError("Bad tap replay")
    bridge["standalone_tail_error_on_same_taps"] = stats(independent, tap)
    bridge["standalone_identical_to_full_depth_pixels"] = int(
        (independent == fitted).sum()
    )
    bridge["max_abs_m_separate_vs_full"] = float(np.abs(independent - fitted).max())
    write_json(DIR / "bridge.json", bridge)
    total_loss = 0.0
    violations = 0
    transitions = 0
    upwards = 0
    prev = None
    per = []
    worst = None
    twostep_matches = 0
    single_matches = 0
    twostep_one_ulp = 0
    single_one_ulp = 0
    for i in range(NMAP):
        d = disp_codes(i)
        p = dirs / f"Result_{i}/depth_m.raw"
        flat = np.fromfile(p, "<f4")
        if flat.size != SIZE or not np.isfinite(flat).all() or np.any(flat <= 0):
            raise ValueError(f"Invalid map {i}")
        if np.any(flat[len(d) :] != flat[len(d) - 1]):
            raise ValueError("Nonidentical padded bin")
        z = flat[: len(d)]
        s = stats(z, d)
        idx = s["index"]
        s.update(
            {
                "map": i,
                "sha256": digest(p),
                "at_bits": hex(int(d[idx].view("u4"))),
                "row": idx // FULL[-1],
                "col": idx % FULL[-1],
            }
        )
        if worst is None or s["max_pct"] > worst["max_pct"]:
            worst = s.copy()
        total_loss += s["mean_pct"] * len(d)
        violations += s["at_or_over_0_15pct"]
        transitions += int(np.count_nonzero(np.diff(z))) + int(
            prev is not None and prev != z[0]
        )
        upwards += int((np.diff(z) > 0).sum()) + int(prev is not None and z[0] > prev)
        prev = z[-1]
        twice = (
            np.clip(1 / d.astype(np.float16).astype("f8"), LO, HI)
            .astype(np.float16)
            .astype("f4")
        )
        single = np.clip(1 / d.astype("f8"), LO, HI).astype(np.float16).astype("f4")
        twostep_matches += int((twice == z).sum())
        single_matches += int((single == z).sum())
        twostep_one_ulp += int((np.abs(twice - z) <= np.spacing(twice)).sum())
        single_one_ulp += int((np.abs(single - z) <= np.spacing(single)).sum())
        per.append(s)
    assert sum(s["pixels"] for s in per) == TOTAL
    result = {
        "count": TOTAL,
        "maps": NMAP,
        "criterion": "abs(device_z / clamp(1 / float64(disparity), .15, 25) - 1) < .0015",
        "mean_rel_pct": total_loss / TOTAL,
        "max_rel_pct": worst["max_pct"],
        "at_or_over_0_15pct": violations,
        "worst": worst,
        "distinct_adjacent_runs": transitions + 1,
        "monotonicity_upward_steps": upwards,
        "matches_CPU_RN16_input_and_reciprocal": twostep_matches,
        "matches_CPU_RN16_reciprocal_only": single_matches,
        "within_one_fp32_ulp_RN16_input_and_reciprocal": twostep_one_ulp,
        "within_one_fp32_ulp_RN16_reciprocal_only": single_one_ulp,
        "bridge": bridge,
        "ordered_output_file_records": per,
    }
    if upwards == 0:
        result["distinct_depth_levels"] = transitions + 1
    witness_input = DIR / "inputs" / f"codes_{worst['map']:03d}.raw"
    witness_device = dirs / f"Result_{worst['map']}/depth_m.raw"
    for path, name in (
        (witness_input, "worst_preinv.raw"),
        (witness_device, "worst_depth.raw"),
        (dirs / f"Result_{NMAP}/depth_m.raw", "bridge_depth.raw"),
    ):
        shutil.copyfile(path, DIR / name)
    result["witness_files_sha256"] = {
        k: digest(DIR / k)
        for k in ("worst_preinv.raw", "worst_depth.raw", "bridge_depth.raw")
    }
    write_json(DIR / "sweep.json", result)
    print(
        {k: v for k, v in result.items() if k not in {"ordered_output_file_records"}},
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="phase", required=True)
    b = sub.add_parser("prepare")
    b.add_argument("--sdk", type=Path, required=True)
    b.add_argument("--python", type=Path, required=True)
    r = sub.add_parser("device")
    r.add_argument("--adb", type=Path, required=True)
    r.add_argument("--serial", required=True)
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.phase == "prepare":
        prepare(args.sdk, args.python)
    elif args.phase == "device":
        run_device(args.adb, args.serial)
    else:
        analyze()
