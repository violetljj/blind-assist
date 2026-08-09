#!/usr/bin/env python3
"""Run one frozen D1 quality chunk on the bound SM8650 saved context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_DEPTH_BYTES = 608 * 448 * 4


def chunk_schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    chunk_size = int(protocol["execution"]["chunk_size_frames"])
    require(chunk_size == 50 and 300 % chunk_size == 0, "chunk size drift")
    return [
        {"chunk_index": session_index * (300 // chunk_size) + start // chunk_size,
         "session_index": session_index, "visit_id": identity["visit_id"],
         "video_id": identity["video_id"], "frame_start": start, "frame_stop": start + chunk_size}
        for session_index, identity in enumerate(protocol["cohort"]["ordered_sessions"])
        for start in range(0, 300, chunk_size)
    ]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def run(command: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None, check=False)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout or ''}")
    return result


def adb(serial: str, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["adb", "-s", serial, *arguments], check=check)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    require(not path.exists() and not temporary.exists(), f"output already exists: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_completed(receipt: dict[str, Any]) -> None:
    for output in receipt["outputs"]:
        path = Path(output["path"])
        require(path.is_file() and path.stat().st_size == output["bytes"] and sha256(path) == output["sha256"],
                f"completed output drift: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--runner-repair", type=Path)
    args = parser.parse_args()
    protocol_path, activation_path = args.protocol.resolve(), args.activation_receipt.resolve()
    protocol, activation = load_json(protocol_path), load_json(activation_path)
    require(activation.get("status") == "OUTCOME_ACCESS_ACTIVATED" and activation.get("execution_authorized") is True,
            "D1 outcome access is not activated")
    require(activation["protocol_sha256"] == sha256(protocol_path), "activation/protocol drift")
    runner_sha = sha256(Path(__file__))
    repair_identity = None
    if args.runner_repair is None:
        require(protocol["bindings"]["device_runner"]["sha256"] == runner_sha, "runner SHA drift")
    else:
        repair_path = args.runner_repair.resolve()
        repair = load_json(repair_path)
        require(repair.get("schema") == "blindassist_depthart_task_preserving_d1_quality_runner_repair_v1",
                "runner repair schema drift")
        require(repair.get("base_protocol_sha256") == sha256(protocol_path), "repair/base protocol drift")
        require(repair.get("activation_receipt_sha256") == sha256(activation_path), "repair/activation drift")
        require(repair.get("prior_runner_sha256") == protocol["bindings"]["device_runner"]["sha256"],
                "repair prior runner drift")
        require(repair.get("repaired_runner_sha256") == runner_sha, "repaired runner SHA drift")
        require(repair.get("only_change") == "ADSP_LIBRARY_PATH_SEPARATOR_COLON_TO_QUOTED_SEMICOLON",
                "repair scope drift")
        require(repair.get("candidate_data_policy_or_gate_changed") is False, "repair changed frozen semantics")
        repair_identity = {"path": str(repair_path), "sha256": sha256(repair_path)}
    chunks = chunk_schedule(protocol)
    require(0 <= args.chunk_index < len(chunks), "chunk outside frozen schedule")
    chunk = chunks[args.chunk_index]
    chunk_root = args.quality_root.resolve() / f"chunk-{args.chunk_index:02d}"
    materialization_path = chunk_root / "materialization-receipt.json"
    materialization = load_json(materialization_path)
    require(materialization["chunk"] == chunk, "materialization chunk drift")
    receipt_path = chunk_root / "device-run-receipt.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        verify_completed(receipt)
        print(json.dumps({"chunk": args.chunk_index, "status": "RESUMED_VALID", "outputs": len(receipt["outputs"])}))
        return 0

    serial = protocol["device"]["serial"]
    state = adb(serial, "get-state").stdout.strip()
    require(state == "device", f"device is not ready: {state}")
    properties = {
        "build_fingerprint": adb(serial, "shell", "getprop", "ro.build.fingerprint").stdout.strip(),
        "model": adb(serial, "shell", "getprop", "ro.product.model").stdout.strip(),
        "device": adb(serial, "shell", "getprop", "ro.product.device").stdout.strip(),
        "soc": adb(serial, "shell", "getprop", "ro.soc.model").stdout.strip(),
        "abi": adb(serial, "shell", "getprop", "ro.product.cpu.abi").stdout.strip(),
    }
    for key, value in properties.items():
        require(value == str(protocol["device"][key]), f"device {key} drift: {value}")

    source_remote = protocol["device"]["validated_preflight_workspace"]
    for asset in protocol["device"]["remote_assets"]:
        output = adb(serial, "shell", "sha256sum", f"{source_remote}/{asset['path']}").stdout.strip().split()
        require(output and output[0].upper() == asset["sha256"], f"remote bound asset drift: {asset['path']}")

    remote_base = str(protocol["execution"]["remote_quality_root"])
    require(remote_base.startswith("/data/local/tmp/depthart_d1_quality_") and ".." not in remote_base,
            "unsafe remote quality root")
    remote_root = f"{remote_base}/chunk-{args.chunk_index:02d}"
    adb(serial, "shell", "rm", "-rf", remote_root)
    adb(serial, "shell", "mkdir", "-p", remote_root)
    adb(serial, "push", str(chunk_root / "inputs"), f"{remote_root}/inputs")
    adb(serial, "push", str(chunk_root / "input-list.txt"), f"{remote_root}/input-list.txt")

    expected_hashes = {record["candidate_inputs"][name]["path"].replace(str(chunk_root) + os.sep, "").replace(os.sep, "/"):
                       record["candidate_inputs"][name]["sha256"]
                       for record in materialization["records"] for name in record["candidate_inputs"]}
    expected_hashes["input-list.txt"] = materialization["input_list"]["sha256"]
    remote_hash_output = adb(serial, "shell",
                             f"cd {remote_root} && find inputs -type f -name '*.raw' -print0 | sort -z | xargs -0 sha256sum && sha256sum input-list.txt").stdout
    observed = {}
    for line in remote_hash_output.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            observed[parts[1].lstrip("*./")] = parts[0].upper()
    require(observed == expected_hashes, "device input SHA set differs from host receipt")

    provider = "DepthArtSelectiveScanPackageInterfaceProvider"
    source = source_remote
    command = (
        f"cd {remote_root} && "
        f"export LD_LIBRARY_PATH={source}/arm64 && "
        f"export ADSP_LIBRARY_PATH='{source}/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && "
        f"{source}/arm64/qnn-net-run --backend {source}/arm64/libQnnHtp.so "
        f"--retrieve_context {source}/context/depthart-d1-608x448-sm8650-v75.bin "
        f"--input_list input-list.txt --output_dir output "
        f"--op_packages libQnnDepthArtSelectiveScanPackage.so:{provider}:CPU,"
        f"libQnnDepthArtSelectiveScanPackage.so:{provider}:HTP --log_level info"
    )
    result = adb(serial, "shell", command, check=False)
    log_path = chunk_root / "device-run.log"
    log_path.write_text(result.stdout or "", encoding="utf-8", newline="\n")
    require(result.returncode == 0, f"qnn-net-run failed: {result.returncode}\n{result.stdout}")
    local_output = chunk_root / "candidate-output"
    require(not local_output.exists(), f"candidate output already exists: {local_output}")
    adb(serial, "pull", f"{remote_root}/output", str(local_output))
    outputs = []
    records = materialization["records"]
    for index, record in enumerate(records):
        candidates = sorted((local_output / f"Result_{index}").glob("*.raw"))
        require(len(candidates) == 1, f"expected one output for Result_{index}")
        path = candidates[0]
        require(path.stat().st_size == EXPECTED_DEPTH_BYTES, f"candidate output byte drift: {path}")
        values = np.fromfile(path, dtype=np.float32)
        require(values.size == 608 * 448 and np.all(np.isfinite(values)), f"candidate output invalid: {path}")
        outputs.append({"frame_id": record["frame_id"], "result_index": index,
                        "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path),
                        "shape": [1, 608, 448], "dtype": "float32"})
    receipt = {
        "schema": "blindassist_depthart_task_preserving_d1_quality_device_chunk_v1",
        "protocol_sha256": sha256(protocol_path), "activation_receipt_sha256": sha256(activation_path),
        "materialization_receipt_sha256": sha256(materialization_path), "chunk": chunk,
        "runner_repair": repair_identity,
        "device": properties, "remote_root": remote_root,
        "remote_input_sha256_verified": True,
        "command": command, "exit_code": result.returncode,
        "log": {"path": str(log_path.resolve()), "bytes": log_path.stat().st_size, "sha256": sha256(log_path)},
        "outputs": outputs, "candidate_outcome_accessed": True, "r2_cohort_accessed": False,
    }
    atomic_json(receipt_path, receipt)
    adb(serial, "shell", "rm", "-rf", remote_root)
    print(json.dumps({"chunk": args.chunk_index, "status": "DEVICE_PASS", "outputs": len(outputs),
                      "receipt_sha256": sha256(receipt_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
