#!/usr/bin/env python3
"""Run exactly one frozen 50-frame D2 Development saved-context chunk."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (  # noqa: E402
    TruthReaderPolicy,
    load_manifest_frame,
    parse_trajectory,
)
from scripts.research.hftf.deployment.depthart.export_depthart_camera_external import (  # noqa: E402
    install_timm_compat,
)
from scripts.research.hftf.deployment.depthart.run_depthart_task_preserving_d2_train_only import (  # noqa: E402
    EXPECTED_DEPTH_BYTES,
    INPUT_NAMES,
    adb,
    atomic_json,
    atomic_raw,
    compact_truth,
    load_json,
    require,
    run,
    sha256,
    verify_device,
    verify_file_binding,
)


PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D2_DEVELOPMENT_BASELINE_AND_FROZEN_HEAD_QUALITY"


def chunk_schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    chunk_size = int(protocol["execution"]["chunk_size_frames"])
    require(chunk_size == 50 and 300 % chunk_size == 0, "chunk size drift")
    return [
        {
            "chunk_index": session_index * (300 // chunk_size) + start // chunk_size,
            "session_index": session_index,
            "visit_id": identity["visit_id"],
            "video_id": identity["video_id"],
            "frame_start": start,
            "frame_stop": start + chunk_size,
        }
        for session_index, identity in enumerate(protocol["development_scope"])
        for start in range(0, 300, chunk_size)
    ]


def load_development_video(manifest: dict[str, Any], protocol: dict[str, Any], session_index: int) -> dict[str, Any]:
    roles = [row for row in manifest["roles"] if row["role"] == "D2_DEVELOPMENT_SEALED"]
    require(
        [(row["visit_id"], row["video_id"]) for row in roles]
        == [(row["visit_id"], row["video_id"]) for row in protocol["development_scope"]],
        "Development roster drift",
    )
    role = roles[session_index]
    checkpoint_path = Path(role["checkpoint_path"])
    require(sha256(checkpoint_path) == role["checkpoint_sha256"], "Development checkpoint SHA drift")
    video = load_json(checkpoint_path)
    require(
        video["role"] == "D2_DEVELOPMENT_SEALED"
        and video["visit_id"] == role["visit_id"]
        and video["video_id"] == role["video_id"],
        "Development checkpoint role drift",
    )
    require(len(video["selected_frame_stems"]) == 300, "Development frame count drift")
    expected_stems = hashlib_stems(video["selected_frame_stems"])
    require(expected_stems == protocol["development_scope"][session_index]["frame_stems_sha256"],
            "Development frame schedule drift")
    for entries in video["extracted"].values():
        require(len(entries) == 300, "Development modality count drift")
        require(all("source\\development\\" in str(entry["path"]).lower() for entry in entries),
                "non-Development source path crossed execution scope")
    return video


def hashlib_stems(stems: list[str]) -> str:
    import hashlib
    return hashlib.sha256(("\n".join(stems) + "\n").encode()).hexdigest().upper()


def materialize_chunk(chunk: dict[str, Any], video: dict[str, Any], chunk_root: Path,
                      model: torch.nn.Module, preprocess: Any) -> dict[str, Any]:
    receipt_path = chunk_root / "materialization-receipt.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        require(receipt["chunk"] == chunk, "materialization schedule drift")
        return receipt
    require(not chunk_root.exists(), f"partial chunk without receipt: {chunk_root}")
    chunk_root.mkdir(parents=True)
    trajectory = parse_trajectory(Path(video["trajectory"]["path"]))
    records: list[dict[str, Any]] = []
    input_lines: list[str] = []
    for frame_index in range(int(chunk["frame_start"]), int(chunk["frame_stop"])):
        frame = load_manifest_frame(video, frame_index, trajectory, TruthReaderPolicy())
        require(int(frame["orientation"]["rotation_index"]) in (1, 3), "non-portrait Development frame")
        bgr = cv2.cvtColor(np.asarray(frame["rgb_upright"], dtype=np.uint8), cv2.COLOR_RGB2BGR)
        image, intrinsics = preprocess(
            bgr, np.asarray(frame["intrinsics_upright"], dtype=np.float32), 448, 448
        )
        require(tuple(image.shape) == (1, 3, 608, 448), "input tensor shape drift")
        image, intrinsics = image.cuda(), intrinsics.cuda()
        with torch.inference_mode():
            cameras = model.cam_embedder(intrinsics, 608, 448, "cuda")
        stem = frame["identity"]["frame_stem"]
        require(stem == video["selected_frame_stems"][frame_index], "selected frame stem drift")
        frame_root = chunk_root / "inputs" / stem
        arrays = {
            "image": image.detach().cpu().numpy(),
            **{
                name: value.detach().cpu().numpy()
                for name, value in zip(INPUT_NAMES[1:], cameras, strict=True)
            },
        }
        input_receipts = {name: atomic_raw(frame_root / f"{name}.raw", arrays[name]) for name in INPUT_NAMES}
        records.append({
            "parent_id": video["visit_id"],
            "session_id": video["video_id"],
            "frame_id": stem,
            "frame_index": frame_index,
            "timestamp_ns": int(Decimal(stem.rsplit("_", 1)[1]) * 1_000_000_000),
            "orientation_index": int(frame["orientation"]["rotation_index"]),
            "up_camera": frame["orientation"]["up_camera"],
            "intrinsics_tensor": intrinsics.detach().cpu().numpy()[0].tolist(),
            "truth_bands": compact_truth(frame["truth"]),
            "candidate_inputs": input_receipts,
        })
        input_lines.append(" ".join(f"{name}:=inputs/{stem}/{name}.raw" for name in INPUT_NAMES))
    input_list = chunk_root / "input-list.txt"
    input_list.write_text("\n".join(input_lines) + "\n", encoding="utf-8", newline="\n")
    receipt = {
        "schema": "blindassist_depthart_task_preserving_d2_development_chunk_materialization_v1",
        "chunk": chunk,
        "records": records,
        "input_list": {
            "path": str(input_list.resolve()), "bytes": input_list.stat().st_size,
            "sha256": sha256(input_list),
        },
        "development_truth_accessed": True,
        "reference_model_output_accessed": False,
        "training_or_tuning": False,
        "r2_accessed": False,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def run_device_chunk(protocol: dict[str, Any], chunk: dict[str, Any], chunk_root: Path,
                     materialization: dict[str, Any]) -> dict[str, Any]:
    receipt_path = chunk_root / "device-run-receipt.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        require(receipt["chunk"] == chunk, "device receipt schedule drift")
        for output in receipt["outputs"]:
            path = Path(output["path"])
            require(
                path.is_file() and path.stat().st_size == output["bytes"] and sha256(path) == output["sha256"],
                f"completed output drift: {path}",
            )
        return receipt
    properties = verify_device(protocol)
    serial = protocol["device"]["serial"]
    remote_base = protocol["execution"]["remote_root"]
    require(remote_base.startswith("/data/local/tmp/depthart_d2_development_") and ".." not in remote_base,
            "unsafe remote root")
    remote_root = f"{remote_base}/chunk-{chunk['chunk_index']:02d}"
    adb(serial, "shell", "rm", "-rf", remote_root)
    adb(serial, "shell", "mkdir", "-p", remote_root)
    adb(serial, "push", str(chunk_root / "inputs"), f"{remote_root}/inputs")
    adb(serial, "push", str(chunk_root / "input-list.txt"), f"{remote_root}/input-list.txt")
    expected = {
        record["candidate_inputs"][name]["path"].replace(str(chunk_root) + os.sep, "").replace(os.sep, "/"):
        record["candidate_inputs"][name]["sha256"]
        for record in materialization["records"] for name in INPUT_NAMES
    }
    expected["input-list.txt"] = materialization["input_list"]["sha256"]
    output = adb(
        serial, "shell",
        f"cd {remote_root} && find inputs -type f -name '*.raw' -print0 | sort -z | "
        "xargs -0 sha256sum && sha256sum input-list.txt",
    ).stdout
    observed: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            observed[parts[1].lstrip("*./")] = parts[0].upper()
    require(observed == expected, "device input SHA set differs from host receipt")
    source = protocol["device"]["validated_preflight_workspace"]
    provider = "DepthArtSelectiveScanPackageInterfaceProvider"
    command = (
        f"cd {remote_root} && export LD_LIBRARY_PATH={source}/arm64 && "
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
    require(not local_output.exists(), f"candidate output exists without receipt: {local_output}")
    adb(serial, "pull", f"{remote_root}/output", str(local_output))
    outputs = []
    for index, record in enumerate(materialization["records"]):
        candidates = sorted((local_output / f"Result_{index}").glob("*.raw"))
        require(len(candidates) == 1, f"expected one output for Result_{index}")
        path = candidates[0]
        values = np.fromfile(path, dtype=np.float32)
        require(
            path.stat().st_size == EXPECTED_DEPTH_BYTES and values.size == 608 * 448
            and np.all(np.isfinite(values)),
            f"candidate output invalid: {path}",
        )
        outputs.append({
            "frame_id": record["frame_id"], "result_index": index,
            "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path),
            "shape": [1, 608, 448], "dtype": "float32",
        })
    receipt = {
        "schema": "blindassist_depthart_task_preserving_d2_development_device_chunk_v1",
        "chunk": chunk, "device": properties, "remote_root": remote_root,
        "remote_input_sha256_verified": True, "command": command, "exit_code": result.returncode,
        "log": {"path": str(log_path.resolve()), "bytes": log_path.stat().st_size, "sha256": sha256(log_path)},
        "outputs": outputs, "development_base_output_accessed": True,
        "training_or_tuning": False, "performance_measured": False, "r2_accessed": False,
    }
    atomic_json(receipt_path, receipt)
    adb(serial, "shell", "rm", "-rf", remote_root)
    shutil.rmtree(chunk_root / "inputs")
    atomic_json(chunk_root / "input-cleanup-receipt.json", {
        "generated_inputs_removed_after_verified_device_pass": True,
        "materialization_receipt_sha256": sha256(chunk_root / "materialization-receipt.json"),
        "device_receipt_sha256": sha256(receipt_path),
    })
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    activation_path = args.activation_receipt.resolve()
    protocol, activation = load_json(protocol_path), load_json(activation_path)
    require(protocol["protocol_id"] == PROTOCOL_ID and protocol["status"] == "FROZEN_DEVELOPMENT_QUALITY_EXECUTION",
            "execution protocol drift")
    require(activation["status"] == "DEVELOPMENT_QUALITY_EXECUTION_ACTIVATED"
            and activation["execution_authorized"] is True, "Development execution is not activated")
    require(activation["protocol_sha256"] == sha256(protocol_path), "activation/protocol drift")
    require(protocol["bindings"]["chunk_runner"]["sha256"] == sha256(Path(__file__)), "runner SHA drift")
    for binding in protocol["bindings"].values():
        if isinstance(binding, dict) and {"path", "bytes", "sha256"}.issubset(binding):
            verify_file_binding(binding)
    require(sha256(args.reference_checkpoint.resolve()) == protocol["reference"]["checkpoint_sha256"],
            "reference checkpoint SHA drift")
    require(protocol["truth_reader_policy"] == json.loads(json.dumps(asdict(TruthReaderPolicy()))),
            "truth reader policy drift")
    schedule = chunk_schedule(protocol)
    require(0 <= args.chunk_index < len(schedule), "chunk index out of range")
    chunk = schedule[args.chunk_index]
    output_root = args.output_root.resolve()
    attempt_path = output_root / "attempt.json"
    attempt = {
        "schema": "blindassist_depthart_task_preserving_d2_development_attempt_v1",
        "protocol_sha256": sha256(protocol_path),
        "activation_receipt_sha256": sha256(activation_path),
        "chunk_runner_sha256": sha256(Path(__file__)),
        "development_identity_count": 4, "frame_count": 1200,
        "training_or_tuning": False, "r2_accessed": False,
    }
    if attempt_path.exists():
        require(load_json(attempt_path) == attempt, "attempt binding drift")
    else:
        require(not output_root.exists(), f"output root exists without attempt: {output_root}")
        output_root.mkdir(parents=True)
        atomic_json(attempt_path, attempt)
    chunk_root = output_root / f"chunk-{chunk['chunk_index']:02d}"
    if (chunk_root / "device-run-receipt.json").exists():
        device = run_device_chunk(protocol, chunk, chunk_root, load_json(chunk_root / "materialization-receipt.json"))
        print(json.dumps({"completed": chunk["chunk_index"] + 1, "total": len(schedule),
                          "status": "RESUMED_VALID", "outputs": len(device["outputs"])}))
        return 0

    source_manifest = load_json(Path(protocol["bindings"]["source_manifest"]["path"]))
    video = load_development_video(source_manifest, protocol, int(chunk["session_index"]))
    install_timm_compat()
    source = args.source_root.resolve()
    require(run(["git", "-C", str(source), "rev-parse", "HEAD"]).stdout.strip()
            == protocol["reference"]["source_git_commit"], "DepthART source commit drift")
    require(not run(["git", "-C", str(source), "status", "--short"]).stdout.strip(),
            "DepthART source tree is dirty")
    runtime = protocol["reference"]["host_runtime"]
    require(torch.__version__ == runtime["torch"] and torch.version.cuda == runtime["cuda"]
            and cv2.__version__ == runtime["opencv"] and np.__version__ == runtime["numpy"],
            "host runtime drift")
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
    from common import preprocess as official_preprocess  # type: ignore
    from depthart_selective_scan import install_depthart  # type: ignore
    from model import load_model  # type: ignore
    from network import tvimblock  # type: ignore
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    model = load_model(args.reference_checkpoint.resolve(), "S", "indoor", "cuda").eval()
    install_depthart(tvimblock)
    materialization = materialize_chunk(chunk, video, chunk_root, model, official_preprocess)
    device = run_device_chunk(protocol, chunk, chunk_root, materialization)
    print(json.dumps({
        "completed": chunk["chunk_index"] + 1, "total": len(schedule),
        "visit_id": chunk["visit_id"], "video_id": chunk["video_id"],
        "status": "DEVICE_PASS", "outputs": len(device["outputs"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
