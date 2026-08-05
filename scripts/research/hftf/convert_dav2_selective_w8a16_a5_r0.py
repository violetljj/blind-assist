#!/usr/bin/env python3
"""Convert the single frozen DA V2 A5 selective W8A16 DLC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def converter_command(
    python: Path,
    converter: Path,
    onnx_path: Path,
    override_path: Path,
    output_path: Path,
) -> list[str]:
    return [
        str(python),
        str(converter),
        "--input_network",
        str(onnx_path),
        "--output_path",
        str(output_path),
        "--quantization_overrides",
        str(override_path),
        "--float_bitwidth",
        "16",
        "--float_bias_bitwidth",
        "16",
        "--target_backend",
        "HTP",
        "--target_soc_model",
        "SM8650",
        "--quantizer_log",
        str(output_path.with_suffix(".quantizer.csv")),
        "--quantizer_log_level",
        "INFO",
    ]


def qairt_environment(qairt_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    bin_path = qairt_root / "bin" / "x86_64-windows-msvc"
    lib_path = qairt_root / "lib" / "x86_64-windows-msvc"
    python_path = qairt_root / "lib" / "python"
    environment["QAIRT_SDK_ROOT"] = str(qairt_root)
    environment["QNN_SDK_ROOT"] = str(qairt_root)
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONPATH"] = str(python_path)
    environment["PATH"] = os.pathsep.join(
        [str(bin_path), str(lib_path), environment.get("PATH", "")]
    )
    return environment


def run_logged(command: list[str], environment: dict[str, str], log_path: Path) -> None:
    result = subprocess.run(
        command,
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        json.dumps({"command": command, "returncode": result.returncode}, indent=2)
        + "\n\nSTDOUT\n"
        + result.stdout
        + "\nSTDERR\n"
        + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"A5 command failed with {result.returncode}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--qairt-root", type=Path, required=True)
    parser.add_argument("--qairt-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_A5_DLC_CONVERSION":
        raise ValueError("A5 protocol is not frozen")
    bindings = protocol["bindings"]
    for path, key in (
        (args.onnx, "onnx_sha256"),
        (args.overrides, "override_sha256"),
        (Path(__file__).resolve(), "converter_source_sha256"),
    ):
        if sha256_file(path) != bindings[key]:
            raise ValueError(f"A5 hash mismatch: {key}")
    qairt_root = args.qairt_root.resolve()
    converter = qairt_root / "bin" / "x86_64-windows-msvc" / "qairt-converter"
    dlc_info = qairt_root / "bin" / "x86_64-windows-msvc" / "qairt-dlc-info"
    if not converter.is_file() or not dlc_info.is_file() or not args.qairt_python.is_file():
        raise FileNotFoundError("A5 QAIRT runtime is incomplete")
    args.output_root.mkdir(parents=True)
    output_path = args.output_root / "dav2-a5-selective-w8a16.dlc"
    environment = qairt_environment(qairt_root)
    command = converter_command(
        args.qairt_python.resolve(),
        converter,
        args.onnx.resolve(),
        args.overrides.resolve(),
        output_path.resolve(),
    )
    run_logged(command, environment, args.output_root / "converter.log")
    run_logged(
        [str(args.qairt_python.resolve()), str(dlc_info), "-i", str(output_path.resolve())],
        environment,
        args.output_root / "dlc-info.log",
    )
    receipt: dict[str, Any] = {
        "schema": "blindassist_dav2_selective_w8a16_a5_r0_conversion_receipt",
        "protocol_sha256": sha256_file(args.protocol),
        "onnx_sha256": sha256_file(args.onnx),
        "override_sha256": sha256_file(args.overrides),
        "converter_source_sha256": sha256_file(Path(__file__).resolve()),
        "qairt_version": "2.47.0.260601",
        "target_backend": "HTP",
        "target_soc_model": "SM8650",
        "float_bitwidth": 16,
        "float_bias_bitwidth": 16,
        "dlc": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "converter_log_sha256": sha256_file(args.output_root / "converter.log"),
        "dlc_info_log_sha256": sha256_file(args.output_root / "dlc-info.log"),
    }
    receipt_path = args.output_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
