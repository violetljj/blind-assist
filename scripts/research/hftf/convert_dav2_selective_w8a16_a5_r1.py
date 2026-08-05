#!/usr/bin/env python3
"""Convert A5S with generic HTP after host SoC-name preflight rejection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert_dav2_selective_w8a16_a5_r0 import (
    qairt_environment,
    run_logged,
    sha256_file,
)


def generic_htp_command(
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
        "--quantizer_log",
        str(output_path.with_suffix(".quantizer.csv")),
        "--quantizer_log_level",
        "INFO",
    ]


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
    if protocol.get("status") != "FROZEN_BEFORE_A5_R1_GENERIC_HTP_CONVERSION":
        raise ValueError("A5S R1 protocol is not frozen")
    bindings = protocol["bindings"]
    for path, key in (
        (args.onnx, "onnx_sha256"),
        (args.overrides, "override_sha256"),
        (Path(__file__).resolve(), "converter_r1_source_sha256"),
    ):
        if sha256_file(path) != bindings[key]:
            raise ValueError(f"A5S R1 hash mismatch: {key}")
    qairt_root = args.qairt_root.resolve()
    converter = qairt_root / "bin" / "x86_64-windows-msvc" / "qairt-converter"
    dlc_info = qairt_root / "bin" / "x86_64-windows-msvc" / "qairt-dlc-info"
    if not converter.is_file() or not dlc_info.is_file() or not args.qairt_python.is_file():
        raise FileNotFoundError("A5S R1 QAIRT runtime is incomplete")
    args.output_root.mkdir(parents=True)
    output_path = args.output_root / "dav2-a5s-r1-selective-w8a16.dlc"
    environment = qairt_environment(qairt_root)
    run_logged(
        generic_htp_command(
            args.qairt_python.resolve(),
            converter,
            args.onnx.resolve(),
            args.overrides.resolve(),
            output_path.resolve(),
        ),
        environment,
        args.output_root / "converter.log",
    )
    run_logged(
        [str(args.qairt_python.resolve()), str(dlc_info), "-i", str(output_path.resolve())],
        environment,
        args.output_root / "dlc-info.log",
    )
    receipt = {
        "schema": "blindassist_dav2_selective_w8a16_a5s_r1_conversion_receipt",
        "protocol_sha256": sha256_file(args.protocol),
        "onnx_sha256": sha256_file(args.onnx),
        "override_sha256": sha256_file(args.overrides),
        "converter_r1_source_sha256": sha256_file(Path(__file__).resolve()),
        "qairt_version": "2.47.0.260601",
        "target_backend": "GENERIC_HTP",
        "target_device_cache_required": True,
        "dlc": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "converter_log_sha256": sha256_file(args.output_root / "converter.log"),
        "dlc_info_log_sha256": sha256_file(args.output_root / "dlc-info.log"),
    }
    (args.output_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
