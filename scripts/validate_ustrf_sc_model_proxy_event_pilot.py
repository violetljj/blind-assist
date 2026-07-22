#!/usr/bin/env python3
"""Stable adapter for the USTRF model-proxy route-event pilot validator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path


_IMPLEMENTATION_PATH = Path(__file__).parent / "research" / "ustrf_sc" / "validate_model_proxy_event_pilot.py"
_IMPLEMENTATION_SPEC = importlib.util.spec_from_file_location("ustrf_sc_model_proxy_event_pilot_validator", _IMPLEMENTATION_PATH)
if _IMPLEMENTATION_SPEC is None or _IMPLEMENTATION_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load model-proxy validator: {_IMPLEMENTATION_PATH}")
_IMPLEMENTATION = importlib.util.module_from_spec(_IMPLEMENTATION_SPEC)
_IMPLEMENTATION_SPEC.loader.exec_module(_IMPLEMENTATION)

ContractError = _IMPLEMENTATION.ContractError
decoded_rows = _IMPLEMENTATION.decoded_rows
probe_image = _IMPLEMENTATION.probe_image
probe_media = _IMPLEMENTATION.probe_media
sha256 = _IMPLEMENTATION.sha256
toolchain_receipt = _IMPLEMENTATION.toolchain_receipt
validate = _IMPLEMENTATION.validate


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config_path = args.config.resolve()
        manifest_path = args.manifest.resolve()
        ffmpeg_path = args.ffmpeg.resolve()
        ffprobe_path = args.ffprobe.resolve()
        output_path = args.output.resolve()
        if output_path in {config_path, manifest_path, ffmpeg_path, ffprobe_path}:
            raise ContractError("output must not overwrite an input or media tool")
        if not ffmpeg_path.is_file() or not ffprobe_path.is_file():
            raise ContractError("ffmpeg and ffprobe must be local files")
        actual_toolchain = toolchain_receipt(ffmpeg_path, ffprobe_path)
        report = validate(
            load(config_path),
            load(manifest_path),
            root=manifest_path.parent,
            config_sha256=sha256(config_path),
            manifest_sha256=sha256(manifest_path),
            actual_toolchain=actual_toolchain,
            frame_provider=lambda path: decoded_rows(ffmpeg_path, path),
            media_provider=lambda path: probe_media(ffprobe_path, path),
            image_provider=lambda path: probe_image(ffprobe_path, path),
        )
    except (
        ContractError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        subprocess.SubprocessError,
    ) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "proxy_full_matrix_expansion_eligible": report["proxy_full_matrix_expansion_eligible"],
        "proxy_u0_evaluation_eligible": report["proxy_u0_evaluation_eligible"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
