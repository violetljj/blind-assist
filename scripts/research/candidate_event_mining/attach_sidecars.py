#!/usr/bin/env python3
"""Attach hash-bound model sidecars to an existing canonical frame trace.

This is a post-inference join. It never decodes frames or reruns the base
adapter; it only joins sidecar rows on the canonical
``source_id/session_id/frame_index`` identity and emits a new validated trace.
The accepted sidecars are the real SegFormer semantic sidecar and the real
HFTF student sidecar. Missing channels remain missing and duplicate signal
keys fail closed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.research.candidate_event_mining.pipeline import (
    ContractError,
    normalize_frames,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


ATTACHED_SCHEMA = "blindassist_candidate_event_mining_attached_sidecars_manifest_v1"
SEGMENTATION_SCHEMA = "blindassist_candidate_event_mining_segmentation_sidecar_manifest_v1"
HFTF_SCHEMA = "blindassist_candidate_event_mining_hftf_sidecar_manifest_v1"


def _key(row: dict[str, Any], where: str) -> tuple[str, str, int]:
    source_id = row.get("source_id")
    session_id = row.get("session_id")
    frame_index = row.get("frame_index")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ContractError(f"{where}: source_id must be a non-empty string")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ContractError(f"{where}: session_id must be a non-empty string")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ContractError(f"{where}: frame_index must be a non-negative integer")
    return source_id, session_id, frame_index


def _load_sidecar(
    path: Path,
    manifest_path: Path,
    input_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str]:
    manifest = read_json(manifest_path)
    schema = manifest.get("schema")
    if schema == SEGMENTATION_SCHEMA:
        prefix = "segmentation."
        real_flag = "real_semantic_segmentation"
    elif schema == HFTF_SCHEMA:
        prefix = "hftf."
        real_flag = "real_hftf_model_inference"
    else:
        raise ContractError(f"unsupported sidecar manifest schema: {schema!r}")
    if manifest.get(real_flag) is not True or manifest.get("proxy") is True:
        raise ContractError(f"sidecar is not marked as real non-proxy inference: {manifest_path}")
    manifest_input = manifest.get("input_trace")
    if not isinstance(manifest_input, dict) or manifest_input.get("sha256") != input_sha256:
        raise ContractError(f"sidecar input trace hash mismatch: {manifest_path}")
    manifest_sidecar = manifest.get("sidecar")
    if not isinstance(manifest_sidecar, dict) or manifest_sidecar.get("sha256") != sha256_file(path):
        raise ContractError(f"sidecar hash mismatch: {path}")
    rows = read_jsonl(path)
    expected_count = manifest_sidecar.get("frame_count")
    if expected_count != len(rows):
        raise ContractError(f"sidecar frame count mismatch: {path}")
    return rows, manifest, prefix, real_flag


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input_trace.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest_output.resolve()
    if output_path.exists() or manifest_path.exists():
        raise ContractError("refusing to overwrite attached sidecar outputs")
    if len(args.sidecar) != len(args.sidecar_manifest):
        raise ContractError("each --sidecar must have one matching --sidecar-manifest")
    if not args.sidecar:
        raise ContractError("at least one sidecar is required")

    base_rows = normalize_frames(read_jsonl(input_path))
    input_sha256 = sha256_file(input_path)
    base_by_key = {_key(row, "base trace"): row for row in base_rows}
    merged_by_key = {key: dict(row, signals=dict(row["signals"])) for key, row in base_by_key.items()}
    sidecar_receipts: list[dict[str, Any]] = []
    real_flags: list[str] = []
    added_signal_count = 0

    for sidecar_arg, manifest_arg in zip(args.sidecar, args.sidecar_manifest):
        sidecar_path = sidecar_arg.resolve()
        sidecar_manifest_path = manifest_arg.resolve()
        if not sidecar_path.is_file() or not sidecar_manifest_path.is_file():
            raise ContractError(f"sidecar or manifest is missing: {sidecar_path}, {sidecar_manifest_path}")
        rows, sidecar_manifest, prefix, real_flag = _load_sidecar(
            sidecar_path,
            sidecar_manifest_path,
            input_sha256,
        )
        seen: set[tuple[str, str, int]] = set()
        for index, row in enumerate(rows, 1):
            key = _key(row, f"sidecar {sidecar_path}:{index}")
            if key in seen:
                raise ContractError(f"duplicate sidecar frame identity: {key}")
            seen.add(key)
            if key not in merged_by_key:
                raise ContractError(f"sidecar frame is not present in base trace: {key}")
            signals = row.get("signals")
            if not isinstance(signals, dict) or not signals:
                raise ContractError(f"sidecar signals must be a non-empty object: {sidecar_path}:{index}")
            for signal_key, value in signals.items():
                if not isinstance(signal_key, str) or not signal_key.startswith(prefix):
                    raise ContractError(
                        f"sidecar signal {signal_key!r} does not match {prefix!r}: {sidecar_path}:{index}"
                    )
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ContractError(f"sidecar signal must be numeric: {signal_key}")
                value_float = float(value)
                if not math.isfinite(value_float) or not 0.0 <= value_float <= 1.0:
                    raise ContractError(f"sidecar signal must be finite and in [0,1]: {signal_key}")
                if signal_key in merged_by_key[key]["signals"]:
                    raise ContractError(f"sidecar would overwrite existing signal: {signal_key}")
                merged_by_key[key]["signals"][signal_key] = value_float
                added_signal_count += 1
        if seen != set(base_by_key):
            missing = sorted(set(base_by_key) - seen)[:3]
            extra = sorted(seen - set(base_by_key))[:3]
            raise ContractError(f"sidecar frame coverage mismatch: missing={missing}, extra={extra}")
        sidecar_receipts.append(
            {
                "path": str(sidecar_path),
                "sha256": sha256_file(sidecar_path),
                "manifest_path": str(sidecar_manifest_path),
                "manifest_sha256": sha256_file(sidecar_manifest_path),
                "schema": sidecar_manifest["schema"],
                "sidecar_id": sidecar_manifest.get("sidecar_id"),
                "frame_count": len(rows),
                "signal_prefix": prefix,
                "real_inference_flag": real_flag,
            }
        )
        real_flags.append(real_flag)

    merged_rows = normalize_frames(list(merged_by_key.values()))
    write_jsonl(output_path, merged_rows)
    manifest = {
        "schema": ATTACHED_SCHEMA,
        "attachment_id": "cem-real-segmentation-hftf-sidecars-r0",
        "input_trace": {
            "path": str(input_path),
            "sha256": input_sha256,
            "frame_count": len(base_rows),
        },
        "sidecars": sidecar_receipts,
        "output_trace": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "frame_count": len(merged_rows),
        },
        "real_sidecars_attached": sorted(real_flags),
        "signals_added_count": added_signal_count,
        "post_inference_join": True,
        "data_role": "THESIS_DEVELOPMENT_CONSUMED_DISCOVERY",
        "authorization": {
            "event_truth": False,
            "training": False,
            "production": False,
            "safety": False,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-trace", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, action="append", required=True)
    parser.add_argument("--sidecar-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        manifest = run(parse_args())
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "frame_count": manifest["output_trace"]["frame_count"],
                "signals_added_count": manifest["signals_added_count"],
                "output": manifest["output_trace"]["path"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
