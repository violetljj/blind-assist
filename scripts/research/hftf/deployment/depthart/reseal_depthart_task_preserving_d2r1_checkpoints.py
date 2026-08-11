#!/usr/bin/env python3
"""Reseal unchanged D2R1 checkpoints after documented Windows CRLF translation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "blindassist_depthart_task_preserving_d2r1_checkpoint_reseal_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object required")
    return value


def verify_pair(receipt: Path, old_sidecar: Path, expected: dict[str, Any]) -> dict[str, Any]:
    actual_value = load_json(receipt)
    require(actual_value == expected, f"receipt semantic drift: {receipt}")
    old_seal = load_json(old_sidecar)
    canonical = canonical_json_bytes(actual_value)
    canonical_sha = hashlib.sha256(canonical).hexdigest().upper()
    require(len(canonical) == int(old_seal["bytes"]), f"old sidecar canonical byte drift: {receipt}")
    require(canonical_sha == old_seal["sha256"], f"old sidecar canonical SHA drift: {receipt}")
    actual = receipt.read_bytes()
    require(actual.replace(b"\r\n", b"\n") == canonical, f"translation is not pure CRLF: {receipt}")
    return {
        "receipt_name": receipt.name,
        "old_sidecar_name": old_sidecar.name,
        "old_canonical_lf_bytes": int(old_seal["bytes"]),
        "old_canonical_lf_sha256": old_seal["sha256"],
        "actual_crlf_bytes": len(actual),
        "actual_crlf_sha256": hashlib.sha256(actual).hexdigest().upper(),
        "semantic_manifest_match": True,
        "only_lf_to_crlf_translation": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")
    manifest = load_json(args.manifest)
    rows = []
    for video in manifest["videos"]:
        receipt = args.receipt_root / f"{int(video['phase_a_order']):02d}-{video['video_id']}.json"
        sidecar = receipt.with_suffix(".sha256.json")
        rows.append(verify_pair(receipt, sidecar, video))
    require(len(rows) == 16, "reseal row count drift")
    value = {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "reason": "Windows text-mode LF-to-CRLF translation occurred after the v1 sidecar hashed canonical JSON bytes",
        "body_protocol_sha256": sha256_file(args.body_protocol),
        "manifest_bytes": args.manifest.stat().st_size,
        "manifest_sha256": sha256_file(args.manifest),
        "receipt_count": len(rows),
        "receipts": rows,
        "receipt_content_modified": False,
        "threshold_or_role_change": False,
        "truth_or_model_reexecution": False,
        "authority": "CHECKPOINT_BYTE_RESEAL_ONLY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in value.items() if key != "receipts"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
